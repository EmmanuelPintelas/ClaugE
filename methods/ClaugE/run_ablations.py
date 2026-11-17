"""ClaugE ablations, on the two extremes of the four benchmarks (Camelyon17 and RxRx1).

    python run_ablations.py prepare         --benchmark camelyon17 --variant random
    python run_ablations.py train           --benchmark camelyon17 --variant random
    python run_ablations.py committee       --benchmark camelyon17 --variant random
    python run_ablations.py committee-level --benchmark camelyon17
    python run_ablations.py paths           --benchmark camelyon17
    python run_ablations.py table           --benchmark camelyon17

Variants that retrain the members (prepare, then train, then committee):

  nopartition   every path trains under the whole palette: one recipe of all the draws
  random        the draws split at random into k groups of the partition's own sizes
  data          the training images split into k groups (k-means of the frozen encoder's
                features); path c trains on group c alone, under the whole palette
  oneref        the partition of the first reference image alone, without consensus
  ssl           the partition redone with a self-distilled encoder (DINO ResNet-50)
  identity0     the partition's recipes with identity share 0 instead of 1/k
  identity0.5   the partition's recipes with identity share 1/2 instead of 1/k
  k20           the finer partition: k = 20 paths, one candidate each, at k_max = 25

prepare builds a variant's recipes in work/<benchmark>/ablations/<variant>/ (for data also
the image groups, for ssl the whole partition again: about two hours); train trains its
pools in pools_ablations/<benchmark>/<backbone>/<variant>/recipe_<c>/ with the main pools'
seeds, so that member j of a variant has the training settings and seeds of ClaugE's member
j (resumable); committee writes its committees, with alpha and the fit radius chosen on the
selection split as for ClaugE, to outputs_ablations/<benchmark>_<backbone>_kmax<k>_<variant>/,
and, when the variant trains fewer candidates per path than ClaugE (configs/ablations.json),
the reference: ClaugE restricted to the same first candidates of its own pools.

Committee-level variants need no training and are computed from the main pools: random
candidates per path instead of the best on validation, weights per member instead of per
path, uniform weights, fixed alpha = 1 and r = 1/4, the prior alone, the unbounded fit, the
committee without the path worst on validation, and the budget of five spent on the best
path alone (committee-level -> outputs_ablations/<benchmark>_committee_level.json).

paths reports what every training path yields on its own (no training either): the test
score of single networks by recipe, next to the networks of the retrained variants under the
same settings and seeds, the five-candidate model of each path, and two-network ensembles
within and across the two paths best on validation (-> outputs_ablations/<benchmark>_paths.json).

table prints every available variant minus its reference, with paired 95% intervals from
the cluster bootstrap of the shared bench, also with equal weights for the retrained
variants, and writes outputs_ablations/<benchmark>_ablations.json.

committee-level, paths and table read predictions one network at a time and can run beside
a training.
"""
from __future__ import annotations

import argparse
import collections
import itertools
import json
import math
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_claugE as RC  # noqa: E402
from claugE import committee as Cm  # noqa: E402
from claugE import drawer, partition, recipes  # noqa: E402
from claugE.partition import log  # noqa: E402

sys.path.insert(0, RC.PROJECT)
from bench import evaluate as E  # noqa: E402

TRAINED = {
    "nopartition": {"k": 5, "recipes": "whole",
                    "row": "whole palette for every member (no partition)"},
    "random": {"k": 5, "recipes": "random",
               "row": "random partition of the transforms into k groups"},
    "data": {"k": 5, "recipes": "whole", "data": True,
             "row": "partition of the data instead of the augmentations"},
    "oneref": {"k": 5, "recipes": "oneref", "row": "one reference image instead of ten"},
    "ssl": {"k": 5, "recipes": "ssl",
            "row": "self-supervised encoder instead of the supervised ResNet-50"},
    "identity0": {"k": 5, "recipes": "main", "identity_p": 0.0,
                  "row": "identity share 0 instead of 1/k"},
    "identity0.5": {"k": 5, "recipes": "main", "identity_p": 0.5,
                    "row": "identity share 1/2 instead of 1/k"},
    "k20": {"k": 20, "recipes": "main", "candidates": 1,
            "row": "k = 20 paths with one candidate each, at k_max = 25"},
}


def ablation_settings():
    with open(os.path.join(HERE, "configs", "ablations.json"), encoding="utf-8") as f:
        return json.load(f)


def variant_dir(b, v):
    return os.path.join(RC.work_dir(b), "ablations", v)


def variant_pool(b, bb, v, c):
    return os.path.join(HERE, "pools_ablations", b, bb, v, f"recipe_{c}")


def ablation_out(b, bb, kmax, name):
    return os.path.join(HERE, "outputs_ablations", f"{b}_{bb}_kmax{kmax}_{name}")


def main_committee(b, bb, kmax):
    return os.path.join(HERE, "outputs", f"{b}_{bb}_kmax{kmax}")


def candidates(b, v, AB):
    return int(TRAINED[v].get("candidates") or AB["candidates"].get(b, 5))


def recipe_path(b, v, c):
    spec = TRAINED[v]
    if spec["recipes"] == "main":
        return os.path.join(RC.work_dir(b, spec["k"]), f"recipe_{c}.json")
    if spec["recipes"] == "whole":
        return os.path.join(variant_dir(b, v), "recipe_all.json")
    return os.path.join(variant_dir(b, v), f"recipe_{c}.json")


# ---------------------------------------------------------------------- prepare
def ssl_labels(b, S, AB, d, draws):
    import torch
    from claugE.encoder import StageGAP
    from netkit import data as D
    weights = os.path.join(HERE, AB["ssl_weights"])
    if not os.path.exists(weights):
        raise SystemExit(f"the self-supervised encoder needs its weights file: {weights}")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    enc = StageGAP(AB["ssl_encoder"], weights=weights).to(dev)
    bench = D.load(b, RC.data_root(S, b))
    with open(os.path.join(RC.work_dir(b), "references.json"), encoding="utf-8") as f:
        refs = json.load(f)["indices"]
    label_sets = []
    for j, i in enumerate(refs):
        zp = os.path.join(d, f"reduced_ref{j}.npy")
        if not os.path.exists(zp):
            log(f"  {AB['ssl_encoder']}: reference {j + 1}/{len(refs)}, embedding the draws")
            feats = partition.embed_draws(enc, partition.to_unit(bench.source_train[i][0]), draws,
                                          dev, seed=S["seed"])
            z = partition.reduce(feats, dims=S["reduced_dims"], pca_dims=S["pca_dims"],
                                 seed=S["seed"])
            np.save(zp, z.astype(np.float32))
        label_sets.append(partition.cluster(np.load(zp), 5, seed=S["seed"]))
    labels, agreement = partition.consensus(label_sets, 5, seed=S["seed"])
    log(f"  ssl partition: sizes {np.bincount(labels, minlength=5).tolist()}, agreement (ARI) "
        f"{np.round(agreement, 2).tolist()}")
    return labels


def data_groups(b, S, AB, d):
    """k-means groups of the training images in the frozen encoder's feature space."""
    import torch
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from claugE.encoder import StageGAP
    from netkit import data as D
    if all(os.path.exists(os.path.join(d, f"train_group_{c}.npy")) for c in range(5)):
        return
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bench = D.load(b, RC.data_root(S, b))
    n = len(bench.source_train)
    fp = os.path.join(d, "train_features.npy")
    if not os.path.exists(fp):
        log(f"  encoding the {n} training images")
        enc = StageGAP(S["encoder"]).to(dev)
        np.save(fp, partition.embed_dataset(enc, bench.source_train, np.arange(n), dev,
                                            batch=256).astype(np.float16))
    X = np.load(fp).astype(np.float32)
    rng = np.random.default_rng(S["seed"])
    fit = np.sort(rng.choice(n, min(AB["data_clusters"]["fit_images"], n), replace=False))
    pca = PCA(n_components=AB["data_clusters"]["pca_dims"], random_state=S["seed"]).fit(X[fit])
    Z = np.concatenate([pca.transform(X[s:s + 50000]) for s in range(0, n, 50000)])
    km = KMeans(n_clusters=5, n_init=10, random_state=S["seed"]).fit(Z[fit])
    lab = np.concatenate([km.predict(Z[s:s + 50000]) for s in range(0, n, 50000)])
    y = bench.source_train.y if hasattr(bench.source_train, "y") else None
    info = {"sizes": np.bincount(lab, minlength=5).tolist()}
    for c in range(5):
        idx = np.flatnonzero(lab == c)
        np.save(os.path.join(d, f"train_group_{c}.npy"), idx)
        if y is not None:
            info[f"classes_{c}"] = np.bincount(np.asarray(y)[idx]).tolist()
    with open(os.path.join(d, "train_groups.json"), "w", encoding="utf-8") as f:
        json.dump(info, f)
    log(f"  image groups: sizes {info['sizes']}")


def cmd_prepare(a, S, AB):
    b, v = a.benchmark, a.variant
    spec = TRAINED[v]
    if spec["recipes"] == "main":
        log(f"{v}: uses the main recipes of k = {spec['k']}; nothing to prepare")
        return
    d = variant_dir(b, v)
    os.makedirs(d, exist_ok=True)
    draws = drawer.load(os.path.join(RC.work_dir(b), "draws.jsonl"))
    meta = {"benchmark": b, "ablation": v}
    if spec["recipes"] == "whole":
        path = recipe_path(b, v, 0)
        if not os.path.exists(path):
            recipes.build(draws, np.zeros(len(draws), dtype=int), 1)[0].save(
                path, meta={**meta, "k": 1, "cluster": 0})
        log(f"{v}: the whole palette as one recipe, {len(draws)} draws")
    else:
        if spec["recipes"] == "random":
            with open(os.path.join(RC.work_dir(b, 5), "partition.json"), encoding="utf-8") as f:
                main = np.asarray(json.load(f)["labels"])
            labels = np.random.default_rng(S["seed"]).permutation(main)
        elif spec["recipes"] == "oneref":
            labels = np.load(os.path.join(RC.work_dir(b, 5), "labels_per_reference.npy"))[0]
        else:
            labels = ssl_labels(b, S, AB, d, draws)
        labels = np.asarray(labels).astype(int)
        sizes = np.bincount(labels, minlength=5)
        if (sizes == 0).any():
            raise SystemExit(f"{v}: an empty group, sizes {sizes.tolist()}")
        with open(os.path.join(d, "partition.json"), "w", encoding="utf-8") as f:
            json.dump({"k": 5, "ablation": v, "sizes": sizes.tolist(), "labels": labels.tolist()}, f)
        for c, rec in enumerate(recipes.build(draws, labels, 5)):
            rec.save(recipe_path(b, v, c), meta={**meta, "k": 5, "cluster": c})
        log(f"{v}: 5 recipes, group sizes {sizes.tolist()}")
    if spec.get("data"):
        data_groups(b, S, AB, d)


# ------------------------------------------------------------------------ train
def variant_config(S, AB, b, v, bb, c):
    """The engine configuration of path c of a variant: the main pool's, with the change."""
    spec, n = TRAINED[v], candidates(b, v, AB)
    k, d = spec["k"], variant_dir(b, v)
    rp = recipe_path(b, v, c)
    if not os.path.exists(rp):
        raise SystemExit(f"{rp} missing: run prepare first")
    if spec.get("data") and not os.path.exists(os.path.join(d, f"train_group_{c}.npy")):
        raise SystemExit("image groups missing: run prepare first")
    cfg = RC.engine_config(S, b, bb, k, c)
    cfg["augment"]["claugE_recipe"] = rp
    if "identity_p" in spec:
        cfg["augment"]["identity_p"] = float(spec["identity_p"])
    cfg["pool_size"] = n
    cfg["max_draws"] = max(n, int(round(S["max_draws_per_recipe"] * n / S["candidates_per_recipe"])))
    if spec.get("data"):
        cfg["train_indices"] = os.path.join(d, f"train_group_{c}.npy")
    cfg["notes"] = (f"ClaugE ablation {v} ({spec['row']}), {b} {bb}, path {c} of {k}: the "
                    "main pool's configuration and seed with the variant's change.")
    return cfg


def cmd_train(a, S, AB):
    b, v, bb = a.benchmark, a.variant, a.backbone
    spec, n = TRAINED[v], candidates(a.benchmark, a.variant, AB)
    k = spec["k"]
    d = variant_dir(b, v)
    os.makedirs(d, exist_ok=True)
    for c in range(k):
        cfg = variant_config(S, AB, b, v, bb, c)
        cfg_path = os.path.join(d, f"engine_{bb}_recipe_{c}.json")
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=1)
        out = variant_pool(b, bb, v, c)
        cmd = [sys.executable, os.path.join(RC.ENGINE, "build_pool.py"), "--workers",
               str(a.workers), "--data", RC.data_root(S, b)]
        if os.path.exists(os.path.join(out, "pool_config.json")):
            cmd += ["--out", out]
        else:
            cmd += ["--config", cfg_path, "--pool-seed", str(1000 * k + c), "--out", out]
        log(f"{b} {bb} ablation {v}: path {c + 1}/{k}, {n} candidate(s)")
        r = subprocess.run(cmd, cwd=RC.ENGINE)
        if r.returncode != 0:
            raise SystemExit(f"engine stopped on path {c} (exit {r.returncode}); rerun to resume")


# -------------------------------------------------------------------- committee
def cmd_committee(a, S, AB):
    b, v, bb = a.benchmark, a.variant, a.backbone
    spec, n = TRAINED[v], candidates(a.benchmark, a.variant, AB)
    k = spec["k"]
    pools = [variant_pool(b, bb, v, c) for c in range(k)]
    for p in pools:
        if not os.path.exists(os.path.join(p, "predictions", "index.json")):
            raise SystemExit(f"{p}: not trained yet")
    for kmax in (5, 25):
        m = kmax // k
        if m < 1 or m > n:
            continue
        out = ablation_out(b, bb, kmax, v)
        if os.path.exists(os.path.join(out, "committee.json")) and not a.overwrite:
            log(f"{os.path.basename(out)}: exists, left as it is (--overwrite to rebuild)")
            continue
        com = Cm.build(pools, out, {"method": "ClaugE", "benchmark": b, "backbone": bb,
                                    "k_max": kmax, "variant": v, "candidates_per_path": n},
                       per_recipe=m, seed=S["seed"])
        log(f"{os.path.basename(out)}: {k} paths x {m}, alpha {com['alpha']:g}, radius "
            f"{com['fit']['radius']}, weights {[round(r['weight'], 3) for r in com['recipe_models']]}")
    if k == 5 and n < S["candidates_per_recipe"]:
        ref = ablation_out(b, bb, 5, f"full_c{n}")
        if not os.path.exists(os.path.join(ref, "committee.json")) or a.overwrite:
            Cm.build([RC.pool_dir(b, bb, 5, c) for c in range(5)], ref,
                     {"method": "ClaugE", "benchmark": b, "backbone": bb, "k_max": 5,
                      "variant": f"full_c{n}", "candidates_per_path": n},
                     per_recipe=1, seed=S["seed"], limit=n)
            log(f"{os.path.basename(ref)}: the reference, ClaugE on the first {n} candidate(s) "
                "of each of its pools")


def reference_of(b, bb, v, kmax, n):
    if TRAINED[v]["k"] != 5 or n >= 5:
        return main_committee(b, bb, kmax)
    return ablation_out(b, bb, kmax, f"full_c{n}")


# ---------------------------------------------------------- committee-level variants
class Pools:
    """The predictions of a set of pools (the main ones, or a variant's).  Probabilities are
    read when needed and never kept, so that a committee costs a few hundred megabytes even
    on RxRx1 and the stage can run beside a training."""

    def __init__(self, pools):
        self.pools = pools
        index, _ = Cm._read_pool(pools[0])
        self.K = int(index["num_classes"])
        self.metric = Cm.metric_for(self.K)
        self.sel, self.test = Cm.roles(index)
        rd = lambda s, f: np.load(os.path.join(pools[0], "predictions", s, f))  # noqa: E731
        self.y_sel, self.d_sel = rd(self.sel, "labels.npy").astype(np.int64), rd(self.sel, "domains.npy")
        self.y_test, self.d_test = rd(self.test, "labels.npy").astype(np.int64), rd(self.test, "domains.npy")
        self.cands = [Cm.ranked(p, self.metric)[0] for p in pools]

    def _file(self, c, mid, split):
        return os.path.join(self.pools[c], "predictions", split, f"member_{mid:04d}.npy")

    def probs(self, c, mid, split):
        return Cm.softmax(np.load(self._file(c, mid, split)).astype(np.float64)).astype(np.float32)

    def mean_probs(self, group, split):
        """Uniform average of the members (c, id) of one path model, one member at a time."""
        total = None
        for c, m in group:
            p = self.probs(c, m, split)
            if total is None:
                total = p
            else:
                total += p
        total /= len(group)
        return total

    def path_scores(self, m):
        """Selection-split score of every path model made of its m best candidates."""
        return [Cm.decision_score(self.mean_probs([(c, mid) for mid, _ in self.cands[c][:m]],
                                                  self.sel).argmax(1), self.y_sel, self.K, self.metric)
                for c in range(len(self.pools))]

    def single_test(self, c, mid):
        """Test score of one candidate alone."""
        return self.score(np.load(self._file(c, mid, self.test)).argmax(1))

    def balanced_error(self, c, mid):
        """One minus the mean per-class recall of one candidate on the test split."""
        pred = np.load(self._file(c, mid, self.test)).argmax(1)
        return 1.0 - E.decision_metrics(pred, self.y_test, self.K)["balanced_acc"]

    def true_probs(self, c, mid):
        """Probability one candidate gives the true class of every test point."""
        p = self.probs(c, mid, self.test)
        return p[np.arange(len(self.y_test)), self.y_test].astype(np.float64)

    def first(self):
        """The first candidate (lowest id) of every path.  Path c of a variant has the pool
        seed of ClaugE's path c, so these are the same slots, with the same training
        settings and seeds, in the main pools and in a variant's (two paths of one set of
        pools have different pool seeds, hence different settings and seeds)."""
        return [(c, min(mid for mid, _ in self.cands[c])) for c in range(len(self.pools))]

    def disagreement(self, members):
        """Mean share of test points on which two of the networks predict differently."""
        preds = [np.load(self._file(c, mid, self.test)).argmax(1) for c, mid in members]
        return float(np.mean([np.mean(preds[i] != preds[j])
                              for i in range(len(preds)) for j in range(i + 1, len(preds))]))

    def five_networks(self):
        """Test score of five of these pools' networks averaged uniformly: the five candidates
        of a path (mean over the paths), or the five paths' networks when each has one."""
        if all(len(c) >= 5 for c in self.cands):
            return float(np.mean([self.score(self.mean_probs([(c, mid) for mid, _ in self.cands[c][:5]],
                                                             self.test).argmax(1))
                                  for c in range(len(self.pools))]))
        return self.score(self.mean_probs(self.first(), self.test).argmax(1))

    def committee(self, groups, alpha="auto", radius="auto", seed=0):
        """Test predictions of the committee of path models `groups` (lists of (c, member))."""
        P = np.stack([self.mean_probs(g, self.sel) for g in groups])
        a, r = alpha, radius
        if a == "auto" or r == "auto":
            a, r, _ = Cm.select_settings(P, self.y_sel, self.d_sel, self.K, self.metric,
                                         Cm.ALPHA_GRID if a == "auto" else (a,),
                                         Cm.FIT_RADIUS_GRID if r == "auto" else (r,),
                                         Cm.FIT_ITERS, True, seed)
        _, _, w, fit = Cm.committee_weights(P, self.y_sel, self.d_sel, self.K, self.metric, a, r,
                                            Cm.FIT_ITERS, True, seed)
        del P
        total = None
        for wg, g in zip(w, groups):
            pg = self.mean_probs(g, self.test)
            pg *= float(wg)
            if total is None:
                total = pg
            else:
                total += pg
        return total.argmax(1), {"alpha": a, "radius": r if math.isfinite(r) else "inf",
                                 "fit_kept": fit["kept"], "weights": [float(x) for x in w]}

    def score(self, pred):
        return E.decision_metrics(pred, self.y_test, self.K)[self.metric]


def balanced_weights(y, idx=None):
    """Weights of the class-balanced average of the fit (every class present counts equally)
    over the points y, or over a resample idx of them, returned per original point."""
    yy = y if idx is None else y[idx]
    counts = np.bincount(yy)
    w = 1.0 / ((counts > 0).sum() * counts[yy])
    return w if idx is None else np.bincount(idx, weights=w, minlength=len(y))


def log_loss(q, a):
    """Weighted log loss, from the probabilities q given to the true class."""
    return float(a @ -np.log(np.clip(q, 1e-12, None)))


def quality_diversity(qs, w, a):
    """The identity of the analysis on a split: a committee's log loss is the weighted mean
    of its members' log losses (quality) minus a non-negative diversity term."""
    w = np.asarray(w, dtype=float)
    quality = float(sum(wi * log_loss(q, a) for wi, q in zip(w, qs)))
    loss = log_loss(sum(wi * q for wi, q in zip(w, qs)), a)
    return {"quality": quality, "diversity": quality - loss, "loss": loss}


def paired(pa, pb, y, d, K, metric, B, seed=0):
    """A - B with the shared bench's cluster bootstrap (domains, then points within them)."""
    rng = np.random.default_rng(seed)
    doms = np.unique(d)
    by = [np.flatnonzero(d == u) for u in doms]
    f = lambda pr, idx: E.decision_metrics(pr[idx], y[idx], K)[metric]  # noqa: E731
    point = f(pa, np.arange(len(y))) - f(pb, np.arange(len(y)))
    reps = []
    for _ in range(B):
        pick = rng.integers(len(doms), size=len(doms))
        idx = np.concatenate([rng.choice(by[i], size=len(by[i])) for i in pick])
        reps.append(f(pa, idx) - f(pb, idx))
    lo, hi = np.percentile(reps, [2.5, 97.5])
    return float(point), float(lo), float(hi)


def cmd_committee_level(a, S, AB):
    b, bb = a.benchmark, a.backbone
    P = Pools([RC.pool_dir(b, bb, 5, c) for c in range(5)])
    B, seed = AB["bootstrap"], S["seed"]
    best = lambda m: [[(c, mid) for mid, _ in P.cands[c][:m]] for c in range(5)]  # noqa: E731
    rows = []
    for kmax in (5, 25):
        m = kmax // 5
        ref_pred, ref_info = P.committee(best(m), seed=seed)
        ref = P.score(ref_pred)
        official = main_committee(b, bb, kmax)
        if os.path.exists(os.path.join(official, "committee.json")):
            o = E.evaluate(official)["primary"]
            log(f"k_max={kmax}: full ClaugE {ref:.4f} (committee on disk {o:.4f})")
        variants = [("uniform weights across paths instead of Eq. (weights)", best(m), 0.0, 0.0),
                    ("fixed alpha = 1, r = 1/4 instead of the selected ones", best(m), 1.0, 0.25),
                    ("prior alone (delta = 0) instead of Eq. (fit)", best(m), "auto", 0.0),
                    ("unbounded fit instead of Eq. (fit)", best(m), "auto", math.inf)]
        if kmax == 25:
            variants.append(("weights per member instead of per path (uniform within a path)",
                             [[(c, mid)] for c in range(5) for mid, _ in P.cands[c][:5]], "auto", "auto"))
        path_sel = P.path_scores(m)
        worst = int(np.argmin(path_sel))
        variants.append(("without the path worst on validation (four paths)",
                         [g for c, g in enumerate(best(m)) if c != worst], "auto", "auto"))
        for name, groups, al, r in variants:
            pred, info = P.committee(groups, al, r, seed=seed)
            diff, lo, hi = paired(pred, ref_pred, P.y_test, P.d_test, P.K, P.metric, B)
            if name.startswith("without the path"):
                info = dict(info, dropped_path=worst, path_selection_scores=path_sel)
            rows.append({"variant": name, "k_max": kmax, "value": P.score(pred), "reference": ref,
                         "delta": diff, "ci95": [lo, hi], "settings": info})
            log(f"  {name}: {rows[-1]['value']:.4f}, delta {diff * 100:+.2f} [{lo * 100:+.2f}, {hi * 100:+.2f}]")
        if kmax == 5:
            # the same budget spent on one recipe: the five candidates of the path whose
            # five-candidate model is best on the selection split, averaged uniformly
            sel5 = P.path_scores(5)
            top = int(np.argmax(sel5))
            pred = P.mean_probs([(top, mid) for mid, _ in P.cands[top][:5]], P.test).argmax(1)
            diff, lo, hi = paired(pred, ref_pred, P.y_test, P.d_test, P.K, P.metric, B)
            rows.append({"variant": "the best path on validation alone, its five candidates",
                         "k_max": 5, "value": P.score(pred), "reference": ref, "delta": diff,
                         "ci95": [lo, hi], "settings": {"path": top, "path_selection_scores": sel5}})
            log(f"  the best path alone (path {top}, five candidates): {rows[-1]['value']:.4f}, delta "
                f"{diff * 100:+.2f} [{lo * 100:+.2f}, {hi * 100:+.2f}]")
            draws = []
            rng = np.random.default_rng(seed)
            for t in range(AB["random_candidate_draws"]):
                groups = [[(c, int(rng.choice([mid for mid, _ in P.cands[c]])))] for c in range(5)]
                pred, _ = P.committee(groups, seed=seed)
                draws.append(P.score(pred) - ref)
            rows.append({"variant": "random candidates per path instead of the best on validation",
                         "k_max": 5, "reference": ref, "delta": float(np.mean(draws)),
                         "delta_sd": float(np.std(draws)), "delta_range": [float(min(draws)), float(max(draws))],
                         "draws": len(draws)})
            log(f"  random candidates: delta {np.mean(draws) * 100:+.2f} (sd {np.std(draws) * 100:.2f}, "
                f"{len(draws)} draws)")
    out = os.path.join(HERE, "outputs_ablations", f"{b}_committee_level.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"benchmark": b, "metric": P.metric, "rows": rows}, f, indent=1)
    log(f"written {out}")


def group_operators(draws, labels, k, top=3):
    """Size and most frequent operators (share of the group's transforms that contain them)
    of each group of a partition of the draws."""
    out = []
    for c in range(k):
        count, n = collections.Counter(), 0
        for d, lab in zip(draws, labels):
            if lab == c:
                n += 1
                # each operator once, in its order within the draw (a set's order varies
                # between processes), so that ties keep the same order in every run
                count.update(list(dict.fromkeys(name for name, _ in d)))
        out.append({"draws": n, "operators": [[name, v / n] for name, v in count.most_common(top)]})
    return out


def recipe_operators(b, k=5, top=3, draws=None):
    """Size and most frequent operators of each recipe of the main partition."""
    draws = draws if draws is not None else drawer.load(os.path.join(RC.work_dir(b), "draws.jsonl"))
    with open(os.path.join(RC.work_dir(b, k), "partition.json"), encoding="utf-8") as f:
        labels = json.load(f)["labels"]
    return group_operators(draws, labels, k, top)


def variant_partition(b, v, draws, top=3):
    """A variant's own partition of the draws (random groups, one reference, the
    self-supervised encoder): its sizes, its agreement with the main partition (adjusted
    Rand index) and the most frequent operators of its groups; None for the other variants."""
    from sklearn.metrics import adjusted_rand_score
    path = os.path.join(variant_dir(b, v), "partition.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        labels = np.asarray(json.load(f)["labels"])
    with open(os.path.join(RC.work_dir(b, 5), "partition.json"), encoding="utf-8") as f:
        main = np.asarray(json.load(f)["labels"])
    return {"sizes": np.bincount(labels, minlength=5).tolist(),
            "ari_to_main": float(adjusted_rand_score(main, labels)),
            "groups": group_operators(draws, labels.tolist(), 5, top)}


def cmd_paths(a, S, AB):
    """What each training path yields on its own: single-network test scores by recipe, next
    to the networks of the retrained variants (paired by path and candidate: same training
    settings and seeds), the five-candidate model of each path, and two-network ensembles
    within and across the two paths best on validation."""
    b, bb = a.benchmark, a.backbone
    P = Pools([RC.pool_dir(b, bb, 5, c) for c in range(5)])
    draws = drawer.load(os.path.join(RC.work_dir(b), "draws.jsonl"))
    ops = recipe_operators(b, draws=draws)
    sel5 = P.path_scores(5)
    single = [{mid: P.single_test(c, mid) for mid, _ in P.cands[c]} for c in range(5)]
    rep = {"benchmark": b, "metric": P.metric, "paths": [], "variants": {}}
    log(f"{b} ({P.metric}, %): single networks by training path")
    for c in range(5):
        t = list(single[c].values())
        five = P.score(P.mean_probs([(c, mid) for mid, _ in P.cands[c][:5]], P.test).argmax(1))
        rep["paths"].append({"path": c, **ops[c], "candidates": len(t),
                             "single_test_mean": float(np.mean(t)), "single_test_min": float(min(t)),
                             "single_test_max": float(max(t)),
                             "single_selection_mean": float(np.mean([s for _, s in P.cands[c]])),
                             "five_candidates_test": five, "five_candidates_selection": sel5[c]})
        log(f"  path {c} ({', '.join(f'{n} {v:.0%}' for n, v in ops[c]['operators'])}): single "
            f"{np.mean(t) * 100:.2f} ({min(t) * 100:.2f}-{max(t) * 100:.2f}), five candidates {five * 100:.2f}, "
            f"selection {sel5[c] * 100:.2f}")
    rep["first_candidates"] = {"disagreement": P.disagreement(P.first()),
                               "uniform_test": P.score(P.mean_probs(P.first(), P.test).argmax(1))}
    log(f"  first candidate of every path: disagreement {rep['first_candidates']['disagreement'] * 100:.1f}, "
        f"averaged uniformly {rep['first_candidates']['uniform_test'] * 100:.2f}")
    for v in [v for v, spec in TRAINED.items() if spec["k"] == 5]:
        vp = [variant_pool(b, bb, v, c) for c in range(5)]
        if not all(os.path.exists(os.path.join(p, "predictions", "index.json")) for p in vp):
            continue
        V = Pools(vp)
        vs = [{mid: V.single_test(c, mid) for mid, _ in V.cands[c]} for c in range(5)]
        allv = [x for d in vs for x in d.values()]
        # the same path and candidate in ClaugE's pools: same settings and seeds, other recipe
        diff = [[single[c][mid] - vs[c][mid] for mid in vs[c] if mid in single[c]] for c in range(5)]
        rep["variants"][v] = {"networks": len(allv), "single_test_mean": float(np.mean(allv)),
                              "single_test_min": float(min(allv)), "single_test_max": float(max(allv)),
                              "per_path_mean": [float(np.mean(list(d.values()))) for d in vs],
                              "claugE_minus_variant_per_path": [float(np.mean(d)) if d else None for d in diff],
                              "paired_networks": int(sum(len(d) for d in diff)),
                              "five_networks_test": V.five_networks(),
                              "first_candidates_disagreement": V.disagreement(V.first())}
        log(f"  {v}: single {np.mean(allv) * 100:.2f} ({min(allv) * 100:.2f}-{max(allv) * 100:.2f}), "
            f"{len(allv)} networks; five networks {rep['variants'][v]['five_networks_test'] * 100:.2f}; "
            f"disagreement {rep['variants'][v]['first_candidates_disagreement'] * 100:.1f}; ClaugE's path "
            f"minus the variant's, same settings and seeds: "
            f"{[None if x is None else round(x * 100, 2) for x in rep['variants'][v]['claugE_minus_variant_per_path']]}")
        part = variant_partition(b, v, draws)
        if part is not None:
            rep["variants"][v]["partition"] = part
            log(f"      its partition: sizes {part['sizes']}, agreement with the main one (ARI) "
                f"{part['ari_to_main']:.2f}; groups: " + "; ".join(
                    ", ".join(f"{n} {s:.0%}" for n, s in g["operators"]) for g in part["groups"]))
        if v in ("nopartition", "random"):
            # the mixture model of the analysis: a network trained under the whole palette
            # against the share-weighted average of the paths' networks (same slots), in a
            # risk that is an average over examples (balanced error)
            share = np.array([o["draws"] for o in ops], dtype=float)
            share /= share.sum()
            ev = [[V.balanced_error(c, mid) for mid, _ in V.cands[c]] for c in range(5)]
            ep = [float(np.mean([P.balanced_error(c, mid) for mid, _ in V.cands[c] if mid in single[c]]))
                  for c in range(5)]
            rep["variants"][v]["mixture_check"] = {
                "variant_balanced_error": float(np.mean([x for e in ev for x in e])),
                "paths_balanced_error": ep, "shares": [float(x) for x in share],
                "share_weighted_average": float(share @ np.array(ep)), "best_path": float(min(ep))}
            mc = rep["variants"][v]["mixture_check"]
            log(f"      balanced error: {v} {mc['variant_balanced_error'] * 100:.2f}; paths "
                f"{[round(x * 100, 2) for x in ep]}, share-weighted average {mc['share_weighted_average'] * 100:.2f}, "
                f"best {mc['best_path'] * 100:.2f}")

    # the finer partition (k = 20, one candidate per path): what each of its paths yields
    # alone, and how its groups sit inside the five of the main partition
    k20 = TRAINED["k20"]["k"]
    vp = [variant_pool(b, bb, "k20", c) for c in range(k20)]
    if all(os.path.exists(os.path.join(p, "predictions", "index.json")) for p in vp):
        V = Pools(vp)
        with open(os.path.join(RC.work_dir(b, 5), "partition.json"), encoding="utf-8") as f:
            main5 = np.asarray(json.load(f)["labels"])
        with open(os.path.join(RC.work_dir(b, k20), "partition.json"), encoding="utf-8") as f:
            lab20 = np.asarray(json.load(f)["labels"])
        ops20 = group_operators(draws, lab20.tolist(), k20)
        fine = []
        for c in range(k20):
            mid, sel = V.cands[c][0]
            inside = np.bincount(main5[lab20 == c], minlength=5)
            fine.append({"path": c, **ops20[c], "single_test": V.single_test(c, mid),
                         "single_selection": float(sel), "main_group": int(inside.argmax()),
                         "share_in_main_group": float(inside.max() / inside.sum())})
        share = np.array([f["draws"] for f in fine], dtype=float)
        rep["k20"] = {"paths": fine,
                      "nested_share": float(share @ np.array([f["share_in_main_group"] for f in fine])
                                            / share.sum()),
                      "single_test_mean": float(np.mean([f["single_test"] for f in fine])),
                      "first_candidates_disagreement": V.disagreement(V.first())}
        log(f"  k = 20: single networks {rep['k20']['single_test_mean'] * 100:.2f} on average "
            f"({min(f['single_test'] for f in fine) * 100:.2f}-{max(f['single_test'] for f in fine) * 100:.2f}); "
            f"{rep['k20']['nested_share']:.0%} of the draws lie in their group's main k = 5 group; "
            f"disagreement {rep['k20']['first_candidates_disagreement'] * 100:.1f}")
        for f in fine:
            log(f"    path {f['path']:>2} ({f['draws']} draws, {f['share_in_main_group']:.0%} in main group "
                f"{f['main_group']}; {', '.join(f'{n} {s:.0%}' for n, s in f['operators'])}): single "
                f"{f['single_test'] * 100:.2f}, selection {f['single_selection'] * 100:.2f}")

    # do two recipes combine better than two seeds of one recipe?
    p1, p2 = [int(x) for x in np.argsort(sel5)[::-1][:2]]
    pairs = {"across": [], f"within {p1}": [], f"within {p2}": []}
    for i, _ in P.cands[p1]:
        pi = P.probs(p1, i, P.test)
        for j, _ in P.cands[p2]:
            pj = P.probs(p2, j, P.test)
            pj += pi
            pairs["across"].append(P.score(pj.argmax(1)))
        del pi
    for c in (p1, p2):
        for (i, _), (j, _) in itertools.combinations(P.cands[c], 2):
            pj = P.probs(c, j, P.test)
            pj += P.probs(c, i, P.test)
            pairs[f"within {c}"].append(P.score(pj.argmax(1)))
    rep["pairs"] = {"paths": [p1, p2], **{k: {"n": len(v), "mean": float(np.mean(v)), "min": float(min(v)),
                                             "max": float(max(v))} for k, v in pairs.items()}}
    log(f"  two-network ensembles, paths {p1} and {p2} (the two best on validation): "
        + ", ".join(f"{k} {np.mean(v) * 100:.2f} ({len(v)} pairs)" for k, v in pairs.items()))
    out = os.path.join(HERE, "outputs_ablations", f"{b}_paths.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=1)
    log(f"written {out}")


def cmd_allocation(a, S, AB):
    """Spending the budget of five: one member per path (the committee) or the five
    candidates of the path best on validation.  The choice is made on the selection split
    alone, by 2-fold cross-fitting as for the fit: on each half the committee's weights are
    fitted and the best path is chosen, and both are scored on the other half."""
    b, bb = a.benchmark, a.backbone
    P = Pools([RC.pool_dir(b, bb, 5, c) for c in range(5)])
    y, d, K, metric, seed = P.y_sel, P.d_sel, P.K, P.metric, S["seed"]
    groups = [[(c, P.cands[c][0][0])] for c in range(5)]
    one = np.stack([P.mean_probs(g, P.sel) for g in groups])
    five = np.stack([P.mean_probs([(c, mid) for mid, _ in P.cands[c][:5]], P.sel) for c in range(5)])
    halves_ = Cm.halves(d, seed)
    cv = {"committee": [], "best path": []}
    for fit, ev in (halves_, halves_[::-1]):
        al, r, _ = Cm.select_settings(one[:, fit], y[fit], d[fit], K, metric, Cm.ALPHA_GRID,
                                      Cm.FIT_RADIUS_GRID, Cm.FIT_ITERS, True, seed)
        _, _, w, _ = Cm.committee_weights(one[:, fit], y[fit], d[fit], K, metric, al, r,
                                          Cm.FIT_ITERS, True, seed)
        cv["committee"].append(Cm.decision_score(Cm.committee_pred(one[:, ev], w), y[ev], K, metric))
        top = int(np.argmax([Cm.decision_score(q[fit].argmax(1), y[fit], K, metric) for q in five]))
        cv["best path"].append(Cm.decision_score(five[top][ev].argmax(1), y[ev], K, metric))
    cv = {k: float(np.mean(v)) for k, v in cv.items()}
    chosen = "best path" if round(cv["best path"], 6) > round(cv["committee"], 6) else "committee"
    top = int(np.argmax([Cm.decision_score(q.argmax(1), y, K, metric) for q in five]))
    del one, five
    ref_pred, info = P.committee(groups, seed=seed)
    best_pred = P.mean_probs([(top, mid) for mid, _ in P.cands[top][:5]], P.test).argmax(1)
    diff, lo, hi = paired(best_pred, ref_pred, P.y_test, P.d_test, K, metric, AB["bootstrap"])
    rep = {"benchmark": b, "metric": metric, "cross_fitted_selection": cv, "chosen": chosen,
           "best_path": top, "committee_test": P.score(ref_pred), "best_path_test": P.score(best_pred),
           "best_minus_committee": diff, "ci95": [lo, hi]}
    rep["chosen_test"] = rep["best_path_test"] if chosen == "best path" else rep["committee_test"]
    # the two options in the terms of the analysis: quality and diversity of their log loss
    a_t = balanced_weights(P.y_test)
    rep["log_loss"] = {
        "committee": quality_diversity([P.true_probs(c, mid) for (c, mid), in groups], info["weights"], a_t),
        "best_path": quality_diversity([P.true_probs(top, mid) for mid, _ in P.cands[top][:5]], [0.2] * 5, a_t),
        "committee_weights": info["weights"]}
    for name in ("committee", "best_path"):
        q = rep["log_loss"][name]
        log(f"  log loss, {name}: quality {q['quality']:.3f}, diversity {q['diversity']:.3f}, loss {q['loss']:.3f}")
    log(f"{b} k_max=5: committee {rep['committee_test'] * 100:.2f}, best path alone (path {top}) "
        f"{rep['best_path_test'] * 100:.2f}, difference {diff * 100:+.2f} [{lo * 100:+.2f}, {hi * 100:+.2f}]; "
        f"cross-fitted on the selection split: committee {cv['committee'] * 100:.2f}, best path "
        f"{cv['best path'] * 100:.2f} -> {chosen}")
    out = os.path.join(HERE, "outputs_ablations", f"{b}_allocation.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=1)
    log(f"written {out}")


def cmd_complementarity(a, S, AB):
    """Do networks of different recipes combine better than networks of one recipe, at
    matched quality?  For every two paths a, b and two candidates of each, the four
    across-path pairs and the two within-path pairs are made of the same four networks, so
    the two sides average the same networks with the same multiplicity; the difference of
    their mean pair-ensemble test scores is the across-path effect, read where the quality
    gap between the two paths is small (the difference also grows with that gap, since a
    strong and a weak network differ more than two equal ones).  Every path has its own pool
    seed, so two candidates of the same slot in two paths share neither their seeds nor their
    training settings.

    The same difference is taken in the class-balanced log loss of the pairs (within minus
    across, so that positive means the pairs across paths are better).  By the
    quality-diversity identity the mean member loss is the same on both sides, so this
    difference is exactly the diversity of the pairs across paths minus that of the pairs
    within them: the definition of complementarity in the paper's analysis."""
    b, bb = a.benchmark, a.backbone
    P = Pools([RC.pool_dir(b, bb, 5, c) for c in range(5)])
    y, d, K, metric = P.y_test, P.d_test, P.K, P.metric
    sel = {(c, mid): s for c in range(5) for mid, s in P.cands[c]}
    nets = sorted(sel)
    single = {m: np.load(P._file(*m, P.test)).argmax(1).astype(np.int32) for m in nets}
    pair, qv = {}, {}
    log(f"{b}: pair ensembles of {len(nets)} networks")
    for i, mi in enumerate(nets):
        pi = P.probs(*mi, P.test)
        qv[mi] = pi[np.arange(len(y)), y].astype(np.float64)
        for mj in nets[i + 1:]:
            pj = P.probs(*mj, P.test)
            pj += pi
            pair[(mi, mj)] = pj.argmax(1).astype(np.int32)
            del pj
        del pi
    key = lambda u, v: (u, v) if u < v else (v, u)  # noqa: E731
    pkeys = sorted(pair)
    prow = {k: i for i, k in enumerate(pkeys)}
    # log loss of every pair at every test point (pairs x points)
    PL = np.stack([-np.log(np.clip((qv[u] + qv[v]) / 2, 1e-12, None)).astype(np.float32) for u, v in pkeys])

    # the quality gap between the two paths of a quadruplet, three ways: the metric on the
    # selection split, and on the test split itself the metric and the log loss.  The
    # definition of complementarity asks for equal quality on the split that is analysed,
    # and the selection split can hide a gap (a path close to the others on validation and
    # far behind on test), so the test gaps are the ones to read.
    w_all = balanced_weights(y)
    t_score = {m: E.decision_metrics(single[m], y, K)[metric] for m in nets}
    t_loss = {m: log_loss(qv[m], w_all) for m in nets}
    quads = []
    for pa, pb in itertools.combinations(range(5), 2):
        ca = [m for m in nets if m[0] == pa]
        cb = [m for m in nets if m[0] == pb]
        for a1, a2 in itertools.combinations(ca, 2):
            for b1, b2 in itertools.combinations(cb, 2):
                gap = lambda f: abs((f[a1] + f[a2]) / 2 - (f[b1] + f[b2]) / 2)  # noqa: E731
                quads.append({"a": (a1, a2), "b": (b1, b2),
                              "gaps": {"selection metric": gap(sel), "test metric": gap(t_score),
                                       "test log loss": gap(t_loss)}})

    def stats(idx):
        sc = {k: E.decision_metrics(v[idx], y[idx], K)[metric] for k, v in pair.items()}
        df = {k: float(np.mean((single[k[0]][idx] != y[idx]) & (single[k[1]][idx] != y[idx])))
              for k in pair}
        ll = PL @ balanced_weights(y, idx)
        out = []
        for q in quads:
            (a1, a2), (b1, b2) = q["a"], q["b"]
            within = [key(a1, a2), key(b1, b2)]
            across = [key(a1, b1), key(a1, b2), key(a2, b1), key(a2, b2)]
            row = {"score": np.mean([sc[k] for k in across]) - np.mean([sc[k] for k in within]),
                   "double_fault": np.mean([df[k] for k in across]) - np.mean([df[k] for k in within]),
                   "loglik": float(np.mean([ll[prow[k]] for k in within]) - np.mean([ll[prow[k]] for k in across]))}
            out.append(row)
        return out

    matchings, cuts = {}, {}
    for mname in ("selection metric", "test metric", "test log loss"):
        g = np.array([q["gaps"][mname] for q in quads])
        t10, t1, t2 = np.quantile(g, [0.1, 1 / 3, 2 / 3])
        cuts[mname] = [float(t10), float(t1), float(t2)]
        matchings[mname] = {"all": np.ones(len(quads), bool), "smallest tenth": g <= t10,
                            "smallest third": g <= t1, "middle third": (g > t1) & (g <= t2),
                            "largest third": g > t2}
    skip = ("n", "share_positive", "share_positive_loglik")

    def summarise(rows):
        col = {f: np.array([r[f] for r in rows]) for f in ("score", "double_fault", "loglik")}
        s = {}
        for mname, groups in matchings.items():
            s[mname] = {}
            for gname, mask in groups.items():
                s[mname][gname] = {"n": int(mask.sum()), "score": float(col["score"][mask].mean()),
                                   "double_fault": float(col["double_fault"][mask].mean()),
                                   "loglik": float(col["loglik"][mask].mean()),
                                   "share_positive": float((col["score"][mask] > 0).mean()),
                                   "share_positive_loglik": float((col["loglik"][mask] > 0).mean())}
        return s

    point = summarise(stats(np.arange(len(y))))
    rng = np.random.default_rng(S["seed"])
    doms = np.unique(d)
    by = [np.flatnonzero(d == u) for u in doms]
    reps = []
    B = int(AB.get("complementarity_bootstrap", 200))
    for t in range(B):
        pick = rng.integers(len(doms), size=len(doms))
        idx = np.concatenate([rng.choice(by[i], size=len(by[i])) for i in pick])
        reps.append(summarise(stats(idx)))
    ci = {}
    for mname, groups in matchings.items():
        ci[mname] = {}
        for gname in groups:
            ci[mname][gname] = {}
            for fld in point[mname][gname]:
                if fld in skip:
                    continue
                vals = [r[mname][gname][fld] for r in reps if fld in r[mname][gname]]
                ci[mname][gname][fld] = [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]
    log(f"{b} ({metric}): pairs across two paths minus pairs within them, same four networks; "
        f"score in points, diversity in log loss x100")
    for mname, groups in matchings.items():
        log(f"  quadruplets grouped by the gap in the {mname}")
        for gname in groups:
            p, c_ = point[mname][gname], ci[mname][gname]
            line = (f"    {gname:<15} n {p['n']:>4}  score {p['score'] * 100:+.2f} [{c_['score'][0] * 100:+.2f}, "
                    f"{c_['score'][1] * 100:+.2f}] positive {p['share_positive']:.0%}  both wrong "
                    f"{p['double_fault'] * 100:+.2f} [{c_['double_fault'][0] * 100:+.2f}, "
                    f"{c_['double_fault'][1] * 100:+.2f}]  diversity {p['loglik'] * 100:+.2f} "
                    f"[{c_['loglik'][0] * 100:+.2f}, {c_['loglik'][1] * 100:+.2f}] positive "
                    f"{p['share_positive_loglik']:.0%}")
            log(line)
    out = os.path.join(HERE, "outputs_ablations", f"{b}_complementarity.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"benchmark": b, "metric": metric, "quadruplets": len(quads), "bootstrap": B,
                   "gap_quantiles_10_33_67": cuts, "point": point, "ci95": ci}, f, indent=1)
    log(f"written {out}")


def uniform_test(path):
    """Test predictions of a committee's members averaged with equal weights."""
    c = E.Committee(path)
    c.weights = np.full(len(c.weights), 1.0 / len(c.weights))
    test = c.roles["test"]
    return c.probs(test).argmax(1), c.labels(test), c.domains(test), c.K


def cmd_table(a, S, AB):
    b, bb = a.benchmark, a.backbone
    rows = []
    for v, spec in TRAINED.items():
        n = candidates(b, v, AB)
        for kmax in (5, 25):
            vd = ablation_out(b, bb, kmax, v)
            if not os.path.exists(os.path.join(vd, "committee.json")):
                continue
            ref = reference_of(b, bb, v, kmax, n)
            if not os.path.exists(os.path.join(ref, "committee.json")):
                log(f"{v} k_max={kmax}: reference {ref} missing")
                continue
            res = E.paired_bootstrap(vd, ref, B=AB["bootstrap"])
            # the same comparison with equal weights, to separate the members from the weights
            pv, y, d, K = uniform_test(vd)
            pr = uniform_test(ref)[0]
            metric = E.primary_metric(b)
            du, lo, hi = paired(pv, pr, y, d, K, metric, AB["bootstrap"])
            rows.append({"variant": spec["row"], "name": v, "k_max": kmax, "candidates_per_path": n,
                         "value": E.evaluate(vd)["primary"], "reference": E.evaluate(ref)["primary"],
                         "reference_dir": os.path.basename(ref), "delta": res["difference"],
                         "ci95": res["ci95"],
                         "uniform_value": E.decision_metrics(pv, y, K)[metric],
                         "uniform_reference": E.decision_metrics(pr, y, K)[metric],
                         "uniform_delta": du, "uniform_ci95": [lo, hi]})
    cl = os.path.join(HERE, "outputs_ablations", f"{b}_committee_level.json")
    if os.path.exists(cl):
        with open(cl, encoding="utf-8") as f:
            rows += [dict(r, name="committee-level") for r in json.load(f)["rows"]]
    for r in rows:
        ci = (f"[{r['ci95'][0] * 100:+.2f}, {r['ci95'][1] * 100:+.2f}]" if "ci95" in r
              else f"(sd {r['delta_sd'] * 100:.2f} over {r['draws']} draws)")
        uni = (f"   equal weights {r['uniform_delta'] * 100:+.2f} [{r['uniform_ci95'][0] * 100:+.2f}, "
               f"{r['uniform_ci95'][1] * 100:+.2f}]" if "uniform_delta" in r else "")
        print(f"k_max {r['k_max']:>2}  {r['delta'] * 100:+6.2f} {ci:<20} {r['variant']}{uni}")
    out = os.path.join(HERE, "outputs_ablations", f"{b}_ablations.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"benchmark": b, "rows": rows}, f, indent=1)
    log(f"written {out}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["prepare", "train", "committee", "committee-level", "paths",
                                      "allocation", "complementarity", "table"])
    ap.add_argument("--benchmark", required=True, choices=["camelyon17", "rxrx1", "iwildcam", "kather"])
    ap.add_argument("--variant", choices=sorted(TRAINED))
    ap.add_argument("--backbone", default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args(argv)
    S, AB = RC.settings(), ablation_settings()
    a.backbone = a.backbone or AB["backbone"]
    if a.stage in ("prepare", "train", "committee") and not a.variant:
        ap.error(f"--variant is required for {a.stage}")
    {"prepare": cmd_prepare, "train": cmd_train, "committee": cmd_committee,
     "committee-level": cmd_committee_level, "paths": cmd_paths, "allocation": cmd_allocation,
     "complementarity": cmd_complementarity, "table": cmd_table}[a.stage](a, S, AB)
    return 0


if __name__ == "__main__":
    sys.exit(main())
