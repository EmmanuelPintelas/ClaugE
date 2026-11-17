"""ProtoNeX: from GenE's pool to the GeNeX committee (Algorithm 2 of the GeNeX paper),
extended from two classes to many.

1. Behavioral clustering.  Every network of the pool is run on the selection split; its
   prediction signature is its softmax output on a fixed random subset of that split,
   all K class probabilities concatenated (with two classes this carries the same
   information as the positive-class probability GeNeX uses).  The signatures are
   projected by UMAP and clustered by a Gaussian mixture; the number of clusters is the
   one of largest silhouette score among k_min..min(k_max, N - 1), so that the committee
   never exceeds the budget.
2. Expert election per cluster, by GeNeX's five criteria: the best on the selection split,
   the most robust (smallest drop of the selection metric under Gaussian input noise), the
   closest to the cluster centre, the farthest from it, and the farthest on average from
   the other clusters' centres; duplicates count once.
3. Prototype synthesis: the experts' weights averaged layer by layer, then the output head
   trained for one epoch on the training set with the backbone frozen.
4. Prototype weights by SLSQP on the simplex, maximizing the selection metric of the
   weighted average of the prototypes' probabilities, from uniform weights.

The selection metric is the benchmark's: geometric-mean recall up to ten classes,
macro-F1 above.  The test split is predicted only to be stored for the shared evaluation.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil

import numpy as np
from scipy.optimize import minimize

from . import nets


def metric_for(K):
    return "gm" if K <= 10 else "macro_f1"


def softmax(logits):
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def score(pred, labels, K, metric):
    present = np.unique(labels)
    tc = np.bincount(labels, minlength=K)[present].astype(float)
    tp = np.bincount(labels[pred == labels], minlength=K)[present].astype(float)
    rec = tp / tc
    if metric == "gm":
        return float(np.exp(np.mean(np.log(np.clip(rec, 1e-6, None)))))
    pc = np.bincount(pred, minlength=K)[present].astype(float)
    prec = np.divide(tp, pc, out=np.zeros_like(tp), where=pc > 0)
    s = prec + rec
    f1 = np.divide(2 * prec * rec, s, out=np.zeros_like(s), where=s > 0)
    return float(f1.mean())


# ------------------------------------------------------------- pool on selection
def profile_pool(pool, bench, cfg, work, device, workers, log=print, max_batches=None):
    """Selection-split logits, score and noise robustness of every network, cached."""
    cache = os.path.join(work, "profile")
    os.makedirs(cache, exist_ok=True)
    K, metric = bench.num_classes, metric_for(bench.num_classes)
    y = bench.selection.y
    rows = []
    for i, path in enumerate(pool):
        name = os.path.splitext(os.path.basename(path))[0]
        lp, jp = os.path.join(cache, f"{name}.npy"), os.path.join(cache, f"{name}.json")
        if not (os.path.exists(lp) and os.path.exists(jp)):
            model = nets.load(path, device)
            lg = nets.predict(model, bench.selection, device=device, workers=workers,
                              max_batches=max_batches)
            noisy = nets.predict(model, bench.selection, device=device, workers=workers,
                                 noise_std=cfg["noise_std"], seed=cfg["seed"],
                                 max_batches=max_batches)
            yy = y[:len(lg)]
            s, sn = score(lg.argmax(1), yy, K, metric), score(noisy.argmax(1), yy, K, metric)
            np.save(lp, lg.astype(np.float16))
            with open(jp, "w", encoding="utf-8") as f:
                json.dump({"score": s, "noisy_score": sn}, f)
            log(f"  profiled {i + 1}/{len(pool)}: {metric} {s:.4f}, under noise {sn:.4f}")
            del model
        with open(jp, encoding="utf-8") as f:
            rows.append(json.load(f))
    logits = [np.load(os.path.join(cache, os.path.splitext(os.path.basename(p))[0] + ".npy"))
              for p in pool]
    return logits, rows


def signatures(logits, n, seed):
    """Softmax outputs on a fixed random subset of the selection split, concatenated."""
    N = logits[0].shape[0]
    idx = np.sort(np.random.default_rng(seed).choice(N, min(n, N), replace=False))
    return np.stack([softmax(lg[idx].astype(np.float64)).ravel() for lg in logits]).astype(np.float32)


def embed(sigs, cfg):
    import umap
    n = len(sigs)
    reducer = umap.UMAP(n_components=cfg["umap_dims"], n_neighbors=min(cfg["umap_neighbors"], n - 1),
                        min_dist=cfg["umap_min_dist"], random_state=cfg["seed"])
    return reducer.fit_transform(sigs)


def cluster(coords, k_lo, k_hi, seed):
    """Gaussian mixture with the number of components of largest silhouette."""
    from sklearn.metrics import silhouette_score
    from sklearn.mixture import GaussianMixture
    table, best = {}, None
    for k in range(k_lo, k_hi + 1):
        gmm = GaussianMixture(n_components=k, covariance_type="full", reg_covar=1e-4,
                              n_init=5, random_state=seed).fit(coords)
        lab = gmm.predict(coords)
        used = np.unique(lab)
        if len(used) < 2 or len(used) >= len(coords):
            table[k] = None
            continue
        s = float(silhouette_score(coords, lab))
        table[k] = s
        if best is None or s > best[0] + 1e-12:
            best = (s, k, lab, gmm.means_)
    if best is None:
        raise RuntimeError("no clustering with at least two clusters")
    _, k, lab, means = best
    used = np.unique(lab)
    remap = {c: i for i, c in enumerate(used)}
    return k, np.array([remap[c] for c in lab]), means[used], table


def elect(coords, labels, centres, scores, drops, max_experts):
    """GeNeX's five criteria per cluster; returns the expert indices of each cluster."""
    out = []
    for c in range(len(centres)):
        members = np.flatnonzero(labels == c)
        d_own = np.linalg.norm(coords[members] - centres[c], axis=1)
        others = [centres[o] for o in range(len(centres)) if o != c]
        d_other = (np.mean([np.linalg.norm(coords[members] - o, axis=1) for o in others], axis=0)
                   if others else np.zeros(len(members)))
        picks = [members[np.argmax(scores[members])],          # top on the selection split
                 members[np.argmin(drops[members])],           # most robust to input noise
                 members[np.argmin(d_own)],                    # cluster representative
                 members[np.argmax(d_own)],                    # intra-cluster anomalous
                 members[np.argmax(d_other)]]                  # inter-cluster anomalous
        uniq = list(dict.fromkeys(int(p) for p in picks))
        out.append(uniq[:max_experts])
    return out


def slsqp_weights(probs, labels, K, metric):
    """GeNeX's ensemble weights: SLSQP on the simplex from uniform weights."""
    n = len(probs)

    def objective(w):
        w = np.clip(w, 0, 1)
        w = w / max(w.sum(), 1e-12)
        return -score(np.tensordot(w, probs, axes=1).argmax(1), labels, K, metric)

    res = minimize(objective, np.full(n, 1.0 / n), method="SLSQP", bounds=[(0, 1)] * n,
                   constraints={"type": "eq", "fun": lambda w: np.sum(w) - 1})
    w = np.clip(res.x, 0, 1)
    return w / w.sum(), -float(res.fun)


# ---------------------------------------------------------------- the committee
def prototype(expert_paths, bench, cfg, work, device, workers, log=print, max_batches=None):
    """Averaged experts with a retrained head, cached by the set of experts."""
    names = sorted(os.path.basename(p) for p in expert_paths)
    key = hashlib.sha1("|".join(names).encode()).hexdigest()[:12]
    path = os.path.join(work, "prototypes", f"p_{key}.pt")
    if os.path.exists(path):
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    states = [nets.state(p) for p in expert_paths]
    model = nets.from_state(nets.average(states), cfg["arch"], bench.num_classes, cfg["image_size"])
    seed = int(key, 16) % (2 ** 31)
    nets.train(model, bench.train, np.arange(len(bench.train)), epochs=cfg["proto_epochs"],
               lr=cfg["proto_lr"], weight_decay=cfg["proto_weight_decay"],
               batch_size=cfg["proto_batch_size"], device=device, workers=workers, seed=seed,
               head_only=True, max_batches=max_batches)
    nets.save(model.cpu(), path)
    with open(os.path.join(work, "prototypes", f"p_{key}.json"), "w", encoding="utf-8") as f:
        json.dump({"experts": names}, f)
    log(f"  prototype of {len(names)} experts trained")
    return path


def build(pool, bench, cfg, work, out_dir, kmax, meta, device, workers, log=print,
          max_batches=None):
    K, metric = bench.num_classes, metric_for(bench.num_classes)
    logits, rows = profile_pool(pool, bench, cfg, work, device, workers, log, max_batches)
    y_sel = bench.selection.y[:len(logits[0])]
    scores = np.array([r["score"] for r in rows])
    drops = np.array([r["score"] - r["noisy_score"] for r in rows])
    coords = embed(signatures(logits, cfg["signature_points"], cfg["seed"]), cfg)
    k_hi = min(kmax, len(pool) - 1)
    k_lo = min(cfg["k_min"], k_hi)
    k, labels, centres, sil = cluster(coords, k_lo, k_hi, cfg["seed"])
    experts = elect(coords, labels, centres, scores, drops, cfg["experts_per_cluster"])
    log(f"  k_max={kmax}: {k} clusters (silhouette {sil[k]:.3f}), experts per cluster "
        f"{[len(e) for e in experts]}")
    protos = [prototype([pool[i] for i in e], bench, cfg, work, device, workers, log, max_batches)
              for e in experts]

    os.makedirs(os.path.join(out_dir, "models"), exist_ok=True)
    splits = {bench.selection_name: (bench.selection, bench.selection_role),
              bench.test_name: (bench.test, "test")}
    sel_probs = []
    index = {"num_classes": K, "splits": {}, "members": list(range(len(protos)))}
    for name, (ds, role) in splits.items():
        d = os.path.join(out_dir, "predictions", name)
        os.makedirs(d, exist_ok=True)
        n = None
        for j, p in enumerate(protos):
            lg = nets.predict(nets.load(p, device), ds, device=device, workers=workers,
                              max_batches=max_batches)
            n = len(lg)
            np.save(os.path.join(d, f"member_{j:04d}.npy"), lg.astype(np.float16))
            if name == bench.selection_name:
                sel_probs.append(softmax(lg.astype(np.float64)))
        np.save(os.path.join(d, "labels.npy"), ds.y[:n])
        np.save(os.path.join(d, "domains.npy"), ds.domains[:n])
        index["splits"][name] = {"role": role, "n": int(n), "labels": f"{name}/labels.npy",
                                 "domains": f"{name}/domains.npy"}
    for j, p in enumerate(protos):
        shutil.copy2(p, os.path.join(out_dir, "models", f"member_{j:04d}.pt"))
    w, sel_score = slsqp_weights(np.stack(sel_probs), y_sel, K, metric)

    with open(os.path.join(work, "gene_done.json"), encoding="utf-8") as f:
        gene = json.load(f)
    committee = dict(
        meta, k=len(protos), num_classes=K, metric=metric,
        selection_split=bench.selection_name, test_split=bench.test_name,
        aggregation="weighted average of probabilities",
        clusters=int(k), silhouette={str(a): b for a, b in sil.items()},
        selection_score=sel_score, models_trained=gene["networks_trained"],
        prototype_heads_trained=len(protos), pool=len(pool), extra_data="none",
        members=[{"id": j, "cluster": j, "experts": [os.path.basename(pool[i]) for i in experts[j]],
                  "weight": float(w[j])} for j in range(len(protos))])
    with open(os.path.join(out_dir, "predictions", "index.json"), "w", encoding="utf-8") as f:
        json.dump(index, f, indent=1)
    with open(os.path.join(out_dir, "committee.json"), "w", encoding="utf-8") as f:
        json.dump(committee, f, indent=1)
    return committee
