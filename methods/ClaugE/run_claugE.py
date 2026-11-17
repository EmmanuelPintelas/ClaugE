"""ClaugE, end to end, in three stages per benchmark.

    python run_claugE.py partition --benchmark kather                 (k in partition_k)
    python run_claugE.py train     --benchmark kather --backbone efficientnet_b0 --k 5
    python run_claugE.py committee --benchmark kather --backbone efficientnet_b0 [--kmax 5 25]
                                   [--alpha auto] [--fit-radius auto] [--fit-iters 100]
                                   [--no-fit-check] [--tag NAME] [--overwrite] [--prune]

k is the number of recipes (clusters), k_max the committee budget; the two are independent.

partition  draws the transforms, chooses the reference images, embeds every draw on every
           reference with the frozen encoder, clusters per reference and takes the
           consensus; writes work/<benchmark>/k<k>/recipe_<c>.json for every k of
           partition_k (5, and 20 for the ablation of a finer partition).  Draws, references and reduced
           embeddings are cached in work/<benchmark>/ and shared by every k.  The partition
           does not depend on the backbone of the members.
train      trains, for every one of the k recipes (default: recipes), a pool of candidates
           with the engine (the training distributions of engine/configs/pools; augmentation = identity with
           probability 1/k, else the recipe) in pools/<benchmark>/<backbone>/k<k>/recipe_<c>/.
           Resumable.
committee  for every budget k_max (default: k_max), keeps the k_max / k best candidates of
           every recipe on the selection split, averages them uniformly into one model per
           recipe, and weights the recipes: a softmax of their selection scores in robust
           standard deviations, of strength --alpha (0 = uniform), then a light fit of the
           log-weights on the selection split, each moving at most --fit-radius (0 = no fit,
           inf = unbounded), for at most --fit-iters iterations (it stops earlier once
           converged), kept only if it beats the prior on cross-fitted halves of the split
           (--no-fit-check keeps it regardless).  By default (auto) alpha and the radius are
           chosen from their grids by the committee's score on the selection split.  Writes
           the committee in the shared output format to
           outputs/<benchmark>_<backbone>_kmax<k_max>[_<tag>]/; settings other than the
           defaults get an automatic tag (e.g. _alpha0.5_fit0) unless --tag names one, and an
           existing committee is never rebuilt without --overwrite.  --prune deletes the
           candidate pools after the last budget.

Paths in configs/claugE.json are relative to the repository root.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(os.path.dirname(HERE))
ENGINE = os.path.join(HERE, "engine")
sys.path.insert(0, HERE)
sys.path.insert(0, ENGINE)

from claugE import committee as Cm  # noqa: E402
from claugE import drawer, partition, recipes  # noqa: E402
from claugE.partition import log  # noqa: E402


def settings():
    with open(os.path.join(HERE, "configs", "claugE.json"), encoding="utf-8") as f:
        return json.load(f)


def data_root(S, benchmark):
    return os.path.normpath(os.path.join(PROJECT, S["benchmarks"][benchmark]["data"]))


def work_dir(benchmark, k=None):
    d = os.path.join(HERE, "work", benchmark)
    return os.path.join(d, f"k{k}") if k is not None else d


# -------------------------------------------------------------------- partition
def cmd_partition(a, S):
    import torch
    from claugE.encoder import StageGAP
    from netkit import data as D

    wd = work_dir(a.benchmark)
    os.makedirs(wd, exist_ok=True)
    draws_path = os.path.join(wd, "draws.jsonl")
    if not os.path.exists(draws_path):
        draws = drawer.draw_many(S["draws"], seed=S["seed"], continue_prob=S["continue_prob"])
        drawer.save(draws, draws_path)
        with open(os.path.join(wd, "draws_summary.json"), "w", encoding="utf-8") as f:
            json.dump(drawer.summary(draws), f, indent=1)
    draws = drawer.load(draws_path)
    log(f"{a.benchmark}: {len(draws)} draws")

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = StageGAP(S["encoder"]).to(dev)
    bench = D.load(a.benchmark, data_root(S, a.benchmark))
    refs_path = os.path.join(wd, "references.json")
    if not os.path.exists(refs_path):
        refs = partition.choose_references(encoder, bench.source_train, S["references"], dev,
                                           n_max=S["reference_pool"], seed=S["seed"])
        with open(refs_path, "w", encoding="utf-8") as f:
            json.dump({"split": "source_train", "indices": refs}, f, indent=1)
    refs = json.load(open(refs_path, encoding="utf-8"))["indices"]

    timing_path = os.path.join(wd, "timing.json")
    timing = json.load(open(timing_path, encoding="utf-8")) if os.path.exists(timing_path) else {}
    for j, i in enumerate(refs):
        zp = os.path.join(wd, f"reduced_ref{j}.npy")
        if os.path.exists(zp):
            continue
        log(f"  reference {j + 1}/{len(refs)} (train image {i}): embedding the draws")
        ref = partition.to_unit(bench.source_train[i][0])
        t0 = time.time()
        feats = partition.embed_draws(encoder, ref, draws, dev, seed=S["seed"])
        t1 = time.time()
        z = partition.reduce(feats, dims=S["reduced_dims"], pca_dims=S["pca_dims"],
                             seed=S["seed"])
        t2 = time.time()
        np.save(zp, z.astype(np.float32))
        timing[f"ref{j}"] = {"embed_s": t1 - t0, "reduce_s": t2 - t1}
        with open(timing_path, "w", encoding="utf-8") as f:
            json.dump(timing, f, indent=1)
        log(f"    embedded in {t1 - t0:.0f} s, reduced in {t2 - t1:.0f} s")

    for k in a.k:
        kd = work_dir(a.benchmark, k)
        os.makedirs(kd, exist_ok=True)
        label_sets = []
        t0 = time.time()
        for j in range(len(refs)):
            z = np.load(os.path.join(wd, f"reduced_ref{j}.npy"))
            label_sets.append(partition.cluster(z, k, seed=S["seed"]))
        labels, agreement = partition.consensus(label_sets, k, seed=S["seed"])
        timing[f"cluster_k{k}_s"] = time.time() - t0
        with open(timing_path, "w", encoding="utf-8") as f:
            json.dump(timing, f, indent=1)
        np.save(os.path.join(kd, "labels_per_reference.npy"), np.stack(label_sets).astype(np.int16))
        with open(os.path.join(kd, "partition.json"), "w", encoding="utf-8") as f:
            json.dump({"k": k, "references": refs, "agreement_ari": agreement,
                       "sizes": np.bincount(labels, minlength=k).tolist(),
                       "labels": labels.tolist()}, f)
        for c, rec in enumerate(recipes.build(draws, labels, k)):
            rec.save(os.path.join(kd, f"recipe_{c}.json"),
                     meta={"benchmark": a.benchmark, "k": k, "cluster": c})
        log(f"  k={k}: cluster sizes {np.bincount(labels, minlength=k).tolist()}, "
            f"agreement with references (ARI) {np.round(agreement, 2).tolist()}")


# ------------------------------------------------------------------------ train
def engine_config(S, benchmark, backbone, k, c):
    base = os.path.join(ENGINE, "configs", "pools",
                        S["backbones"][backbone].format(benchmark=benchmark))
    with open(base, encoding="utf-8") as f:
        cfg = json.load(f)
    b = S["benchmarks"][benchmark]
    cfg = copy.deepcopy(cfg)
    cfg.pop("refine", None)
    cfg["notes"] = (f"ClaugE {benchmark} {backbone} k={k} recipe {c}: {os.path.basename(base)} "
                    "with the ClaugE overrides of configs/claugE.json.")
    cfg["image_size"] = b["image_size"]
    cfg["pool_size"] = S["candidates_per_recipe"]
    cfg["max_draws"] = S["max_draws_per_recipe"]
    cfg["augment"] = {"claugE_recipe": os.path.join(work_dir(benchmark, k), f"recipe_{c}.json"),
                      "identity_p": 1.0 / k}
    cfg["recipe"]["augment"] = {"fixed": "claugE"}
    if b.get("batch_size"):
        cfg["recipe"]["batch_size"] = {"choice": list(b["batch_size"])}
    return cfg


def pool_dir(benchmark, backbone, k, c):
    return os.path.join(HERE, "pools", benchmark, backbone, f"k{k}", f"recipe_{c}")


def cmd_train(a, S):
    for k in a.k:
        for c in range(k):
            out = pool_dir(a.benchmark, a.backbone, k, c)
            cfg_path = os.path.join(work_dir(a.benchmark, k), f"engine_{a.backbone}_recipe_{c}.json")
            with open(cfg_path, "w", encoding="utf-8") as f:
                json.dump(engine_config(S, a.benchmark, a.backbone, k, c), f, indent=1)
            cmd = [sys.executable, os.path.join(ENGINE, "build_pool.py"), "--workers", str(a.workers),
                   "--data", data_root(S, a.benchmark)]
            if os.path.exists(os.path.join(out, "pool_config.json")):
                cmd += ["--out", out]
            else:
                cmd += ["--config", cfg_path, "--pool-seed", str(1000 * k + c), "--out", out]
            log(f"{a.benchmark} {a.backbone} k={k}: recipe {c + 1}/{k}")
            r = subprocess.run(cmd, cwd=ENGINE)
            if r.returncode != 0:
                raise SystemExit(f"engine stopped on recipe {c} (exit {r.returncode}); rerun to resume")


# -------------------------------------------------------------------- committee
def setting(text):
    """'auto' or a non-negative number."""
    if text == "auto":
        return text
    v = float(text)
    if v < 0:
        raise argparse.ArgumentTypeError("must be >= 0 or auto")
    return v


def variant_tag(a):
    """Output suffix for committee settings other than the defaults."""
    if a.tag:
        return a.tag
    parts = []
    if a.alpha != Cm.ALPHA:
        parts.append(f"alpha{a.alpha:g}")
    if a.fit_radius != Cm.FIT_RADIUS:
        parts.append(f"fit{a.fit_radius:g}")
    if a.fit_iters != Cm.FIT_ITERS:
        parts.append(f"iters{a.fit_iters}")
    if not a.fit_check:
        parts.append("nocheck")
    return "_".join(parts)


def cmd_committee(a, S):
    if a.fit_iters < 0:
        raise SystemExit("--fit-iters must be >= 0")
    k = a.k[0]
    pools = [pool_dir(a.benchmark, a.backbone, k, c) for c in range(k)]
    tag = variant_tag(a)
    for kmax in a.kmax:
        if kmax % k or not 1 <= kmax // k <= S["candidates_per_recipe"]:
            raise SystemExit(f"k_max={kmax} must be k={k} times 1..{S['candidates_per_recipe']}")
        name = f"{a.benchmark}_{a.backbone}_kmax{kmax}" + (f"_{tag}" if tag else "")
        out = os.path.join(HERE, "outputs", name)
        if os.path.exists(os.path.join(out, "committee.json")) and not a.overwrite:
            log(f"{name}: exists, left as it is (--overwrite to rebuild)")
            continue
        meta = {"method": "ClaugE", "benchmark": a.benchmark, "backbone": a.backbone,
                "k_max": kmax, "variant": tag or None}
        com = Cm.build(pools, out, meta, per_recipe=kmax // k, alpha=a.alpha,
                       radius=a.fit_radius, iters=a.fit_iters, check=a.fit_check,
                       seed=S["seed"])
        fit = com["fit"]
        if "iterations" in fit:
            status = (f"fit {'kept' if fit['kept'] else 'not kept'} "
                      f"({fit['iterations']} iterations, "
                      f"{'converged' if fit['converged'] else 'stopped at --fit-iters'}; "
                      f"log-likelihood {fit['loglik_prior']:.4f} -> {fit['loglik_fit']:.4f}"
                      + (f"; check prior {fit['check_table']['prior']:.4f} vs fit "
                         f"{fit['check_table']['fit']:.4f}" if fit["check_table"] else "") + ")")
        else:
            status = "no fit"
        sel = com.get("settings_selection")
        chosen = (f"alpha {com['alpha']:g}, radius {fit['radius']}"
                  + (" (chosen on the selection split)" if sel else ""))
        log(f"{name}: {k} recipes x {kmax // k} members, {chosen}, {status}")
        if sel:
            log("  selection-split score per (alpha, radius): " + ", ".join(
                f"({t['alpha']:g}, {t['radius']:g}) {t['score']:.4f}" for t in sel))
        log(f"  prior weights {[round(r['prior_weight'], 3) for r in com['recipe_models']]}, "
            f"recipe weights {[round(r['weight'], 3) for r in com['recipe_models']]}")
    if a.prune:
        Cm.prune(pools)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["partition", "train", "committee"])
    ap.add_argument("--benchmark", required=True,
                    choices=["camelyon17", "iwildcam", "rxrx1", "kather"])
    ap.add_argument("--backbone", choices=["efficientnet_b0"])
    ap.add_argument("--k", type=int, nargs="+",
                    help="number of recipes; default: partition_k for partition, recipes "
                         "for train and committee (configs/claugE.json)")
    ap.add_argument("--kmax", type=int, nargs="+", help="committee budgets; default: k_max")
    ap.add_argument("--workers", type=int, default=4)
    g = ap.add_argument_group("committee weights")
    g.add_argument("--alpha", type=setting, default=Cm.ALPHA,
                   help="strength of the prior: weight ratio e^alpha per robust standard "
                        "deviation of score; 0 = uniform; auto = chosen from "
                        f"{list(Cm.ALPHA_GRID)} on the selection split (default %(default)s)")
    g.add_argument("--fit-radius", type=setting, default=Cm.FIT_RADIUS,
                   help="how far the fit may move each log-weight from the prior's; "
                        "0 = no fit, inf = unbounded; auto = chosen from "
                        f"{list(Cm.FIT_RADIUS_GRID)} on the selection split (default %(default)s)")
    g.add_argument("--fit-iters", type=int, default=Cm.FIT_ITERS,
                   help="maximum iterations of the fit; it stops earlier once converged "
                        "(default %(default)s)")
    g.add_argument("--fit-check", action=argparse.BooleanOptionalAction, default=True,
                   help="keep the fit only if it beats the prior on cross-fitted halves of "
                        "the selection split (default on)")
    g.add_argument("--tag", help="output suffix; default: automatic for non-default settings")
    g.add_argument("--overwrite", action="store_true",
                   help="rebuild a committee whose output folder already exists")
    ap.add_argument("--prune", action="store_true")
    a = ap.parse_args(argv)
    S = settings()
    a.k = a.k or (S["partition_k"] if a.stage == "partition" else [S["recipes"]])
    a.kmax = a.kmax or S["k_max"]
    if a.stage != "partition" and not a.backbone:
        ap.error("--backbone is required for train and committee")
    {"partition": cmd_partition, "train": cmd_train, "committee": cmd_committee}[a.stage](a, S)
    return 0


if __name__ == "__main__":
    sys.exit(main())
