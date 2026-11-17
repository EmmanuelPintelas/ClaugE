"""Hyper-deep ensembles (Wenzel et al., 2020), the baseline.

    python run_hyper_ensembles.py search    --benchmark kather   (random search; resumable)
    python run_hyper_ensembles.py stratify  --benchmark kather   (retrain the selected configs)
    python run_hyper_ensembles.py committee --benchmark kather   (k_max 5 and 25)
    python run_hyper_ensembles.py smoke     --benchmark kather

search     `search_size` networks, each with weight decay, label smoothing and dropout drawn
           from `space` (the rest is the base recipe), in work/<benchmark>/search_XX/.
stratify   hyper_ens selects `stratify_k` networks of the search (greedy, with replacement,
           on the selection-split NLL); every distinct configuration among them is retrained
           from `stratify_seeds` new seeds, in work/<benchmark>/strat_XX_s/.
committee  k_max = stratify_k: hyper_ens over the stratified networks (Algorithm 1);
           larger k_max: hyper_ens over the search and stratified networks together.
           Members are the distinct networks selected, weighted by their selection counts.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import core  # noqa: E402

NAME = "hyper_ensembles"


def draw(space, rng):
    out = {}
    for k, spec in space.items():
        if "loguniform" in spec:
            lo, hi = spec["loguniform"]
            out[k] = float(math.exp(rng.uniform(math.log(lo), math.log(hi))))
        else:
            lo, hi = spec["uniform"]
            out[k] = float(rng.uniform(lo, hi))
    return out


def hyper_ens(Q, K):
    """Greedy selection with replacement minimizing the NLL of the averaged probabilities
    (Caruana et al., 2004, as used by Wenzel et al., 2020).  Q (networks, N): each network's
    probability of the true class, which is all the NLL of an average needs.  Returns
    selection counts."""
    counts = np.zeros(len(Q), int)
    total = np.zeros(Q.shape[1])
    for t in range(1, K + 1):
        v = -np.mean(np.log(np.clip((total[None, :] + Q) / t, 1e-12, None)), axis=1)
        i = int(np.argmin(v))
        counts[i] += 1
        total += Q[i]
    return counts


def train_one(run, B, S, recipe, seed, device, workers, mb, info):
    core.seed_all(seed)
    model = core.build(B, S, drop_rate=recipe["drop_rate"])
    res = core.train(model, B, recipe, device=device, workers=workers, seed=seed, max_batches=mb)
    core.store_network(model, B, run, device=device, workers=workers,
                       info={"seed": seed, "recipe": recipe, **info, **res}, max_batches=mb)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["search", "stratify", "committee", "smoke"])
    ap.add_argument("--benchmark", required=True, choices=["camelyon17", "iwildcam", "rxrx1", "kather"])
    ap.add_argument("--kmax", type=int, nargs="+")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args(argv)
    S = core.settings(HERE, NAME)
    B = core.load_benchmark(a.benchmark, S)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n, sk, ss = S["search_size"], S["stratify_k"], S["stratify_seeds"]
    kmax, mb, work_root, out_root = a.kmax or S["k_max"], None, "work", "outputs"
    base = core.base_recipe(S, a.benchmark)
    if a.stage == "smoke":
        n, sk, ss, kmax, mb, work_root, out_root = 3, 2, 2, [2, 3], 3, "work_smoke", "outputs_smoke"
        base["epochs"] = 1
    work = os.path.join(HERE, work_root, a.benchmark)
    search = [os.path.join(work, f"search_{i:02d}") for i in range(n)]
    labels = B.selection.y

    def true_class_probs(runs):
        out = []
        for r in runs:
            p = core.softmax(core.selection_logits(r))
            out.append(p[np.arange(len(p)), labels[:len(p)]])
        return np.stack(out)

    if a.stage in ("search", "smoke"):
        for i, run in enumerate(search):
            if core.stored(run):
                continue
            hp = draw(S["space"], random.Random(f"{S['seed']}|search|{i}"))
            core.log(f"{a.benchmark}: search {i + 1}/{n} {hp}")
            train_one(run, B, S, {**base, **hp}, S["seed"] * 1000 + i, device, a.workers, mb,
                      {"hyperparameters": hp})

    strat_file = os.path.join(work, "stratified.json")
    if a.stage in ("stratify", "smoke"):
        if any(not core.stored(r) for r in search):
            raise SystemExit("the random search is not complete")
        counts = hyper_ens(true_class_probs(search), sk)
        chosen = [int(i) for i in np.flatnonzero(counts)]
        core.log(f"hyper_ens on the search: configurations {chosen}, counts {counts[chosen].tolist()}")
        strat = []
        for i in chosen:
            hp = core.read_info(search[i])["hyperparameters"]
            for s in range(ss):
                run = os.path.join(work, f"strat_{i:02d}_{s}")
                strat.append(run)
                if core.stored(run):
                    continue
                core.log(f"{a.benchmark}: configuration {i}, seed {s + 1}/{ss}")
                train_one(run, B, S, {**base, **hp}, S["seed"] * 1000 + 100 * (i + 1) + s,
                          device, a.workers, mb, {"hyperparameters": hp, "configuration": i})
        with open(strat_file, "w", encoding="utf-8") as f:
            json.dump({"configurations": chosen, "runs": [os.path.basename(r) for r in strat]}, f)

    if a.stage in ("committee", "smoke"):
        with open(strat_file, encoding="utf-8") as f:
            strat = [os.path.join(work, r) for r in json.load(f)["runs"]]
        trained = len(search) + len(strat)
        for k in kmax:
            pool = strat if k <= sk else search + strat
            counts = hyper_ens(true_class_probs(pool), k)
            idx = [int(i) for i in np.flatnonzero(counts)]
            out = os.path.join(HERE, out_root, f"{a.benchmark}_{S['arch']}_kmax{k}")
            com = core.write_committee(
                out, B, [pool[i] for i in idx], counts[idx].astype(float),
                {"method": "Hyperparameter ensembles", "benchmark": a.benchmark,
                 "backbone": S["arch"], "k_max": k, "aggregation": "weighted average of probabilities",
                 "models_trained": trained, "selected_from": "stratified" if k <= sk else "search+stratified",
                 "selection_counts": counts[idx].tolist()})
            core.log(f"{os.path.basename(out)}: {com['k']} distinct members, counts {counts[idx].tolist()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
