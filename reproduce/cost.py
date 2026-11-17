"""Accuracy against cost on Camelyon17 (the numbers of Fig. 4 and Section 6.2).

Every method's committees are scored with the shared bench (test primary metric, networks
run at inference, inference GFLOPs), and training time is summed from each method's own
logs: the epochs of every network a committee needs (for deep ensembles, ensembles of
averages and augmentation-diverse ensembles at k_max = 5, the five runs they average;
otherwise every network the method trains), and for ClaugE also the partition at k = 5
(embedding, PCA and UMAP for every reference, the mixtures and the consensus).  Not
logged, hence not counted: DiWA's linear probe, GeNeX's prototype stage, and the final
predictions of every method.

    python reproduce/cost.py [--benchmark camelyon17]

Writes results/cost/<benchmark>.json; reproduce/figures.py draws Fig. 4 from it.  Needs
the committees and the training logs of every method (methods/<Method>/outputs, work and,
for ClaugE, pools and work/<benchmark>/timing.json).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from bench import evaluate as E  # noqa: E402

M = os.path.join(ROOT, "methods")
METHODS = [("Deep ensembles", "DeepEnsembles"), ("Hyperparameter ens.", "HyperEnsembles"),
           ("Augmentation-diverse", "AugDiverse"), ("Ensemble of averages", "EoA"),
           ("DiWA", "DiWA"), ("Lookaround", "Lookaround"), ("GeNeX", "GeNeX"), ("ClaugE", "ClaugE")]
# the runs a committee of five averages, when it does not need every trained network
FIRST_FIVE = {"DeepEnsembles": "seed_0{}", "EoA": "seed_0{}", "AugDiverse": "member_0{}"}


def committee(folder, bench, kmax, suffix=""):
    return os.path.join(M, folder, "outputs", f"{bench}_efficientnet_b0_kmax{kmax}{suffix}")


def log_hours(folder, bench, runs=None):
    """Summed epoch time of a baseline's runs (all of them, or the named ones)."""
    s = 0.0
    for p in glob.glob(os.path.join(M, folder, "work", bench, "**", "info.json"), recursive=True):
        if runs is not None and os.path.basename(os.path.dirname(p)) not in runs:
            continue
        with open(p, encoding="utf-8") as f:
            s += sum(h.get("seconds", 0) for h in json.load(f).get("history", []))
    return s / 3600


def claugE_hours(bench):
    """Training of the 25 candidates, and the partition at k = 5."""
    train = 0.0
    for p in glob.glob(os.path.join(M, "ClaugE", "pools", bench, "efficientnet_b0", "k5",
                                    "recipe_*", "manifest.jsonl")):
        with open(p, encoding="utf-8") as f:
            train += sum(json.loads(line)["seconds"]["train"] for line in f)
    with open(os.path.join(M, "ClaugE", "work", bench, "timing.json"), encoding="utf-8") as f:
        t = json.load(f)
    part = sum(v["embed_s"] + v["reduce_s"] for k, v in t.items() if k.startswith("ref"))
    part += t["cluster_k5_s"]
    return train / 3600, part / 3600


def genex_hours(bench):
    with open(os.path.join(M, "GeNeX", "work", bench, "gene_log.jsonl"), encoding="utf-8") as f:
        return sum(json.loads(line).get("seconds", 0) for line in f) / 3600


def collect(bench, claugE_suffix=""):
    rows = []
    for name, folder in METHODS:
        for kmax in (5, 25):
            r = E.evaluate(committee(folder, bench, kmax, claugE_suffix if folder == "ClaugE" else ""))
            extra = {}
            if folder == "ClaugE":
                train, part = claugE_hours(bench)
                hours, extra = train + part, {"training_hours": train, "partition_hours": part}
            elif folder == "GeNeX":
                hours = genex_hours(bench)
            elif kmax == 5 and folder in FIRST_FIVE:
                hours = log_hours(folder, bench, {FIRST_FIVE[folder].format(i) for i in range(5)})
            else:
                hours = log_hours(folder, bench)
            trained = 5 if kmax == 5 and folder in FIRST_FIVE else r["models_trained"]
            rows.append({"method": name, "folder": folder, "k_max": kmax,
                         "test_primary": r["primary"], "networks": r["k"],
                         "inference_gflops": r["inference_gflops"],
                         "networks_trained": trained, "hours": hours, **extra})
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--benchmark", default="camelyon17")
    ap.add_argument("--claugE-suffix", default="", help="output tag of ClaugE's committees")
    a = ap.parse_args(argv)
    rows = collect(a.benchmark, a.claugE_suffix)
    out = os.path.join(ROOT, "results", "cost", f"{a.benchmark}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"benchmark": a.benchmark, "rows": rows}, f, indent=1)
    for r in rows:
        print(f"{r['method']:<22} k_max {r['k_max']:>2}: {r['test_primary'] * 100:6.2f}  "
              f"{r['networks']:>2} networks, {r['inference_gflops']:.3f} GFLOPs, "
              f"{r['networks_trained']} trained, {r['hours']:.2f} h")
    print(f"written {out}")


if __name__ == "__main__":
    main()
