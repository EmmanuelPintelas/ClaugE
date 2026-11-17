"""GeNeX, the baseline, in two stages per benchmark.

    python run_genex.py gene     --benchmark kather            (the pool; resumable)
    python run_genex.py protonex --benchmark kather [--kmax 5 25]
    python run_genex.py smoke    --benchmark kather            (a few batches, end to end)

gene      GenE: N networks of generation 0, then G generations of crossover, mutation and
          one-epoch training; N networks end in work/<benchmark>/M.  Rerun the same command
          to resume after an interruption.
protonex  ProtoNeX for every budget k_max: behavioral clustering of the pool on the selection
          split (UMAP, Gaussian mixture, number of clusters by silhouette in k_min..k_max),
          expert election, prototype fusion, SLSQP weights; writes the committee in the
          shared output format to outputs/<benchmark>_efficientnet_b0_kmax<k_max>/.
smoke     runs both stages on a handful of batches with N = 4, G = 2 into work_smoke/ and
          outputs_smoke/, to check the code path on a benchmark; delete them afterwards.

Settings in configs/genex.json, where a benchmark's entry overrides any setting; data paths
are relative to the repository root.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)

from genex import data, gene, protonex  # noqa: E402


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["gene", "protonex", "smoke"])
    ap.add_argument("--benchmark", required=True, choices=["camelyon17", "iwildcam", "rxrx1", "kather"])
    ap.add_argument("--kmax", type=int, nargs="+", help="committee budgets; default: k_max")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args(argv)
    with open(os.path.join(HERE, "configs", "genex.json"), encoding="utf-8") as f:
        S = json.load(f)
    b = S["benchmarks"][a.benchmark]
    cfg = {**S, **{k: v for k, v in b.items() if k != "data"}}   # a benchmark may override any setting
    bench = data.load(a.benchmark, os.path.normpath(os.path.join(PROJECT, b["data"])))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kmax = a.kmax or S["k_max"]
    max_batches, work_root, out_root = None, "work", "outputs"
    if a.stage == "smoke":
        cfg = copy.deepcopy(cfg)
        cfg.update(N=4, G=2, parent_pairs=2, children_per_pair=1, epochs=[1],
                   signature_points=200, umap_neighbors=3, k_min=2)
        max_batches, work_root, out_root = 3, "work_smoke", "outputs_smoke"
        kmax = [3]
    work = os.path.join(HERE, work_root, a.benchmark)
    os.makedirs(work, exist_ok=True)
    log(f"{a.benchmark}: {bench.num_classes} classes, train {len(bench.train)}, "
        f"selection {bench.selection_name} {len(bench.selection)}, test {bench.test_name} "
        f"{len(bench.test)}, {cfg['image_size']} px, {device}")

    if a.stage in ("gene", "smoke"):
        gene.run(bench, cfg, work, device, a.workers, log, max_batches)
    if a.stage in ("protonex", "smoke"):
        pool = sorted(os.path.join(work, "M", f) for f in os.listdir(os.path.join(work, "M")))
        for k in kmax:
            out = os.path.join(HERE, out_root, f"{a.benchmark}_{cfg['arch']}_kmax{k}")
            meta = {"method": "GeNeX", "benchmark": a.benchmark, "backbone": cfg["arch"],
                    "k_max": k}
            com = protonex.build(pool, bench, cfg, work, out, k, meta, device, a.workers, log,
                                 max_batches)
            log(f"{os.path.basename(out)}: {com['k']} prototypes, weights "
                f"{[round(m['weight'], 3) for m in com['members']]}, selection "
                f"{com['metric']} {com['selection_score']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
