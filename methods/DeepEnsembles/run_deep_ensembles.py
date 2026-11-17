"""Deep ensembles (Lakshminarayanan et al., 2017), the baseline.

    python run_deep_ensembles.py train     --benchmark kather     (25 seeds; resumable)
    python run_deep_ensembles.py committee --benchmark kather     (k_max 5 and 25)
    python run_deep_ensembles.py smoke     --benchmark kather     (a few batches, end to end)

train      trains `members` networks with the base recipe, differing only in their seed,
           each stored in work/<benchmark>/seed_XX/ with its logits on the selection and
           test splits.
committee  the first k_max seeds, averaged uniformly, in the shared output format in
           outputs/<benchmark>_efficientnet_b0_kmax<k_max>/.
smoke      both stages with 3 seeds and 3 batches into work_smoke/, outputs_smoke/.
"""
from __future__ import annotations

import argparse
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import core  # noqa: E402

NAME = "deep_ensembles"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["train", "committee", "smoke"])
    ap.add_argument("--benchmark", required=True, choices=["camelyon17", "iwildcam", "rxrx1", "kather"])
    ap.add_argument("--kmax", type=int, nargs="+")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args(argv)
    S = core.settings(HERE, NAME)
    B = core.load_benchmark(a.benchmark, S)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    members, kmax, mb, work_root, out_root = S["members"], a.kmax or S["k_max"], None, "work", "outputs"
    if a.stage == "smoke":
        members, kmax, mb, work_root, out_root = 3, [2, 3], 3, "work_smoke", "outputs_smoke"
    work = os.path.join(HERE, work_root, a.benchmark)
    recipe = core.base_recipe(S, a.benchmark)
    if a.stage == "smoke":
        recipe["epochs"] = 1

    if a.stage in ("train", "smoke"):
        for i in range(members):
            run = os.path.join(work, f"seed_{i:02d}")
            if core.stored(run):
                continue
            seed = S["seed"] * 1000 + i
            core.log(f"{a.benchmark}: seed {i + 1}/{members}")
            core.seed_all(seed)
            model = core.build(B, S, drop_rate=recipe["drop_rate"])
            res = core.train(model, B, recipe, device=device, workers=a.workers, seed=seed,
                             max_batches=mb)
            core.store_network(model, B, run, device=device, workers=a.workers,
                               info={"seed": seed, "recipe": recipe, **res}, max_batches=mb)
    if a.stage in ("committee", "smoke"):
        runs = [os.path.join(work, f"seed_{i:02d}") for i in range(members)]
        missing = [r for r in runs if not core.stored(r)]
        if missing:
            raise SystemExit(f"{len(missing)} seeds not trained yet")
        for k in kmax:
            out = os.path.join(HERE, out_root, f"{a.benchmark}_{S['arch']}_kmax{k}")
            com = core.write_committee(
                out, B, runs[:k], [1.0] * k,
                {"method": "Deep ensembles", "benchmark": a.benchmark, "backbone": S["arch"],
                 "k_max": k, "aggregation": "average of probabilities", "models_trained": members})
            core.log(f"{os.path.basename(out)}: {com['k']} members")
    return 0


if __name__ == "__main__":
    sys.exit(main())
