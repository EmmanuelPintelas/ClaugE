"""Ensemble of averages (Arpit et al., 2022), the baseline.

    python run_eoa.py train     --benchmark kather     (25 seeds; resumable)
    python run_eoa.py committee --benchmark kather     (k_max 5 and 25)
    python run_eoa.py smoke     --benchmark kather

train      trains `members` runs with the base recipe for all its epochs (no early stopping,
           as the moving average replaces checkpoint selection) and keeps, for each, the
           simple moving average of its weights from step `sma_start` on; stored in
           work/<benchmark>/seed_XX/ with its logits on the selection and test splits.
committee  the first k_max averaged networks, averaged uniformly in output space.
"""
from __future__ import annotations

import argparse
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import core  # noqa: E402

NAME = "eoa"


class SMA:
    """Running mean of a network's floating-point state from step `start` on."""

    def __init__(self, start):
        self.start, self.n, self.state = start, 0, None

    @torch.no_grad()
    def __call__(self, model, step):
        if step < self.start:
            return
        sd = model.state_dict()
        if self.state is None:
            self.state = {k: v.detach().float().clone() for k, v in sd.items() if v.is_floating_point()}
            self.n = 1
            return
        self.n += 1
        for k, v in self.state.items():
            v.add_(sd[k].detach().float() - v, alpha=1.0 / self.n)

    def load_into(self, model):
        sd = model.state_dict()
        for k, v in self.state.items():
            sd[k].copy_(v.to(sd[k].dtype))


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
    start = S["sma_start"]
    recipe = core.base_recipe(S, a.benchmark)
    if a.stage == "smoke":
        members, kmax, mb, work_root, out_root, start = 3, [2, 3], 3, "work_smoke", "outputs_smoke", 1
        recipe["epochs"] = 2
    work = os.path.join(HERE, work_root, a.benchmark)

    if a.stage in ("train", "smoke"):
        for i in range(members):
            run = os.path.join(work, f"seed_{i:02d}")
            if core.stored(run):
                continue
            seed = S["seed"] * 1000 + i
            core.log(f"{a.benchmark}: seed {i + 1}/{members}")
            core.seed_all(seed)
            model = core.build(B, S, drop_rate=recipe["drop_rate"])
            sma = SMA(start)
            res = core.train(model, B, recipe, device=device, workers=a.workers, seed=seed,
                             select="last", step_hook=sma, max_batches=mb)
            if sma.state is None:
                raise SystemExit(f"run ended after {res['steps']} steps, before sma_start={start}")
            sma.load_into(model)
            core.store_network(model, B, run, device=device, workers=a.workers,
                               info={"seed": seed, "recipe": recipe, "sma_start": start,
                                     "sma_steps": sma.n, **res}, max_batches=mb)
    if a.stage in ("committee", "smoke"):
        runs = [os.path.join(work, f"seed_{i:02d}") for i in range(members)]
        if any(not core.stored(r) for r in runs):
            raise SystemExit("not every seed is trained yet")
        for k in kmax:
            out = os.path.join(HERE, out_root, f"{a.benchmark}_{S['arch']}_kmax{k}")
            com = core.write_committee(
                out, B, runs[:k], [1.0] * k,
                {"method": "EoA", "benchmark": a.benchmark, "backbone": S["arch"], "k_max": k,
                 "aggregation": "average of probabilities", "models_trained": members})
            core.log(f"{os.path.basename(out)}: {com['k']} members")
    return 0


if __name__ == "__main__":
    sys.exit(main())
