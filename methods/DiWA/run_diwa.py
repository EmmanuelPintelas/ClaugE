"""Diverse weight averaging (Rame et al., 2022), the baseline.

    python run_diwa.py train     --benchmark kather     (linear probe, then the runs; resumable)
    python run_diwa.py committee --benchmark kather     (one averaged network, k_max 5 and 25)
    python run_diwa.py smoke     --benchmark kather

train      the linear-probed initialization in work/<benchmark>/lp.pt, then `runs`
           fine-tunings from it with learning rate, weight decay and dropout drawn from
           `space`, for the base number of epochs, last weights kept, in work/<benchmark>/run_XX/.
committee  the weight average of the runs (uniform, or restricted: runs added in order of
           selection score while the average does not get worse on the selection split),
           batch-normalization statistics re-estimated on training images, written as a
           committee of one network for every budget.
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

NAME = "diwa"


def draw(space, rng):
    out = {}
    for k, spec in space.items():
        if "loguniform" in spec:
            lo, hi = spec["loguniform"]
            out[k] = float(math.exp(rng.uniform(math.log(lo), math.log(hi))))
        elif "choice" in spec:
            out[k] = rng.choice(spec["choice"])
        else:
            out[k] = float(rng.uniform(*spec["uniform"]))
    return out


def averaged(runs, B, S, device, workers, mb, seed):
    states = [torch.load(os.path.join(r, "model.pt"), map_location="cpu", weights_only=False)["state_dict"]
              for r in runs]
    model = core.build(B, S, pretrained=False)
    model.load_state_dict(core.average_states(states))
    core.recalibrate_bn(model, B, device=device, workers=workers,
                        batches=3 if mb else S["bn_batches"], seed=seed)
    return model


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["train", "committee", "smoke"])
    ap.add_argument("--benchmark", required=True, choices=["camelyon17", "iwildcam", "rxrx1", "kather"])
    ap.add_argument("--kmax", type=int, nargs="+")
    ap.add_argument("--variant", choices=["uniform", "restricted"])
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args(argv)
    S = core.settings(HERE, NAME)
    B = core.load_benchmark(a.benchmark, S)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n, kmax, mb, work_root, out_root = S["runs"], a.kmax or S["k_max"], None, "work", "outputs"
    base = core.base_recipe(S, a.benchmark)
    if a.stage == "smoke":
        n, kmax, mb, work_root, out_root = 3, [5], 3, "work_smoke", "outputs_smoke"
        base["epochs"] = 1
    work = os.path.join(HERE, work_root, a.benchmark)
    runs = [os.path.join(work, f"run_{i:02d}") for i in range(n)]
    variant = a.variant or S["variant"]

    if a.stage in ("train", "smoke"):
        os.makedirs(work, exist_ok=True)
        lp = os.path.join(work, "lp.pt")
        if not os.path.exists(lp):
            core.log(f"{a.benchmark}: linear probe")
            seed = S["seed"] * 1000
            core.seed_all(seed)
            model = core.build(B, S, drop_rate=0.0)
            for p in model.backbone.parameters():
                p.requires_grad_(False)
            core.train(model, B, {**base, "epochs": S["lp_epochs"] if not mb else 1}, device=device,
                       workers=a.workers, seed=seed, select="last", max_batches=mb)
            for p in model.backbone.parameters():
                p.requires_grad_(True)
            core.save(model.cpu(), lp)
        for i, run in enumerate(runs):
            if core.stored(run):
                continue
            hp = draw(S["space"], random.Random(f"{S['seed']}|run|{i}"))
            seed = S["seed"] * 1000 + i + 1
            core.log(f"{a.benchmark}: run {i + 1}/{n} {hp}")
            core.seed_all(seed)
            model = core.load(lp)
            model.backbone.drop_rate = hp["drop_rate"]
            recipe = {**base, **hp}
            res = core.train(model, B, recipe, device=device, workers=a.workers, seed=seed,
                             select="last", max_batches=mb)
            core.store_network(model, B, run, device=device, workers=a.workers,
                               info={"seed": seed, "hyperparameters": hp, "recipe": recipe, **res},
                               max_batches=mb)

    if a.stage in ("committee", "smoke"):
        if any(not core.stored(r) for r in runs):
            raise SystemExit("not every run is trained yet")
        y = B.selection.y
        metric = core.metric_for(B.num_classes)
        if variant == "uniform":
            chosen = runs
        else:
            order = sorted(runs, key=lambda r: -core.score(core.selection_logits(r).argmax(1),
                                                           y[:len(core.selection_logits(r))],
                                                           B.num_classes, metric))
            chosen, best = [], None
            for r in order:
                trial = averaged(chosen + [r], B, S, device, a.workers, mb, S["seed"])
                lg = core.predict(trial, B.selection, device=device, workers=a.workers, max_batches=mb)
                s = core.score(lg.argmax(1), y[:len(lg)], B.num_classes, metric)
                if best is None or s >= best:
                    chosen, best = chosen + [r], s
                core.log(f"  restricted: {len(chosen)} runs, selection {metric} {best:.4f}")
        model = averaged(chosen, B, S, device, a.workers, mb, S["seed"])
        avg_dir = os.path.join(work, f"average_{variant}")
        core.store_network(model, B, avg_dir, device=device, workers=a.workers,
                           info={"variant": variant, "runs": [os.path.basename(r) for r in chosen]},
                           max_batches=mb)
        for k in kmax:
            out = os.path.join(HERE, out_root, f"{a.benchmark}_{S['arch']}_kmax{k}")
            core.write_committee(
                out, B, [avg_dir], [1.0],
                {"method": "DiWA", "benchmark": a.benchmark, "backbone": S["arch"], "k_max": k,
                 "aggregation": "single model", "variant": variant, "averaged_runs": len(chosen),
                 "models_trained": n})
            core.log(f"{os.path.basename(out)}: one network, the average of {len(chosen)} runs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
