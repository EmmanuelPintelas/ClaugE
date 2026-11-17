"""Augmentation-diverse ensembles (Stickland and Murray, 2020), the baseline.

    python run_aug_diverse.py train     --benchmark kather     (25 members; resumable)
    python run_aug_diverse.py committee --benchmark kather     (k_max 5 and 25)
    python run_aug_diverse.py smoke     --benchmark kather

train      member i is trained with the base recipe under AugMix of severity
           severities[i mod 4]; stored in work/<benchmark>/member_XX/.
committee  the first k_max members (severities 1, 2, 3, 4, 1, ...), averaged uniformly.
"""
from __future__ import annotations

import argparse
import os
import random
import sys

import torch
import torchvision.transforms as T

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import core  # noqa: E402

NAME = "aug_diverse"


class _AugmentedMix(T.AugMix):
    """AugMix whose output is the convex mix of the augmentation chains alone: the paper
    replaces AugMix's Beta-weighted blend with the original by a Bernoulli choice, which
    MemberAugMix makes."""

    def _sample_dirichlet(self, params):
        if params.shape[-1] == 2:
            w = torch.zeros_like(params)
            w[..., 1] = 1.0
            return w
        return super()._sample_dirichlet(params)


class MemberAugMix:
    """With probability p the augmented mix, otherwise the image unchanged."""

    def __init__(self, severity, p, width):
        self.mix = _AugmentedMix(severity=int(severity), mixture_width=int(width), chain_depth=-1,
                                 alpha=1.0, all_ops=False)
        self.p = float(p)

    def __call__(self, img):
        return self.mix(img) if random.random() < self.p else img


def member_transform(size, severity, S):
    return T.Compose(core.light_ops(size) + [MemberAugMix(severity, S["augment_prob"],
                                                          S["mixture_width"])]
                     + core.to_tensor_ops())


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
    recipe = core.base_recipe(S, a.benchmark)
    if a.stage == "smoke":
        members, kmax, mb, work_root, out_root = 3, [2, 3], 3, "work_smoke", "outputs_smoke"
        recipe["epochs"] = 1
    work = os.path.join(HERE, work_root, a.benchmark)
    sev = S["severities"]

    if a.stage in ("train", "smoke"):
        for i in range(members):
            run = os.path.join(work, f"member_{i:02d}")
            if core.stored(run):
                continue
            seed, s = S["seed"] * 1000 + i, sev[i % len(sev)]
            core.log(f"{a.benchmark}: member {i + 1}/{members}, AugMix severity {s}")
            core.seed_all(seed)
            model = core.build(B, S, drop_rate=recipe["drop_rate"])
            res = core.train(model, B, recipe, device=device, workers=a.workers, seed=seed,
                             transform=member_transform(B.image_size, s, S), max_batches=mb)
            core.store_network(model, B, run, device=device, workers=a.workers,
                               info={"seed": seed, "severity": s, "recipe": recipe, **res},
                               max_batches=mb)
    if a.stage in ("committee", "smoke"):
        runs = [os.path.join(work, f"member_{i:02d}") for i in range(members)]
        if any(not core.stored(r) for r in runs):
            raise SystemExit("not every member is trained yet")
        for k in kmax:
            out = os.path.join(HERE, out_root, f"{a.benchmark}_{S['arch']}_kmax{k}")
            com = core.write_committee(
                out, B, runs[:k], [1.0] * k,
                {"method": "Augmentation-diverse", "benchmark": a.benchmark,
                 "backbone": S["arch"], "k_max": k, "aggregation": "average of probabilities",
                 "models_trained": members, "severities": [sev[i % len(sev)] for i in range(k)]})
            core.log(f"{os.path.basename(out)}: {com['k']} members")
    return 0


if __name__ == "__main__":
    sys.exit(main())
