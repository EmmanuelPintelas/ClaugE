"""The benchmarks of the comparison and what every method shares on them.

Every method uses, per benchmark, the same image size, the same selection split (OOD
validation where the benchmark has one, source validation for Kather) and the same test
split, found by their roles in a committee's predictions/index.json.  Inference cost is
reported as members x GFLOPs per image at the benchmark's image size.
"""
from __future__ import annotations

BENCHMARKS = {
    "camelyon17": {"classes": 2, "image_size": 96, "selection_role": "tuning",
                   "wilds_metric": "acc"},
    "iwildcam": {"classes": 182, "image_size": 224, "selection_role": "tuning",
                 "wilds_metric": "macro_f1"},
    "rxrx1": {"classes": 1139, "image_size": 224, "selection_role": "tuning",
              "wilds_metric": "acc"},
    "kather": {"classes": 9, "image_size": 224, "selection_role": "source_val",
               "wilds_metric": "acc"},
}

BACKBONES = ("efficientnet_b0",)
BUDGETS = (5, 25)

# GFLOPs (multiply-adds) per image at 224 x 224; scaled by (size / 224)^2
GFLOPS_224 = {"efficientnet_b0": 0.39, "convnext_atto": 0.55}


def primary_metric(benchmark):
    """Geometric-mean recall below 11 classes, macro-F1 otherwise."""
    return "gm" if BENCHMARKS[benchmark]["classes"] <= 10 else "macro_f1"


def gflops(backbone, benchmark):
    return GFLOPS_224[backbone] * (BENCHMARKS[benchmark]["image_size"] / 224) ** 2
