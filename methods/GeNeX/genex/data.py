"""The four benchmarks, read from the shared packed arrays, with GeNeX's own transforms.

Splits are named exactly as in every other method's committee (the paired comparison of
the shared bench matches them by name): Camelyon17 ood_val_hospital1 / ood_test_hospital2,
iWildCam and RxRx1 ood_val / ood_test, Kather source_val / its test split from the
manifest.  The selection split is the OOD validation split where the benchmark has one and
source validation for Kather; the test split is only ever predicted, never read for
training or selection.
"""
from __future__ import annotations

import json
import os
import random

import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image, ImageEnhance
from torch.utils.data import Dataset

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class Packed(Dataset):
    """One split of a packed benchmark: uint8 images in a memory-mapped array."""

    def __init__(self, x_path, y_path, d_path=None, transform=None):
        self.x_path = x_path
        self.y = np.load(y_path).astype(np.int64)
        self.domains = np.load(d_path) if d_path and os.path.exists(d_path) else np.zeros(len(self.y), np.int64)
        self.transform = transform
        self._x = None

    def __len__(self):
        return len(self.y)

    @property
    def x(self):
        if self._x is None:
            self._x = np.load(self.x_path, mmap_mode="r")
        return self._x

    def with_transform(self, transform):
        d = Packed.__new__(Packed)
        d.x_path, d.y, d.domains, d.transform, d._x = self.x_path, self.y, self.domains, transform, None
        return d

    def __getitem__(self, i):
        img = Image.fromarray(np.asarray(self.x[i]))
        return (self.transform(img) if self.transform else img), int(self.y[i])


class Benchmark:
    def __init__(self, name, num_classes, train, selection, test, selection_name,
                 selection_role, test_name):
        self.name, self.num_classes = name, num_classes
        self.train, self.selection, self.test = train, selection, test
        self.selection_name, self.selection_role, self.test_name = selection_name, selection_role, test_name


def load(name, root):
    """Train, selection and test splits of a benchmark from its packed arrays."""
    if name == "camelyon17":
        def split(key):
            return Packed(*(os.path.join(root, f"{key}_{s}.npy") for s in ("x", "y", "center")))
        return Benchmark(name, 2, split("train"), split("ood_val"), split("ood_test"),
                         "ood_val_hospital1", "tuning", "ood_test_hospital2")
    with open(os.path.join(root, "manifest.json"), encoding="utf-8") as f:
        man = json.load(f)

    def split(key):
        return Packed(*(os.path.join(root, f"{key}_{s}.npy") for s in ("x", "y", "domain")))

    targets = man.get("targets") or {k: k for k in man["counts"] if k.startswith("ood")}
    train = split(man.get("source_train", "train"))
    test_key = next(k for k in targets if "test" in k)
    if "ood_val" in targets:          # WILDS: select on the OOD validation split
        sel, sel_name, role = split("ood_val"), targets["ood_val"], "tuning"
    else:                             # Kather: no OOD validation, select on source validation
        sel, sel_name, role = split(man.get("source_val", "id_val")), "source_val", "source_val"
    return Benchmark(name, int(man["num_classes"]), train, sel, split(test_key), sel_name, role,
                     targets[test_key])


# ------------------------------------------------------------------- transforms
class RandomRotation:
    def __init__(self, angles=(0, 90, 180, 270)):
        self.angles = list(angles)

    def __call__(self, img):
        return TF.rotate(img, random.choice(self.angles))


class RandomSharpen:
    def __init__(self, p=0.5, factor=2.0):
        self.p, self.factor = p, factor

    def __call__(self, img):
        return ImageEnhance.Sharpness(img).enhance(self.factor) if random.random() < self.p else img


def train_transform(size):
    """GeNeX's training augmentation (rotation by a multiple of 90 degrees, then
    sharpening with probability 1/2), with ImageNet normalization for the pretrained
    backbone."""
    return T.Compose([T.Resize(size, antialias=True), RandomRotation(), RandomSharpen(),
                      T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])


def eval_transform(size, normalize=True):
    ops = [T.Resize(size, antialias=True), T.ToTensor()]
    return T.Compose(ops + ([T.Normalize(IMAGENET_MEAN, IMAGENET_STD)] if normalize else []))


def normalize_batch(x):
    """ImageNet normalization of a batch in [0, 1] (after input noise is added)."""
    m = torch.tensor(IMAGENET_MEAN, device=x.device).view(1, 3, 1, 1)
    s = torch.tensor(IMAGENET_STD, device=x.device).view(1, 3, 1, 1)
    return (x - m) / s


# ----------------------------------------------------- class-balanced subsampling
def balanced_indices(labels, ratio, rng):
    """GeNeX's down-sampling, extended from two classes to many.  With two classes GeNeX
    keeps the minority class whole and draws min(ratio x minority, majority) images of the
    majority at random.  Here the largest class is drawn down to min(ratio x the
    second-largest class, its size) and every other class is kept whole, which is the same
    rule for two classes; on a long-tailed label set it removes the dominant class's excess
    without shrinking the rare classes.  The random ratio (0.9, 1, 1.1) varies the subset
    between members."""
    counts = np.bincount(labels)
    order = np.argsort(counts)[::-1]
    top, second = int(order[0]), int(counts[order[1]])
    keep = [np.flatnonzero(labels != top)]
    idx = np.flatnonzero(labels == top)
    n = min(len(idx), max(1, int(ratio * second)))
    keep.append(rng.choice(idx, n, replace=False))
    return np.sort(np.concatenate(keep))
