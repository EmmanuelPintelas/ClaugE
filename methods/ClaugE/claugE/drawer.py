"""The drawer: T random transforms over the palette, single operators and composites.

A draw starts with one operator chosen uniformly among the N of the palette; then, with
probability ``continue_prob`` (1/2), a further operator is chosen uniformly among those
not yet in the draw, and so on, so a draw never repeats an operator and
P(length >= m) = continue_prob^(m - 1).  Operators are applied in the order drawn.  Each
operator's parameters are drawn independently within its ranges (palette.py), and every
draw is recorded with all its parameters so that a cluster of draws maps back to
operators and parameter regions exactly.
"""
from __future__ import annotations

import collections
import json

import numpy as np

from .palette import PALETTE


def draw_one(rng, palette=PALETTE, continue_prob=0.5):
    remaining = list(range(len(palette)))
    draw = []
    while True:
        j = remaining.pop(int(rng.integers(len(remaining))))
        op = palette[j]
        draw.append((op.name, op.sample(rng)))
        if not remaining or rng.random() >= continue_prob:
            return draw


def draw_many(T, seed=0, palette=PALETTE, continue_prob=0.5):
    rng = np.random.default_rng(seed)
    return [draw_one(rng, palette, continue_prob) for _ in range(T)]


def save(draws, path):
    with open(path, "w", encoding="utf-8") as f:
        for d in draws:
            f.write(json.dumps([[name, params] for name, params in d]) + "\n")


def load(path):
    with open(path, encoding="utf-8") as f:
        return [[(name, params) for name, params in json.loads(line)] for line in f if line.strip()]


def summary(draws):
    """Operator frequencies and the distribution of composite lengths."""
    lengths = collections.Counter(len(d) for d in draws)
    ops = collections.Counter(name for d in draws for name, _ in d)
    return {"T": len(draws), "lengths": dict(sorted(lengths.items())),
            "operator_counts": dict(ops.most_common())}
