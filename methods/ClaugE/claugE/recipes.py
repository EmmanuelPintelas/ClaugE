"""Recipes: one training augmentation per cluster of draws.

A recipe samples from the distribution of its cluster's draws: it picks one of the
cluster's draws uniformly and jitters each continuous parameter by a Gaussian with the
Silverman bandwidth of that parameter within the cluster (1.06 * std * n^(-1/5)),
clipped to the operator's range; discrete parameters (rotation angle, flip type, dropped
channel) and the operator sequence are kept as drawn.  This reproduces the cluster's
distribution of operators, composites, orders and parameters without summarising it.

``ClaugeAugment`` is the training transform of member c: the untransformed image with
probability 1/k, otherwise a draw from recipe c.  ``summary`` gives the readable form of a
recipe (operator frequencies, composite lengths and sequences, parameter quantiles).
"""
from __future__ import annotations

import collections
import copy
import json
import random

import numpy as np

from .palette import OPS, apply_draw, flat_params, set_flat


class Recipe:
    def __init__(self, draws):
        if not draws:
            raise ValueError("a recipe needs at least one draw")
        self.draws = [[(name, dict(params)) for name, params in d] for d in draws]
        values = collections.defaultdict(list)
        for d in self.draws:
            for name, params in d:
                for key, v in flat_params(params).items():
                    if key in OPS[name].ranges:
                        values[(name, key)].append(v)
        self.bandwidth = {}
        for key, vs in values.items():
            vs = np.asarray(vs)
            self.bandwidth[key] = (1.06 * float(vs.std()) * len(vs) ** -0.2
                                   if len(vs) > 1 else 0.0)

    def sample(self, rnd=random):
        draw = copy.deepcopy(self.draws[rnd.randrange(len(self.draws))])
        for name, params in draw:
            for key, v in flat_params(params).items():
                h = self.bandwidth.get((name, key), 0.0)
                if h > 0:
                    lo, hi = OPS[name].ranges[key]
                    set_flat(params, key, min(hi, max(lo, v + rnd.gauss(0.0, h))))
        return draw

    # ------------------------------------------------------------------ files
    def save(self, path, meta=None):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"meta": meta or {}, "summary": summary(self.draws),
                       "draws": [[[n, p] for n, p in d] for d in self.draws]}, f)

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as f:
            obj = json.load(f)
        return cls([[(n, p) for n, p in d] for d in obj["draws"]])


class ClaugeAugment:
    """Training transform on a [0, 1] tensor: identity with probability ``identity_p``,
    otherwise one draw of the recipe."""

    def __init__(self, recipe_path, identity_p):
        self.recipe = Recipe.load(recipe_path)
        self.identity_p = float(identity_p)

    def __call__(self, x):
        if random.random() < self.identity_p:
            return x
        return apply_draw(x, self.recipe.sample(random))


def build(draws, labels, k):
    """The k recipes of a partition of the draws."""
    labels = np.asarray(labels)
    return [Recipe([draws[i] for i in np.where(labels == c)[0]]) for c in range(k)]


def summary(draws, top=10):
    n = len(draws)
    # operators in their order within each draw (not a set, whose order varies between
    # processes), so that ties keep the same order in every run
    contains = collections.Counter(name for d in draws for name in dict.fromkeys(nm for nm, _ in d))
    lengths = collections.Counter(len(d) for d in draws)
    seqs = collections.Counter(" > ".join(nm for nm, _ in d) for d in draws)
    values = collections.defaultdict(list)
    for d in draws:
        for name, params in d:
            for key, v in flat_params(params).items():
                values[f"{name}.{key}"].append(v)
            for key in OPS[name].discrete:
                values[f"{name}.{key}"].append(float(params[key]))
    quant = {k: [float(q) for q in np.quantile(v, [0.05, 0.5, 0.95])]
             for k, v in sorted(values.items())}
    return {"n_draws": n,
            "operator_share": {k: v / n for k, v in contains.most_common()},
            "length_share": {str(k): v / n for k, v in sorted(lengths.items())},
            "top_sequences": [[s, c / n] for s, c in seqs.most_common(top)],
            "parameter_quantiles_5_50_95": quant}
