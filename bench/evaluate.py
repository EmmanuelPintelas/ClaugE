"""Score any method's committee from its stored predictions, the same way for every method.

A committee directory holds

    committee.json                 method, benchmark, backbone, k_max, members with weights,
                                   aggregation ("weighted average of probabilities",
                                   "average of probabilities", or "single model" for one
                                   merged network), models_trained, extra_data
    predictions/index.json         num_classes and the splits with their roles
    predictions/<split>/labels.npy, domains.npy, member_XXXX.npy   (N, K) logits

    python -m bench.evaluate methods/ClaugE/outputs/kather_efficientnet_b0_k5
    python -m bench.evaluate A_dir --against B_dir      (paired difference with interval)

Metrics: the primary metric of the benchmark (geometric-mean recall below 11 classes,
macro-F1 otherwise), the WILDS metric, accuracy and ECE, on the test split.  Paired
cluster bootstrap: resample domains with replacement, then points within each, 1,000
replicates, 95% percentile interval.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from .benchmarks import BENCHMARKS, gflops, primary_metric


def softmax(logits):
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def decision_metrics(pred, labels, K):
    present = np.unique(labels)
    tc = np.bincount(labels, minlength=K)[present].astype(float)
    pc = np.bincount(pred, minlength=K)[present].astype(float)
    tp = np.bincount(labels[pred == labels], minlength=K)[present].astype(float)
    rec = tp / tc
    prec = np.divide(tp, pc, out=np.zeros_like(tp), where=pc > 0)
    s = prec + rec
    f1 = np.divide(2 * prec * rec, s, out=np.zeros_like(s), where=s > 0)
    return {"acc": float(np.mean(pred == labels)), "balanced_acc": float(rec.mean()),
            "macro_f1": float(f1.mean()),
            "gm": float(np.exp(np.mean(np.log(np.clip(rec, 1e-6, None)))))}


def ece(probs, labels, n_bins=15):
    conf = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == labels).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(total)


class Committee:
    def __init__(self, path):
        self.path = path
        with open(os.path.join(path, "committee.json"), encoding="utf-8") as f:
            self.meta = json.load(f)
        with open(os.path.join(path, "predictions", "index.json"), encoding="utf-8") as f:
            self.index = json.load(f)
        self.K = int(self.index["num_classes"])
        self.benchmark = self.meta["benchmark"]
        self.weights = np.array([m.get("weight", 1.0) for m in self.meta["members"]], float)
        self.weights /= self.weights.sum()
        self.roles = {m["role"]: s for s, m in self.index["splits"].items()}

    def split_dir(self, split):
        return os.path.join(self.path, "predictions", split)

    def labels(self, split):
        return np.load(os.path.join(self.split_dir(split), "labels.npy")).astype(np.int64)

    def domains(self, split):
        return np.load(os.path.join(self.split_dir(split), "domains.npy"))

    def probs(self, split):
        out = None
        for j, w in enumerate(self.weights):
            lg = np.load(os.path.join(self.split_dir(split), f"member_{j:04d}.npy"))
            p = w * softmax(lg.astype(np.float64))
            out = p if out is None else out + p
        return out


def evaluate(path, split=None):
    c = Committee(path)
    split = split or c.roles["test"]
    y, p = c.labels(split), c.probs(split)
    m = decision_metrics(p.argmax(axis=1), y, c.K)
    k = len(c.meta["members"])
    return {"method": c.meta["method"], "benchmark": c.benchmark,
            "backbone": c.meta.get("backbone"), "k_max": c.meta.get("k_max"), "k": k,
            "split": split, "primary_metric": primary_metric(c.benchmark),
            "primary": m[primary_metric(c.benchmark)],
            "wilds_metric": m[BENCHMARKS[c.benchmark]["wilds_metric"]], **m,
            "ece": ece(p, y),
            "inference_gflops": (1 if c.meta.get("aggregation") == "single model" else k)
            * gflops(c.meta["backbone"], c.benchmark) if c.meta.get("backbone") else None,
            "models_trained": c.meta.get("models_trained"),
            "extra_data": c.meta.get("extra_data", "none")}


def paired_bootstrap(path_a, path_b, split=None, B=1000, seed=0):
    """Difference A - B in the primary metric, with a paired cluster-bootstrap interval."""
    a, b = Committee(path_a), Committee(path_b)
    split = split or a.roles["test"]
    y, d = a.labels(split), a.domains(split)
    if not np.array_equal(y, b.labels(split)):
        raise ValueError("the two committees were scored on different points")
    pa, pb = a.probs(split).argmax(1), b.probs(split).argmax(1)
    metric = primary_metric(a.benchmark)

    def score(pred, idx):
        return decision_metrics(pred[idx], y[idx], a.K)[metric]

    rng = np.random.default_rng(seed)
    doms = np.unique(d)
    by_dom = [np.where(d == u)[0] for u in doms]
    point = score(pa, np.arange(len(y))) - score(pb, np.arange(len(y)))
    reps = []
    for _ in range(B):
        pick = rng.integers(len(doms), size=len(doms))
        idx = np.concatenate([rng.choice(by_dom[i], size=len(by_dom[i])) for i in pick])
        reps.append(score(pa, idx) - score(pb, idx))
    lo, hi = np.percentile(reps, [2.5, 97.5])
    return {"difference": float(point), "ci95": [float(lo), float(hi)],
            "excludes_zero": bool(lo > 0 or hi < 0)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("committee")
    ap.add_argument("--against")
    ap.add_argument("--split")
    a = ap.parse_args(argv)
    out = evaluate(a.committee, a.split)
    if a.against:
        out["against"] = os.path.basename(os.path.normpath(a.against))
        out.update(paired_bootstrap(a.committee, a.against, a.split))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
