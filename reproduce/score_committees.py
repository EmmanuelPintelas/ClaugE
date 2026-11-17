"""Score every committee of the comparison: Table 4 and the intervals quoted in Section 6.

Every method writes its committees to methods/<Method>/outputs/<benchmark>_efficientnet_b0_
kmax<k>/ with the stored predictions of its members.  This script scores them all with the
shared bench (bench/evaluate.py): the test primary metric, the WILDS metric, the expected
calibration error, the selection-split score, the networks run at inference and their
GFLOPs, and, for every method other than ClaugE, the paired difference ClaugE minus the
method with the 95% interval of a cluster bootstrap of the test split (domains resampled,
then points within them; 1,000 replicates, seed 0).

    python reproduce/score_committees.py                       (all four benchmarks)
    python reproduce/score_committees.py --benchmarks camelyon17 kather
    python reproduce/score_committees.py --tag ClaugE:kather=_alpha0

--tag scores, in place of the default committee, one built with other settings, under the
output tag run_claugE.py gives it (here --alpha 0: uniform weights).  Writes
results/main/committees.json; reproduce/tables.py turns it into Table 4.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from bench import evaluate as E  # noqa: E402

METHODS = ["DeepEnsembles", "HyperEnsembles", "AugDiverse", "EoA", "DiWA", "Lookaround", "GeNeX",
           "ClaugE"]
BENCHMARKS = ["camelyon17", "iwildcam", "rxrx1", "kather"]
BOOTSTRAP, SEED = 1000, 0


def committee_dir(method, benchmark, kmax, tags):
    suffix = tags.get((method, benchmark), "")
    return os.path.join(ROOT, "methods", method, "outputs",
                        f"{benchmark}_efficientnet_b0_kmax{kmax}{suffix}")


def bootstrap(pa, pb, y, d, K, metric, B=BOOTSTRAP, seed=SEED):
    """A - B in the metric, with the shared bench's cluster bootstrap."""
    rng = np.random.default_rng(seed)
    doms = np.unique(d)
    by = [np.flatnonzero(d == u) for u in doms]
    f = lambda pr, idx: E.decision_metrics(pr[idx], y[idx], K)[metric]  # noqa: E731
    point = f(pa, np.arange(len(y))) - f(pb, np.arange(len(y)))
    reps = []
    for _ in range(B):
        pick = rng.integers(len(doms), size=len(doms))
        idx = np.concatenate([rng.choice(by[i], size=len(by[i])) for i in pick])
        reps.append(f(pa, idx) - f(pb, idx))
    lo, hi = np.percentile(reps, [2.5, 97.5])
    return float(point), [float(lo), float(hi)]


def parse_tags(items):
    tags = {}
    for item in items or []:
        key, _, suffix = item.partition("=")
        method, _, benchmark = key.partition(":")
        if method not in METHODS or benchmark not in BENCHMARKS or not suffix:
            raise SystemExit(f"--tag {item}: expected Method:benchmark=suffix")
        tags[(method, benchmark)] = suffix
    return tags


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--benchmarks", nargs="+", default=BENCHMARKS, choices=BENCHMARKS)
    ap.add_argument("--tag", action="append", help="Method:benchmark=suffix")
    a = ap.parse_args(argv)
    tags = parse_tags(a.tag)
    out_path = os.path.join(ROOT, "results", "main", "committees.json")
    rows = []
    if os.path.exists(out_path):   # keep the benchmarks this run does not score
        with open(out_path, encoding="utf-8") as f:
            rows = [r for r in json.load(f)["rows"] if r["benchmark"] not in a.benchmarks]
    for bench in a.benchmarks:
        metric = E.primary_metric(bench)
        for kmax in (5, 25):
            ref = committee_dir("ClaugE", bench, kmax, tags)
            cr = E.Committee(ref)
            test = cr.roles["test"]
            y, d = cr.labels(test), cr.domains(test)
            pa = cr.probs(test).argmax(1)
            for method in METHODS:
                path = committee_dir(method, bench, kmax, tags)
                if not os.path.exists(os.path.join(path, "committee.json")):
                    print(f"{bench} k_max {kmax} {method}: no committee at {path}", flush=True)
                    continue
                r = E.evaluate(path)
                c = E.Committee(path)
                role = E.BENCHMARKS[bench]["selection_role"]
                sel = next(n for n, m in c.index["splits"].items() if m["role"] == role)
                row = {"benchmark": bench, "k_max": kmax, "method": method, "metric": metric,
                       "test": r["primary"], "wilds_metric": r["wilds_metric"], "ece": r["ece"],
                       "selection": E.decision_metrics(c.probs(sel).argmax(1), c.labels(sel),
                                                       c.K)[metric],
                       "networks": r["k"], "inference_gflops": r["inference_gflops"],
                       "networks_trained": r["models_trained"]}
                if method != "ClaugE":
                    if not np.array_equal(c.labels(test), y):
                        raise SystemExit(f"{path}: test labels differ from ClaugE's")
                    pb = c.probs(test).argmax(1)
                    row["claugE_minus"], row["claugE_minus_ci95"] = bootstrap(pa, pb, y, d, c.K,
                                                                              metric)
                rows.append(row)
                gap = (f"  ClaugE minus it {row['claugE_minus'] * 100:+.2f} "
                       f"[{row['claugE_minus_ci95'][0] * 100:+.2f}, {row['claugE_minus_ci95'][1] * 100:+.2f}]"
                       if "claugE_minus" in row else "")
                print(f"{bench:<11}{kmax:>3} {method:<15} test {row['test'] * 100:6.2f}  selection "
                      f"{row['selection'] * 100:6.2f}  ECE {row['ece']:.3f}  {row['networks']:>2} "
                      f"networks{gap}", flush=True)
    order = {b: i for i, b in enumerate(BENCHMARKS)}
    rows.sort(key=lambda r: (order[r["benchmark"]], r["k_max"], METHODS.index(r["method"])))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"bootstrap": BOOTSTRAP, "seed": SEED, "rows": rows}, f, indent=1)
    print(f"written {out_path}")


if __name__ == "__main__":
    main()
