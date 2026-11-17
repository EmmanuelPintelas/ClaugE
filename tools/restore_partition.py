"""Restore the paper's partitions into methods/ClaugE/work/, without recomputing them.

    python tools/restore_partition.py                 (all four benchmarks)
    python tools/restore_partition.py --benchmarks kather

The partition stage (run_claugE.py partition) embeds 10^6 transformed images and runs UMAP
ten times per benchmark; UMAP's output can differ across library versions and machines.
This tool writes, from results/partition/, the files that stage wrote for the paper, with
the same content: the 10^5 draws (regenerated from the drawer's seed in configs/claugE.json
and checked against the stored summary), the references, and for k = 5 and k = 20 the
consensus labels and the recipes of every group (rebuilt from the draws and the labels).
It also restores the partitions of the ablation variants (random groups, one reference,
the self-supervised encoder) and the image groups of the data variant, so that
run_ablations.py train can start without its prepare stage for those variants.

Then train as usual:  python methods/ClaugE/run_claugE.py train --benchmark kather
--backbone efficientnet_b0.  Existing files are never overwritten (--force replaces them).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLAUGE = os.path.join(ROOT, "methods", "ClaugE")
sys.path.insert(0, CLAUGE)
sys.path.insert(0, os.path.join(CLAUGE, "engine"))

from claugE import drawer, recipes  # noqa: E402

BENCHMARKS = ["camelyon17", "kather", "iwildcam", "rxrx1"]
SRC = os.path.join(ROOT, "results", "partition")


def settings():
    with open(os.path.join(CLAUGE, "configs", "claugE.json"), encoding="utf-8") as f:
        return json.load(f)


def write(path, force, fn):
    if os.path.exists(path) and not force:
        return False
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fn(path)
    return True


def restore(b, S, force):
    src, wd = os.path.join(SRC, b), os.path.join(CLAUGE, "work", b)
    if not os.path.isdir(src):
        raise SystemExit(f"{src}: missing")
    draws_path = os.path.join(wd, "draws.jsonl")
    write(draws_path, force, lambda p: drawer.save(
        drawer.draw_many(S["draws"], seed=S["seed"], continue_prob=S["continue_prob"]), p))
    draws = drawer.load(draws_path)
    with open(os.path.join(src, "draws_summary.json"), encoding="utf-8") as f:
        if json.loads(json.dumps(drawer.summary(draws))) != json.load(f):
            raise SystemExit(f"{b}: the regenerated draws differ from the paper's")
    for name in ("references.json", "timing.json", "draws_summary.json"):
        write(os.path.join(wd, name), force, lambda p, n=name: shutil.copyfile(os.path.join(src, n), p))
    for k in (5, 20):
        with open(os.path.join(src, f"partition_k{k}.json"), encoding="utf-8") as f:
            part = json.load(f)
        kd = os.path.join(wd, f"k{k}")
        write(os.path.join(kd, "partition.json"), force,
              lambda p: shutil.copyfile(os.path.join(src, f"partition_k{k}.json"), p))
        for c, rec in enumerate(recipes.build(draws, part["labels"], k)):
            write(os.path.join(kd, f"recipe_{c}.json"), force,
                  lambda p, r=rec, c=c: r.save(p, meta={"benchmark": b, "k": k, "cluster": c}))
    ad = os.path.join(src, "ablations")
    restored = []
    if os.path.isdir(ad):
        for v in ("random", "oneref", "ssl"):
            p = os.path.join(ad, f"{v}.json")
            if not os.path.exists(p):
                continue
            with open(p, encoding="utf-8") as f:
                part = json.load(f)
            vd = os.path.join(wd, "ablations", v)
            write(os.path.join(vd, "partition.json"), force, lambda q: shutil.copyfile(p, q))
            for c, rec in enumerate(recipes.build(draws, part["labels"], 5)):
                write(os.path.join(vd, f"recipe_{c}.json"), force,
                      lambda q, r=rec, c=c: r.save(q, meta={"benchmark": b, "ablation": v, "k": 5,
                                                            "cluster": c}))
            restored.append(v)
        g = os.path.join(ad, "data_groups.npz")
        if os.path.exists(g):
            groups = np.load(g)
            dd = os.path.join(wd, "ablations", "data")
            for c in range(5):
                write(os.path.join(dd, f"train_group_{c}.npy"), force,
                      lambda q, c=c: np.save(q, groups[f"group_{c}"].astype(np.int64)))
            write(os.path.join(dd, "train_groups.json"), force,
                  lambda q: shutil.copyfile(os.path.join(ad, "data_groups.json"), q))
            restored.append("data")
    print(f"{b}: draws, references, k = 5 and k = 20 partitions and recipes"
          + (f"; ablation variants {', '.join(restored)}" if restored else ""))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--benchmarks", nargs="+", default=BENCHMARKS, choices=BENCHMARKS)
    ap.add_argument("--force", action="store_true", help="replace existing files")
    a = ap.parse_args(argv)
    S = settings()
    for b in a.benchmarks:
        restore(b, S, a.force)


if __name__ == "__main__":
    main()
