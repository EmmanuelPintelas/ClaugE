"""Gather what the paper reports from the run folders into results/.

    python reproduce/collect.py [--skip-partitions] [--skip-maps] [--skip-images]

results/ablations/   the reports of methods/ClaugE/run_ablations.py (committee-level, paths,
                     allocation, complementarity, table) for every benchmark that has them
results/partition/<benchmark>/
    partition_k5.json, partition_k20.json   the consensus labels of the 10^5 draws, the
                     references and the agreement of each reference with the consensus
    references.json, timing.json, draws_summary.json
    ablations/<variant>.json     the partitions of the random, one-reference and
                     self-supervised variants; ablations/data_groups.npz, the image groups
                     of the data variant (Camelyon17 and RxRx1)
    transform_map.npz            the 2-D UMAP of 20,000 draws on the first reference
                     (Fig. 2b; needs umap-learn and work/<benchmark>/reduced_ref0.npy)
    reference0.png               the first reference image (Fig. 3; needs the dataset)

The draws and the recipes are not stored: tools/restore_partition.py rebuilds them, byte
for byte, from the drawer's seed and these labels.  reproduce/tables.py and
reproduce/figures.py read only results/.
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

BENCHMARKS = ["camelyon17", "kather", "iwildcam", "rxrx1"]
REPORTS = ["ablations", "committee_level", "paths", "allocation", "complementarity"]
VARIANT_PARTITIONS = ["random", "oneref", "ssl"]
OUT = os.path.join(ROOT, "results")


def copy(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copyfile(src, dst)


def reports():
    src = os.path.join(CLAUGE, "outputs_ablations")
    n = 0
    for b in BENCHMARKS:
        for r in REPORTS:
            p = os.path.join(src, f"{b}_{r}.json")
            if os.path.exists(p):
                copy(p, os.path.join(OUT, "ablations", f"{b}_{r}.json"))
                n += 1
    print(f"ablation reports: {n}")


def partitions():
    for b in BENCHMARKS:
        wd = os.path.join(CLAUGE, "work", b)
        dst = os.path.join(OUT, "partition", b)
        for name in ("references.json", "timing.json", "draws_summary.json"):
            copy(os.path.join(wd, name), os.path.join(dst, name))
        for k in (5, 20):
            copy(os.path.join(wd, f"k{k}", "partition.json"), os.path.join(dst, f"partition_k{k}.json"))
        ad = os.path.join(wd, "ablations")
        for v in VARIANT_PARTITIONS:
            p = os.path.join(ad, v, "partition.json")
            if os.path.exists(p):
                copy(p, os.path.join(dst, "ablations", f"{v}.json"))
        gd = os.path.join(ad, "data")
        if os.path.exists(os.path.join(gd, "train_groups.json")):
            groups = {f"group_{c}": np.load(os.path.join(gd, f"train_group_{c}.npy")).astype(np.int32)
                      for c in range(5)}
            os.makedirs(os.path.join(dst, "ablations"), exist_ok=True)
            np.savez_compressed(os.path.join(dst, "ablations", "data_groups.npz"), **groups)
            copy(os.path.join(gd, "train_groups.json"), os.path.join(dst, "ablations", "data_groups.json"))
        print(f"partition of {b}")


def transform_maps(n=20000):
    """The 2-D map of Fig. 2b: 20,000 draws on the first reference, UMAP of their
    50-dimensional embeddings (random state 0)."""
    import umap
    for b in BENCHMARKS:
        z = np.load(os.path.join(CLAUGE, "work", b, "reduced_ref0.npy"))
        idx = np.random.default_rng(0).choice(len(z), size=min(n, len(z)), replace=False)
        xy = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.1, random_state=0).fit_transform(z[idx])
        np.savez_compressed(os.path.join(OUT, "partition", b, "transform_map.npz"),
                            index=idx.astype(np.int32), xy=xy.astype(np.float32))
        print(f"transform map of {b}")


def reference_images():
    """The first reference image of every benchmark, as the encoder saw it, in 8 bits."""
    from PIL import Image
    import run_claugE as RC
    from claugE import partition
    from netkit import data as D
    S = RC.settings()
    for b in BENCHMARKS:
        with open(os.path.join(CLAUGE, "work", b, "references.json"), encoding="utf-8") as f:
            i = json.load(f)["indices"][0]
        x = partition.to_unit(D.load(b, RC.data_root(S, b)).source_train[i][0])
        arr = x.permute(1, 2, 0).clamp(0, 1).numpy()
        q = np.round(arr * 255)
        if np.abs(q / 255 - arr).max() > 1e-5:
            raise SystemExit(f"{b}: the reference image is not 8-bit; store it otherwise")
        Image.fromarray(q.astype(np.uint8)).save(os.path.join(OUT, "partition", b, "reference0.png"))
        print(f"reference image of {b}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-partitions", action="store_true",
                    help="leave results/partition as it is (it needs methods/ClaugE/work)")
    ap.add_argument("--skip-maps", action="store_true")
    ap.add_argument("--skip-images", action="store_true")
    a = ap.parse_args(argv)
    reports()
    if not a.skip_partitions:
        partitions()
    if not a.skip_maps:
        transform_maps()
    if not a.skip_images:
        reference_images()


if __name__ == "__main__":
    main()
