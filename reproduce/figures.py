"""Every results figure of the paper, from results/ alone (no download, no GPU).

    python reproduce/figures.py [--out results/figures]

operator_shares.pdf     Fig. 2a  per benchmark, the share of each recipe's transforms that
                                 use each operator (k = 5)
transform_maps.pdf      Fig. 2b  the transforms on the first reference image, 2-D UMAP,
                                 coloured by consensus group
recipes_<benchmark>.jpg Fig. 3   the first reference image and six draws of every recipe
cost_camelyon17.pdf     Fig. 4   accuracy against inference and training cost

The draws are regenerated from the drawer's seed and the recipes from the stored labels,
with the method's own code (methods/ClaugE/claugE), exactly as the partition wrote them.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FixedLocator, NullLocator  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLAUGE = os.path.join(ROOT, "methods", "ClaugE")
sys.path.insert(0, CLAUGE)
sys.path.insert(0, os.path.join(CLAUGE, "engine"))
RES = os.path.join(ROOT, "results")

BENCHES = ["camelyon17", "kather", "iwildcam", "rxrx1"]
NAMES = {"camelyon17": "Camelyon17", "kather": "Kather", "iwildcam": "iWildCam", "rxrx1": "RxRx1"}
OP_LABELS = ["rotation", "shift+flip", "resized crop", "sharpen", "CLAHE", "grayscale",
             "Gaussian noise", r"$\ell_\infty$ noise", "photometric", "stain", "gain+gamma",
             "channel drop"]
K = 5
SAMPLES = 6


def load(*parts):
    with open(os.path.join(RES, *parts), encoding="utf-8") as f:
        return json.load(f)


_DRAWS = None


def draws():
    """The 10^5 draws, regenerated from the drawer's seed (the same on every benchmark)."""
    global _DRAWS
    if _DRAWS is None:
        from claugE import drawer
        with open(os.path.join(CLAUGE, "configs", "claugE.json"), encoding="utf-8") as f:
            S = json.load(f)
        raw = drawer.draw_many(S["draws"], seed=S["seed"], continue_prob=S["continue_prob"])
        _DRAWS = [[(name, json.loads(json.dumps(params))) for name, params in d] for d in raw]
    return _DRAWS


def recipe_list(bench, k=K):
    from claugE import recipes
    return recipes.build(draws(), load("partition", bench, f"partition_k{k}.json")["labels"], k)


# ---------------------------------------------------------------- Figs. 2 and 3
def shares_figure(out):
    from claugE.palette import PALETTE
    from claugE.recipes import summary
    fig, axes = plt.subplots(1, len(BENCHES), figsize=(7.2, 2.4), sharey=True)
    for ax, bench in zip(axes, BENCHES):
        m = np.zeros((K, len(PALETTE)))
        for c, rec in enumerate(recipe_list(bench)):
            shares = summary(rec.draws)["operator_share"]
            m[c] = [shares.get(op.name, 0.0) for op in PALETTE]
        im = ax.imshow(m.T, vmin=0, vmax=1, cmap="Blues", aspect="auto")
        ax.set_title(NAMES[bench], fontsize=8)
        ax.set_xticks(range(K))
        ax.set_xticklabels([str(c + 1) for c in range(K)], fontsize=7)
        ax.set_xlabel("recipe", fontsize=7)
        ax.set_yticks(range(len(PALETTE)))
        ax.set_yticklabels(OP_LABELS, fontsize=7)
    fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02).ax.tick_params(labelsize=6)
    fig.savefig(os.path.join(out, "operator_shares.pdf"), bbox_inches="tight")
    plt.close(fig)


def maps_figure(out):
    fig, axes = plt.subplots(1, len(BENCHES), figsize=(7.2, 1.9))
    cmap = plt.get_cmap("tab10")
    for ax, bench in zip(axes, BENCHES):
        m = np.load(os.path.join(RES, "partition", bench, "transform_map.npz"))
        lab = np.array(load("partition", bench, f"partition_k{K}.json")["labels"])
        ax.scatter(m["xy"][:, 0], m["xy"][:, 1], s=0.3, c=[cmap(int(l)) for l in lab[m["index"]]],
                   rasterized=True)
        ax.set_title(NAMES[bench], fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout(pad=0.3)
    fig.savefig(os.path.join(out, "transform_maps.pdf"), dpi=300)
    plt.close(fig)


def recipe_figure(out, bench):
    import torch
    from PIL import Image
    from claugE.palette import apply_draw
    from claugE.partition import to_unit

    def show(ax, x):
        ax.imshow(x.permute(1, 2, 0).clamp(0, 1).numpy())
        ax.set_xticks([])
        ax.set_yticks([])

    ref = to_unit(Image.open(os.path.join(RES, "partition", bench, "reference0.png")).convert("RGB"))
    fig, axes = plt.subplots(K, SAMPLES + 1, figsize=(1.15 * (SAMPLES + 1), 1.2 * K))
    rnd = random.Random(0)
    for c, rec in enumerate(recipe_list(bench)):
        show(axes[c, 0], ref)
        axes[c, 0].set_ylabel(f"recipe {c + 1}", fontsize=8)
        for s in range(SAMPLES):
            gen = torch.Generator().manual_seed(1000 * c + s)
            show(axes[c, s + 1], apply_draw(ref.clone(), rec.sample(rnd), gen))
    axes[0, 0].set_title("reference", fontsize=8)
    fig.tight_layout(pad=0.2)
    fig.savefig(os.path.join(out, f"recipes_{bench}.jpg"), dpi=220, pil_kwargs={"quality": 90})
    plt.close(fig)


# ---------------------------------------------------------------------- Fig. 4
COST_METHODS = [("Deep ensembles", "DeepEnsembles", "s"), ("Hyperparameter ens.", "HyperEnsembles", "^"),
                ("Augmentation-diverse", "AugDiverse", "D"), ("Ensemble of averages", "EoA", "v"),
                ("DiWA", "DiWA", "P"), ("Lookaround", "Lookaround", "X"), ("GeNeX", "GeNeX", "*"),
                ("ClaugE", "ClaugE", "o")]
ACCENT, GRAY, LINE_GRAY = "#2a78d6", "#52514e", "#898781"
GRID, AXIS, INK = "#e1e0d9", "#c3c2b7", "#0b0b0b"


def _mark(ax, x, y, marker, color, size, filled, z):
    """Open marker: k_max = 5; filled, ringed in the surface colour so that overlapping
    marks stay apart: k_max = 25, or the same model at both budgets."""
    size = size * (1.45 if marker == "*" else 1.0)
    if filled:
        ax.plot(x, y, marker=marker, ms=size + 1.0, ls="none", mfc=color, mec="white", mew=0.6, zorder=z)
    else:
        ax.plot(x, y, marker=marker, ms=size, ls="none", mfc="white", mec=color, mew=0.9, zorder=z)


def _cost_panel(ax, rows, x_key):
    for _, folder, marker in COST_METHODS:
        a, b = [r for r in rows if r["folder"] == folder]
        accent = folder == "ClaugE"
        color, size, z = (ACCENT, 6.5, 4) if accent else (GRAY, 5.5, 3)
        if a["test_primary"] == b["test_primary"] and a["networks"] == b["networks"]:
            _mark(ax, a[x_key], a["test_primary"] * 100, marker, color, size, True, z)
            continue
        ax.plot([a[x_key], b[x_key]], [a["test_primary"] * 100, b["test_primary"] * 100],
                color=ACCENT if accent else LINE_GRAY, lw=1.1 if accent else 0.7, zorder=z - 1)
        for r, filled in ((a, False), (b, True)):
            _mark(ax, r[x_key], r["test_primary"] * 100, marker, color, size, filled, z)
    ax.set_xscale("log")
    ax.set_ylim(64, 100)
    ax.yaxis.set_major_locator(FixedLocator([70, 80, 90, 100]))
    ax.grid(axis="y", color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
    ax.tick_params(colors=INK, labelsize=6.5, length=2.5, width=0.5)
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_ylabel("test GM (%)", fontsize=7, color=INK)


def cost_figure(out, bench="camelyon17"):
    rows = load("cost", f"{bench}.json")["rows"]
    plt.rcParams.update({"font.size": 7, "pdf.fonttype": 42})
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.45, 3.9))
    _cost_panel(ax1, rows, "inference_gflops")
    gf = sorted({r["inference_gflops"] for r in rows if r["networks"] in (1, 5, 25)})
    nets = {r["inference_gflops"]: r["networks"] for r in rows}
    ax1.xaxis.set_major_locator(FixedLocator(gf))
    ax1.set_xticklabels([f"{g:.2f} ({nets[g]})" for g in gf])
    ax1.set_xlim(0.055, 2.6)
    ax1.set_xlabel("inference GFLOPs per image (networks run)", fontsize=7, color=INK)
    _cost_panel(ax2, rows, "hours")
    ticks = [0.5, 1, 2, 5, 10, 20]
    ax2.xaxis.set_major_locator(FixedLocator(ticks))
    ax2.set_xticklabels([f"{t:g}" for t in ticks])
    ax2.set_xlim(0.45, 26)
    ax2.set_xlabel("training hours on one laptop", fontsize=7, color=INK)
    c5 = next(r for r in rows if r["folder"] == "ClaugE" and r["k_max"] == 5)
    for ax, key in ((ax1, "inference_gflops"), (ax2, "hours")):
        ax.annotate("ClaugE", (c5[key], c5["test_primary"] * 100), xytext=(-6, 4),
                    textcoords="offset points", ha="right", va="bottom", fontsize=6.5, color=INK)
    for label, ax in (("(a)", ax1), ("(b)", ax2)):
        ax.text(-0.16, 1.04, label, transform=ax.transAxes, fontsize=7, color=INK, va="bottom")
    handles = [Line2D([], [], marker=mk, ls="none", ms=5 * (1.45 if mk == "*" else 1.0),
                      mfc=ACCENT if f == "ClaugE" else GRAY, mec="white", mew=0.4, label=n)
               for n, f, mk in COST_METHODS]
    fig.legend(handles=handles, loc="upper center", ncol=3, fontsize=6.2, frameon=False,
               handletextpad=0.2, columnspacing=0.9, borderaxespad=0.1)
    fig.subplots_adjust(left=0.14, right=0.98, top=0.84, bottom=0.1, hspace=0.62)
    fig.savefig(os.path.join(out, f"cost_{bench}.pdf"))
    plt.close(fig)
    plt.rcParams.update(plt.rcParamsDefault)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(RES, "figures"))
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    for bench in BENCHES:
        recipe_figure(a.out, bench)
    shares_figure(a.out)
    maps_figure(a.out)
    cost_figure(a.out)
    print("figures written to", a.out)


if __name__ == "__main__":
    main()
