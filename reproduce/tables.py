"""Every results table of the paper, from results/ alone (no download, no GPU).

    python reproduce/tables.py                        (writes results/tables/*.tex)
    python reproduce/tables.py --compare paper.tex    (and compares them with a LaTeX source)

main.tex             Table 4   test primary metric of every method, both budgets
partition.tex        Table 5   group sizes, agreement with the consensus, time
ablations.tex        Table 6   ablations on Camelyon17 and RxRx1
paths.tex            Table 7   what each training path yields alone
complementarity.tex  Table 8   pairs across two paths against pairs within them
identity.tex         Table 9   quality and diversity of the two ways to spend five members

Each file holds the table's tabular environment; captions are the paper's.  --compare
finds, in the given source, the tabular of the table labelled tab:<name> and reports every
line that differs.  No number in these tables is typed by hand.
"""
from __future__ import annotations

import argparse
import json
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
OUT = os.path.join(RES, "tables")

NAMES = {"camelyon17": "Camelyon17", "kather": "Kather", "iwildcam": "iWildCam", "rxrx1": "RxRx1"}
METHODS = [("DeepEnsembles", "Deep ensembles"), ("HyperEnsembles", "Hyperparameter ensembles"),
           ("AugDiverse", "Augmentation-diverse"), ("EoA", "Ensemble of averages"),
           ("DiWA", "DiWA"), ("Lookaround", "Lookaround"), ("GeNeX", "GeNeX")]
# the names the paper gives the recipes, from their most frequent operators (Table 7)
RECIPE_NAMES = {
    "camelyon17": ["geometry and sharpening", "pixel noise", "colour and contrast",
                   "channel dropout", "grayscale"],
    "rxrx1": ["geometry and sharpening", "pixel noise", "colour, contrast and channels",
              "grayscale and channel dropout", "photometric and shifts"],
}
RETRAINED = [("nopartition", "whole palette for every path (no partition)"),
             ("random", "random partition of the transforms"),
             ("data", "partition of the data instead"),
             ("oneref", "one reference image instead of ten"),
             ("ssl", "self-supervised encoder (DINO)"),
             ("identity0", r"identity share $0$ instead of $1/k$"),
             ("identity0.5", r"identity share $\frac12$ instead of $1/k$"),
             ("k20", r"$k=20$ paths, one candidate each")]
COMMITTEE = [("the best path on validation alone, its five candidates",
              "the best path alone, its five candidates"),
             ("random candidates per path instead of the best on validation",
              "random candidate per path, not the best"),
             ("weights per member instead of per path (uniform within a path)",
              "weights per member, not per path"),
             ("uniform weights across paths instead of Eq. (weights)", "uniform weights across paths"),
             ("fixed alpha = 1, r = 1/4 instead of the selected ones", r"fixed $\alpha=1$, $\rho=\frac14$"),
             ("prior alone (delta = 0) instead of Eq. (fit)", r"prior alone ($\delta=0$)"),
             ("unbounded fit instead of Eq. (fit)", r"unbounded fit ($\rho=\infty$)")]


def load(*parts):
    with open(os.path.join(RES, *parts), encoding="utf-8") as f:
        return json.load(f)


def signed(v, ci=None, digits=2):
    """Points with their sign, starred when the interval excludes zero; zero unsigned."""
    text = f"{v:+.{digits}f}"
    if float(text) == 0:
        text = f"{0:.{digits}f}"
    star = r"^\ast" if ci is not None and (ci[0] > 0 or ci[1] < 0) else ""
    return f"${text}{star}$"


# ------------------------------------------------------------------ Table 4
def table_main():
    rows = load("main", "committees.json")["rows"]
    val = {(r["method"], r["benchmark"], r["k_max"]): r["test"] for r in rows}
    cols = [(b, k) for b in ("camelyon17", "iwildcam", "rxrx1", "kather") for k in (5, 25)]
    best = {c: max(v for (m, b, k), v in val.items() if (b, k) == c) for c in cols}

    def cell(m, c):
        v = val[(m, *c)]
        s = f"{v * 100:.1f}"
        return rf"\textbf{{{s}}}" if v == best[c] else s

    out = [r"\begin{tabular}{@{}lcccccccc@{}}", r"\toprule",
           r"& \multicolumn{2}{c}{Camelyon17} & \multicolumn{2}{c}{iWildCam} & "
           r"\multicolumn{2}{c}{RxRx1} & \multicolumn{2}{c}{Kather} \\",
           r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}",
           r"method & $k\le5$ & $k\le25$ & $k\le5$ & $k\le25$ & $k\le5$ & $k\le25$ & $k\le5$ & $k\le25$ \\",
           r"\midrule"]
    for m, name in METHODS:
        out.append(f"{name} & " + " & ".join(cell(m, c) for c in cols) + r" \\")
    out += [r"\midrule", r"\method{} & " + " & ".join(cell("ClaugE", c) for c in cols) + r" \\",
            r"\bottomrule", r"\end{tabular}"]
    return out


# ------------------------------------------------------------------ Table 5
def table_partition():
    out = [r"\begin{tabular}{@{}lccccc@{}}", r"\toprule",
           r"& \multicolumn{2}{c}{$k=5$} & \multicolumn{2}{c}{$k=20$} & \\",
           r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
           r"benchmark & sizes & agreement & sizes & agreement & time \\", r"\midrule"]
    for b in ("camelyon17", "kather", "iwildcam", "rxrx1"):
        cells = [NAMES[b]]
        for k in (5, 20):
            p = load("partition", b, f"partition_k{k}.json")
            s, a = p["sizes"], p["agreement_ari"]
            cells += [f"{min(s) / 1000:.1f}--{max(s) / 1000:.1f}",
                      f"{sum(a) / len(a):.2f} ({min(a):.2f}--{max(a):.2f})"]
        t = load("partition", b, "timing.json")
        sec = sum(v["embed_s"] + v["reduce_s"] for key, v in t.items() if key.startswith("ref"))
        sec += t["cluster_k5_s"] + t["cluster_k20_s"]
        cells.append(f"{sec / 3600:.1f} h")
        out.append(" & ".join(cells) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}"]
    return out


# ------------------------------------------------------------------ Table 6
def table_ablations():
    benches = ("camelyon17", "rxrx1")
    rows = {b: load("ablations", f"{b}_ablations.json")["rows"] for b in benches}

    def retrained(b, name, kmax):
        r = next((r for r in rows[b] if r.get("name") == name and r["k_max"] == kmax), None)
        return "---" if r is None else signed(r["delta"] * 100, [c * 100 for c in r["ci95"]])

    def committee(b, variant, kmax):
        r = next((r for r in rows[b] if r.get("name") == "committee-level"
                  and r["variant"] == variant and r["k_max"] == kmax), None)
        if r is None:
            return "---"
        return signed(r["delta"] * 100, [c * 100 for c in r["ci95"]] if "ci95" in r else None)

    out = [r"\begin{tabular}{@{}p{0.47\columnwidth}cccc@{}}", r"\toprule",
           r"& \multicolumn{2}{c}{Camelyon17} & \multicolumn{2}{c}{RxRx1} \\",
           r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
           r"variant & $k\le5$ & $k\le25$ & $k\le5$ & $k\le25$ \\", r"\midrule",
           r"\multicolumn{5}{@{}l}{\emph{training paths (members retrained)}} \\"]
    for name, label in RETRAINED:
        out.append(f"{label} & " + " & ".join(retrained(b, name, k) for b in benches for k in (5, 25))
                   + r" \\")
    out += [r"\midrule", r"\multicolumn{5}{@{}l}{\emph{committee (same members)}} \\"]
    for variant, label in COMMITTEE:
        out.append(f"{label} & " + " & ".join(committee(b, variant, k) for b in benches for k in (5, 25))
                   + r" \\")
    out += [r"\bottomrule", r"\end{tabular}"]
    return out


# ------------------------------------------------------------------ Table 7
def table_paths():
    out = [r"\begin{tabular}{@{}lcc@{}}", r"\toprule", r"training recipe & one network & five \\",
           r"\midrule"]
    for i, b in enumerate(("camelyon17", "rxrx1")):
        p = load("ablations", f"{b}_paths.json")
        if i:
            out.append(r"\midrule")
        out.append(rf"\multicolumn{{3}}{{@{{}}l}}{{\emph{{{NAMES[b]}}}}} \\")
        for c, name in enumerate(RECIPE_NAMES[b]):
            r = p["paths"][c]
            out.append(f"{name} & {r['single_test_mean'] * 100:.1f} & {r['five_candidates_test'] * 100:.1f} \\\\")
        for v, name in (("nopartition", "whole palette (no partition)"), ("random", "random groups")):
            r = p["variants"][v]
            out.append(f"{name} & {r['single_test_mean'] * 100:.1f} & {r['five_networks_test'] * 100:.1f} \\\\")
    out += [r"\bottomrule", r"\end{tabular}"]
    return out


# ------------------------------------------------------------------ Table 8
def table_complementarity():
    benches = ("camelyon17", "kather", "iwildcam", "rxrx1")
    rep = {b: load("ablations", f"{b}_complementarity.json") for b in benches}

    def cell(b, matching, group, field):
        p, ci = rep[b]["point"][matching][group][field], rep[b]["ci95"][matching][group][field]
        return signed(p * 100, [c * 100 for c in ci])

    out = [r"\begin{tabular}{@{}lcccc@{}}", r"\toprule",
           "& " + " & ".join(NAMES[b] for b in benches) + r" \\", r"\midrule",
           r"\multicolumn{5}{@{}l}{\emph{decision metric, across minus within (points)}} \\"]
    for label, group, field in (("all quadruplets", "all", "score"),
                                ("matched quality", "smallest third", "score"),
                                ("both wrong, matched", "smallest third", "double_fault")):
        out.append(f"{label} & " + " & ".join(cell(b, "test metric", group, field) for b in benches) + r" \\")
    out += [r"\midrule", r"\multicolumn{5}{@{}l}{\emph{diversity, across minus within (log loss $\times100$)}} \\"]
    for label, group in (("all quadruplets", "all"), ("matched quality", "smallest third")):
        out.append(f"{label} & " + " & ".join(cell(b, "test log loss", group, "loglik") for b in benches)
                   + r" \\")
    out += [r"\bottomrule", r"\end{tabular}"]
    return out


# ------------------------------------------------------------------ Table 9
def table_identity():
    benches = ("camelyon17", "kather", "iwildcam", "rxrx1")
    rep = {b: load("ablations", f"{b}_allocation.json")["log_loss"] for b in benches}
    out = [r"\begin{tabular}{@{}lcccc@{}}", r"\toprule",
           "& " + " & ".join(NAMES[b] for b in benches) + r" \\", r"\midrule"]
    for i, (key, label) in enumerate((("committee", "committee of the five paths"),
                                      ("best_path", "five candidates of the best path"))):
        if i:
            out.append(r"\midrule")
        out.append(rf"\multicolumn{{5}}{{@{{}}l}}{{\emph{{{label}}}}} \\")
        for field in ("quality", "diversity", "loss"):
            out.append(f"{field} & " + " & ".join(f"{rep[b][key][field]:.3f}" for b in benches) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}"]
    return out


TABLES = {"main": table_main, "partition": table_partition, "ablations": table_ablations,
          "paths": table_paths, "complementarity": table_complementarity, "identity": table_identity}


def paper_tabular(source, name):
    """The lines of the tabular environment of the table labelled tab:<name>."""
    i = source.find(rf"\label{{tab:{name}}}")
    if i < 0:
        return None
    j = source.find(r"\begin{tabular}", i)
    k = source.find(r"\end{tabular}", j) + len(r"\end{tabular}")
    return [ln.strip() for ln in source[j:k].splitlines() if ln.strip()]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--compare", help="a LaTeX source whose tables to compare with")
    a = ap.parse_args(argv)
    os.makedirs(OUT, exist_ok=True)
    source = None
    if a.compare:
        with open(a.compare, encoding="utf-8") as f:
            source = f.read()
    differences = 0
    for name, fn in TABLES.items():
        lines = fn()
        with open(os.path.join(OUT, f"{name}.tex"), "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        if source is None:
            print(f"written results/tables/{name}.tex")
            continue
        theirs = paper_tabular(source, name)
        if theirs is None:
            print(f"{name}: no table labelled tab:{name} in {a.compare}")
            continue
        norm = lambda s: re.sub(r"\s+", " ", s)  # noqa: E731
        diff = [(m, t) for m, t in zip(lines, theirs) if norm(m) != norm(t)]
        if len(lines) != len(theirs):
            diff.append((f"{len(lines)} lines", f"{len(theirs)} lines"))
        differences += len(diff)
        print(f"{name}: {'identical' if not diff else f'{len(diff)} differing lines'}")
        for m, t in diff:
            print(f"   generated: {m}\n   source:    {t}")
    if source is not None:
        print("all tables identical" if not differences else f"{differences} differing lines")


if __name__ == "__main__":
    main()
