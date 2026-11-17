"""From recipe pools to the ClaugE committee: selection within recipes, weights across them.

The k recipes (5) each have a pool of candidates (5) trained by the engine.  A budget of
k_max members is spent as m = k_max / k members per recipe, so the number of recipes does
not grow with the budget: k_max = 5 takes the best candidate of every recipe, k_max = 25
all five.  Within recipe c the m candidates best on the selection split (OOD validation
where the benchmark has one, source validation otherwise) are averaged uniformly into the
recipe model M_c; across recipes the committee averages the recipe models with a softmax
of their scores measured in robust standard deviations.  With s_c the selection-split score
of M_c, med the median of the scores and sigma their robust standard deviation (1.4826
times the median absolute deviation, floored at 3% of the median so that a tight pack is
weighted nearly uniformly),

    z_c = (s_c - med) / sigma,     w_c proportional to exp(alpha * z_c):

a recipe one robust standard deviation above another gets e^alpha times its weight.
Recipes with close scores get close but graded weights, so each contributes its
complementary invariances; a recipe far below the pack gets exponentially little.  The
median and MAD are not moved by that one weak recipe, so it cannot widen the scale it is
measured on.  How strongly the scores should count differs between benchmarks (a tight
pack on Camelyon17, scores far apart on RxRx1), so the strength alpha, with the radius of
the fit below, is chosen from ALPHA_GRID x FIT_RADIUS_GRID by the selection-split score of
the resulting committee (ties to the smaller alpha, then the smaller radius); either can
also be fixed.

These prior weights are then refined by a light fit on the selection split: w_c is
proportional to exp(alpha z_c + d_c), each correction |d_c| <= r = FIT_RADIUS (1/4 of a
robust standard deviation at alpha = 1), so the ratio of any two weights changes by at most
e^(2r); within that box d maximises the class-balanced log-likelihood of the committee (the
mean over classes of the mean log-probability of the true class), by L-BFGS-B from d = 0
with an exact gradient, for at most FIT_ITERS iterations (it stops earlier once converged).
The log-likelihood is strictly concave in the weights and rewards recipes that are right
where the others are wrong; a smooth version of the metric itself (mean probability of the
true class) is nearly linear in the weights and drives them to one recipe.  The fit is kept
only if it helps out of sample: on 2-fold cross-fitting over the selection split (halves of
its domains, or random halves when it has one domain), prior and fit from one half,
committee scored with the actual metric on the other, the fit must beat the prior on
average; otherwise d = 0.  Wider boxes are not the default: on the one-hospital validation
split of Camelyon17, whose random halves cannot detect over-fitting to that hospital, a box
of 1 moved 76% of the weight onto one recipe at k_max = 25.  Since averaging is linear, the
committee is a weighted average of its k_max members with weight w_c / m each, which is how
it is written for the shared evaluation.
"""
from __future__ import annotations

import json
import math
import os
import shutil

import numpy as np
from scipy.optimize import minimize

ALPHA = "auto"             # strength: weight ratio e^alpha per robust standard deviation
ALPHA_GRID = (0.1, 0.3, 1.0, 3.0, 10.0)
SIGMA_FLOOR = 0.03         # relative to the median score
FIT_RADIUS = "auto"        # bound on each correction of the log-weights; 0 = no fit
FIT_RADIUS_GRID = (0.25, 0.5)
FIT_ITERS = 100            # maximum L-BFGS-B iterations of the fit


def metric_for(K):
    return "gm" if K <= 10 else "macro_f1"


def softmax(logits):
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def decision_score(pred, labels, K, metric):
    """Geometric-mean recall or macro-F1 over the classes present in ``labels``."""
    present = np.unique(labels)
    tc = np.bincount(labels, minlength=K)[present].astype(float)
    tp = np.bincount(labels[pred == labels], minlength=K)[present].astype(float)
    rec = tp / tc
    if metric == "gm":
        return float(np.exp(np.mean(np.log(np.clip(rec, 1e-6, None)))))
    pc = np.bincount(pred, minlength=K)[present].astype(float)
    prec = np.divide(tp, pc, out=np.zeros_like(tp), where=pc > 0)
    s = prec + rec
    f1 = np.divide(2 * prec * rec, s, out=np.zeros_like(s), where=s > 0)
    return float(f1.mean())


def robust_z(scores):
    s = np.asarray(scores, dtype=float)
    med = float(np.median(s))
    sigma = max(1.4826 * float(np.median(np.abs(s - med))), SIGMA_FLOOR * abs(med), 1e-12)
    return (s - med) / sigma


def weights_for(scores, alpha=1.0):
    """Softmax of the scores in robust standard deviations; alpha = 0 gives uniform."""
    z = alpha * robust_z(scores)
    w = np.exp(z - z.max())
    return w / w.sum()


def _softmax1(t):
    w = np.exp(t - t.max())
    return w / w.sum()


def balanced_loglik(w, Q, a):
    """Class-balanced log-likelihood of the committee sum_c w_c P_c and its gradient in w.
    Q (k, N): probability each recipe model gives the true class; a (N,): 1 / (number of
    classes present x size of the sample's class), so that sum_i a_i log q_i is the mean over
    classes of the mean log-probability of the true class."""
    q = np.clip(w @ Q, 1e-12, None)
    return float(a @ np.log(q)), Q @ (a / q)


def fit_weights(probs, labels, K, metric, alpha=1.0, radius=0.25, iters=FIT_ITERS):
    """Recipe scores, prior weights, fitted weights and a record of the fit, from
    recipe-model probabilities (k, N, K) on one split.  The log-weights move at most
    ``radius`` from the prior's (inf: unbounded); at most ``iters`` iterations."""
    scores = [decision_score(p.argmax(1), labels, K, metric) for p in probs]
    w0 = weights_for(scores, alpha)
    present, inv = np.unique(labels, return_inverse=True)
    a = 1.0 / (len(present) * np.bincount(inv)[inv])
    Q = probs[:, np.arange(len(labels)), labels].astype(np.float64)
    ll0, _ = balanced_loglik(w0, Q, a)
    info = {"loglik_prior": ll0}
    if radius == 0 or iters == 0:
        return scores, w0, w0, info
    base = np.log(w0)

    def neg(d):
        w = _softmax1(base + d)
        ll, g = balanced_loglik(w, Q, a)
        return -ll, -(w * (g - w @ g))       # chain rule through the softmax

    bounds = None if math.isinf(radius) else [(-radius, radius)] * len(w0)
    res = minimize(neg, np.zeros(len(w0)), jac=True, method="L-BFGS-B", bounds=bounds,
                   options={"maxiter": int(iters)})
    w = _softmax1(base + res.x)
    info.update(loglik_fit=-float(res.fun), iterations=int(res.nit),
                converged=bool(res.success), message=str(res.message),
                corrections=[float(x) for x in res.x])
    return scores, w0, w, info


def committee_pred(probs, w):
    """probs: (k, N, K) recipe-model probabilities; w: (k,) weights."""
    return np.tensordot(np.asarray(w), probs, axes=1).argmax(axis=1)


def halves(domains, seed=0):
    """Two halves of a split: whole domains to each, or at random within one domain."""
    rng = np.random.default_rng(seed)
    doms = np.unique(domains)
    if len(doms) > 1:
        rng.shuffle(doms)
        a = np.isin(domains, doms[: len(doms) // 2])
    else:
        a = rng.random(len(domains)) < 0.5
    return np.where(a)[0], np.where(~a)[0]


def check_fit(probs, labels, domains, K, metric, alpha=1.0, radius=0.25,
              iters=FIT_ITERS, seed=0):
    """Does the fit beat the prior out of sample?  Prior and fit from one half of the split,
    committee scored with the actual metric on the other, both directions averaged; a tie
    keeps the prior."""
    A, B = halves(domains, seed)
    vals = {"prior": [], "fit": []}
    for fit, ev in ((A, B), (B, A)):
        _, w0, w, _ = fit_weights(probs[:, fit], labels[fit], K, metric, alpha, radius, iters)
        for name, ww in (("prior", w0), ("fit", w)):
            vals[name].append(decision_score(committee_pred(probs[:, ev], ww), labels[ev], K,
                                             metric))
    table = {name: float(np.mean(v)) for name, v in vals.items()}
    return round(table["fit"], 6) > round(table["prior"], 6), table


def committee_weights(probs, labels, domains, K, metric, alpha, radius, iters, check, seed):
    """Recipe scores, prior weights, final weights and the record of the fit for one alpha
    and one radius."""
    scores, w0, w, info = fit_weights(probs, labels, K, metric, alpha, radius, iters)
    fit = {"radius": radius if math.isfinite(radius) else "inf", "max_iterations": iters,
           "check": check, "check_table": None, "kept": not (radius == 0 or iters == 0)}
    if fit["kept"] and check:
        fit["kept"], fit["check_table"] = check_fit(probs, labels, domains, K, metric, alpha,
                                                    radius, iters, seed)
    if not fit["kept"]:
        w = w0
    fit.update(info)
    return scores, w0, w, fit


def select_settings(probs, labels, domains, K, metric, alphas, radii, iters, check, seed):
    """alpha and radius of the committee best on the selection split; ties go to the
    smaller alpha (more even weights), then the smaller radius."""
    table = []
    for a in alphas:
        for r in radii:
            _, _, w, fit = committee_weights(probs, labels, domains, K, metric, a, r, iters,
                                             check, seed)
            s = decision_score(committee_pred(probs, w), labels, K, metric)
            table.append({"alpha": a, "radius": r, "score": s, "fit_kept": fit["kept"]})
    best = max(table, key=lambda t: (round(t["score"], 6), -t["alpha"], -t["radius"]))
    return best["alpha"], best["radius"], table


# ------------------------------------------------------------------ engine pools
def _read_pool(pool_dir):
    with open(os.path.join(pool_dir, "predictions", "index.json"), encoding="utf-8") as f:
        index = json.load(f)
    rows = {}
    with open(os.path.join(pool_dir, "manifest.jsonl"), encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                rows[int(r["member_id"])] = r
    return index, rows


def roles(index):
    by_role = {m["role"]: s for s, m in index["splits"].items()}
    return by_role.get("tuning") or by_role.get("source_val"), by_role.get("test")


def ranked(pool_dir, metric):
    """Accepted candidates of one recipe pool, best first on the selection split."""
    index, rows = _read_pool(pool_dir)
    sel, _ = roles(index)
    scored = [(int(m), float(rows[int(m)]["metrics"][sel][metric])) for m in index["accepted"]]
    return sorted(scored, key=lambda t: (-t[1], t[0])), len(rows)


def build(recipe_pools, out_dir, meta, per_recipe, alpha=ALPHA, radius=FIT_RADIUS,
          iters=FIT_ITERS, check=True, seed=0, limit=None):
    """Assemble the committee of len(recipe_pools) * per_recipe members in the shared
    output format (predictions, checkpoints, committee.json).  alpha: strength of the prior
    (0 = uniform, "auto" = chosen from ALPHA_GRID on the selection split); radius: bound of
    the fit (0 = prior only, inf = unbounded, "auto" = chosen from FIT_RADIUS_GRID); iters:
    its maximum iterations; check: keep the fit only if it beats the prior out of sample;
    limit: consider only the first `limit` accepted candidates of every pool (by member id,
    i.e. the first slots of its seed), as the ablations with fewer candidates do."""
    first_index, _ = _read_pool(recipe_pools[0])
    K = int(first_index["num_classes"])
    metric = metric_for(K)
    sel_split, test_split = roles(first_index)
    chosen, trained = [], 0
    for c, pool in enumerate(recipe_pools):
        cands, n_rows = ranked(pool, metric)
        if limit is not None:
            first = sorted(cands, key=lambda t: t[0])[:limit]
            cands = sorted(first, key=lambda t: (-t[1], t[0]))
            n_rows = len(first)
        trained += n_rows
        if len(cands) < per_recipe:
            raise ValueError(f"recipe {c} has {len(cands)} candidates, {per_recipe} needed")
        chosen.append(cands[:per_recipe])

    members = [(c, mid, s) for c, picks in enumerate(chosen) for mid, s in picks]
    os.makedirs(os.path.join(out_dir, "models"), exist_ok=True)
    for split in first_index["splits"]:
        d = os.path.join(out_dir, "predictions", split)
        os.makedirs(d, exist_ok=True)
        for f in ("labels.npy", "domains.npy"):
            shutil.copy2(os.path.join(recipe_pools[0], "predictions", split, f), d)
        for j, (c, mid, _) in enumerate(members):
            shutil.copy2(os.path.join(recipe_pools[c], "predictions", split, f"member_{mid:04d}.npy"),
                         os.path.join(d, f"member_{j:04d}.npy"))
    for j, (c, mid, _) in enumerate(members):
        src = os.path.join(recipe_pools[c], "models", f"member_{mid:04d}.pt")
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(out_dir, "models", f"member_{j:04d}.pt"))

    pdir = os.path.join(out_dir, "predictions", sel_split)
    labels = np.load(os.path.join(pdir, "labels.npy")).astype(np.int64)
    domains = np.load(os.path.join(pdir, "domains.npy"))
    k = len(recipe_pools)
    probs = np.zeros((k, len(labels), K), dtype=np.float32)
    for j, (c, _, _) in enumerate(members):
        lg = np.load(os.path.join(pdir, f"member_{j:04d}.npy")).astype(np.float64)
        probs[c] += softmax(lg).astype(np.float32) / per_recipe
    selection = None
    if alpha == "auto" or radius == "auto":
        alpha, radius, selection = select_settings(
            probs, labels, domains, K, metric,
            ALPHA_GRID if alpha == "auto" else (alpha,),
            FIT_RADIUS_GRID if radius == "auto" else (radius,), iters, check, seed)
    recipe_scores, w0, w, fit = committee_weights(probs, labels, domains, K, metric, alpha,
                                                  radius, iters, check, seed)

    committee = dict(
        meta, recipes=k, per_recipe=per_recipe, k=len(members), num_classes=K, metric=metric,
        selection_split=sel_split, test_split=test_split,
        aggregation=("weighted average of recipe models, each the uniform average of its "
                     "members"),
        alpha=alpha, sigma_floor=SIGMA_FLOOR, fit=fit, settings_selection=selection,
        models_trained=trained,
        recipe_models=[{"recipe": c, "selection_score": recipe_scores[c],
                        "prior_weight": float(w0[c]), "weight": float(w[c]),
                        "candidates": [mid for mid, _ in chosen[c]]} for c in range(k)],
        members=[{"id": j, "recipe": c, "candidate": mid, "selection_score": s,
                  "weight": float(w[c]) / per_recipe}
                 for j, (c, mid, s) in enumerate(members)])
    with open(os.path.join(out_dir, "predictions", "index.json"), "w", encoding="utf-8") as f:
        json.dump({"num_classes": K, "splits": first_index["splits"],
                   "members": list(range(len(members)))}, f, indent=1)
    with open(os.path.join(out_dir, "committee.json"), "w", encoding="utf-8") as f:
        json.dump(committee, f, indent=1)
    return committee


def prune(recipe_pools):
    """Delete the candidate pools once every committee has been written."""
    for p in recipe_pools:
        shutil.rmtree(p)
