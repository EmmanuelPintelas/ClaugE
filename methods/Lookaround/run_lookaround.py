"""Lookaround (Zhang et al., 2023), the baseline.

    python run_lookaround.py train     --benchmark kather
    python run_lookaround.py committee --benchmark kather     (one network, k_max 5 and 25)
    python run_lookaround.py smoke     --benchmark kather

train      one Lookaround run: d copies of the network, each on its own augmentation of the
           same mini-batches, averaged every `average_every` steps; the averaged network of
           the best epoch on source validation is stored in work/<benchmark>/run/.
committee  that network, as a committee of one for every budget.
"""
from __future__ import annotations

import argparse
import copy
import os
import sys
import time

import torch
import torch.nn as nn
import torchvision.transforms as T

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import core  # noqa: E402

NAME = "lookaround"


def aug(name):
    if name == "hflip":
        return T.RandomHorizontalFlip()
    if name == "randaugment":
        return T.RandAugment()
    if name == "autoaugment":
        return T.AutoAugment(T.AutoAugmentPolicy.IMAGENET)
    raise ValueError(name)


class Views:
    """The same image under each copy's augmentation."""

    def __init__(self, size, names):
        self.tfs = [T.Compose([T.Resize((size, size), antialias=True), aug(n)] + core.to_tensor_ops())
                    for n in names]

    def __call__(self, img):
        return [tf(img) for tf in self.tfs]


@torch.no_grad()
def average_into(models):
    avg = core.average_states([m.state_dict() for m in models])
    for m in models:
        m.load_state_dict(avg)


def train_lookaround(B, S, recipe, *, device, workers, seed, max_batches=None):
    recipe = core.resolve(recipe, seed)
    core.seed_all(seed)
    names = S["augmentations"]
    base = core.build(B, S, drop_rate=recipe["drop_rate"])
    models = [copy.deepcopy(base).to(device) for _ in names]
    opts = [torch.optim.Adam(m.parameters(), lr=float(recipe["lr"]),
                             weight_decay=float(recipe["weight_decay"])) for m in models]
    scheds = [torch.optim.lr_scheduler.ReduceLROnPlateau(
        o, mode="max", patience=int(recipe["sched_patience"]), factor=float(recipe["sched_factor"]))
        for o in opts]
    scalers = [torch.amp.GradScaler("cuda", enabled=device.type == "cuda") for _ in names]
    loss_fn = nn.CrossEntropyLoss(label_smoothing=float(recipe["label_smoothing"]))
    dl = core.loader(B.train.with_transform(Views(B.image_size, names)), int(recipe["batch_size"]),
                     True, workers, seed, drop_last=True)
    metric = core.metric_for(B.num_classes)
    k = int(S["average_every"])
    best, history, step, since = None, [], 0, 0
    for ep in range(1, int(recipe["epochs"]) + 1):
        t0 = time.time()
        for m in models:
            m.train()
        run, n = 0.0, 0
        for bi, (views, y) in enumerate(dl):
            if max_batches is not None and bi >= max_batches:
                break
            y = y.to(device, non_blocking=True)
            for m, o, sc, x in zip(models, opts, scalers, views):
                o.zero_grad(set_to_none=True)
                with torch.autocast("cuda", enabled=sc.is_enabled()):
                    loss = loss_fn(m(x.to(device, non_blocking=True)), y)
                if not torch.isfinite(loss):
                    continue
                sc.scale(loss).backward()
                sc.step(o)
                sc.update()
                run += float(loss.detach())
                n += 1
            step += 1
            if step % k == 0:
                average_into(models)
        average_into(models)
        lg = core.predict(models[0], B.source_val, device=device, workers=workers, max_batches=max_batches)
        s = core.score(lg.argmax(1), B.source_val.y[:len(lg)], B.num_classes, metric)
        for sch in scheds:
            sch.step(s)
        history.append({"epoch": ep, "loss": run / max(1, n), "source_val": s,
                        "seconds": round(time.time() - t0, 1)})
        improved = best is None or s > best[0]
        core.log(f"    ep {ep:02d}  loss {run / max(1, n):.4f}  source val {metric} {s:.4f}"
                 f"{'  *' if improved else ''}  [{time.time() - t0:.0f}s]")
        if improved:
            best = (s, ep, {kk: v.detach().cpu().clone() for kk, v in models[0].state_dict().items()})
            since = 0
        else:
            since += 1
        if since >= int(recipe["stop_patience"]):
            break
    out = models[0]
    out.load_state_dict(best[2])
    out.eval()
    return out, {"history": history, "best_epoch": best[1], "source_val": best[0], "steps": step}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["train", "committee", "smoke"])
    ap.add_argument("--benchmark", required=True, choices=["camelyon17", "iwildcam", "rxrx1", "kather"])
    ap.add_argument("--kmax", type=int, nargs="+")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args(argv)
    S = core.settings(HERE, NAME)
    B = core.load_benchmark(a.benchmark, S)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kmax, mb, work_root, out_root = a.kmax or S["k_max"], None, "work", "outputs"
    recipe = core.base_recipe(S, a.benchmark)
    if a.stage == "smoke":
        kmax, mb, work_root, out_root = [5], 3, "work_smoke", "outputs_smoke"
        recipe["epochs"] = 1
    run = os.path.join(HERE, work_root, a.benchmark, "run")

    if a.stage in ("train", "smoke") and not core.stored(run):
        core.log(f"{a.benchmark}: Lookaround, {len(S['augmentations'])} copies, average every "
                 f"{S['average_every']} steps")
        seed = S["seed"] * 1000
        model, res = train_lookaround(B, S, recipe, device=device, workers=a.workers, seed=seed,
                                      max_batches=mb)
        core.store_network(model, B, run, device=device, workers=a.workers,
                           info={"seed": seed, "recipe": recipe, **res}, max_batches=mb)
    if a.stage in ("committee", "smoke"):
        if not core.stored(run):
            raise SystemExit("not trained yet")
        for k in kmax:
            out = os.path.join(HERE, out_root, f"{a.benchmark}_{S['arch']}_kmax{k}")
            core.write_committee(out, B, [run], [1.0],
                                 {"method": "Lookaround", "benchmark": a.benchmark,
                                  "backbone": S["arch"], "k_max": k, "aggregation": "single model",
                                  "models_trained": 1, "copies": len(S["augmentations"])})
            core.log(f"{os.path.basename(out)}: one network")
    return 0


if __name__ == "__main__":
    sys.exit(main())
