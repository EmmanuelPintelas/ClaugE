"""GenE: the genetic generation of GeNeX's model pool (Algorithm 1 of the GeNeX paper).

Generation 0 trains N networks from ImageNet weights under randomly drawn settings for a
few epochs, with no validation monitoring.  Each of G generations then draws random
parent pairs from the evolving pool, makes children by uniform crossover and mutation of
their weights, fine-tunes every child for one epoch, and moves N / G children chosen at
random into the output pool M.  The other children, with N - N / G fresh networks, are
trained for one epoch each and form the next evolving pool.  After G generations M holds N
networks from different stages of evolution.

Every step writes its network to disk before the next starts and every random choice is
seeded by its position, so an interrupted run resumes where it stopped.  A generation's
folders are deleted once the next evolving pool is complete.
"""
from __future__ import annotations

import glob
import json
import os
import random
import shutil
import time

import numpy as np
import torch

from . import nets
from .data import balanced_indices


def _rng(*key):
    return random.Random("|".join(str(k) for k in key))


def _seed(*key):
    return _rng(*key).randrange(2 ** 31)


def _log_row(work, row):
    with open(os.path.join(work, "gene_log.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def _train_one(model, bench, cfg, rng, *, epochs, lr, batch_size, device, workers, seed,
               max_batches):
    wd = rng.choice(cfg["weight_decays"])
    ratio = rng.choice(cfg["ratios"])
    idx = balanced_indices(bench.train.y, ratio, np.random.default_rng(seed))
    t0 = time.time()
    losses = nets.train(model, bench.train, idx, epochs=epochs, lr=lr, weight_decay=wd,
                        batch_size=batch_size, device=device, workers=workers, seed=seed,
                        max_batches=max_batches)
    return {"epochs": epochs, "lr": lr, "weight_decay": wd, "ratio": ratio,
            "batch_size": batch_size, "images": int(len(idx)), "losses": losses,
            "seconds": round(time.time() - t0, 1)}


def run(bench, cfg, work, device, workers, log=print, max_batches=None):
    arch, K, size, seed = cfg["arch"], bench.num_classes, cfg["image_size"], cfg["seed"]
    N, G = cfg["N"], cfg["G"]
    sel = N // G
    pairs, per_pair = cfg["parent_pairs"], cfg["children_per_pair"]
    if pairs * per_pair < sel:
        raise ValueError(f"{pairs} pairs x {per_pair} children < {sel} to select per generation")
    out_pool = os.path.join(work, "M")
    os.makedirs(out_pool, exist_ok=True)
    if os.path.exists(os.path.join(work, "gene_done.json")):
        log("GenE already complete")
        return sorted(glob.glob(os.path.join(out_pool, "*.pt")))

    # ---- generation 0: N networks trained from ImageNet weights, no validation
    evol = os.path.join(work, "evol0")
    os.makedirs(evol, exist_ok=True)
    if not os.path.exists(os.path.join(evol, "done")):
        for i in range(N):
            path = os.path.join(evol, f"m{i:03d}.pt")
            if os.path.exists(path):
                continue
            rng = _rng(seed, "gen0", i)
            s = _seed(seed, "gen0-train", i)
            nets.seed_all(s)
            model = nets.build(arch, K, size, pretrained=True)
            row = _train_one(model, bench, cfg, rng, epochs=rng.choice(cfg["epochs"]),
                             lr=rng.choice(cfg["learning_rates"]),
                             batch_size=rng.choice(cfg["batch_sizes"]), device=device,
                             workers=workers, seed=s, max_batches=max_batches)
            nets.save(model.cpu(), path)
            _log_row(work, {"stage": "gen0", "model": i, **row})
            log(f"  gen 0: network {i + 1}/{N} ({row['epochs']} ep, lr {row['lr']:g}, "
                f"{row['images']} images, {row['seconds']:.0f} s)")
        open(os.path.join(evol, "done"), "w").close()

    # ---- evolution
    for g in range(G):
        gdir = os.path.join(work, f"g{g}")
        if os.path.exists(os.path.join(gdir, "done")):
            evol = os.path.join(work, f"evol{g + 1}")
            continue
        parents = sorted(glob.glob(os.path.join(evol, "*.pt")))
        cdir = os.path.join(gdir, "children")
        os.makedirs(cdir, exist_ok=True)
        for j in range(pairs):
            a, b = _rng(seed, "pair", g, j).sample(parents, 2)
            for t in range(per_pair):
                path = os.path.join(cdir, f"c{j:03d}_{t}.pt")
                if os.path.exists(path):
                    continue
                s = _seed(seed, "child", g, j, t)
                gen = torch.Generator().manual_seed(s)
                sd = nets.mutate(nets.uniform_crossover(nets.state(a), nets.state(b), gen), gen,
                                 cfg["mutation_rate"], cfg["mutation_sigma"])
                child = nets.from_state(sd, arch, K, size)
                row = _train_one(child, bench, cfg, _rng(seed, "child-cfg", g, j, t),
                                 epochs=1, lr=cfg["child_lr"], batch_size=cfg["child_batch_size"],
                                 device=device, workers=workers, seed=s, max_batches=max_batches)
                nets.save(child.cpu(), path)
                _log_row(work, {"stage": f"g{g}-child", "pair": j, "child": t,
                                "parents": [os.path.basename(a), os.path.basename(b)], **row})
            log(f"  generation {g + 1}/{G}: children of pair {j + 1}/{pairs} done")
        children = sorted(glob.glob(os.path.join(cdir, "*.pt")))
        chosen = sorted(_rng(seed, "select", g).sample(children, sel))
        for i, c in enumerate(chosen):
            dst = os.path.join(out_pool, f"m_g{g}_{i}.pt")
            if not os.path.exists(dst):
                shutil.copy2(c, dst)
        with open(os.path.join(gdir, "selected.json"), "w", encoding="utf-8") as f:
            json.dump([os.path.basename(c) for c in chosen], f)
        log(f"  generation {g + 1}/{G}: {sel} children moved to the output pool")

        if g < G - 1:
            # the unselected children and N - sel fresh networks, one epoch each
            nxt = os.path.join(work, f"evol{g + 1}")
            os.makedirs(nxt, exist_ok=True)
            rest = [c for c in children if c not in chosen] + [None] * (N - sel)
            for i, src in enumerate(rest):
                path = os.path.join(nxt, f"r{i:03d}.pt")
                if os.path.exists(path):
                    continue
                s = _seed(seed, "rest", g, i)
                nets.seed_all(s)
                model = nets.load(src) if src else nets.build(arch, K, size, pretrained=True)
                rng = _rng(seed, "rest-cfg", g, i)
                row = _train_one(model, bench, cfg, rng, epochs=1,
                                 lr=rng.choice(cfg["learning_rates"]),
                                 batch_size=rng.choice(cfg["batch_sizes"]), device=device,
                                 workers=workers, seed=s, max_batches=max_batches)
                nets.save(model.cpu(), path)
                _log_row(work, {"stage": f"g{g}-rest", "model": i,
                                "from": os.path.basename(src) if src else "fresh", **row})
            log(f"  generation {g + 1}/{G}: next evolving pool of {len(rest)} trained")
        open(os.path.join(gdir, "done"), "w").close()
        if not cfg.get("keep_intermediate", False):
            shutil.rmtree(evol, ignore_errors=True)
            shutil.rmtree(cdir, ignore_errors=True)
        evol = os.path.join(work, f"evol{g + 1}")

    pool = sorted(glob.glob(os.path.join(out_pool, "*.pt")))
    trained = N + G * pairs * per_pair + (G - 1) * (pairs * per_pair - sel + N - sel)
    with open(os.path.join(work, "gene_done.json"), "w", encoding="utf-8") as f:
        json.dump({"pool": len(pool), "networks_trained": trained, "N": N, "G": G,
                   "parent_pairs": pairs, "children_per_pair": per_pair}, f, indent=1)
    log(f"GenE complete: {len(pool)} networks in the output pool, {trained} trained in all")
    return pool
