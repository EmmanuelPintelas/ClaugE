"""Standard supervised training, prediction and committee writing for the baselines.

The same file is copied into every baseline folder (methods/<name>/core.py), so that each
method stays self-contained while all of them share one base training procedure: the
benchmark's base recipe (configs/<method>.json, "base"), EfficientNet-B0 from ImageNet
weights, Adam, cross-entropy, a plateau learning-rate schedule, the light augmentation of
the comparison (rotation by 0, 90 or 270 degrees and a horizontal flip), and the
checkpoint of the best epoch on source validation, unless a method says otherwise.

Splits are named exactly as in every other committee (the shared bench pairs committees by
split name): Camelyon17 ood_val_hospital1 / ood_test_hospital2, iWildCam and RxRx1
ood_val / ood_test, Kather source_val / ood_test_crc_val_7k.  The selection split is the
OOD validation split where the benchmark has one and source validation for Kather; the
test split is predicted only to be stored for the shared evaluation.
"""
from __future__ import annotations

import json
import os
import random
import time
from collections import OrderedDict

import numpy as np
import timm
import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(os.path.dirname(HERE))


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


# ------------------------------------------------------------------------- data
class Packed(Dataset):
    """One split of a packed benchmark: uint8 images in a memory-mapped array."""

    def __init__(self, x_path, y_path, d_path=None, transform=None):
        self.x_path = x_path
        self.y = np.load(y_path).astype(np.int64)
        self.domains = (np.load(d_path) if d_path and os.path.exists(d_path)
                        else np.zeros(len(self.y), np.int64))
        self.transform = transform
        self._x = None

    def __len__(self):
        return len(self.y)

    @property
    def x(self):
        if self._x is None:
            self._x = np.load(self.x_path, mmap_mode="r")
        return self._x

    def with_transform(self, transform):
        d = Packed.__new__(Packed)
        d.x_path, d.y, d.domains, d.transform, d._x = self.x_path, self.y, self.domains, transform, None
        return d

    def __getitem__(self, i):
        img = Image.fromarray(np.asarray(self.x[i]))
        return (self.transform(img) if self.transform else img), int(self.y[i])


class Benchmark:
    pass


def load_benchmark(name, settings):
    """train, source_val (checkpoint selection), selection (the method's choices) and test."""
    b = settings["benchmarks"][name]
    root = os.path.normpath(os.path.join(PROJECT, b["data"]))
    B = Benchmark()
    B.name, B.image_size = name, b["image_size"]
    if name == "camelyon17":
        def split(key):
            return Packed(*(os.path.join(root, f"{key}_{s}.npy") for s in ("x", "y", "center")))
        B.num_classes, B.train, B.source_val = 2, split("train"), split("id_val")
        B.selection, B.selection_name, B.selection_role = split("ood_val"), "ood_val_hospital1", "tuning"
        B.test, B.test_name = split("ood_test"), "ood_test_hospital2"
        return B
    with open(os.path.join(root, "manifest.json"), encoding="utf-8") as f:
        man = json.load(f)

    def split(key):
        return Packed(*(os.path.join(root, f"{key}_{s}.npy") for s in ("x", "y", "domain")))

    targets = man.get("targets") or {k: k for k in man["counts"] if k.startswith("ood")}
    B.num_classes = int(man["num_classes"])
    B.train, B.source_val = split(man.get("source_train", "train")), split(man.get("source_val", "id_val"))
    if "ood_val" in targets:
        B.selection, B.selection_name, B.selection_role = split("ood_val"), targets["ood_val"], "tuning"
    else:
        B.selection, B.selection_name, B.selection_role = B.source_val, "source_val", "source_val"
    test_key = next(k for k in targets if "test" in k)
    B.test, B.test_name = split(test_key), targets[test_key]
    return B


class RandomRightAngle:
    def __init__(self, angles=(0, 90, 270)):
        self.angles = list(angles)

    def __call__(self, img):
        return TF.rotate(img, random.choice(self.angles))


def light_ops(size):
    """The light augmentation of the comparison, before tensor conversion."""
    return [T.Resize((size, size), antialias=True), RandomRightAngle(), T.RandomHorizontalFlip()]


def to_tensor_ops():
    return [T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]


def light_transform(size):
    return T.Compose(light_ops(size) + to_tensor_ops())


def eval_transform(size):
    return T.Compose([T.Resize((size, size), antialias=True)] + to_tensor_ops())


# ----------------------------------------------------------------------- models
class Net(nn.Module):
    def __init__(self, arch, num_classes, pretrained=True, drop_rate=0.0, image_size=224):
        super().__init__()
        self.arch, self.num_classes, self.image_size = arch, num_classes, image_size
        self.backbone = timm.create_model(arch, pretrained=pretrained, num_classes=0,
                                          drop_rate=drop_rate)
        self.backbone.eval()
        with torch.no_grad():
            width = int(self.backbone(torch.zeros(1, 3, image_size, image_size)).shape[1])
        self.head = nn.Linear(width, num_classes)

    def forward(self, x):
        return self.head(self.backbone(x))


def build(B, settings, drop_rate=0.0, pretrained=True):
    return Net(settings["arch"], B.num_classes, pretrained, drop_rate, B.image_size)


def save(model, path, extra=None):
    payload = {"arch": model.arch, "num_classes": model.num_classes,
               "image_size": model.image_size, "state_dict": model.state_dict()}
    payload.update(extra or {})
    torch.save(payload, path)


def load(path, device="cpu"):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = Net(ck["arch"], ck["num_classes"], pretrained=False, image_size=ck["image_size"])
    m.load_state_dict(ck["state_dict"])
    return m.to(device)


def average_states(states, weights=None):
    """Weighted average of several networks' floating-point tensors (weights and
    normalization statistics); integer buffers come from the first network."""
    w = np.full(len(states), 1.0 / len(states)) if weights is None else np.asarray(weights, float)
    out = OrderedDict()
    for k, v in states[0].items():
        if v.is_floating_point():
            out[k] = sum(float(wi) * s[k].float() for wi, s in zip(w, states)).to(v.dtype)
        else:
            out[k] = v.clone()
    return out


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed % 2 ** 32)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _worker_init(_):
    s = torch.utils.data.get_worker_info().seed % 2 ** 32
    random.seed(s)
    np.random.seed(s)


def loader(ds, batch_size, shuffle, workers, seed=0, drop_last=False):
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=workers,
                      pin_memory=torch.cuda.is_available(), drop_last=drop_last,
                      generator=g if shuffle else None,
                      worker_init_fn=_worker_init if workers > 0 else None)


# ---------------------------------------------------------------------- metrics
def metric_for(K):
    return "gm" if K <= 10 else "macro_f1"


def softmax(logits):
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def score(pred, labels, K, metric):
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


def nll(probs, labels):
    return float(-np.mean(np.log(np.clip(probs[np.arange(len(labels)), labels], 1e-12, None))))


# --------------------------------------------------------------------- training
@torch.no_grad()
def predict(model, ds, *, device, workers, batch_size=256, max_batches=None):
    model.to(device).eval()
    out = []
    for bi, (x, _) in enumerate(loader(ds.with_transform(eval_transform(model.image_size)),
                                       batch_size, False, workers)):
        if max_batches is not None and bi >= max_batches:
            break
        with torch.autocast("cuda", enabled=device.type == "cuda"):
            lg = model(x.to(device, non_blocking=True))
        out.append(lg.float().cpu().numpy())
    return np.concatenate(out)


def train(model, B, recipe, *, device, workers, seed, transform=None, indices=None,
          select="best", step_hook=None, max_batches=None, log_fn=log):
    """Adam, cross-entropy with label smoothing, plateau schedule on the source-validation
    metric, early stopping after ``stop_patience`` epochs without improvement.  select:
    "best" keeps the best epoch on source validation, "last" the final weights (and trains
    all epochs).  step_hook(model, step) runs after every optimizer step."""
    recipe = resolve(recipe, seed)
    seed_all(seed)
    tf = transform or light_transform(B.image_size)
    ds = B.train.with_transform(tf)
    if indices is not None:
        ds = Subset(ds, list(indices))
    dl = loader(ds, int(recipe["batch_size"]), True, workers, seed, drop_last=True)
    opt = torch.optim.Adam(model.parameters(), lr=float(recipe["lr"]),
                           weight_decay=float(recipe["weight_decay"]))
    sched = (torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="max", patience=int(recipe.get("sched_patience", 1)),
        factor=float(recipe.get("sched_factor", 0.5)))
        if recipe.get("scheduler", "plateau") == "plateau" else None)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=float(recipe.get("label_smoothing", 0.0)))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    metric = metric_for(B.num_classes)
    model.to(device)
    best, history, step, since = None, [], 0, 0
    patience = int(recipe.get("stop_patience", 10 ** 9)) if select == "best" else 10 ** 9
    for ep in range(1, int(recipe["epochs"]) + 1):
        t0 = time.time()
        model.train()
        run, n = 0.0, 0
        for bi, (x, y) in enumerate(dl):
            if max_batches is not None and bi >= max_batches:
                break
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=scaler.is_enabled()):
                loss = loss_fn(model(x), y)
            if not torch.isfinite(loss):
                continue
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            step += 1
            if step_hook is not None:
                step_hook(model, step)
            run += float(loss.detach())
            n += 1
        lg = predict(model, B.source_val, device=device, workers=workers, max_batches=max_batches)
        s = score(lg.argmax(1), B.source_val.y[:len(lg)], B.num_classes, metric)
        if sched is not None:
            sched.step(s)
        history.append({"epoch": ep, "loss": run / max(1, n), "source_val": s,
                        "lr": opt.param_groups[0]["lr"], "seconds": round(time.time() - t0, 1)})
        improved = best is None or s > best[0]
        log_fn(f"    ep {ep:02d}  loss {run / max(1, n):.4f}  source val {metric} {s:.4f}"
               f"{'  *' if improved else ''}  [{time.time() - t0:.0f}s]")
        if improved:
            best = (s, ep, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
            since = 0
        else:
            since += 1
        if since >= patience:
            break
    if select == "best" and best is not None:
        model.load_state_dict(best[2])
    model.eval()
    return {"history": history, "best_epoch": best[1] if best else None,
            "source_val": best[0] if best else None, "steps": step,
            "epochs": int(recipe["epochs"])}


def resolve(recipe, seed):
    """A recipe entry given as {"choice": [...]} (e.g. the epochs on iWildCam, as in
    ClaugE's pool settings) is drawn per network, from the network's seed."""
    rng = random.Random(f"recipe|{seed}")
    return {k: (rng.choice(v["choice"]) if isinstance(v, dict) and "choice" in v else v)
            for k, v in recipe.items()}


@torch.no_grad()
def recalibrate_bn(model, B, *, device, workers, batches=200, seed=0):
    """Re-estimate batch-normalization statistics on training images (after averaging
    weights), with the light augmentation, in train mode without gradient steps."""
    for m in model.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            m.reset_running_stats()
            m.momentum = None                      # cumulative average
    model.to(device).train()
    dl = loader(B.train.with_transform(light_transform(B.image_size)), 128, True, workers, seed)
    for bi, (x, _) in enumerate(dl):
        if bi >= batches:
            break
        with torch.autocast("cuda", enabled=device.type == "cuda"):
            model(x.to(device, non_blocking=True))
    model.eval()


# ------------------------------------------------------------- stored networks
def store_network(model, B, run_dir, *, device, workers, info, max_batches=None):
    """Checkpoint and logits on the selection and test splits of one trained network."""
    os.makedirs(run_dir, exist_ok=True)
    save(model.cpu(), os.path.join(run_dir, "model.pt"))
    for ds, name in ((B.selection, "selection"), (B.test, "test")):
        lg = predict(model, ds, device=device, workers=workers, max_batches=max_batches)
        np.save(os.path.join(run_dir, f"{name}.npy"), lg.astype(np.float16))
    with open(os.path.join(run_dir, "info.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, indent=1)
    model.cpu()


def stored(run_dir):
    return os.path.exists(os.path.join(run_dir, "info.json"))


def read_info(run_dir):
    with open(os.path.join(run_dir, "info.json"), encoding="utf-8") as f:
        return json.load(f)


def selection_logits(run_dir):
    return np.load(os.path.join(run_dir, "selection.npy")).astype(np.float64)


def write_committee(out_dir, B, run_dirs, weights, meta, n=None):
    """The shared output format: committee.json, predictions/index.json and, per split,
    labels.npy, domains.npy and member_XXXX.npy (logits), plus models/member_XXXX.pt."""
    import shutil
    os.makedirs(os.path.join(out_dir, "models"), exist_ok=True)
    index = {"num_classes": B.num_classes, "splits": {}, "members": list(range(len(run_dirs)))}
    for ds, name, role, key in ((B.selection, B.selection_name, B.selection_role, "selection"),
                                (B.test, B.test_name, "test", "test")):
        d = os.path.join(out_dir, "predictions", name)
        os.makedirs(d, exist_ok=True)
        m = None
        for j, r in enumerate(run_dirs):
            lg = np.load(os.path.join(r, f"{key}.npy"))
            m = len(lg)
            np.save(os.path.join(d, f"member_{j:04d}.npy"), lg)
        np.save(os.path.join(d, "labels.npy"), ds.y[:m])
        np.save(os.path.join(d, "domains.npy"), ds.domains[:m])
        index["splits"][name] = {"role": role, "n": int(m), "labels": f"{name}/labels.npy",
                                 "domains": f"{name}/domains.npy"}
    for j, r in enumerate(run_dirs):
        shutil.copy2(os.path.join(r, "model.pt"), os.path.join(out_dir, "models", f"member_{j:04d}.pt"))
    w = np.asarray(weights, float)
    w = w / w.sum()
    committee = dict(meta, k=len(run_dirs), num_classes=B.num_classes,
                     selection_split=B.selection_name, test_split=B.test_name, extra_data="none",
                     members=[{"id": j, "run": os.path.basename(os.path.normpath(r)),
                               "weight": float(w[j])} for j, r in enumerate(run_dirs)])
    with open(os.path.join(out_dir, "predictions", "index.json"), "w", encoding="utf-8") as f:
        json.dump(index, f, indent=1)
    with open(os.path.join(out_dir, "committee.json"), "w", encoding="utf-8") as f:
        json.dump(committee, f, indent=1)
    return committee


def settings(method_dir, name):
    with open(os.path.join(method_dir, "configs", f"{name}.json"), encoding="utf-8") as f:
        return json.load(f)


def base_recipe(S, benchmark, **override):
    r = dict(S["base"]["all"])
    r.update(S["base"].get(benchmark, {}))
    r.update(override)
    return r
