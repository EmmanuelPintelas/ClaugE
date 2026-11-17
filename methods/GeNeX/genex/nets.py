"""Networks, training, prediction and the weight-level operators of GeNeX."""
from __future__ import annotations

import random
from collections import OrderedDict

import numpy as np
import timm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from .data import eval_transform, normalize_batch, train_transform


class Net(nn.Module):
    """A timm backbone without its classifier and a linear head of K classes, as in GeNeX."""

    def __init__(self, arch, num_classes, pretrained=True, image_size=224):
        super().__init__()
        self.arch, self.num_classes, self.image_size = arch, num_classes, image_size
        self.model = timm.create_model(arch, pretrained=pretrained, num_classes=0)
        self.model.eval()
        with torch.no_grad():
            width = int(self.model(torch.zeros(1, 3, image_size, image_size)).shape[1])
        self.head = nn.Linear(width, num_classes)

    def forward(self, x):
        return self.head(self.model(x))


def build(arch, num_classes, image_size, pretrained=True):
    return Net(arch, num_classes, pretrained, image_size)


def save(model, path):
    torch.save({"arch": model.arch, "num_classes": model.num_classes,
                "image_size": model.image_size, "state_dict": model.state_dict()}, path)


def load(path, device="cpu"):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = Net(ck["arch"], ck["num_classes"], pretrained=False, image_size=ck["image_size"])
    m.load_state_dict(ck["state_dict"])
    return m.to(device)


def state(path):
    return torch.load(path, map_location="cpu", weights_only=False)["state_dict"]


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed % 2 ** 32)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _loader(ds, batch_size, shuffle, workers, seed=0):
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=workers,
                      pin_memory=torch.cuda.is_available(), persistent_workers=False,
                      generator=g if shuffle else None)


def train(model, dataset, indices, *, epochs, lr, weight_decay, batch_size, device, workers,
          seed, head_only=False, amp=True, max_batches=None):
    """Plain supervised training with Adam and cross-entropy, no validation monitoring
    (GeNeX trains for a fixed, small number of epochs).  head_only trains the output head
    with the backbone frozen; the model stays in train mode, so the batch-normalization
    statistics follow the training data, as in GeNeX's prototype refinement."""
    seed_all(seed)
    ds = Subset(dataset.with_transform(train_transform(model.image_size)), indices.tolist())
    loader = _loader(ds, batch_size, True, workers, seed)
    for p in model.model.parameters():
        p.requires_grad_(not head_only)
    params = list(model.head.parameters()) if head_only else list(model.parameters())
    opt = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=amp and device.type == "cuda")
    model.to(device).train()
    losses = []
    for _ in range(int(epochs)):
        run, n = 0.0, 0
        for bi, (x, y) in enumerate(loader):
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
            run += float(loss.detach())
            n += 1
        losses.append(run / max(1, n))
    for p in model.parameters():
        p.requires_grad_(True)
    model.eval()
    return losses


@torch.no_grad()
def predict(model, dataset, *, device, workers, batch_size=256, noise_std=0.0, seed=0,
            amp=True, max_batches=None):
    """Logits on a split, in order.  noise_std > 0 adds Gaussian noise to the images in
    [0, 1] (clamped) before normalization: GeNeX's robustness probe."""
    ds = dataset.with_transform(eval_transform(model.image_size, normalize=noise_std == 0))
    loader = _loader(ds, batch_size, False, workers)
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    model.to(device).eval()
    out = []
    for bi, (x, _) in enumerate(loader):
        if max_batches is not None and bi >= max_batches:
            break
        x = x.to(device, non_blocking=True)
        if noise_std > 0:
            noise = torch.randn(x.shape, generator=gen, device=device) * noise_std
            x = normalize_batch((x + noise).clamp(0.0, 1.0))
        with torch.autocast("cuda", enabled=amp and device.type == "cuda"):
            lg = model(x)
        out.append(lg.float().cpu().numpy())
    return np.concatenate(out)


# ------------------------------------------------------------ genetic operators
def uniform_crossover(sa, sb, gen):
    """Each element of every tensor from one parent or the other, with equal odds."""
    child = OrderedDict()
    for k, a in sa.items():
        if "num_batches_tracked" in k:
            continue
        if a.is_floating_point():
            mask = torch.rand(a.shape, generator=gen) > 0.5
            child[k] = torch.where(mask, a, sb[k])
        else:
            child[k] = a.clone()
    return child


def mutate(sd, gen, rate=0.05, sigma=0.01):
    """Gaussian noise of standard deviation sigma on a weight tensor (2-D or more) with
    probability rate, tensor by tensor."""
    for k, v in sd.items():
        if "weight" in k and v.dim() >= 2 and torch.rand(1, generator=gen).item() < rate:
            sd[k] = v + torch.randn(v.shape, generator=gen) * sigma
    return sd


def average(states):
    """Layer-wise average of several networks' weights (and normalization statistics)."""
    out = OrderedDict()
    for k, v in states[0].items():
        if "num_batches_tracked" in k:
            continue
        if v.is_floating_point():
            out[k] = torch.stack([s[k] for s in states]).mean(dim=0)
        else:
            out[k] = v.clone()
    return out


def from_state(sd, arch, num_classes, image_size):
    m = Net(arch, num_classes, pretrained=False, image_size=image_size)
    m.load_state_dict(sd, strict=False)
    return m
