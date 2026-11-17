"""Partition the augmentation space: references, embeddings, clustering and consensus.

For one benchmark and one k:

1. ``choose_references``: r images of the source training split, the medoids of an
   r-means clustering of their encoder features (r = 1: the image closest to the
   centroid).
2. ``embed_draws``: every draw is applied to one reference image and the result embedded
   by the frozen encoder.  Noise patterns are seeded per draw, so a draw is the same
   transform on every reference.  Draws are applied by CPU threads while the GPU encodes.
3. ``reduce`` and ``cluster``: features -> PCA (256) -> UMAP (50) -> GMM with k
   components, separately for each reference, so the clusters reflect what a transform
   does to that image rather than differences between images.
4. ``consensus``: each draw has one label per reference; their one-hot codes are
   concatenated and clustered into k groups, one partition of the draws.
"""
from __future__ import annotations

import collections
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import umap
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture
from torch.utils.data import Dataset

from .palette import apply_draw


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def to_unit(img):
    """PIL image or uint8 HWC array -> (3, H, W) float tensor in [0, 1]."""
    a = np.asarray(img, dtype=np.uint8)
    if a.ndim == 2:
        a = np.repeat(a[:, :, None], 3, axis=2)
    return torch.from_numpy(a[:, :, :3].copy()).permute(2, 0, 1).float() / 255.0


def _encode(encoder, x, device):
    x = x.to(device, non_blocking=True)
    if device.type == "cuda":
        with torch.autocast("cuda", dtype=torch.float16):
            return encoder(x).float().cpu().numpy()
    return encoder(x).float().cpu().numpy()


@torch.no_grad()
def embed_dataset(encoder, dataset, idx, device, batch=128):
    """Features of dataset[i] for i in idx, loaded batch by batch."""
    feats = []
    for s in range(0, len(idx), batch):
        x = torch.stack([to_unit(dataset[int(i)][0]) for i in idx[s:s + batch]])
        feats.append(_encode(encoder, x, device))
    return np.concatenate(feats)


def choose_references(encoder, dataset, r, device, n_max=10000, seed=0):
    """Indices into ``dataset`` (items (image, label)) of r representative images."""
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(dataset), size=min(n_max, len(dataset)), replace=False))
    feats = embed_dataset(encoder, dataset, idx, device)
    if r == 1:
        centre = feats.mean(axis=0, keepdims=True)
        return [int(idx[np.argmin(((feats - centre) ** 2).sum(1))])]
    km = KMeans(n_clusters=r, n_init=10, random_state=seed).fit(feats)
    refs = []
    for c in range(r):
        members = np.where(km.labels_ == c)[0]
        d = ((feats[members] - km.cluster_centers_[c]) ** 2).sum(1)
        refs.append(int(idx[members[np.argmin(d)]]))
    return refs


class DrawSet(Dataset):
    """Draw t applied to the reference image, its noise seeded by (seed, t)."""

    def __init__(self, ref, draws, seed):
        self.ref, self.draws, self.seed = ref, draws, seed

    def __len__(self):
        return len(self.draws)

    def __getitem__(self, t):
        gen = torch.Generator().manual_seed(self.seed * 1_000_003 + t)
        return apply_draw(self.ref.clone(), self.draws[t], gen)


@torch.no_grad()
def embed_draws(encoder, ref, draws, device, batch=256, seed=0, threads=4, ahead=3):
    """(T, F) features of every draw applied to the reference image ``ref`` in [0, 1].

    A pool of threads applies the draws (torch and OpenCV release the GIL) while the GPU
    encodes; at most ``ahead`` batches are prepared in advance.  Threads, not processes:
    on Windows, moving float image batches between processes costs more than computing
    them."""
    ds = DrawSet(ref.cpu(), draws, seed)
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    starts = iter(range(0, len(ds), batch))
    feats = []
    try:
        with ThreadPoolExecutor(threads) as ex:
            def submit(s):
                return [ex.submit(ds.__getitem__, i) for i in range(s, min(s + batch, len(ds)))]
            pending = collections.deque()
            for _ in range(ahead):
                s = next(starts, None)
                if s is not None:
                    pending.append(submit(s))
            while pending:
                futures = pending.popleft()
                s = next(starts, None)
                if s is not None:
                    pending.append(submit(s))
                feats.append(_encode(encoder, torch.stack([f.result() for f in futures]), device))
    finally:
        torch.set_num_threads(previous)
    return np.concatenate(feats)


def reduce(features, dims=50, pca_dims=256, seed=0):
    """PCA to ``pca_dims`` (for speed), then UMAP to ``dims``."""
    x = features - features.mean(axis=0, keepdims=True)
    x = PCA(n_components=min(pca_dims, x.shape[1], x.shape[0] - 1),
            random_state=seed).fit_transform(x)
    return umap.UMAP(n_components=dims, n_neighbors=30, min_dist=0.0, metric="euclidean",
                     random_state=seed).fit_transform(x)


def cluster(z, k, seed=0):
    """GMM labels of the reduced embeddings of one reference.

    Transforms with identical effect (e.g. a lone 90-degree rotation) map to the same
    point, so a component can have zero variance in some directions; the coordinates are
    standardised and the covariances floored at 1e-3 (on unit-variance data)."""
    z = np.asarray(z, dtype=np.float64)
    z = (z - z.mean(axis=0)) / np.clip(z.std(axis=0), 1e-12, None)
    gmm = GaussianMixture(n_components=k, covariance_type="full", reg_covar=1e-3, n_init=3,
                          random_state=seed).fit(z)
    return gmm.predict(z)


def consensus(label_sets, k, seed=0):
    """One partition of the draws from their labels under every reference."""
    T = len(label_sets[0])
    codes = np.zeros((T, len(label_sets) * k), dtype=np.float32)
    for j, labels in enumerate(label_sets):
        codes[np.arange(T), j * k + np.asarray(labels)] = 1.0
    if len(label_sets) == 1:
        final = np.asarray(label_sets[0])
    else:
        final = KMeans(n_clusters=k, n_init=20, random_state=seed).fit_predict(codes)
    agreement = [float(adjusted_rand_score(final, labels)) for labels in label_sets]
    return final, agreement
