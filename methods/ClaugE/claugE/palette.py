"""The operator palette: twelve image transforms, each deterministic given its parameters.

Every operator acts on a float image tensor of shape (3, H, W) with values in [0, 1] and
returns one of the same shape, clamped to [0, 1].  ``sample(rng)`` draws a parameter
dictionary uniformly within the operator's ranges (discrete parameters uniformly over
their values); ``apply(x, params, gen)`` applies exactly those parameters.  The only
randomness left inside ``apply`` is the pattern of the two noise operators, taken from
``gen`` so that a draw can be replayed.

Operators carry no inner "apply or not" probability: the untransformed image enters a
member's training only through the identity probability of its recipe.

    f1  rotation           angle in {90, 180, 270}
    f2  shift_flip         flip in {none, horizontal, vertical}; shift (dx, dy) in
                           [-0.25, 0.25] of the side, reflect padding
    f3  resized_crop       scale in [0.5, 1], log aspect in [log 3/4, log 4/3], centre
    f4  sharpen            factor in [1.5, 3]
    f5  clahe              clip limit in [1, 4], 8 x 8 tiles
    f6  grayscale          no parameter
    f7  gauss              sigma in [0.05, 0.25]
    f8  linf               eps in [0.05, 0.25]
    f9  photometric        eps in [0.2, 0.8]; brightness, contrast and saturation factors
                           in [1 - eps, 1 + eps], blur mix in [0, 1]
    f10 stain              eps in [0.05, 0.35]; per-channel gain in [1 - eps, 1 + eps],
                           bias in [-eps, eps]
    f11 channel_gain_gamma per-channel gain in [0.7, 1.3], log gamma in [-0.4, 0.4]
    f12 channel_dropout    channel in {0, 1, 2}
"""
from __future__ import annotations

import math

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF


def _u(rng, lo, hi):
    return float(rng.uniform(lo, hi))


class Op:
    """An operator: name, parameter ranges for jitter and clipping, sample and apply."""
    name = ""
    # continuous parameters: name -> (lo, hi); vector parameters use name[i]
    ranges: dict = {}
    discrete: tuple = ()

    def sample(self, rng):
        raise NotImplementedError

    def apply(self, x, p, gen=None):
        raise NotImplementedError


class Rotation(Op):
    name = "rotation"
    discrete = ("angle",)

    def sample(self, rng):
        return {"angle": int(rng.choice([90, 180, 270]))}

    def apply(self, x, p, gen=None):
        return torch.rot90(x, k=p["angle"] // 90, dims=(1, 2))


class ShiftFlip(Op):
    name = "shift_flip"
    ranges = {"dx": (-0.25, 0.25), "dy": (-0.25, 0.25)}
    discrete = ("flip",)

    def sample(self, rng):
        return {"flip": int(rng.integers(3)), "dx": _u(rng, -0.25, 0.25),
                "dy": _u(rng, -0.25, 0.25)}

    def apply(self, x, p, gen=None):
        if p["flip"] == 1:
            x = torch.flip(x, dims=(2,))
        elif p["flip"] == 2:
            x = torch.flip(x, dims=(1,))
        _, h, w = x.shape
        sy, sx = int(round(p["dy"] * h)), int(round(p["dx"] * w))
        pad = max(abs(sy), abs(sx), 1)
        big = F.pad(x.unsqueeze(0), (pad, pad, pad, pad), mode="reflect").squeeze(0)
        top, left = pad - sy, pad - sx
        return big[:, top:top + h, left:left + w].contiguous()


class ResizedCrop(Op):
    name = "resized_crop"
    ranges = {"scale": (0.5, 1.0), "log_ratio": (math.log(3 / 4), math.log(4 / 3)),
              "cx": (0.0, 1.0), "cy": (0.0, 1.0)}

    def sample(self, rng):
        return {"scale": _u(rng, 0.5, 1.0),
                "log_ratio": _u(rng, math.log(3 / 4), math.log(4 / 3)),
                "cx": _u(rng, 0.0, 1.0), "cy": _u(rng, 0.0, 1.0)}

    def apply(self, x, p, gen=None):
        _, h, w = x.shape
        ratio = math.exp(p["log_ratio"])
        cw = max(1, min(w, int(round(w * math.sqrt(p["scale"] * ratio)))))
        ch = max(1, min(h, int(round(h * math.sqrt(p["scale"] / ratio)))))
        top = int(round(p["cy"] * (h - ch)))
        left = int(round(p["cx"] * (w - cw)))
        crop = x[:, top:top + ch, left:left + cw].unsqueeze(0)
        out = F.interpolate(crop, size=(h, w), mode="bilinear", align_corners=False,
                            antialias=True)
        return out.squeeze(0)


class Sharpen(Op):
    name = "sharpen"
    ranges = {"factor": (1.5, 3.0)}

    def sample(self, rng):
        return {"factor": _u(rng, 1.5, 3.0)}

    def apply(self, x, p, gen=None):
        return TF.adjust_sharpness(x, p["factor"])


class Clahe(Op):
    name = "clahe"
    ranges = {"clip": (1.0, 4.0)}

    def sample(self, rng):
        return {"clip": _u(rng, 1.0, 4.0)}

    def apply(self, x, p, gen=None):
        img = (x.permute(1, 2, 0).cpu().numpy() * 255.0 + 0.5).astype(np.uint8)
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        l = cv2.createCLAHE(clipLimit=p["clip"], tileGridSize=(8, 8)).apply(l)
        rgb = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2RGB)
        return torch.from_numpy(rgb).to(x.device, x.dtype).permute(2, 0, 1) / 255.0


class Grayscale(Op):
    name = "grayscale"

    def sample(self, rng):
        return {}

    def apply(self, x, p, gen=None):
        return TF.rgb_to_grayscale(x, num_output_channels=3)


class Gauss(Op):
    name = "gauss"
    ranges = {"sigma": (0.05, 0.25)}

    def sample(self, rng):
        return {"sigma": _u(rng, 0.05, 0.25)}

    def apply(self, x, p, gen=None):
        noise = torch.randn(x.shape, generator=gen, dtype=x.dtype).to(x.device)
        return x + p["sigma"] * noise


class Linf(Op):
    name = "linf"
    ranges = {"eps": (0.05, 0.25)}

    def sample(self, rng):
        return {"eps": _u(rng, 0.05, 0.25)}

    def apply(self, x, p, gen=None):
        noise = torch.rand(x.shape, generator=gen, dtype=x.dtype).to(x.device)
        return x + p["eps"] * (2.0 * noise - 1.0)


class Photometric(Op):
    """Brightness, contrast, saturation and a mix with a 3 x 3 binomial blur."""
    name = "photometric"
    ranges = {"brightness": (0.2, 1.8), "contrast": (0.2, 1.8), "saturation": (0.2, 1.8),
              "blur": (0.0, 1.0)}

    def sample(self, rng):
        eps = _u(rng, 0.2, 0.8)
        return {"eps": eps, "brightness": _u(rng, 1 - eps, 1 + eps),
                "contrast": _u(rng, 1 - eps, 1 + eps),
                "saturation": _u(rng, 1 - eps, 1 + eps), "blur": _u(rng, 0.0, 1.0)}

    def apply(self, x, p, gen=None):
        out = (x * p["brightness"]).clamp(0.0, 1.0)
        mean = out.mean()
        out = ((out - mean) * p["contrast"] + mean).clamp(0.0, 1.0)
        grey = out.mean(dim=0, keepdim=True)
        out = ((out - grey) * p["saturation"] + grey).clamp(0.0, 1.0)
        if x.shape[-1] >= 16 and p["blur"] > 0:
            k = torch.tensor([[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]],
                             dtype=x.dtype, device=x.device)
            k = (k / k.sum()).view(1, 1, 3, 3).expand(3, 1, 3, 3)
            blurred = F.conv2d(F.pad(out.unsqueeze(0), (1, 1, 1, 1), mode="reflect"), k,
                               groups=3).squeeze(0)
            out = (1.0 - p["blur"]) * out + p["blur"] * blurred
        return out


class Stain(Op):
    """Per-channel gain and bias."""
    name = "stain"
    ranges = {"gain[0]": (0.65, 1.35), "gain[1]": (0.65, 1.35), "gain[2]": (0.65, 1.35),
              "bias[0]": (-0.35, 0.35), "bias[1]": (-0.35, 0.35), "bias[2]": (-0.35, 0.35)}

    def sample(self, rng):
        eps = _u(rng, 0.05, 0.35)
        return {"eps": eps, "gain": [_u(rng, 1 - eps, 1 + eps) for _ in range(3)],
                "bias": [_u(rng, -eps, eps) for _ in range(3)]}

    def apply(self, x, p, gen=None):
        g = torch.tensor(p["gain"], dtype=x.dtype, device=x.device).view(3, 1, 1)
        b = torch.tensor(p["bias"], dtype=x.dtype, device=x.device).view(3, 1, 1)
        return x * g + b


class ChannelGainGamma(Op):
    """Per-channel gain and gamma, with black kept black."""
    name = "channel_gain_gamma"
    ranges = {"gain[0]": (0.7, 1.3), "gain[1]": (0.7, 1.3), "gain[2]": (0.7, 1.3),
              "log_gamma[0]": (-0.4, 0.4), "log_gamma[1]": (-0.4, 0.4),
              "log_gamma[2]": (-0.4, 0.4)}

    def sample(self, rng):
        return {"gain": [_u(rng, 0.7, 1.3) for _ in range(3)],
                "log_gamma": [_u(rng, -0.4, 0.4) for _ in range(3)]}

    def apply(self, x, p, gen=None):
        g = torch.tensor(p["gain"], dtype=x.dtype, device=x.device).view(3, 1, 1)
        e = torch.exp(torch.tensor(p["log_gamma"], dtype=x.dtype, device=x.device)).view(3, 1, 1)
        return x.clamp_min(0.0) ** e * g


class ChannelDropout(Op):
    name = "channel_dropout"
    discrete = ("channel",)

    def sample(self, rng):
        return {"channel": int(rng.integers(3))}

    def apply(self, x, p, gen=None):
        out = x.clone()
        out[p["channel"]] = 0.0
        return out


PALETTE = [Rotation(), ShiftFlip(), ResizedCrop(), Sharpen(), Clahe(), Grayscale(), Gauss(),
           Linf(), Photometric(), Stain(), ChannelGainGamma(), ChannelDropout()]
OPS = {op.name: op for op in PALETTE}


def apply_draw(x, draw, gen=None):
    """Apply a draw, a list of (operator name, parameters) in order, to a [0, 1] image."""
    for name, params in draw:
        x = OPS[name].apply(x, params, gen).clamp(0.0, 1.0)
    return x


# ------------------------------------------------------------- vector parameters
def flat_params(params):
    """{'gain': [a, b, c], 'eps': e} -> {'gain[0]': a, ..., 'eps': e} (numbers only)."""
    out = {}
    for k, v in params.items():
        if isinstance(v, (list, tuple)):
            for i, vi in enumerate(v):
                out[f"{k}[{i}]"] = float(vi)
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            out[k] = float(v)
    return out


def set_flat(params, key, value):
    """Write one flat key back into a parameter dictionary."""
    if key.endswith("]"):
        name, idx = key[:-1].split("[")
        params[name][int(idx)] = value
    else:
        params[key] = value
