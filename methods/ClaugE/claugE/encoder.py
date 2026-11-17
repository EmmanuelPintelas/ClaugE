"""The frozen encoder that embeds transformed reference images for clustering.

The encoder is never trained on the task.  It must be sensitive to what the palette's
operators change, so encoders trained to be invariant to augmentations (contrastive or
self-distilled models such as SimCLR, DINO, CLIP) are avoided: they map colour jitter,
blur, grayscale and crops of an image to nearly the same point.  The default is the
supervised ImageNet ResNet-50 of torchvision (IMAGENET1K_V1, trained with random crops and
flips only), with global average pooling at the end of each of its four stages,
L2-normalised per stage and concatenated (256 + 512 + 1024 + 2048 = 3840 features): the
early stages carry texture, colour and noise, the last one shape and content.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class StageGAP(torch.nn.Module):
    """name: "resnet50_v1" (the default) or "dino_resnet50", the self-distilled ResNet-50 of
    DINO (Caron et al., 2021) for the encoder ablation, read from the local file
    ``weights`` (dino_resnet50_pretrain.pth, the backbone state dict DINO releases)."""

    def __init__(self, name="resnet50_v1", input_size=224, weights=None):
        super().__init__()
        from torchvision.models import ResNet50_Weights, resnet50
        if name == "resnet50_v1":
            net = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        elif name == "dino_resnet50":
            if not weights:
                raise ValueError("dino_resnet50 needs the path of its weights file")
            net = resnet50(weights=None)
            sd = torch.load(weights, map_location="cpu", weights_only=True)
            missing, unexpected = net.load_state_dict(sd, strict=False)
            if unexpected or set(missing) - {"fc.weight", "fc.bias"}:
                raise ValueError(f"{weights}: missing {missing}, unexpected {unexpected}")
        else:
            raise ValueError(f"unknown encoder {name!r}")
        net.eval()
        self.stem = torch.nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.stages = torch.nn.ModuleList([net.layer1, net.layer2, net.layer3, net.layer4])
        self.input_size = input_size
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))
        for p in self.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def forward(self, x):
        """x: (B, 3, H, W) in [0, 1] -> (B, 3840) features."""
        if x.shape[-1] != self.input_size or x.shape[-2] != self.input_size:
            x = F.interpolate(x, size=(self.input_size, self.input_size), mode="bilinear",
                              align_corners=False, antialias=True)
        h = self.stem((x - self.mean) / self.std)
        feats = []
        for stage in self.stages:
            h = stage(h)
            feats.append(F.normalize(h.mean(dim=(2, 3)), dim=1))
        return torch.cat(feats, dim=1)
