# Weights of the self-supervised encoder (ablation only)

The ablation that redoes the partition with a self-distilled encoder
(`run_ablations.py prepare --variant ssl`) loads the DINO ResNet-50 backbone from this
folder. Download `dino_resnet50_pretrain.pth` (the ResNet-50 backbone of Caron et al.,
2021, 94 MB) from the DINO repository's release,

    https://dl.fbaipublicfiles.com/dino/dino_resnet50_pretrain/dino_resnet50_pretrain.pth

and place it here. ClaugE itself uses the supervised torchvision ResNet-50, which
torchvision downloads on first use.
