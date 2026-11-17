# Methods

Each method lives in its own folder, reads the shared packed data and writes its committees
in the shared format to `<method>/outputs/<benchmark>_efficientnet_b0_kmax<k_max>/`, scored
by `python -m bench.evaluate <dir> [--against <dir>]` from the repository root. Nothing is shared
between methods: the six baselines below each carry a copy of the same `core.py` (data,
EfficientNet-B0, base training, prediction, committee writer), so they share one base
training procedure without sharing code paths with ClaugE or GeNeX.

Base recipe of the six (the `base` block of every config, from ClaugE's pool
settings): Adam, lr 1e-3 (Camelyon17 7.7e-4), weight decay 1e-5, batch 128, dropout 0.2,
label smoothing 0.05 on Camelyon17 and Kather, plateau schedule (patience 1, factor 0.5),
early stopping after 4 epochs without gain, checkpoint of the best epoch on source
validation; at most 3 epochs on Camelyon17, 10 on Kather, 20 on RxRx1, and on iWildCam 3, 5 or 10
drawn per network (as in ClaugE's pools; a `{"choice": [...]}` entry is drawn from the network's seed); light
augmentation (rotation by 0, 90 or 270 degrees and a horizontal flip).

| folder | method | trains per benchmark | deploys | stages |
|---|---|---|---|---|
| DeepEnsembles | Lakshminarayanan et al. 2017: seeds | 25 | first k_max | train, committee |
| EoA | Arpit et al. 2022: moving average of each run from step 100, all epochs | 25 | first k_max | train, committee |
| AugDiverse | Stickland and Murray 2020: AugMix severity 1-4 per member, p = 0.875 | 25 | first k_max | train, committee |
| HyperEnsembles | Wenzel et al. 2020: random search (wd, label smoothing, dropout), greedy NLL selection, stratification over seeds | 25 + up to 25 | up to k_max distinct | search, stratify, committee |
| DiWA | Rame et al. 2022: linear-probed shared start, 25 mild-hyperparameter runs, uniform weight average, BN re-estimated | 25 + probe | 1 | train, committee |
| Lookaround | Zhang et al. 2023: 3 copies (flip, RandAugment, AutoAugment), averaged every 5 steps | 1 run at 3x cost | 1 | train, committee |

```
cd methods/DeepEnsembles
python run_deep_ensembles.py train --benchmark camelyon17
python run_deep_ensembles.py committee --benchmark camelyon17
```

(`run_eoa.py`, `run_aug_diverse.py`, `run_diwa.py`, `run_lookaround.py` likewise;
`run_hyper_ensembles.py search`, then `stratify`, then `committee`.) Training is resumable:
rerun the same command. `smoke` runs a method end to end on three batches into
`work_smoke/` and `outputs_smoke/`; delete them afterwards. `--workers` sets the data
loader processes (default 4).
