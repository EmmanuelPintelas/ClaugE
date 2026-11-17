# ClaugE: Cluster the Augmentations, Not the Data

Code, results and committees for **"Cluster the Augmentations, Not the Data:
Complementary Training Paths for Ensembles under Distribution Shift"** (Emmanuel Pintelas).

An ensemble improves on its members only where they err differently, and where a network
errs depends on the augmentations it is trained with. ClaugE decides, before any training,
which complementary invariances the members of an ensemble learn. It draws 10^5 transforms
over a palette of twelve operators, embeds what each does to ten reference images of the
training set with a frozen ImageNet encoder, clusters the transforms per reference image
and reconciles the clusterings by consensus into k = 5 groups. Each group becomes a training
recipe, a kernel density over its transforms, and every member trains on the full training
set under one recipe. A committee takes the best candidates of every path on the selection
split and weights the paths by a softmax of their selection scores, refined by a bounded
fit. No test image, labeled or not, is used before prediction.

This repository lets you

* regenerate every table and figure of the paper from the stored results, with no
  download and no GPU;
* recompute those results from the released predictions of every committee and of every
  candidate ClaugE and its ablations trained;
* rerun ClaugE, its ablations and the seven baselines from scratch, on the paper's
  partitions or on new ones.

## Contents

```
reproduce/          the paper's numbers, tables and figures (see section 1 and 3)
results/            every result the paper reports
  main/             the scores of all 64 committees of the comparison (Table 4, Section 6)
  ablations/        the reports of run_ablations.py (Tables 6 to 9, Section 7)
  partition/        the partitions of the 10^5 transforms on each benchmark (Table 5, Figs. 2, 3)
  cost/             accuracy against cost on Camelyon17 (Fig. 4)
  tables/ figures/  generated from the above by reproduce/tables.py and reproduce/figures.py
methods/ClaugE/     the method (see its README): claugE/ (palette, drawer, encoder, partition,
                    recipes, committee), engine/ (training of the candidates), run_claugE.py,
                    run_ablations.py
methods/<baseline>/ DeepEnsembles, HyperEnsembles, AugDiverse, EoA, DiWA, Lookaround, GeNeX,
                    each self-contained (see methods/README.md and methods/GeNeX/README.md)
bench/              the shared evaluation: split roles, metrics, paired cluster bootstrap
data_prep/          download and pack the four benchmarks
tools/              restore the paper's partitions; check and unpack the downloads
```

## Install

Python 3.10 or later. Training needs PyTorch with CUDA; tables, figures and analyses run on
the CPU.

```bash
git clone <repository URL> ClaugE
cd ClaugE
pip install -r requirements.txt
```

## 1. Regenerate the paper's tables and figures (no download)

```bash
python reproduce/tables.py      # results/tables/*.tex: Tables 4 to 9
python reproduce/figures.py     # results/figures/: Figs. 2, 3 and 4
```

No number in them is typed by hand. `python reproduce/tables.py --compare <paper>.tex`
checks the generated tables against those of a LaTeX source, line by line. The draws and
the recipes behind Figs. 2 and 3 are rebuilt from the drawer's seed and the stored
partition labels with the method's own code.

## 2. Download the stored predictions

**https://drive.google.com/drive/folders/15CXeCZfWBGgDPqMB6yJdxPM_WngCRUk5?usp=sharing**

The release holds the stored predictions (pre-softmax logits of every network on every
split, with the labels and domains of each split) as zip parts with a `SHA256SUMS.txt`,
per benchmark:

| Part | Contents | Read by |
|---|---|---|
| `<benchmark>_committees.predictions.*` | the 16 committees of the comparison: every method at k_max = 5 and 25 | `score_committees.py`, `cost.py`, `run_ablations.py table` |
| `<benchmark>_claugE_pools.predictions.*` | ClaugE's 25 candidates (five per path) with their manifests | `run_ablations.py`, `cost.py` |
| `<benchmark>_ablation_pools.predictions.*` | the candidates of the retrained ablation variants (Camelyon17 and RxRx1) | `run_ablations.py committee` and `paths` |
| `<benchmark>_training.logs.*` | the baselines' training records (epoch times, selections) | `cost.py` |

Camelyon17 takes 0.2 GB, Kather 0.1 GB, iWildCam 3.2 GB and RxRx1 28.9 GB: 32.4 GB in 21
parts of at most 3.3 GB. Checkpoints are not part of the release: no table, figure or
analysis needs them. Download the parts you need with `SHA256SUMS.txt` into one folder,
then check and unpack:

```bash
python tools/assemble_release.py --downloads /path/to/downloads
```

Every part unpacks to its place under `methods/`.

## 3. Recompute the results from the downloads

```bash
python tools/restore_partition.py        # the draws, partitions and recipes the analyses read
python reproduce/score_committees.py     # results/main/committees.json (Table 4, intervals)
python reproduce/cost.py                 # results/cost/camelyon17.json (Fig. 4)
cd methods/ClaugE
for v in nopartition random data oneref ssl identity0 identity0.5 k20; do
  python run_ablations.py committee --benchmark camelyon17 --variant $v   # and rxrx1
done
python run_ablations.py committee-level --benchmark camelyon17    # and rxrx1
python run_ablations.py paths           --benchmark camelyon17    # and rxrx1
python run_ablations.py table           --benchmark camelyon17    # and rxrx1
python run_ablations.py allocation      --benchmark camelyon17    # all four benchmarks
python run_ablations.py complementarity --benchmark camelyon17    # all four benchmarks
cd ../..
python reproduce/collect.py --skip-partitions --skip-maps --skip-images   # into results/
python reproduce/tables.py
python reproduce/figures.py
```

`score_committees.py` scores every committee with the shared bench and gives the paired
95% interval of ClaugE minus every other method (cluster bootstrap of the test split,
1,000 replicates). `run_ablations.py committee` builds the committees of a retrained
variant from its candidates, with alpha and the radius chosen on the selection split as
for ClaugE; on RxRx1, where the variants train one candidate per path, it also builds
their reference, ClaugE restricted to the first candidate of each of its paths. The
partition parts of `collect.py` copy what the partition stage wrote (and need its
embeddings and the datasets); their output is already in `results/partition/`.

## 4. Run ClaugE from scratch

Data (from `data_prep/`; every pack lands in `data/` at the paper's resolution):

```bash
cd data_prep
python fetch_camelyon_hf.py                       # Camelyon17-WILDS, HuggingFace mirror
python pack_camelyon.py                           # data/camelyon17, 96 px
python fetch_wilds.py --task iwildcam --root ../data/wilds
python fetch_wilds.py --task rxrx1 --root ../data/wilds
python pack_wilds.py --task iwildcam              # data/iwildcam, 224 px
python pack_wilds.py --task rxrx1                 # data/rxrx1, 224 px
python pack_kather.py --src ../data/kather_raw    # data/kather, 224 px
cd ..
```

For Kather, put `NCT-CRC-HE-100K.zip` and `CRC-VAL-HE-7K.zip` (Zenodo record 1214456) in
`data/kather_raw` first.

The partition: restore the paper's, or compute a new one (about two hours per benchmark on
one laptop GPU, most of it UMAP on the CPU; UMAP's output can differ across versions and
machines, so a new partition is close to the paper's but not identical):

```bash
python tools/restore_partition.py                                  # the paper's partitions
python methods/ClaugE/run_claugE.py partition --benchmark kather   # or a new one
```

Train the candidates and build the committees (k_max = 5 and 25; the strength and the bound
of the weights are chosen on the selection split):

```bash
python methods/ClaugE/run_claugE.py train     --benchmark kather --backbone efficientnet_b0
python methods/ClaugE/run_claugE.py committee --benchmark kather --backbone efficientnet_b0
python -m bench.evaluate methods/ClaugE/outputs/kather_efficientnet_b0_kmax5
```

Training is resumable: rerun the same command. Every candidate's training settings and
seeds are drawn from its pool seed and slot and recorded in its pool's `manifest.jsonl`.
`methods/ClaugE/README.md` describes the settings, the committee options and the ablations
(`run_ablations.py prepare | train | committee` for the retrained variants; after
`tools/restore_partition.py` the random, one-reference and self-supervised variants and
the image groups of the data variant need no `prepare`). The self-supervised variant needs
the DINO ResNet-50 weights in `methods/ClaugE/weights/` (see the README there).

## 5. The baselines

Deep ensembles, hyperparameter ensembles, augmentation-diverse ensembles, ensembles of
averages, DiWA, Lookaround and GeNeX, each in its own folder with its own copy of the
training code, writing its committees in the shared format. Commands and settings are in
`methods/README.md` and `methods/GeNeX/README.md`; for example

```bash
cd methods/DeepEnsembles
python run_deep_ensembles.py train     --benchmark camelyon17
python run_deep_ensembles.py committee --benchmark camelyon17
```

## Data and licenses

The code is released under the MIT license (`LICENSE`). The released predictions and
models are derived from Camelyon17-WILDS, iWildCam2020-WILDS and RxRx1-WILDS and from the
Kather colorectal cohorts (Zenodo record 1214456), and `results/partition/*/reference0.png`
is one image of each; the terms of those datasets apply to anything derived from them.

## Citation

Citation details will be added when the paper is public; see `CITATION.cff`.
