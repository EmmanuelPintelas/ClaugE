# ClaugE

```
claugE/palette.py     the 12 operators, deterministic given their parameters
claugE/drawer.py      T transforms: first operator uniform, continue with prob 1/2 without
                      repetition, parameters uniform in their ranges, all recorded
claugE/encoder.py     frozen ImageNet ResNet-50 (torchvision V1), GAP of its 4 stages,
                      L2-normalised per stage (3840 features)
claugE/partition.py   references (medoids of r-means), embedding of draws per reference,
                      PCA(256) -> UMAP(50) -> GMM(k) per reference, consensus clustering
claugE/recipes.py     recipe = kernel density over a cluster's draws (Silverman bandwidth),
                      readable summary, and the training transform (identity w.p. 1/k)
claugE/committee.py   k_max / k best candidates per recipe on the selection split, averaged
                      uniformly per recipe; recipe weights = softmax of robust z-scores of
                      their selection scores, then a bounded fit of the log-weights
                      (class-balanced log-likelihood), kept if it beats the prior out of sample
configs/claugE.json   settings (T, r, encoder, dims, candidates per recipe, k values,
                      per-benchmark image size, data path, batch sizes)
engine/               the training engine (poolkit, netkit, configs, build_pool.py); the
                      augmentation branch "claugE" of poolkit/augment.py applies a recipe
run_claugE.py         partition | train | committee
run_ablations.py      prepare | train | committee | committee-level | table (the ablations)
```

```
python run_claugE.py partition --benchmark kather            (k = 5 and 20; no training;
                                                              or tools/restore_partition.py)
python run_claugE.py train     --benchmark kather --backbone efficientnet_b0 --k 5
python run_claugE.py committee --benchmark kather --backbone efficientnet_b0     (k_max 5 and 25)
python -m bench.evaluate methods/ClaugE/outputs/kather_efficientnet_b0_kmax5  (from the repository root)
```

Committee weights, all optional (defaults in `claugE/committee.py`):

```
--alpha auto       strength of the prior: weight ratio e^alpha per robust SD of score; 0 = uniform;
                   auto = chosen from {0.1, 0.3, 1, 3, 10} by the committee's selection-split score
--fit-radius auto  how far the fit may move each log-weight from the prior's; 0 = no fit, inf =
                   unbounded; auto = chosen from {1/4, 1/2} together with alpha
--fit-iters 100    maximum iterations of the fit (L-BFGS-B); stops earlier once converged
--no-fit-check     keep the fit even if it does not beat the prior on cross-fitted halves
--tag NAME         output suffix; settings other than the defaults get one automatically
                   (e.g. outputs/kather_efficientnet_b0_kmax25_alpha0.5_fit0)
--overwrite        rebuild a committee whose output folder exists (never done otherwise)
```

## Ablations (`run_ablations.py`, settings in `configs/ablations.json`)

```
python run_ablations.py prepare         --benchmark camelyon17 --variant random
python run_ablations.py train           --benchmark camelyon17 --variant random
python run_ablations.py committee       --benchmark camelyon17 --variant random
python run_ablations.py committee-level --benchmark camelyon17      (no training)
python run_ablations.py paths           --benchmark camelyon17      (no training)
python run_ablations.py allocation      --benchmark camelyon17      (no training)
python run_ablations.py complementarity --benchmark camelyon17      (no training)
python run_ablations.py table           --benchmark camelyon17
```

`allocation` compares, at k_max = 5, the committee (one member per path) with the five
candidates of the path best on validation, and makes the choice on the selection split by
2-fold cross-fitting; it also gives both options in the terms of the paper's analysis
(Proposition 1): class-balanced log loss on test = mean loss of the members (quality)
minus diversity (`outputs_ablations/<benchmark>_allocation.json`). `complementarity`
compares two-network ensembles across two paths with two-network ensembles within each of
them, made of the same four networks (every two paths, every two candidates of each: 1,000
quadruplets), grouped by the quality gap between the two paths (on the selection split,
and on the test split itself in the metric and in log loss: the test gaps are the ones to
read, since validation can hide a gap), with a cluster bootstrap of the test split; in log loss
the difference is exactly the diversity of the pairs across paths minus that of the pairs
within (Corollary 3) (`outputs_ablations/<benchmark>_complementarity.json`). `paths` also
checks the mixture model (Proposition 2): the balanced error of the whole-palette and
random-group networks against the share-weighted average of the paths' networks trained
with the same settings and seeds.

Retrained variants: `nopartition` (whole palette for every path), `random` (draws split at
random into groups of the partition's sizes), `data` (the training images split into 5
k-means groups of the frozen encoder's features, path c trains on group c under the whole
palette), `oneref` (the first reference's clustering alone), `ssl` (the partition redone
with DINO ResNet-50: put `dino_resnet50_pretrain.pth` in `weights/`), `identity0`,
`identity0.5`, `k20` (the k = 20 partition, one candidate per path, at k_max = 25). Each
uses the main pools' seeds, so member j has ClaugE member j's training settings and seeds.
`candidates` in the config sets the candidates per path (5 = ClaugE's; fewer: only
k_max = 5 is built and the reference is ClaugE on its pools' first candidates,
`outputs_ablations/<benchmark>_efficientnet_b0_kmax5_full_c<n>`). Committee-level variants
(random candidates, weights per member, uniform, fixed alpha and r, prior alone, unbounded
fit, the committee without the path worst on validation, the best path alone with its five
candidates) are computed from the main pools. `paths` reports what every path yields alone:
single-network test scores by recipe next to the networks of every retrained variant with
k = 5, trained with the same settings and seeds (for a variant with its own partition of the
draws, also its group sizes, its agreement with the main partition (adjusted Rand index) and
the operators of its groups), what each path of the k = 20 partition yields alone and how
its groups sit inside the five of the main partition, five-network averages, how much the
paths disagree, and two-network ensembles within and across the two best paths
(`outputs_ablations/<benchmark>_paths.json`). Outputs go to `pools_ablations/` and
`outputs_ablations/`; `table` writes `outputs_ablations/<benchmark>_ablations.json`, with
every retrained variant also compared under equal weights. `committee-level`, `paths` and
`table` read one network's predictions at a time (about 1 GB on RxRx1) and can run beside
a training.

The log line of each committee reports whether the fit was kept, its iterations, whether it
converged or stopped at `--fit-iters`, the log-likelihood before and after, and the
out-of-sample check; `committee.json` records the same under `fit`. Run variants before
`--prune`, which deletes the candidate pools.

The partition depends only on the benchmark (the encoder is fixed), not on the backbone of
the members. Candidates train with the training distributions of
`engine/configs/pools/<benchmark>.json` and `engine/configs/global.json`, with the overrides
of `configs/claugE.json`. Training is
resumable: rerun the same command. `--prune` deletes the candidate pools once the committees
(k_max members each, with their predictions and checkpoints) are written to `outputs/`.

Needs `umap-learn` for the partition and downloads the torchvision ResNet-50 weights on first
use.
