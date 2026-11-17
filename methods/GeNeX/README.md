# GeNeX (baseline)

GeNeX (Pintelas and Livieris, IEEE TNNLS) re-implemented for this comparison and extended
from binary to multi-class tasks. Self-contained: it reads the shared packed data and writes
its committees in the shared format for `bench/`; nothing is shared with the other methods.

```
genex/data.py       packed splits (named as in every committee), GeNeX's augmentation
                    (rotation by a multiple of 90 degrees, sharpening w.p. 1/2) with ImageNet
                    normalization, class down-sampling extended to many classes
genex/nets.py       EfficientNet-B0 + linear head, training (Adam, cross-entropy, fixed epochs,
                    no validation monitoring), prediction (optionally under input noise),
                    uniform crossover, mutation, layer-wise averaging
genex/gene.py       GenE (Algorithm 1): generation 0, G generations of crossover, mutation and
                    one-epoch training, N networks in the output pool M; resumable
genex/protonex.py   ProtoNeX (Algorithm 2): behavioral clustering (UMAP + Gaussian mixture,
                    clusters by silhouette in k_min..k_max), five expert criteria, prototype
                    fusion with the head retrained, SLSQP weights
configs/genex.json  settings
run_genex.py        gene | protonex | smoke
```

```
python run_genex.py gene     --benchmark kather
python run_genex.py protonex --benchmark kather            (k_max 5 and 25)
python -m bench.evaluate methods/GeNeX/outputs/kather_efficientnet_b0_kmax5   (from the repository root)
```

Extensions from two classes to K (the original code is binary):

- prediction signatures are the full K-class softmax outputs on a fixed random subset of
  the selection split (`signature_points`), instead of the positive-class probability;
- the selection metric is the benchmark's (geometric-mean recall up to ten classes,
  macro-F1 above), for the top-validation expert, the robustness drop and SLSQP;
- class down-sampling: the largest class is drawn down to ratio x the second-largest and
  the other classes are kept whole, which is GeNeX's rule for two classes;
- the number of clusters is chosen by silhouette in `k_min`..min(k_max, N - 1), so the
  committee never exceeds the budget (`k_min` = 3).

Training budget: N networks in generation 0 (epochs and learning rates drawn from
`epochs` and `learning_rates`; set to 3 or 5 epochs at 1e-4 to 1e-3 for these benchmarks,
where the paper's 1-5 epochs at 1e-5 to 1e-4 leave the many-class tasks undertrained), then per generation
2 x `parent_pairs` children (one epoch each) and, except after the last, the unselected
children plus N - N/G fresh networks (one epoch each). With N = 25, G = 5 and 6 pairs this
is 193 trainings (25 + 60 children + 108) and about 243 epochs per benchmark, plus one
head epoch per prototype. A benchmark's entry in `configs/genex.json` overrides any setting:
on RxRx1, generation 0 trains for 18 to 22 epochs and G = 2 (62 trainings), since three to
five epochs leave its 1,139 classes far from fitted.
`smoke` runs everything on three batches into `work_smoke/` and `outputs_smoke/`.
