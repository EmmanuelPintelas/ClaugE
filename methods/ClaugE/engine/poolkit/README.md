# The training engine

`build_pool.py` trains a pool of candidates from one configuration; `poolkit/` holds its
parts (recipes and seeds, training, the acceptance gate, storage, reports) and `netkit/`
the datasets, augmentations and the network. ClaugE calls it once per path:
`run_claugE.py train` writes the path's configuration (the training distribution of
`configs/pools/<benchmark>.json` and `configs/global.json`, with the path's recipe as the
augmentation and the overrides of `configs/claugE.json`) and runs

```bash
python build_pool.py --config <path config> --pool-seed <1000 k + c> --out <pool> --data <dataset>
```

and `run_ablations.py train` does the same for the ablation variants, with the pool seeds
of ClaugE's own paths. By hand, from this folder:

```bash
python build_pool.py --config configs/pools/kather.json --pool-seed 1 --dry-run 20   # recipes a pool draws
python build_pool.py --out <pool>              # resume after any stop
python build_pool.py --out <pool> --status     # progress, from another terminal
```

## Properties of a pool

1. **Independent members.** A member's training settings and seeds depend only on the pool
   seed and its slot, through NumPy `SeedSequence` streams keyed by parameter name
   (`recipes.py`). There is no adaptive search and no filter that compares members, so
   accepted members are independent draws from the training process. Path c of an ablation
   variant uses the pool seed of ClaugE's path c, so its member j has ClaugE's member j's
   settings and seeds; two paths of one set of pools have different pool seeds.
2. **Frozen configuration.** The configuration is frozen into `pool_config.json` with a
   hash at the first run; a later run with a different one is refused
   (`--allow-config-change` overrides it and records the change).
3. **A minimal gate, on source validation only.** A trained draw is rejected only if it
   failed as a training run: non-finite outputs, a diverged loss or a run stuck at chance,
   balanced accuracy on source validation below 1.5 times chance, or predictions collapsed
   onto one class. The gate never reads a target split and never ranks by quality; every
   rejected draw is committed with its reasons (`gate.py`).
4. **Targets for monitoring only.** Target-split numbers in `pool_report.json` let you
   watch a build; nothing in the build reads them.
5. **Provenance.** Every manifest row records the recipe, both seeds, the per-epoch
   training curve, the gate checks, metrics on every split, weight statistics, timings,
   the code hash and the configuration hash.

## Stopping and resuming

A member is committed only when its manifest row is written. After any stop, the files of
an uncommitted slot are deleted and that slot trains again from scratch with the same
settings and seeds; only the member in progress is lost. A `.lock` file stops two
processes from writing to one pool, and every split's labels are checked against the
stored ones on restart.

## Layout of a pool

```
pool_config.json                      frozen configuration + hash, code hash, environment
manifest.jsonl                        one row per draw, in slot order
pool_report.json                      rewritten after every draw
events.jsonl                          starts, resumes, interruptions, clean-ups
models/member_0000.pt                 accepted checkpoints (netkit.models.load_member)
predictions/index.json                splits, roles, sizes, accepted ids in commit order
predictions/<split>/labels.npy, domains.npy
predictions/<split>/member_0000.npy   logits of the kept checkpoint, float16, (N, K)
rejected/predictions/...              predictions of rejected draws that trained
```

Splits are `source_val` plus the benchmark's targets; their role in `index.json` is
`tuning` for `ood_val*`, `test` for `ood_test*`. Committees (`claugE/committee.py`) read
the stored logits and write the shared committee format of `bench/evaluate.py`.

## Configuration

A recipe parameter is `{"fixed": v}`, `{"choice": [...], "weights": [...]}`,
`{"uniform": [lo, hi]}`, `{"loguniform": [lo, hi]}` or `{"int_uniform": [lo, hi]}`. The
required parameters are `arch`, `pretrained`, `augment`, `optimizer`, `lr`,
`weight_decay`, `batch_size`, `epochs`, `scheduler` (`plateau`, `halve_each_epoch`,
`none`), `sched_patience`, `sched_factor`, `stop_patience`, `label_smoothing`,
`drop_rate`, `train_frac`. `configs/global.json` holds what every pool inherits (merged one
level deep under the benchmark's file): the probability of a two-stage learning rate
(the top of the range for the first half of the epochs, the bottom for the rest, drawn per
member from its own stream) and when a run stuck at chance is abandoned.
