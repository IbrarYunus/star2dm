# Reproducing the paper

Three entry points. Pick the one that matches how much you want to re-derive; each is a
superset of the one above it.

| you want to | you need | download | time |
|---|---|---|---|
| **A.** redraw or restyle a figure, re-read a number | maps + figure cache | ~475 MB | minutes |
| **B.** re-derive figures from predictions | maps + predictions | ~23 GB | hours |
| **C.** regenerate predictions from the models | maps + checkpoints | ~950 MB | ~2 GPU-days |
| **D.** retrain from scratch | maps | 325 MB | + ~1 GPU-day |
| **E.** rebuild the maps | raw snapshots | TB | see `docs/DATA.md` |

Everything below assumes the environment is set up and `scripts/env.sh` has been pointed at
your cluster. That file holds the conda activation line and the seven data roots, and it is
the only file you need to edit to run this somewhere else.

```bash
conda env create -f environment.yml
conda activate stars2dm
pip install -e .

python -m stars2dm_paths                  # where the code expects data
python tests/test_release_integrity.py    # needs no data at all
```

---

## A. Redraw the figures

```bash
python scripts/fetch_data.py --tier maps --tier cache
for S in dreams ood support sweeps aniso; do
    CACHE_MODE=read sbatch --export=ALL,STAGE=$S scripts/07_figures.sh
done
```

`CACHE_MODE=read` requires the cache and fails loudly if an entry is missing, rather than
silently starting a twenty-minute read of prediction files you did not download.

Figures land in `$STARS2DM_FIG_DIR` (default `data/figures/`) as PDF with a PNG preview
alongside. `reference_figures/` holds the published copies to diff against — the panels come
out pixel-identical, so a difference means something real changed.

Check the result against the paper:

```bash
python tests/verify_reproduction.py
```

It rasterizes each regenerated PDF and its published counterpart and compares pixels — a byte
comparison is useless, because matplotlib stamps a creation date into every file. Exit status
is 0 only if all 43 reproduced. Anything else is a real difference worth understanding: the
cache path is verified pixel-for-pixel, so "close enough" is not the expected outcome.

`docs/FIGURE_MAP.md` says which stage makes which figure. Figure 1's schematic is the one
exception: it is TikZ, built by `src/paperfigs/make_schematic.py`, and needs a LaTeX toolchain.

**What the cache will not do:** change the annulus geometry, the histogram binning, or how
draws are combined. Those are upstream of what it stores. All three are part of the cache key,
so asking for a different one produces a miss and a loud failure rather than a quietly wrong
answer — but acting on it means entry point B.

## B. Re-derive the figures from the prediction draws

```bash
python scripts/fetch_data.py --tier maps --tier predictions
bash scripts/submit_all.sh aniso      # the m=2 metric, ~1 h on CPU
bash scripts/submit_all.sh figures    # the five stages, in parallel
```

`CACHE_MODE` defaults to `auto`, so this reads the draws and writes the cache as it goes.
The cache cannot drift from the figures this way: the same run produced both.

`STAGE=dreams` loads 1024 × 100 × 128² — 6.7 GB as float32 before any promotion — which is
why that stage asks for 32 cores. An 8-core request was OOM-killed doing exactly this.

## C. Regenerate the predictions

```bash
python scripts/fetch_data.py --tier maps --tier checkpoints
bash scripts/submit_all.sh infer      # DREAMS in 9 shards, TNG50, FIRE m12
bash scripts/submit_all.sh sweeps     # five parameter sweeps, ~16.5 h each
```

Cost: 1.17 s per galaxy-draw on one v100-16 at Heun-100, measured twice independently. The
DREAMS 100-draw set is 1024 × 100 of those, sharded into ~3.7 h jobs.

The shard boundaries in `submit_all.sh` keep the held-out split `[0, 102)` as its own shard, so
the calibration numbers can be regenerated without re-running the other 922 galaxies.

Then continue from B.

## D. Retrain

```bash
python scripts/fetch_data.py --tier maps
bash scripts/submit_all.sh train      # both models, 12 h each on 2x v100-16
```

Both published runs stopped at step 9054, with val_loss 0.065 (v95) and 0.064 (v97). The
ablation gets the same budget as the fiducial deliberately, so any performance gap is
attributable to removing conditioning rather than to undertraining.

Register a trained checkpoint where the rest of the pipeline looks for it:

```bash
cp checkpoints/v95_fiducial/last.ckpt "$STARS2DM_CKPT_ROOT/last-v95.ckpt"
```

Then continue from C.

Retraining will not reproduce the published draws bit for bit — different hardware, a
different cuDNN, and a different DDP world size all change the arithmetic. It should reproduce
the published *numbers*, which aggregate over 100 draws and up to 1024 galaxies. If a
population statistic moves by more than its quoted uncertainty, that is worth investigating
rather than attributing to nondeterminism.

## E. Rebuild the maps

See `docs/DATA.md`. Short version: it needs the raw snapshots, a second environment, and three
packages that are on neither PyPI nor conda.

---

## Before you trust a number

`docs/CAVEATS.md` lists twelve things that will give you a plausible wrong answer rather than
an error. The three that bite soonest:

- **Radii are 1.86% low.** The maps span ±32.596465 kpc; the analysis axis is ±32. Preserved
  on purpose, because every published number is on that axis (§1).
- **Ensemble size is not free.** Δ is built on a `ddof=0` std, biased low by 0.923 at n=10.
  The same estimator on a perfectly calibrated ensemble gives k = 1.27 at n=10 and 1.03 at
  n=100 (§6).
- **Draws combine with the mean, at both levels**, and the two levels must match (§5).
