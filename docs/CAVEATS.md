# Things that will silently give you a wrong answer

Every item here was live at some point during this project and produced a plausible number
rather than an error. None of them is a style preference. Where the code guards against one,
the guard is named — if you refactor past it, you reintroduce the bug.

---

## 1. The maps span ±32.596465 kpc. Every published radius is on a ±32 kpc axis.

`src/mapgen/redshift_sample_0p35.csv` carries `physical_fov_kpc = 65.192930` for all 1024
rows, and `get_2D_maps.py`'s `--redshift-csv` branch sets `xmin/xmax = ∓fov/2` with no
override path. So the maps are **±32.596465 kpc**.

`build_radial_annulus_geometry` defaults to **±32 kpc**, and every one of its call sites uses
that default. So a radius this code reports is **1.86% low** in absolute kpc — multiply by
1.0186. The outermost annulus centre reads 44.55 kpc where the true map corner is 46.10.

**This is preserved, not fixed.** Every number in the paper is on the ±32 axis; changing the
geometry would move all of them. `src/stars2dm_paths.py` carries both constants and asserts
their ratio, so the disagreement stays visible and a rebuilt map set at a different FOV fires
the assertion instead of silently rescaling every result.

## 2. Normalization is per data variant, and the lookup used to succeed with the wrong one

`merged_norm_dict` overlays with `setdefault`. A normalization profile that introduces new
field names — a surface-density variant defining `sigmaDM` but not `rhoDM` — therefore left
the base `rhoDM` entry in place, and the lookup **succeeded**, returning another variant's
constants.

That is not a harmless offset. Getting both the mean and the std wrong makes the induced
error a function of the pixel's own value, and density falls with radius, so the error maps
onto **radius**: +0.09 dex at the centre, −0.41 at the FOV edge. It manufactured a −0.49 dex
tilt that reads exactly like an outskirt under-prediction — the quantity the analysis
measures. It was reported as a real result before it was found.

Guarded in `pred_data_loaders.rhodm_norm_constants`: a non-empty profile that does not define
`rhoDM` raises. `denormalize_draws` likewise rejects any `field` other than `rhoDM`.

## 3. A gate on one loader says nothing about another

The normalization round-trip check passed at 1.3×10⁻⁶ dex for the entire time the
fall-through in §2 was live, because it exercises a different code path from the one the bias
analysis uses. Passing a self-test is evidence about the thing tested and nothing else.

## 4. The split constants must match the config that trained the model

`DREAMSDataModule.setup` slices by index range, so `TEST_IDX_END` / `TRAIN_IDX_START` in
`pred_data_loaders.py` **are** the split — there is no permutation file to fall back on. The
models in this release used 820/102/102, so test is `[0, 102)`, val `[102, 204)`, train
`[204, 1024)`.

Earlier runs of this project used 860/78/86. Applying those constants here labels galaxies
86–101 "val" when the model held them out as test, which contaminates a test-set result
rather than raising. The integrity test asserts the constants agree with the configs'
`n_test` and `n_val`; if you retrain with different values, change both.

## 5. Draws are combined with the mean, at both levels

`ENSEMBLE_COMBINE` (draws → fiducial prediction) and `PROFILE_AGG` (pixels → annulus value)
are both `mean`. Both were historically `median`. They have to match: the profile-level Δ
needs the same convention in numerator and denominator, and Eq. 1 of the paper defines
σ_pred with the mean.

Mixing them does not raise — it produces a different number for every quantity in the
Results section. The two settings are part of the figure-cache key precisely because they
change values without changing any array shape.

## 6. Ensemble size is not a free parameter

Δ is built on an ensemble standard deviation computed with `ddof=0`, which is biased low by
`sqrt((n-1)/n)·c4(n)` = 0.923 at n=10. The same estimator applied to a *perfectly calibrated*
Gaussian ensemble returns k = 1.27 at n=10 against 1.03 at n=100. A 10-draw number therefore
carries an estimator artifact that has nothing to do with the model.

The fiducial runs are n=100. Two places deliberately are not, and both are stated in the
captions:

- **The parameter sweeps stay at 50 draws.** The ddof bias at n=50 is 0.986, so the residual
  *means* those panels report move by under 2%; converting them would cost ~165 GPU-hours.
- **Appendix D mixes n=10 TNG50 with n=100 FIRE** in one panel, so TNG50's Δ denominators run
  ~5% small relative to FIRE's.

## 7. The FIRE ensemble is not a stochastic ensemble

FIRE m12 has no counterpart to the DREAMS SN1/SN2/BHFF parametrization, so its predictions
marginalize over those under a flat prior: **100 parameter configurations × 1 realization**,
not 100 draws at fixed conditioning. Its spread therefore contains parameter ignorance that
the DREAMS and TNG50 ensembles do not.

Consequences: the DREAMS-derived calibration factor is an *upper bound* on the correction
FIRE needs, not the right divisor for it; and Δ^{A2}, whose denominator is that spread, is
biased toward zero for FIRE. The alignment statistic is normalized and is unaffected. For
amplitudes, compare `A2_pred_median / A2_true_median` from the population CSV instead.

TNG50 is *not* in this category: its cosmology comes from its own parameter file and its
feedback parameters are the TNG fiducial values the DREAMS grid is centred on, so it sits on
the same footing as DREAMS.

## 8. Map axis 0 is x

`radius_map` builds its meshgrid with `indexing="ij"`, so array axis 0 is x, not y. A
transpose puts every peak marker on the wrong feature while still looking entirely plausible.

## 9. Multi-GPU training hangs silently if SLURM_NTASKS is set

Lightning selects `SLURMEnvironment` iff `SLURM_NTASKS` is set. That environment reports
`creates_processes_externally=True`, so `_SubprocessScriptLauncher` never spawns rank 1 and
DDP hangs in rendezvous until TCPStore times out — no error, just a job that burns its whole
allocation doing nothing. `sbatch` inherits the variable from the submitting shell, so
submitting from inside an interactive allocation is enough to trigger it.

`scrub_slurm_ntasks()` in `scripts/env.sh` unsets it; every GPU script calls it.

## 10. Two training jobs must not share a checkpoint directory

`save_last: true` writes a fixed filename. Two full-budget runs sharing a `dirpath` collide
on it and one run's final checkpoint is lost — recoverable only from the `save_top_k`
fallback, which is a different checkpoint. The two configs here write to
`./checkpoints/v95_fiducial/` and `./checkpoints/v97_no_theta/`.

## 11. What the figure cache cannot do

`src/paperfigs/figure_cache.py` stores what the panels are drawn from, which is downstream of
the annulus geometry, the histogram binning and the draw combiner. Changing any of those
means fetching the prediction draws; the cache will not answer the new question.

It cannot answer it *wrongly*, though: all three settings are part of the cache key, so a
changed setting produces a miss, and `--cache-mode read` then fails loudly rather than
serving arrays built under the published settings.

## 12. Cached draws are float32 for a reason

Per-galaxy draws in the cache are stored float32. float16 was tried: near log10(ρ) = 6 its
spacing is 0.0039 dex, and the residual maps divide by an ensemble std of ~0.09 dex, so the
quantization lands at 0.04 in units of Δ — enough to move counts across histogram bin
boundaries. The galaxy-19 histogram and binned residual map came out visibly different from
the same figures built from the draws. The saving had been 3 MB.
