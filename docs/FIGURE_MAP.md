# Which script makes which figure

Generated from `main_v6.tex` at the time of release, not maintained by hand. 32 figure
environments, 41 panel files: 10 main-text figures and 22 in the four appendices
(A model behavior across the test population, B radially resolved residuals,
C simulation-parameter sweeps, D training without parameter conditioning).

The `stage` column is the `STAGE=` value of `scripts/07_figures.sh`. To rebuild one figure,
run its stage -- the stages are independent and parallel:

```bash
sbatch --export=ALL,STAGE=<stage> scripts/07_figures.sh
```

`schematic.pdf` is the exception: it is a TikZ drawing built by `src/paperfigs/make_schematic.py`,
which regenerates its map tiles from the real maps rather than cropping them out of Figure 1.
It needs a LaTeX toolchain, which the rest of this repository does not.

Published copies of every file below are in `reference_figures/`, so a regenerated figure can
be diffed against the one in the paper.

| fig | file | stage | script |
|---|---|---|---|
| **1** | `schematic.pdf` | `-` | `make_schematic.py` |
| **2** | `v95_heun100_n100_galaxy_19_stochastic_draw_panels.pdf` | `dreams` | `plot_radial_profile_figures.py` |
| **3** | `v95_heun100_n100_galaxy_19_deviation_hist.pdf` | `dreams` | `plot_radial_profile_figures.py` |
| ↳ | `v95_heun100_n100_galaxy_19_deviation_N16x8_radial.pdf` | `dreams` | `plot_radial_profile_figures.py` |
| ↳ | `v95_heun100_n100_galaxy_19_deviation_N16x8_azimuthal.pdf` | `dreams` | `plot_radial_profile_figures.py` |
| **4** | `v95_heun100_n100_test_highlight_galaxy_19_prior_posterior_prior_posterior_radial.pdf` | `dreams` | `plot_radial_profile_figures.py` |
| **5** | `v95_heun100_n100_galaxy_19_anisotropy_A2.pdf` | `aniso` | `plot_anisotropy_figures.py` |
| ↳ | `v95_heun100_n100_galaxy_19_anisotropy_phase.pdf` | `aniso` | `plot_anisotropy_figures.py` |
| **6** | `v95_heun100_n100_test_highlight_rhoDM_predict_vs_true_mean_cc_perpixel_std_dreamsall_contours.pdf` | `dreams` | `plot_radial_profile_figures.py` |
| **7** | `v95_heun100_calibration_2x2.pdf` | `support` | `plot_calibration_figures.py` |
| **8** | `v95_heun100_n100_anisotropy_population_alignment_suites.pdf` | `aniso` | `plot_anisotropy_figures.py` |
| ↳ | `v95_heun100_n100_anisotropy_population_delta_suites.pdf` | `aniso` | `plot_anisotropy_figures.py` |
| **9** | `v95_heun100_sn1_sweep_2x2.pdf` | `support` | `plot_calibration_figures.py` |
| **10** | `TNG50_v95_heun100_n100_with_FIREm12_v95_heun100_flat_prior_baryon_rhoDM_predict_vs_true_mean_cc_perpixel_std.pdf` | `ood` | `plot_radial_profile_figures.py` |
| **A1** | `v95_stamp_deviation_map_n16.pdf` | `support` | `plot_paper_stamp_figures.py` |
| **A2** | `v95_stamp_prior_posterior_n16.pdf` | `support` | `plot_paper_stamp_figures.py` |
| **A3** | `v95_heun100_n100_stamp_anisotropy_A2_n16.pdf` | `support` | `plot_paper_stamp_figures.py` |
| **A4** | `v95_heun100_n100_stamp_anisotropy_phase_n16.pdf` | `support` | `plot_paper_stamp_figures.py` |
| **A5** | `v95_stamp_input_stellar_map_n16.pdf` | `support` | `plot_paper_stamp_figures.py` |
| **A6** | `v95_stamp_truth_dm_map_n16.pdf` | `support` | `plot_paper_stamp_figures.py` |
| **A7** | `v95_stamp_pred_dm_map_n16.pdf` | `support` | `plot_paper_stamp_figures.py` |
| **A8** | `v95_stamp_deviation_hist_n16.pdf` | `support` | `plot_paper_stamp_figures.py` |
| **A9** | `v95_stamp_deviation_radial_n16.pdf` | `support` | `plot_paper_stamp_figures.py` |
| **A10** | `v95_stamp_deviation_azimuthal_n16.pdf` | `support` | `plot_paper_stamp_figures.py` |
| **B1** | `v95_heun100_n100_test_highlight_deviation_distribution_mean_cc_perpixel_std.pdf` | `dreams` | `plot_radial_profile_figures.py` |
| **B2** | `TNG50_v95_heun100_n100_with_FIREm12_v95_heun100_flat_prior_baryon_deviation_distribution_mean_cc_perpixel_std.pdf` | `ood` | `plot_radial_profile_figures.py` |
| **C1** | `v95_heun100_om_0p274_deviation_distribution_mean_cc_perpixel_std.pdf` | `sweeps` | `plot_radial_profile_figures.py` |
| ↳ | `v95_heun100_om_0p354_deviation_distribution_mean_cc_perpixel_std.pdf` | `sweeps` | `plot_radial_profile_figures.py` |
| **C2** | `v95_heun100_s8_0p78_deviation_distribution_mean_cc_perpixel_std.pdf` | `sweeps` | `plot_radial_profile_figures.py` |
| ↳ | `v95_heun100_s8_0p888_deviation_distribution_mean_cc_perpixel_std.pdf` | `sweeps` | `plot_radial_profile_figures.py` |
| **C3** | `v95_heun100_sn1_0p25_deviation_distribution_mean_cc_perpixel_std.pdf` | `sweeps` | `plot_radial_profile_figures.py` |
| ↳ | `v95_heun100_sn1_4_deviation_distribution_mean_cc_perpixel_std.pdf` | `sweeps` | `plot_radial_profile_figures.py` |
| **C4** | `v95_heun100_sn2_0p5_deviation_distribution_mean_cc_perpixel_std.pdf` | `sweeps` | `plot_radial_profile_figures.py` |
| ↳ | `v95_heun100_sn2_2_deviation_distribution_mean_cc_perpixel_std.pdf` | `sweeps` | `plot_radial_profile_figures.py` |
| **C5** | `v95_heun100_bhff_0p25_deviation_distribution_mean_cc_perpixel_std.pdf` | `sweeps` | `plot_radial_profile_figures.py` |
| ↳ | `v95_heun100_bhff_4_deviation_distribution_mean_cc_perpixel_std.pdf` | `sweeps` | `plot_radial_profile_figures.py` |
| **D1** | `v95_heun100_n100_with_v97_heun100_n100_anisotropy_alignment_hist.pdf` | `aniso` | `plot_anisotropy_figures.py` |
| **D2** | `v97_heun100_test_highlight_rhoDM_predict_vs_true_mean_cc_perpixel_std_dreamsall_contours.pdf` | `support` | `plot_radial_profile_figures.py` |
| **D3** | `v97_heun100_test_highlight_deviation_distribution_mean_cc_perpixel_std.pdf` | `support` | `plot_radial_profile_figures.py` |
| **D4** | `TNG50_v97_heun100_with_FIREm12_v97_heun100_n100_FIRE_rhoDM_predict_vs_true_mean_cc_perpixel_std.pdf` | `support` | `plot_radial_profile_figures.py` |
| **D5** | `TNG50_v97_heun100_with_FIREm12_v97_heun100_n100_FIRE_deviation_distribution_mean_cc_perpixel_std.pdf` | `support` | `plot_radial_profile_figures.py` |

## Two files you will produce that the paper does not use

`plot_anisotropy_figures.py` emits both a DREAMS-only and a three-suite version of each
population panel. The paper uses the three-suite versions (the `_suites` files, Figure 8); the
DREAMS-only `v95_heun100_n100_anisotropy_population_alignment.pdf` and
`v95_heun100_n100_anisotropy_population_delta.pdf` are kept in `reference_figures/` so that a
diff of a regenerated run comes out empty rather than showing two unexplained extras.

Their copies here were refreshed at release time. The set collected for the paper still held
the 2026-09-04 versions of these two, predating an anisotropy colour and opacity change on
2026-09-11 — because `collect_paper_figures.py` copies what the `.tex` references, and these
two are not referenced, so nothing pulled the newer ones through. The `_suites` versions the
paper does use were current. Caught by `tests/verify_reproduction.py` on the first real run:
five of the seven anisotropy panels came out pixel-identical and exactly these two did not.

## What each figure is

| fig | label | caption (first line) |
|---|---|---|
| **1** | `fig:schematic` | Schematic of the training (top) and inference (bottom) procedures. Relevant sections are noted accordingly.  fig:schemat |
| **2** | `fig:dreams_map` | The left two panels show the condition stellar density input and true dark matter density map in 64 64\,kpc field of vie |
| **3** | `fig:dreams_residual` | The top panel shows the distribution (blue) of the  -residuals for the same galaxy as in Figure~ fig:dreams_map, with th |
| **4** | `fig:dreams_profile_comp` | Azimuthally averaged dark matter density profile for the same held-out DREAMS galaxy shown in Figure~ fig:dreams_map. Gr |
| **5** | `fig:dreams_anisotropy` | Azimuthal m=2 structure of the projected dark matter density for the same held-out DREAMS galaxy shown in Figure~ fig:dr |
| **6** | `fig:dreams_profiles` | Predicted versus true azimuthally averaged dark matter density at six radial locations, for the in-domain DREAMS sample. |
| **7** | `fig:calibration` | Distribution of the profile-level  -residual at r=3.5\,kpc, before (top) and after (bottom) calibration, for the held-ou |
| **8** | `fig:dreams_anisotropy_pop` | Azimuthal m=2 diagnostics for the 102 held-out DREAMS galaxies and for the two out-of-domain samples, TNG50 and . The to |
| **9** | `fig:sim_par_test` | Population-level  -residual distributions when the feedback parameter  e_w is intentionally set to the minimum (top) and |
| **10** | `fig:tng50_fire_profiles` | Same as Figure~ fig:dreams_profiles, but for the out-of-domain TNG50 and samples (circles and diamonds respectively). Th |
| **A1** | `fig:appendix_stamp_deviation` |  -residual maps for 16 randomly selected held-out DREAMS test galaxies, shown in the same format and color scale as the  |
| **A2** | `fig:appendix_stamp_prior_posterior` | Prior/posterior radial dark matter density profile comparison for the same 16 held-out DREAMS test galaxies shown in Fig |
| **A3** | `fig:appendix_stamp_anisotropy_A2` | m=2 amplitude A_2 against projected radius for the same 16 held-out DREAMS test galaxies shown in Figure~ fig:appendix_s |
| **A4** | `fig:appendix_stamp_anisotropy_phase` | Offset  _2 between the predicted and true major-axis position angles for the same 16 galaxies, in the same format as the |
| **A5** | `fig:appendix_stamp_stellar` | Input stellar density conditioning maps for the same 16 held-out DREAMS test galaxies shown in Figure~ fig:appendix_stam |
| **A6** | `fig:appendix_stamp_truth` | True dark matter density maps for the same 16 galaxies, on the same color scale as the corresponding panel of Figure~ fi |
| **A7** | `fig:appendix_stamp_pred` | Median predicted dark matter density maps for the same 16 galaxies, on the same color scale as Figure~ fig:appendix_stam |
| **A8** | `fig:appendix_stamp_hist` | Pixel-level  -residual distributions for the same 16 galaxies, in the same format as the top panel of Figure~ fig:dreams |
| **A9** | `fig:appendix_stamp_radial` | Radial dependence of the binned  -residuals for the same 16 galaxies, in the same format as the middle panel of Figure~  |
| **A10** | `fig:appendix_stamp_azimuthal` | As Figure~ fig:appendix_stamp_radial, but binned in eight azimuthal angle bins, matching the bottom panel of Figure~ fig |
| **B1** | `fig:appendix_dreams_sixpanel` | Uncalibrated profile-level  -residual distributions for the DREAMS sample at six radial locations. The 102 held-out test |
| **B2** | `fig:appendix_ood_sixpanel` | As Figure~ fig:appendix_dreams_sixpanel, but for the out-of-domain TNG50 (light blue) and (dark blue) samples.  fig:appe |
| **C1** | `fig:appendix_sweep_om` | Population-level  -residual distributions with  _m set to the minimum (top,  _m=0.274) and maximum (bottom,  _m=0.354) v |
| **C2** | `fig:appendix_sweep_s8` | As Figure~ fig:appendix_sweep_om, but sweeping  _8 from 0.780 (top) to 0.888 (bottom). This is the weakest response of t |
| **C3** | `fig:appendix_sweep_sn1` | As Figure~ fig:appendix_sweep_om, but sweeping the supernova wind energy  e_w from 0.25 (top) to 4.0 (bottom), and on th |
| **C4** | `fig:appendix_sweep_sn2` | As Figure~ fig:appendix_sweep_om, but sweeping the supernova wind velocity normalization  _w from 0.5 (top) to 2.0 (bott |
| **C5** | `fig:appendix_sweep_bhff` | As Figure~ fig:appendix_sweep_om, but sweeping the AGN coupling efficiency  _f,\, high from 0.25 (top) to 4.0 (bottom).  |
| **D1** | `fig:appendix_noh_alignment` | Distribution of the alignment statistic  A of Equation~ eq:alignment over the 102 held-out DREAMS galaxies, with and wit |
| **D2** | `fig:appendix_noh_scatter` | Predicted versus true azimuthally averaged dark matter density for the model trained without simulation-parameter condit |
| **D3** | `fig:appendix_noh_deviation` | Uncalibrated profile-level  -residual distributions for the model trained without simulation-parameter conditioning, in  |
| **D4** | `fig:appendix_noh_ood_scatter` | Predicted versus true azimuthally averaged dark matter density for the model trained without simulation-parameter condit |
| **D5** | `fig:appendix_noh_ood_deviation` | Uncalibrated profile-level  -residual distributions for the unconditioned model on the out-of-domain TNG50 (light blue)  |
