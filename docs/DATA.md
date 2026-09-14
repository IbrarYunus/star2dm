# The data: what it is, where it goes, what it should hash to

This repository holds code and the 43 published figures. Everything else lives in an archive,
because the smallest piece of it is larger than the whole source tree.

`data/MANIFEST.yaml` is the index: four tiers, with a sha256 and a byte count for every file.
`scripts/fetch_data.py` downloads and verifies against it.

```bash
python scripts/fetch_data.py --list             # tiers, sizes, what each unlocks
python scripts/fetch_data.py --tier maps --tier cache
python scripts/fetch_data.py --verify           # checksum what is already on disk
```

Files land under the `STARS2DM_*` roots. `python -m stars2dm_paths` prints where those
currently resolve; set them before fetching to put the data on a scratch filesystem instead of
beside the source.

## The four tiers

| tier | size | what it unlocks |
|---|---|---|
| `maps` | 325 MB | everything. The training input and the truth for every figure. |
| `cache` | ~150 MB | redrawing all 43 figures. Needs `maps`. |
| `checkpoints` | 625 MB | regenerating predictions without retraining. Needs `maps`. |
| `predictions` | 23.4 GB | changing the annulus geometry, binning, or draw combiner. |

Pick the smallest tier that answers your question. `cache` and `predictions` answer the same
questions about the *published* figures; `predictions` additionally lets you ask different
ones. See `docs/CAVEATS.md` §11 for exactly where the line falls.

## What the maps are

Paired stellar and dark-matter density maps, one pair per galaxy:

| suite | galaxies | role |
|---|---|---|
| DREAMS MW zooms, CDM, SB5 | 1024 | training, validation and test |
| TNG50 | 264 | out of domain, at fixed θ |
| FIRE m12 | 8 | out of domain, θ marginalized under a flat prior |

All are 128×128 on the same grid. Each galaxy is oriented face-on by aligning its stellar
angular momentum within 10 kpc to the line of sight, and the density fields are evaluated **at
the galaxy mid-plane** — a slice, not a line-of-sight integral. Density at each grid point is a
KD-tree adaptive kernel estimate: smoothing length set by the 128th nearest neighbour, particle
masses weighted by a cubic spline. Values are stored as log10 density and standardized by the
data loader.

The field of view is 65.192930 kpc across — ±32.596465 kpc, read per galaxy from
`src/mapgen/redshift_sample_0p35.csv`. This is **not** the ±32 kpc the analysis code uses for
its radial axis, and that disagreement is deliberate and load-bearing: read `docs/CAVEATS.md`
§1 before acting on any radius.

### Filenames encode the dataset

```
Maps_rhoDM_TNG_SB5_128_z=0.00_csv_fov_z0p350_0p350.npy
     ^^^^^ ^^^^^^^ ^^^ ^^^^^^ ^^^^^^^^^^^^^^^^^^^^^
     field suite   px  snapshot   grid provenance
```

The `csv_fov_z0p350_0p350` suffix records that the grid extent came from a redshift table with
min and max both 0.35 — one FOV per galaxy, all at z = 0.35. **Maps with a different suffix are
a different dataset**, and the normalization constants in `baryonbridge/config.py` do not apply
to them. The suffix is what `file_suffix` in the configs selects.

`z=0.00` is the *simulation* snapshot: the galaxies are at z = 0 and are then placed at an
observer redshift of 0.35 to set the angular scale. A pixel is 0.5 kpc, which is 0.1 arcsec at
that redshift — the scale of Euclid/VIS and Roman/WFI.

### The parameter tables

`Parameters/<suite>/CDM_TNG_MW_SB5.txt`, one row per galaxy, five columns: Ωm, σ8, and the
three IllustrisTNG feedback parameters (specific SN energy ē_w, SN wind speed normalization
κ_w, high-accretion AGN coupling ε_f,high). DREAMS samples these on a Sobol sequence; row *i*
is galaxy *i*, and the order **is** the order of the map stack. A stack rebuilt over a partial
box range will not line up with its table.

TNG50's table holds TNG's own cosmology and the fiducial feedback values the DREAMS grid is
centred on, so TNG50 is conditioned on real parameters, not invented ones. FIRE has no such
correspondence, which is why its predictions marginalize instead — `docs/CAVEATS.md` §7.

## The checkpoints

| file | model | used for |
|---|---|---|
| `last-v95.ckpt` | fiducial, θ-conditioned | every main-text figure |
| `last-v97.ckpt` | no-θ ablation | Appendix D |

313 MiB each, Lightning format, loadable with `LightningStochasticInterpolant.load_from_checkpoint`.
They are `save_last` checkpoints: both runs stopped at step 9054 with val_loss 0.065 (v95) and
0.064 (v97), so the ablation comparison is like-for-like.

## The predictions

Per-galaxy npz under `$STARS2DM_PRED_ROOT`, one directory per galaxy:

```
galaxy_0019/galaxy_19_v95_pred_draws_100_heun100_n100.npz
            ^^^^^^^^^ ^^^ ^^^^^^^^^^^^^^ ^^^^^^^^^^^^^^^
            galaxy    tag  draw count     sampler + suffix
```

Each holds `draws` of shape `(n_draws, 128, 128)` in standardized units —
`denormalize_draws` converts to log10 ρ. Out-of-domain files carry the suite in the name
(`galaxy_19_TNG50_v95_heun100_n100_pred_draws_100.npz`) and live under a `TNG50/` or
`FIREm12/` subdirectory.

The sweep files are config-major: `pred_draws_50_heun100_sn1_sweep_5.npz` holds 5 grid points
× 10 draws, so draws `0:10` are grid point 0 and `40:50` are grid point 4. Appendix C plots
only those two extremes.

The tier ships these files **whole**, all five grid points, even though that is 14.3 GB of
the 23.4 GB total — the single largest thing in the archive. Trimming them to the two
plotted slices was the original plan and is wrong twice over:
it would break the published `--draws-slice 40:50` commands, since grid point 4 would no
longer be at index 40; and this tier exists precisely for people who want to ask a question
the paper did not, for whom the three unplotted grid points are the interesting part. If you
only want the published panels, the `cache` tier already has them at a thousandth of the size.

## Publishing the archive

The published archive is **two Zenodo records**, because the prediction draws are 23.4 GB
against 1.05 GB for everything else, and everything else is what a reader actually needs:

| record | tiers | size | role |
|---|---|---|---|
| 1 | maps + checkpoints + cache | 1.05 GB | cited in the paper; reproduces every figure and number |
| 2 | predictions | 23.4 GB | linked from record 1; only for changing binning, geometry or the combiner |

`MANIFEST.yaml` expresses this with `archive_base_url` for record 1 and a per-tier
`base_url` on `predictions` for record 2. Those two lines are the only edits.

Use the **versioned** record URL, not the concept DOI. A concept DOI follows the latest
version, so publishing a new version would silently repoint the URLs at files whose
checksums no longer match the ones in the manifest.

### When to publish

**At acceptance, not at submission.** Once a Zenodo record is published its files are
immutable — metadata stays editable, and you can always publish a *new version* with a new
versioned DOI, but the old version is permanent.

The reason that matters here is that the cache tier is *derived from the figures*. A referee
asking for a different radial binning, a re-coloured panel or one more ablation changes
figures, which changes the cache, which changes its checksum — and that is the tier most
readers will actually download. Maps and checkpoints are stabler, but an ablation request
moves those too.

What to do at submission instead:

- **Reserve the DOIs** on two drafts. Zenodo gives you the DOI string without publishing, so
  it can go in the data-availability statement while nothing becomes immutable.
- **Rehearse on `sandbox.zenodo.org`** — same API, throwaway DOIs. Worth doing for the 23 GB
  record specifically, so its upload misbehaves on a day when that does not matter.
- Publish the code repository if you want; GitHub stays mutable, so code-at-submission and
  data-at-acceptance is a reasonable split.

If the journal or a referee needs the data during review, Zenodo has embargoed and
restricted-access modes: a DOI that resolves, with the files gated until a date or released
on request. Check their current policy rather than trusting this paragraph.

Rebuilding the cache after a change is one `bash scripts/submit_all.sh figures` round plus
`stage_archive.py`, so waiting costs very little.

### Procedure

1. **Stage.** `scripts/stage_archive.py` assembles the upload directory in exactly the layout
   `MANIFEST.yaml` gives each file, so it uploads verbatim.

   ```bash
   python scripts/stage_archive.py --out dist/ --all --dry-run   # the plan
   python scripts/stage_archive.py --out dist/ --tier maps --tier checkpoints --tier cache
   python scripts/stage_archive.py --out dist/ --tier predictions
   ```

   It re-checksums maps and checkpoints against the manifest as it goes, and prints the size
   and sha256 of the two tarballs it builds. Paste those into the manifest.

2. **Create both drafts on Zenodo** (or sandbox.zenodo.org first — same API, throwaway DOIs).
   Note each draft's record id from its URL. Reserve the DOI on each so it can go in the
   paper before publication.

3. **Upload.** The API is more reliable than the browser for the 23 GB record. Each draft has
   a bucket URL in its JSON (`links.bucket`):

   ```bash
   TOKEN=...                      # personal access token, deposit:write scope
   BUCKET=...                     # from the draft's links.bucket
   curl --upload-file dist/maps/dreams/Maps_rhoDM_TNG_SB5_128_z=0.00_csv_fov_z0p350_0p350.npy         "$BUCKET/Maps_rhoDM_TNG_SB5_128_z=0.00_csv_fov_z0p350_0p350.npy?access_token=$TOKEN"
   ```

   The filename in the bucket URL is what the record serves, so it must match the `url` field
   in the manifest, minus the base.

4. **Set the two base URLs** in `MANIFEST.yaml` to the records' file roots, then check the
   pattern against a real file listed on the record page rather than assuming — Zenodo has
   changed it before.

5. **Test before publishing.** Draft files need the token, so either test with it appended or
   publish record 1 first and test that:

   ```bash
   python scripts/fetch_data.py --tier maps
   python scripts/fetch_data.py --verify --tier maps
   ```

6. **Publish.** Files become immutable and the DOIs mint. Put record 1's concept DOI in the
   paper, and link record 2 from record 1's description.

7. **Commit the manifest** and run `python tests/test_release_integrity.py` once more.

Files are hard-linked when the staging directory is on the same filesystem as the source and
copied otherwise — staging 23 GB onto the same filesystem costs no extra space.

The prediction tarball is uncompressed on purpose: npz members are already deflated, so gzip
buys almost nothing and costs hours at that size.

```bash
python scripts/stage_archive.py --out dist/ --all --dry-run   # the plan, touches nothing
python scripts/stage_archive.py --out dist/ --tier maps --tier checkpoints
python scripts/stage_archive.py --out dist/ --tier cache      # needs the cache built
python scripts/stage_archive.py --out dist/ --tier predictions
```

## Regenerating the maps yourself

Possible, and mostly not worth it: it needs the raw DREAMS, TNG50 and FIRE snapshots and group
catalogs (terabytes, not redistributable here), a second environment
(`environment-mapgen.yml`), and three packages that are on neither PyPI nor conda —
`dreams_python` for DREAMS, `gizmopy` and `gizmo_analysis` for FIRE. Snapshot paths are set at
the top of `src/mapgen/get_2D_maps.py` and point at the cluster where the published maps were
built.

If you do: `sbatch --export=ALL,SUITE=DREAMS scripts/01_make_maps.sh`, and run the full box
range in one pass — see the note about row order above.
