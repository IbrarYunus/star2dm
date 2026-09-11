"""Release integrity check: every module imports, and every correctness guard still fires.

    python tests/test_release_integrity.py

Needs no data -- it touches no map, checkpoint or prediction file. Run it after any edit to
pred_data_loaders.py or baryonbridge/config.py. Each assertion below stands for a bug that
was live at some point and produced a plausible wrong answer rather than an error; the
comments say which.
"""
import sys, importlib, pathlib
REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(1, str(REPO / "src" / "paperfigs"))

import baryonbridge
assert str(REPO) in baryonbridge.__file__, f"wrong baryonbridge: {baryonbridge.__file__}"
print("baryonbridge ->", baryonbridge.__file__)

mods = [
    "stars2dm_paths",
    "baryonbridge.config", "baryonbridge.train", "baryonbridge.eval",
    "baryonbridge.interpolants.interpolant", "baryonbridge.interpolants.stochastic",
    "baryonbridge.models.pixels.blocks", "baryonbridge.models.pixels.cunet",
    "baryonbridge.models.pixels.embeddings", "baryonbridge.models.pixels.resnet",
    "baryonbridge.data.dreams", "baryonbridge.data.tng", "baryonbridge.data.fire",
    "baryonbridge.data.transforms",
    "baryonbridge.utils.galaxy_info", "baryonbridge.utils.summaries",
    "pred_data_loaders", "paper_style", "plot_radial_profile_figures",
    "plot_paper_stamp_figures", "plot_calibration_figures", "analyze_anisotropy",
    "plot_anisotropy_figures", "compare_fire_v95_v97", "compare_combiners",
    "make_schematic", "collect_paper_figures",
]
for m in mods:
    importlib.import_module(m)
    print("  ok  ", m)

import pred_data_loaders as pdl
import numpy as np

# --- split: the 820/102/102 ranges every shipped model used ---
assert pdl.split_label(0, "v95") == "test"
assert pdl.split_label(101, "v95") == "test"
assert pdl.split_label(102, "v95") == "val"
assert pdl.split_label(203, "v95") == "val"
assert pdl.split_label(204, "v95") == "train"
assert pdl.split_label(300, "TNG50_v95") == "ood"
print("split_label: 820/102/102 + ood  ok")

# --- normalization: v95/v97 resolve to the published fixed-32 rhoDM constants ---
for tag in ("v95", "v95_heun100", "v97", "TNG50_v95_heun100_n100"):
    mean, std = pdl.rhodm_norm_constants(tag)
    assert (mean, std) == (6.0495, 0.4818), (tag, mean, std)
print("rhodm_norm_constants: (6.0495, 0.4818) for v95/v97  ok")

# --- the fall-through guard still raises, not falls through ---
from baryonbridge.config import NORM_PROFILE
NORM_PROFILE["vFAKE"] = "dwarffire_fov25"
raised = False
try:
    pdl.rhodm_norm_constants("vFAKE")
except KeyError as e:
    raised = True
    assert "not 'rhoDM'" in str(e)
assert raised, "fall-through guard did NOT fire"
del NORM_PROFILE["vFAKE"]
print("norm fall-through guard: raises  ok")

# --- denormalize_draws rejects a non-rhoDM channel ---
raised = False
try:
    pdl.denormalize_draws(np.zeros((2, 4, 4)), version_tag="v95", field="sigmaDM")
except ValueError:
    raised = True
assert raised, "denormalize_draws accepted a non-rhoDM field"
print("denormalize_draws field guard: raises  ok")

# --- annulus geometry: the published +/-32 kpc axis, 32 bins, r_max 45.25 ---
edges, centers, mask = pdl.build_radial_annulus_geometry(128)
assert mask.shape == (32, 128, 128)
assert abs(edges[-1] - np.sqrt(32**2 + 32**2)) < 1e-9
assert abs(centers[-1] - 44.548) < 1e-2, centers[-1]
print(f"annulus geometry: 32 bins, r_max {edges[-1]:.4f} kpc  ok")

# --- the duplicate definition in plot_radial_profile_figures matches the canonical one ---
import plot_radial_profile_figures as prpf
e2, c2, m2 = prpf.build_radial_annulus_geometry(128)
assert np.array_equal(edges, e2) and np.array_equal(mask, m2)
print("duplicate annulus geometry: identical to canonical  ok")

# --- combiner convention is mean/mean, as the published figures used ---
assert (pdl.ENSEMBLE_COMBINE, pdl.PROFILE_AGG) == ("mean", "mean")
print("combiners: mean/mean  ok")
print("\nALL CHECKS PASSED")
