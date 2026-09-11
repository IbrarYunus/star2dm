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

# --- the two configs still describe the two published models ---------------------------
# A config is one edit away from silently becoming a different model. These assert the
# hyperparameters the paper reports, at the YAML level so the check is fast and needs no GPU.
import yaml

def _cfg(name):
    return yaml.safe_load((REPO / "configs" / name).read_text())

fid = _cfg("v95_fiducial.yaml")
abl = _cfg("v97_no_theta.yaml")
for name, cfg in (("v95", fid), ("v97", abl)):
    m, d = cfg["model"]["init_args"], cfg["data"]["init_args"]
    assert cfg["seed_everything"] == 42, name
    assert m["use_radial_loss_weight"] is True, name
    assert m["radial_loss_weight_mode"] == "radial_power", name
    assert m["radial_loss_power"] == 1.0, name          # p = 1
    assert m["radial_loss_r_min_kpc"] == 2.0, name      # R_min = 2 kpc
    assert m["n_sampling_steps"] == 100, name
    assert (d["n_train"], d["n_val"], d["n_test"]) == (820, 102, 102), name
    assert d["x_conditioning_fields"] == ["rhostar"] and d["x1_fields"] == ["rhoDM"], name
    assert d["file_suffix"] == "csv_fov_z0p350_0p350", name
    assert cfg["trainer"]["max_steps"] == 10000, name
    assert d["resolution"] == 128, name

# The split constants in the loader must agree with what the configs trained on, or every
# split label is wrong for the shipped models.
assert (pdl.TEST_IDX_END, pdl.TRAIN_IDX_START) == (
    fid["data"]["init_args"]["n_test"],
    fid["data"]["init_args"]["n_test"] + fid["data"]["init_args"]["n_val"])
print("configs: p=1, R_min=2 kpc, 820/102/102, z0p350 maps, and the loader split agrees  ok")

# theta conditioning is the ONLY difference between them -- both halves matter: a null
# num_h_embeds stops the embedding module being built at all, which is what avoids a DDP
# unused-parameter error from a module that exists but never receives h.
assert fid["model"]["init_args"]["condition_on_parameters"] is True
assert fid["model"]["init_args"]["num_h_embeds"] == 5
assert abl["model"]["init_args"]["condition_on_parameters"] is False
assert abl["model"]["init_args"]["num_h_embeds"] is None
diff = {k for k in set(fid["model"]["init_args"]) | set(abl["model"]["init_args"])
        if fid["model"]["init_args"].get(k) != abl["model"]["init_args"].get(k)}
assert diff == {"condition_on_parameters", "num_h_embeds"}, f"configs differ in more than theta: {diff}"
print("v95 vs v97: differ in theta conditioning and nothing else  ok")

# --- job scripts are syntactically valid shell ------------------------------------------
import subprocess
scripts = sorted((REPO / "scripts").glob("*.sh"))
assert len(scripts) == 9, f"expected 9 job scripts, found {len(scripts)}"
for sh in scripts:
    subprocess.run(["bash", "-n", str(sh)], check=True)
print(f"job scripts: {len(scripts)} parse  ok")

assert (REPO / "environment.yml").exists() and (REPO / "environment-mapgen.yml").exists()
print("environment files present  ok")

print("\nALL CHECKS PASSED")
