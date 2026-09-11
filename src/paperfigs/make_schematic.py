#!/usr/bin/env python3
"""Method schematic for the paper, drawn in TikZ.

Re-creates docs/image_to_dark_matter_draft_source_updated/figures/schematic_v2.png (built
in Keynote) as a vector PDF. The image tiles are regenerated from the real maps rather
than cropped out of Figure 1, so the training row and the testing row can show different
galaxies -- Figure 1's galaxy 19 is held-out, and using it to illustrate *training* would
be wrong.

The layout is computed here, in centimetres, and written out as explicit TikZ coordinates.
The Python side owns the geometry and TikZ only renders, so moving a panel is a number in
LAYOUT rather than a fight with node placement.

Outputs, under --outdir (default docs/image_to_dark_matter_draft_source_updated/schematic):

    tiles/*.pdf     one PDF per map tile, no axes, exact square geometry
    schematic.tex   TikZ source; needs only tiles/, tikz, xcolor, mathptmx
    schematic.pdf   compiled, and copied to figures_pdf/schematic.pdf

Two deliberate departures from the Keynote version, both flagged in the module docstring
because they change what the figure claims:

  * "Interpolent" -> "Interpolant".
  * The drift network is called $v_\\theta$ and its target $r_t$, which is the notation of
    Eq. 2 in the draft. The Keynote slide used BaryonBridge's $b_t$ / $\\dot X_t$, which
    appears nowhere in the text.

And one addition: a time axis under each chain, saying that training forms $X_t$ directly
at a random $t$ while testing integrates from $t=0$ to $t=1$. The Keynote version draws
the two rows identically, which reads as if training also integrates. Pass
--training-style chain to get the Keynote behaviour back.

The boxes at each station spread out towards the middle of the path and close again on
the endpoint, standing for the injected noise. The profile is illustrative, not the
model's own schedule -- see NOISE_PROFILES and --noise-profile.

Run (light: reads ~6 npz files, seconds):
    conda activate <your stars2dm environment>
    cd ML_dm_distribution/training_res && python make_schematic.py
"""

import argparse
import shutil
import subprocess
from pathlib import Path

import numpy as np
import stars2dm_paths
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from pred_data_loaders import NORM_DICT  # noqa: E402

RES = 128
MEAN_RHO = NORM_DICT["2D"]["rhoDM"][RES]["mean"]
STD_RHO = NORM_DICT["2D"]["rhoDM"][RES]["std"]

# Same colour limits as the Figure 1 draw panels (plot_radial_profile_figures.py), so a
# reader can put the schematic and Figure 1 side by side and compare tiles directly.
COND_VMIN, COND_VMAX = 3.0, 11.0
RHO_VMIN, RHO_VMAX = 5.5, 9.5

PRED_ROOT = stars2dm_paths.PRED_ROOT
FIG_OUT = stars2dm_paths.DATA_ROOT / "figures"

SUITES = {
    "DREAMS": {
        "star": stars2dm_paths.DREAMS_RHOSTAR_MAP,
        "dm": stars2dm_paths.DREAMS_RHODM_MAP,
        "draws": lambda g: PRED_ROOT / f"galaxy_{g:04d}"
                         / f"galaxy_{g}_v95_pred_draws_100_heun100_n100.npz",
    },
    "TNG50": {
        "star": stars2dm_paths.TNG_RHOSTAR_MAP,
        "dm": stars2dm_paths.TNG_RHODM_MAP,
        "draws": lambda g: PRED_ROOT / "TNG50" / f"galaxy_{g:04d}"
                         / f"galaxy_{g}_TNG50_v95_heun100_n100_pred_draws_100.npz",
    },
    "FIREm12": {
        "star": stars2dm_paths.FIRE_RHOSTAR_MAP,
        "dm": stars2dm_paths.FIRE_RHODM_MAP,
        "draws": lambda g: PRED_ROOT / "FIREm12" / f"galaxy_{g:04d}"
                         / f"galaxy_{g}_FIREm12_v95_pred_draws_100"
                           "_heun100_FIREm12_flat_prior_baryon_100.npz",
    },
}

# ---------------------------------------------------------------------------------------
# Geometry, in centimetres on the TikZ canvas. Everything downstream reads from here.
# ---------------------------------------------------------------------------------------
LAYOUT = dict(
    width=19.0,
    height=11.40,
    margin=0.15,
    # left-hand data stack
    stack_cx=2.70,
    card_size=1.18,
    card_offset=0.21,
    # the big arrow into X_0
    feed_x0=4.65,
    feed_x1=5.45,
    # endpoint tiles
    endpoint_size=1.90,
    x0_cx=6.50,
    final_cx=16.95,
    # interpolant stations
    station_cx=(8.90, 11.30, 13.70),
    station_size=1.10,
    station_spread=0.62,
    # posterior draw stack
    draw_size=1.36,
    draw_offset=0.42,
)

# Vertical offsets measured from the bottom edge of whichever panel is being drawn.
#
# Budgeted against real text heights, which is the trap here: a node's height is set in
# points and does NOT follow SCALE, so at --width-cm 16.5 a \small line eats 0.50 canvas
# units, not the 0.40 cm it measures on paper. Every gap below allows for that.
DY = dict(
    title=5.10,
    row=3.72,
    tile_label=2.62,
    taxis=1.75,
    text_upper=1.30,
    text_lower=0.72,
    text_single=1.00,
    left_label_upper=2.60,
    left_label_lower=2.10,
)

# Colour ramp along the interpolant path: grey at t=0 through purple and blue to cyan at
# t=1. Same progression as the Keynote slide.
STATION_COLORS = ["6E6E6E", "7C6FD0", "2E86D6", "35C6E8"]

# How far apart the boxes at one station are drawn. The interpolant is
#     X_t = (1-t) X_0 + t^2 X_1 + sqrt(t) sigma_t eps,   sigma_t = 1-t
# (BaryonBridge/baryonbridge/interpolants/stochastic.py), so the injected noise has
# amplitude sqrt(t)(1-t): zero at both endpoints, peaking at t=1/3. The three boxes at a
# station are different noise realisations of the same t, so their spread follows that
# amplitude -- wide through the middle of the path, collapsing back onto X_1 at t=1.
NOISE_GAMMA = 1.0            # set from --noise-gamma; >1 exaggerates the contrast
NOISE_PROFILE = "symmetric"  # set from --noise-profile

# The three boxes at a station are noise realisations of the same t, so their spread stands
# for the injected noise. It is illustrative, not measured. The default "symmetric" profile
# is widest at t=1/2, which is how a reader expects a diffusion to look. "sigma" is the
# schedule the model actually uses, sqrt(t) sigma_t = sqrt(t)(1-t), which peaks at t=1/3 and
# so puts the bulge left of centre. "uniform" draws every station alike.
NOISE_PROFILES = {
    "sigma": (lambda t: np.sqrt(t) * (1.0 - t), 1.0 / 3.0),
    "symmetric": (lambda t: np.sin(np.pi * t), 0.5),
    "uniform": (lambda t: 1.0, 0.5),
}


def station_offsets():
    """Box spread in cm at each station; see NOISE_PROFILES."""
    shape, t_peak = NOISE_PROFILES[NOISE_PROFILE]
    peak = shape(t_peak)
    n = len(LAYOUT["station_cx"])
    ts = [(k + 1) / (n + 1.0) for k in range(n)]
    return [LAYOUT["station_spread"] * (shape(t) / peak) ** NOISE_GAMMA for t in ts]


# ---------------------------------------------------------------------------------------
# Tiles
# ---------------------------------------------------------------------------------------
def load_suite(name):
    spec = SUITES[name]
    for key in ("star", "dm"):
        if not spec[key].exists():
            raise SystemExit(f"missing map file for {name}: {spec[key]}")
    return np.load(spec["star"]), np.load(spec["dm"])


def load_draws(name, galaxy):
    path = SUITES[name]["draws"](galaxy)
    if not path.exists():
        raise SystemExit(f"missing draws for {name} galaxy {galaxy}: {path}")
    with np.load(path) as data:
        draws = data["draws"]
    if draws.ndim == 4:
        draws = draws[:, 0]
    return draws * STD_RHO + MEAN_RHO


def save_tile(data, cmap, vmin, vmax, path, size_in=1.5, dpi=300):
    """One square map tile, no axes, no padding.

    add_axes([0, 0, 1, 1]) with no tight bbox makes the saved page exactly size_in square,
    which is what lets TikZ place the tiles on a grid without per-tile fudge factors.
    """
    fig = plt.figure(figsize=(size_in, size_in))
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, origin="lower",
              interpolation="nearest", aspect="equal")
    ax.set_axis_off()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


def build_tiles(tile_dir, cfg):
    """Write every tile the schematic needs; return {name: relative path}."""
    tiles = {}

    def star(name, arr):
        tiles[name] = save_tile(arr, "gray_r", COND_VMIN, COND_VMAX,
                                tile_dir / f"{name}.pdf")

    def dm(name, arr):
        tiles[name] = save_tile(arr, "viridis", RHO_VMIN, RHO_VMAX,
                                tile_dir / f"{name}.pdf")

    dreams_star, dreams_dm = load_suite("DREAMS")

    # --- training row -------------------------------------------------------------
    for i, g in enumerate(cfg["train_stack"]):
        star(f"train_card{i}_star", dreams_star[g])
        dm(f"train_card{i}_dm", dreams_dm[g])
    star("train_x0", dreams_star[cfg["train_galaxy"]])
    dm("train_x1", dreams_dm[cfg["train_galaxy"]])

    # --- testing row --------------------------------------------------------------
    # The back cards come from the two out-of-domain suites, so the stack itself says
    # "three suites are tested" without needing another label.
    for i, (suite, g) in enumerate(cfg["test_stack"]):
        s_star, s_dm = load_suite(suite)
        star(f"test_card{i}_star", s_star[g])
        dm(f"test_card{i}_dm", s_dm[g])

    test_star, _ = load_suite(cfg["test_suite"])
    star("test_x0", test_star[cfg["test_galaxy"]])

    draws = load_draws(cfg["test_suite"], cfg["test_galaxy"])
    n = draws.shape[0]
    idx = np.linspace(0, n - 1, 4, dtype=int)
    for i, j in enumerate(idx):
        dm(f"test_draw{i}", draws[j])
    cfg["draw_labels"] = [f"draw {int(j) + 1}" for j in idx]
    cfg["n_draws"] = n

    return {k: f"tiles/{v.name}" for k, v in tiles.items()}


# ---------------------------------------------------------------------------------------
# TikZ emission
# ---------------------------------------------------------------------------------------
# Set from --width-cm. Multiplies every length written into the TikZ, and nothing else:
# font sizes and line widths are in points and stay put, so a smaller canvas prints the
# same text larger relative to the figure once it is scaled to \linewidth.
SCALE = 1.0


def fmt(v):
    return f"{v * SCALE:.3f}"


class Picture:
    def __init__(self):
        self.lines = []

    def add(self, line):
        self.lines.append(line)

    def tile(self, path, cx, cy, size, border="black!55", line_width="0.4pt"):
        self.add(
            f"\\node[inner sep=0pt, outer sep=0pt, draw={border}, "
            f"line width={line_width}] at ({fmt(cx)},{fmt(cy)}) "
            f"{{\\includegraphics[width={fmt(size)}cm,height={fmt(size)}cm]{{{path}}}}};"
        )

    def chip(self, cx, cy, body, fill="white", text_color="black", opacity="0.72"):
        """Small label plate inside a tile, as on the Figure 1 map panels."""
        self.add(
            f"\\node[anchor=south west, fill={fill}, fill opacity={opacity}, "
            f"text opacity=1, text={text_color}, inner sep=1.3pt, rounded corners=1pt] "
            f"at ({fmt(cx)},{fmt(cy)}) {{\\scriptsize {body}}};"
        )

    def text(self, cx, cy, body, anchor="center", size="\\small", color="black"):
        self.add(
            f"\\node[anchor={anchor}, text={color}, inner sep=1pt] "
            f"at ({fmt(cx)},{fmt(cy)}) {{{size} {body}}};"
        )

    def rect(self, x0, y0, x1, y1, opts):
        self.add(f"\\draw[{opts}] ({fmt(x0)},{fmt(y0)}) rectangle ({fmt(x1)},{fmt(y1)});")

    def line(self, x0, y0, x1, y1, opts):
        self.add(f"\\draw[{opts}] ({fmt(x0)},{fmt(y0)}) -- ({fmt(x1)},{fmt(y1)});")


def outer_frame(pic, x0, y0, x1, y1, y_split):
    """One box around both rows, divided by a rule.

    Two separate boxes, as in an earlier draft, put a band of dead space straight through
    the middle of the figure and bought nothing; the Keynote original splits one box.
    """
    pic.rect(x0, y0, x1, y1, "rounded corners=7pt, draw=black, line width=1.5pt")
    pic.line(x0, y_split, x1, y_split, "draw=black, line width=1.5pt")


def panel_title(pic, x0, y0, title):
    pic.text(x0 + 0.45, y0 + DY["title"], title, anchor="west",
             size="\\large\\bfseries")


def data_stack(pic, tiles, names, cx, cy):
    """Overlapping (stellar, dark matter) cards, back to front."""
    size = LAYOUT["card_size"]
    off = LAYOUT["card_offset"]
    n = len(names)
    pair_w = 2 * size
    for i, (star_key, dm_key) in enumerate(names):
        # i = 0 is the back card, drawn first and offset up-left.
        dx = (i - (n - 1) / 2.0) * off
        dy = -(i - (n - 1) / 2.0) * off
        left = cx + dx - pair_w / 2.0 + size / 2.0
        # A white plate behind each pair so the card underneath does not show through the
        # 1px seam between the two tiles.
        pic.rect(cx + dx - pair_w / 2.0 - 0.05, cy + dy - size / 2.0 - 0.05,
                 cx + dx + pair_w / 2.0 + 0.05, cy + dy + size / 2.0 + 0.05,
                 "fill=white, draw=black!45, line width=0.4pt")
        pic.tile(tiles[star_key], left, cy + dy, size)
        pic.tile(tiles[dm_key], left + size, cy + dy, size)


def station(pic, cx, cy, color, d):
    """One cluster of empty boxes standing for the distribution of X_t at that t.

    Three boxes rather than one: the dotted and dashed ones are other noise realisations
    of the same t, which is what makes the two arrow families below meaningful. They are
    deliberately left empty -- drawing actual interpolant states here would put six more
    near-identical maps in the figure and hide the flow.

    `d` is the spread between them, set by station_offsets() from sqrt(t)(1-t), so the
    cluster visibly widens along the path and then closes onto the endpoint.
    """
    s = LAYOUT["station_size"]
    boxes = {
        "dotted": (cx - 0.62 * d, cy + 0.62 * d, f"draw={color}!55, densely dotted, line width=0.7pt"),
        "solid": (cx, cy, f"draw={color}, line width=1.3pt"),
        "dashed": (cx + d, cy - d, f"draw={color}!85, densely dashed, line width=1.0pt"),
    }
    for key in ("dotted", "solid", "dashed"):
        bx, by, opts = boxes[key]
        pic.rect(bx - s / 2.0, by - s / 2.0, bx + s / 2.0, by + s / 2.0, opts)
    return {k: (v[0], v[1]) for k, v in boxes.items()}


def arrow(pic, p0, p1, color, dashed=False, width="1.1pt"):
    style = "densely dashed, " if dashed else ""
    pic.add(
        f"\\draw[-latex, {style}line width={width}, draw={color}] "
        f"({fmt(p0[0])},{fmt(p0[1])}) -- ({fmt(p1[0])},{fmt(p1[1])});"
    )


def chain(pic, cy, x0_right, final_left, final_y_solid, final_y_dashed):
    """Arrows through the station clusters: one solid family, one dashed family."""
    s = LAYOUT["station_size"]
    centers = LAYOUT["station_cx"]
    offs = station_offsets()
    stations = []
    for k, (cx, d) in enumerate(zip(centers, offs)):
        stations.append(station(pic, cx, cy, f"st{k}", d))

    def dashed_port(k):
        """Left and right attachment points on station k's dashed box.

        Taken below the solid box on purpose. The dashed box sits only `d` away, so once
        d < s/2 its left edge is hidden behind the solid one and an arrow aimed there
        buries its head inside the solid box instead of touching the dashed one.
        """
        cx, d = centers[k], offs[k]
        # Clamped so the head always clears the solid box, even at the last station where
        # the spread has nearly closed up.
        y = min(cy - d - 0.45 * s / 2.0, cy - s / 2.0 - 0.14)
        return (cx + d - s / 2.0, y), (cx + d + s / 2.0, y)

    # X_0 into the first cluster. The solid arrow crosses the faint dotted box on its way
    # to the solid one, as in the Keynote original.
    arrow(pic, (x0_right + 0.10, cy + 0.18),
          (centers[0] - s / 2.0 - 0.06, cy + 0.06), "st0")
    left0, _ = dashed_port(0)
    arrow(pic, (x0_right + 0.10, cy - 0.18), (left0[0] - 0.06, left0[1]),
          "st0", dashed=True)

    # Cluster to cluster: solid to solid, dashed to dashed. The dashed family slopes
    # because the spread changes with t.
    for k in range(len(centers) - 1):
        arrow(pic, (centers[k] + s / 2.0 + 0.06, cy),
              (centers[k + 1] - s / 2.0 - 0.06, cy), f"st{k + 1}")
        _, out = dashed_port(k)
        nxt, _ = dashed_port(k + 1)
        arrow(pic, (out[0] + 0.06, out[1]), (nxt[0] - 0.06, nxt[1]),
              f"st{k + 1}", dashed=True)

    # Last cluster out to whatever ends the row.
    arrow(pic, (centers[-1] + s / 2.0 + 0.06, cy), (final_left - 0.10, final_y_solid), "st3")
    _, out = dashed_port(len(centers) - 1)
    arrow(pic, (out[0] + 0.06, out[1]), (final_left - 0.10, final_y_dashed),
          "st3", dashed=True)
    return stations


def time_axis(pic, x0, x1, y, label):
    pic.add(
        f"\\draw[-latex, line width=0.8pt, draw=black!55] "
        f"({fmt(x0)},{fmt(y)}) -- ({fmt(x1)},{fmt(y)});"
    )
    pic.add(
        f"\\node[anchor=center, fill=white, inner sep=2pt] "
        f"at ({fmt((x0 + x1) / 2.0)},{fmt(y)}) {{\\footnotesize {label}}};"
    )
    pic.text(x0 - 0.12, y, "$t=0$", anchor="east", size="\\scriptsize", color="black!55")
    pic.text(x1 + 0.12, y, "$t=1$", anchor="west", size="\\scriptsize", color="black!55")


def draw_stack(pic, tiles, labels, cx, cy):
    """Posterior ensemble: overlapping predicted maps in a rounded frame."""
    size = LAYOUT["draw_size"]
    off = LAYOUT["draw_offset"]
    n = len(labels)
    half = (size + (n - 1) * off) / 2.0
    pic.add(
        f"\\draw[rounded corners=5pt, draw=black!45, line width=1.2pt] "
        f"({fmt(cx - half - 0.13)},{fmt(cy - half - 0.13)}) rectangle "
        f"({fmt(cx + half + 0.13)},{fmt(cy + half + 0.13)});"
    )
    fronts = []
    for i, label in enumerate(labels):
        dx = (i - (n - 1) / 2.0) * off
        dy = -(i - (n - 1) / 2.0) * off
        pic.tile(tiles[f"test_draw{i}"], cx + dx, cy + dy, size)
        pic.add(
            f"\\node[anchor=north east, fill=black, fill opacity=0.62, text opacity=1, "
            f"text=white, inner sep=1.3pt, rounded corners=1pt] "
            f"at ({fmt(cx + dx + size / 2.0 - 0.04)},{fmt(cy + dy + size / 2.0 - 0.04)}) "
            f"{{\\scriptsize {label}}};"
        )
        fronts.append((cx + dx - size / 2.0, cy + dy))
    return half, fronts


# ---------------------------------------------------------------------------------------
def build_tex(tiles, cfg):
    W, H = LAYOUT["width"], LAYOUT["height"]
    m = LAYOUT["margin"]
    ph = (H - 2 * m) / 2.0
    px0, px1 = m, W - m
    train_y0 = m + ph
    test_y0 = m

    pic = Picture()
    pic.add(f"\\useasboundingbox (0,0) rectangle ({fmt(W)},{fmt(H)});")
    outer_frame(pic, px0, test_y0, px1, test_y0 + 2 * ph, train_y0)

    es = LAYOUT["endpoint_size"]
    x0_cx = LAYOUT["x0_cx"]
    final_cx = LAYOUT["final_cx"]
    stack_cx = LAYOUT["stack_cx"]

    # =========================== TRAINING =============================================
    y0 = train_y0
    cy = y0 + DY["row"]
    panel_title(pic, px0, y0, "Training BaryonBridge (Sec.~2.1--2.3)")

    data_stack(pic, tiles,
               [(f"train_card{i}_star", f"train_card{i}_dm")
                for i in range(len(cfg["train_stack"]))],
               stack_cx, cy)
    pic.text(stack_cx, y0 + DY["left_label_upper"],
             "820 DREAMS training pairs", anchor="north", size="\\small")

    pic.add(f"\\draw[-latex, line width=2.6pt, draw=black] "
            f"({fmt(LAYOUT['feed_x0'])},{fmt(cy)}) -- ({fmt(LAYOUT['feed_x1'])},{fmt(cy)});")

    pic.tile(tiles["train_x0"], x0_cx, cy, es)
    pic.chip(x0_cx - es / 2.0 + 0.05, cy - es / 2.0 + 0.05, "$64\\times64$ kpc")
    pic.text(x0_cx, y0 + DY["tile_label"],
             "$X_0=\\log_{10}\\rho_{\\mathrm{star}}$", anchor="north", size="\\small")
    pic.tile(tiles["train_x1"], final_cx, cy, es)
    pic.text(final_cx, y0 + DY["tile_label"],
             "$X_1=\\log_{10}\\rho_{\\mathrm{DM}}$", anchor="north", size="\\small")

    chain(pic, cy, x0_cx + es / 2.0, final_cx - es / 2.0, cy + 0.18, cy - 0.18)

    if cfg["training_style"] == "sampled":
        taxis_label = "$t\\sim\\mathcal{U}(0,1)$ sampled"
    else:
        taxis_label = "interpolant path $X_t$"
    time_axis(pic, x0_cx, final_cx, y0 + DY["taxis"], taxis_label)

    pic.text(W / 2.0, y0 + DY["text_upper"],
             "\\textbf{Objective:} the U-Net $v_\\theta(X_t,X_0,t)$ matches the target "
             "velocity $r_t=\\dot X_t$",
             anchor="north", size="\\small")
    pic.text(W / 2.0, y0 + DY["text_lower"],
             "$X_t(X_0,X_1,t)$ is formed directly from the endpoint pair at a random $t$",
             anchor="north", size="\\footnotesize", color="black!55")

    # =========================== TESTING ==============================================
    y0 = test_y0
    cy = y0 + DY["row"]
    panel_title(pic, px0, y0, "Testing (Sec.~2.4, 3.2--3.3)")

    data_stack(pic, tiles,
               [(f"test_card{i}_star", f"test_card{i}_dm")
                for i in range(len(cfg["test_stack"]))],
               stack_cx, cy)
    pic.text(stack_cx, y0 + DY["left_label_upper"],
             "102 held-out DREAMS", anchor="north", size="\\small")
    pic.text(stack_cx, y0 + DY["left_label_lower"],
             "$+$ TNG50, FIRE-m12", anchor="north", size="\\footnotesize")

    pic.add(f"\\draw[-latex, line width=2.6pt, draw=black] "
            f"({fmt(LAYOUT['feed_x0'])},{fmt(cy)}) -- ({fmt(LAYOUT['feed_x1'])},{fmt(cy)});")

    pic.tile(tiles["test_x0"], x0_cx, cy, es)
    pic.chip(x0_cx - es / 2.0 + 0.05, cy - es / 2.0 + 0.05, "$64\\times64$ kpc")
    pic.text(x0_cx, y0 + DY["tile_label"],
             "$X_0=\\log_{10}\\rho_{\\mathrm{star}}$", anchor="north", size="\\small")

    dcx = final_cx
    half, fronts = draw_stack(pic, tiles, cfg["draw_labels"], dcx, cy)
    chain(pic, cy, x0_cx + es / 2.0, dcx - half - 0.13,
          fronts[0][1], fronts[-1][1])

    time_axis(pic, x0_cx, dcx - half - 0.13, y0 + DY["taxis"],
              "integrate $v_\\theta$")

    pic.text(W / 2.0, y0 + DY["text_single"],
             "\\textbf{Inference:} independent noise each run $\\Rightarrow$ "
             f"{cfg['n_draws']} draws from $p(X_1\\mid X_0)$",
             anchor="north", size="\\small")

    colors = "\n".join(
        f"\\definecolor{{st{i}}}{{HTML}}{{{c}}}" for i, c in enumerate(STATION_COLORS)
    )
    body = "\n".join("  " + ln for ln in pic.lines)

    return f"""% Generated by training_res/make_schematic.py -- do not edit by hand.
% Regenerate with:  python make_schematic.py
\\documentclass[10pt]{{article}}
\\usepackage[paperwidth={fmt(W)}cm,paperheight={fmt(H)}cm,margin=0pt]{{geometry}}
\\usepackage[T1]{{fontenc}}
\\usepackage{{amsmath}}
% Times for text and math, matching the STIX face the matplotlib figures use and the
% Times-like body font of the AASTeX class. mathptmx sets both from one package, so the
% labels here cannot drift away from the maths in the surrounding figures.
\\usepackage{{mathptmx}}
\\usepackage{{graphicx}}
\\usepackage{{xcolor}}
\\usepackage{{tikz}}
\\usetikzlibrary{{arrows,calc,positioning}}
\\pagestyle{{empty}}
\\setlength{{\\parindent}}{{0pt}}
\\setlength{{\\parskip}}{{0pt}}
\\setlength{{\\topskip}}{{0pt}}
{colors}
\\begin{{document}}%
\\begin{{tikzpicture}}[x=1cm,y=1cm]
{body}
\\end{{tikzpicture}}%
\\end{{document}}
"""


# ---------------------------------------------------------------------------------------
def compile_pdf(tex_path):
    for _ in range(2):
        proc = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", tex_path.name],
            cwd=tex_path.parent, capture_output=True, text=True,
        )
    if proc.returncode != 0:
        tail = "\n".join(proc.stdout.splitlines()[-40:])
        raise SystemExit(f"pdflatex failed:\n{tail}")
    return tex_path.with_suffix(".pdf")


def parse_ids(spec):
    return [int(v) for v in spec.split(",") if v.strip()]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir", default=str(FIG_OUT / "schematic"))
    ap.add_argument("--width-cm", type=float, default=16.5,
                    help="printed width of the figure. Text is set in points and does not "
                         "scale with it, so a NARROWER canvas prints the labels LARGER "
                         "once the figure is included at width=\\linewidth (18.06 cm in "
                         "an AASTeX figure*).")
    ap.add_argument("--copy-to", default=str(FIG_OUT / "schematic.pdf"),
                    help="where the finished PDF is copied; empty string to skip")
    ap.add_argument("--train-galaxy", type=int, default=500,
                    help="DREAMS galaxy shown as the X_0/X_1 pair in the training row "
                         "(must be in the training split, i.e. index >= 204)")
    ap.add_argument("--train-stack", default="260,640,1000",
                    help="the DREAMS training galaxies stacked behind it")
    ap.add_argument("--test-suite", default="DREAMS", choices=sorted(SUITES))
    ap.add_argument("--test-galaxy", type=int, default=42,
                    help="galaxy carried through the testing row; DREAMS 0-101 is the "
                         "held-out split")
    ap.add_argument("--test-stack", default="TNG50:3,FIREm12:2,DREAMS:42",
                    help="suite:galaxy cards in the testing stack, back to front")
    ap.add_argument("--noise-profile", default="symmetric",
                    choices=sorted(NOISE_PROFILES),
                    help="shape of the station-spread profile, which is illustrative "
                         "rather than measured. 'symmetric' peaks at t=1/2; 'sigma' is the "
                         "model's own sqrt(t)(1-t), which peaks at t=1/3; 'uniform' draws "
                         "every station alike.")
    ap.add_argument("--noise-gamma", type=float, default=1.0,
                    help="exponent on the sqrt(t)(1-t) noise profile that sets how far "
                         "apart the boxes at each station are drawn. 1.0 is the model's "
                         "own schedule; >1 exaggerates the middle bulge, 0 makes every "
                         "station equally wide.")
    ap.add_argument("--training-style", default="sampled", choices=("sampled", "chain"),
                    help="'sampled' labels the training time axis as a random draw of t; "
                         "'chain' reproduces the Keynote wording")
    ap.add_argument("--skip-tiles", action="store_true",
                    help="re-emit the TikZ only, reusing the tiles already on disk")
    args = ap.parse_args()

    global SCALE, NOISE_GAMMA, NOISE_PROFILE
    SCALE = args.width_cm / LAYOUT["width"]
    NOISE_GAMMA = args.noise_gamma
    NOISE_PROFILE = args.noise_profile

    outdir = Path(args.outdir)
    tile_dir = outdir / "tiles"
    outdir.mkdir(parents=True, exist_ok=True)

    test_stack = []
    for item in args.test_stack.split(","):
        suite, gid = item.split(":")
        if suite not in SUITES:
            raise SystemExit(f"unknown suite {suite!r}; choose from {sorted(SUITES)}")
        test_stack.append((suite, int(gid)))

    cfg = dict(
        train_galaxy=args.train_galaxy,
        train_stack=parse_ids(args.train_stack) + [args.train_galaxy],
        test_suite=args.test_suite,
        test_galaxy=args.test_galaxy,
        test_stack=test_stack,
        training_style=args.training_style,
    )

    if args.skip_tiles:
        names = sorted(p.stem for p in tile_dir.glob("*.pdf"))
        if not names:
            raise SystemExit(f"--skip-tiles but no tiles in {tile_dir}")
        tiles = {n: f"tiles/{n}.pdf" for n in names}
        draws = load_draws(cfg["test_suite"], cfg["test_galaxy"])
        idx = np.linspace(0, draws.shape[0] - 1, 4, dtype=int)
        cfg["draw_labels"] = [f"draw {int(j) + 1}" for j in idx]
        cfg["n_draws"] = draws.shape[0]
    else:
        tiles = build_tiles(tile_dir, cfg)

    tex_path = outdir / "schematic.tex"
    tex_path.write_text(build_tex(tiles, cfg))
    print(f"wrote {tex_path}")

    pdf = compile_pdf(tex_path)
    print(f"wrote {pdf}")
    if args.copy_to:
        dest = Path(args.copy_to)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(pdf, dest)
        print(f"copied to {dest}")


if __name__ == "__main__":
    main()
