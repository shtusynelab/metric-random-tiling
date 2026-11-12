# -*- coding: utf-8 -*-
"""fig1_model_classification.py - Fig 1 of the metric-random-tiling paper
(composite figure).
Layout: a classical random tiling | b metric-disordered honeycomb (original
position/size kept, top row 2x1); c 3.4.6.4 | d 3.12.12 | e 4.6.12 (square
crops, the three together span the a-b total width); f ln Omega-N_F
four-carrier readout (moved to the right of the a-e block, stretched
vertically, height aligned with the total height of the a-e block).
Canvas 10.30x6.30 in; font sizes/line widths scaled by K = 10.30/7.205 --
this figure is wider than the 183 mm standard canvas, so after typesetting
scale-down the font sizes return to the 5-8 pt design range (recheck at the
production-typesetting round).
Assets: figs/assets_carriers/ (G1_lozenge_color/G5b_honeycomb_faces/
G4_rth_colorful are canonical in-repo copies; 31212_metric/4612_metric are
re-rendered by gen_metric_patches.py with uniform heavy black edges).
Output: ../figs/fig1_model_classification"""
import math
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.image as mpimg

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import paths_common as PC  # noqa: E402
from l1_style import plt, savefig  # noqa: E402

FIGS = PC.FIGURES
ASSETS = FIGS / "assets_carriers"

# ---------------- layout (units: inches; K = font-size scale) ----------------
WIN, HIN = 10.82, 6.30          # canvas
K = WIN / 7.205                 # 183-mm typesetting scale compensation
AX_A = (0.648, 2.791, 2.853, 2.853)   # a classical random tiling (original spot)
AX_B = (3.716, 2.791, 2.853, 2.853)   # b metric-disordered honeycomb (original spot)
SQ = 1.831                      # c-e square edge: (5.923 - 2x0.215)/3, width = a-b span
AX_C = (0.648, 0.554, SQ, SQ)   # c 3.4.6.4
AX_D = (2.694, 0.554, SQ, SQ)   # d 3.12.12
AX_E = (4.740, 0.554, SQ, SQ)   # e 4.6.12
AX_F = (7.571, 0.554, 2.650, 5.090)   # f: height = a-e total height (0.554 -> 5.644)
AX_CB = (6.725, 3.509, 0.072, 1.417)  # tile-area colorbar of b (b-f gap;

plt.rcParams.update({
    'font.size': 7 * K,
    'axes.labelsize': 7 * K,
    'legend.fontsize': 7 * K,
    'xtick.labelsize': 7 * K, 'ytick.labelsize': 7 * K,
    'axes.linewidth': 0.6 * K,
    'xtick.major.width': 0.6 * K, 'ytick.major.width': 0.6 * K,
    'xtick.major.size': 2.5 * K, 'ytick.major.size': 2.5 * K,
})


def rect(t):
    return (t[0] / WIN, t[1] / HIN, t[2] / WIN, t[3] / HIN)


def load_img(p):
    return mpimg.imread(str(p))


def show(ax, img, crop=None, tgt=1.0, tight=False):
    """Square-crop display (crop = relative window; tgt = target aspect
    ratio; tight = tight-crop to non-white content, so the three patterns
    c-e fill the same fraction of their square windows -- the c asset is
    originally full-bleed while the d/e assets carry ~2.8% white margins)."""
    if crop is not None:
        x0, x1, y0, y1 = crop
        h, w = img.shape[:2]
        img = img[int(h * y0):int(h * y1), int(w * x0):int(w * x1)]
    if tight:
        rgb = img[..., :3]
        sc = 255.0 if rgb.dtype.kind == 'f' else 1.0   # imread -> float [0,1]
        nz = (rgb.sum(axis=2) * sc < 720)
        rows = np.where(nz.any(axis=1))[0]
        cols = np.where(nz.any(axis=0))[0]
        img = img[rows.min():rows.max() + 1, cols.min():cols.max() + 1]
    ih, iw = img.shape[:2]
    if iw / ih > tgt:
        nw = int(ih * tgt)
        img = img[:, (iw - nw) // 2:(iw - nw) // 2 + nw]
    elif iw / ih < tgt:
        nh = int(iw / tgt)
        img = img[(ih - nh) // 2:(ih - nh) // 2 + nh, :]
    ax.imshow(img, interpolation='bilinear')
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


def fig_letter(fig, t, s):
    """Panel letter: placed outside the panel's top-left corner (bold,
    scaled by K)."""
    fig.text((t[0] - 0.108) / WIN, (t[1] + t[3] + 0.012) / HIN, s,
             fontsize=8 * K, fontweight='bold', va='bottom', ha='right')


def main():
    fig = plt.figure(figsize=(WIN, HIN))

    # ---- top row: a classical random tiling / b metric-disordered
    # honeycomb (positions and sizes unchanged) ----
    axL = fig.add_axes(rect(AX_A))
    show(axL, load_img(ASSETS / "G1_lozenge_color.png"), crop=(0.02, 0.98, 0.02, 0.98))
    axR = fig.add_axes(rect(AX_B))
    show(axR, load_img(ASSETS / "G5b_honeycomb_faces.png"), crop=(0.02, 0.98, 0.02, 0.98))
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    cax = fig.add_axes(rect(AX_CB))
    cb = fig.colorbar(ScalarMappable(Normalize(0.662, 1.489),
                                     plt.get_cmap('viridis')), cax=cax)
    cb.set_ticks([])                                   # no numeric ticks
    cax.tick_params(length=0)
    cb_cx = (AX_CB[0] + AX_CB[2] / 2) / WIN            # center large/small on the bar
    fig.text(cb_cx, (AX_CB[1] + AX_CB[3] + 0.035) / HIN, 'large',
             fontsize=7 * K, va='bottom', ha='center')
    fig.text(cb_cx, (AX_CB[1] - 0.035) / HIN, 'small',
             fontsize=7 * K, va='top', ha='center')
    cb.set_label('tile area', fontsize=7 * K, labelpad=1.5 * K)
    fig_letter(fig, AX_A, 'a')
    fig_letter(fig, AX_B, 'b')

    # ---- bottom row: c 3.4.6.4 / d 3.12.12 / e 4.6.12 (square crops,
    # uniform heavy black edges) ----
    specs = [
        (ASSETS / "G4_rth_colorful.png", AX_C, "3.4.6.4", 'c',
         r"locked deficiency $\rightarrow$ leak", r"$s_\infty$ = 0.040"),
        (ASSETS / "31212_metric.png", AX_D, "3.12.12", 'd',
         r"split rows $\rightarrow$ mid-edge", r"$s_3 \approx 0.19$"),
        (ASSETS / "4612_metric.png", AX_E, "4.6.12", 'e',
         r"split rows $\rightarrow$ lattice", r"$s_2 \geq 0.186$ [strict]"),
    ]
    for path, t, nm, letter, mech, val in specs:
        ax = fig.add_axes(rect(t))
        show(ax, load_img(path), tgt=1.0, tight=True)
        ax.text(0.5, 1.04, nm, transform=ax.transAxes, fontsize=7.0 * K,
                ha='center', va='bottom', color='#1a1a1a')
        ax.text(0.5, -0.045, mech + chr(10) + val, transform=ax.transAxes,
                fontsize=7.0 * K, ha="center", va="top", color="#555555",
                linespacing=1.35)
        fig_letter(fig, t, letter)

    # ---- f: four-carrier ln Omega-N_F readout (right, stretched) ----
    axc = fig.add_axes(rect(AX_F))
    tr = pd.read_csv(PC.engine("z4_3464") /
                     PC.fname("RTH_torus_area_law_8pts.csv"))
    pm = pd.read_csv(PC.HC / PC.fname("P0_PM_torus_growth_forms.csv"))
    tse = pd.read_csv(PC.HC /
                      PC.fname("P0_PM_torus_trace_extension_Ly3.csv"))
    da = pm[(pm.alphabet == 'arithmetic') & (pm.Ly == 3)]
    de = tse[(tse.alphabet == 'arithmetic') & (tse.Ly == 3) &
             (tse.Lx % 2 == 0)]
    da = pd.concat([da, de], ignore_index=True).sort_values('F')
    axc.plot(da.F, da.lnOmega, 's', ms=5.1 * K, color='#c0392b',
             label=r'honeycomb $L_y$=3', zorder=4)
    d = tr[tr.type == 'count']
    axc.plot(6 * d.L ** 2, d.lnOmega, '^', ms=5.7 * K, color='#e6811e',
             label=r'3.4.6.4 ($L\times L$)', zorder=4)
    L_arr = d.L.values.astype(float)
    coef = np.polyfit(L_arr, d.lnOmega.values.astype(float), 2)
    nf_fit = np.logspace(math.log10(24), math.log10(600), 100)
    axc.plot(nf_fit, np.polyval(coef, np.sqrt(nf_fit / 6.0)), '-', lw=1.0 * K,
             color='#e6811e', zorder=2)
    data4 = [(1, 6, math.log(2349)), (2, 24, math.log(29723517)),
             (3, 54, math.log(6.56623e12)), (4, 96, math.log(8.351e19))]
    xs = [t[1] for t in data4]
    ys = [t[2] for t in data4]
    axc.plot(xs[:2], ys[:2], 'o', ms=5.7 * K, color='#2166ac', zorder=4,
             label='4.6.12 exact')
    axc.plot(xs[2:], ys[2:], 'o', ms=5.7 * K, mfc='white', mec='#2166ac',
             mew=1.5 * K, zorder=4, label='4.6.12 est.')
    axc.plot(xs, ys, '-', color='#2166ac', lw=0.7 * K, zorder=3)
    d3 = pd.read_csv(PC.engine("z3_31212") / "TSD31212_dp_ternary.csv")
    d3 = d3[d3.W == 2].sort_values("L")
    axc.plot(6 * d3.L, d3.lnOmega_mid3, 'v', ms=5.1 * K, color='#7b3294',
             zorder=4, label='3.12.12 cyl.')
    c_ext = float(da.lnOmega.iloc[-1]) / float(da.F.iloc[-1])
    c_sub = float(d.lnOmega.iloc[0]) / math.sqrt(24.0)
    axc.plot([3.5 / c_ext, 75.0 / c_ext], [3.5, 75.0], ls=(0, (5, 2.2)),
             lw=0.8 * K, color='#8a8a8a', zorder=1.2, label='_nolegend_')
    c_law = 1.34 * c_sub        # raised as a whole to avoid overlap with 3.4.6.4
                                # (1.34x: leaves ~10 pt to the triangle markers at
                                # the right end and stays above the series)
    x_hi = min((75.0 / c_law) ** 2, 720.0)   # y must follow when x is cut (the
    axc.plot([3.6, x_hi],                # old hard-coded 75 gave a true slope of
             [c_law * math.sqrt(3.6), c_law * math.sqrt(x_hi)],
             ls=(0, (1.2, 1.6)), lw=0.8 * K,
             color='#8a8a8a', zorder=1.2, label='_nolegend_')
    axc.set_xlabel(r"$N_F$", fontsize=7 * K)
    axc.set_ylabel(r"$\ln\Omega$", fontsize=7 * K)
    axc.set_xscale("log")
    axc.set_yscale("log")
    axc.set_xlim(4, 750)   # was 260: cut off the L=7..10 points of 3.4.6.4
    from matplotlib.ticker import LogLocator, FuncFormatter, NullFormatter
    axc.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 3.0, 4.0, 6.0)))
    axc.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    axc.yaxis.set_minor_formatter(NullFormatter())   # plain numbers, major ticks only
    # slope labels written directly on the two dashed lines (the two slope
    # legend entries removed to free space; labels rotated along the lines,
    # placed in the gap above each line)
    def _disp_ang(x0_, y0_, x1_, y1_):
        p0 = axc.transData.transform((x0_, y0_))
        p1 = axc.transData.transform((x1_, y1_))
        return math.degrees(math.atan2(p1[1] - p0[1], p1[0] - p0[0]))
    x1l = 55.0
    y1l = c_ext * x1l
    axc.text(x1l, y1l * 1.10, 'slope 1',
             rotation=_disp_ang(x1l, y1l, 1.6 * x1l, 1.6 * y1l),
             rotation_mode='anchor', ha='left', va='bottom',
             fontsize=7.0 * K, color='#8a8a8a')
    x2l = 130.0
    y2l = c_law * math.sqrt(x2l)
    axc.text(x2l, y2l * 1.12, r'slope $\frac{1}{2}$',
             rotation=_disp_ang(x2l, y2l, 1.6 * x2l,
                                c_law * math.sqrt(1.6 * x2l)),
             rotation_mode='anchor', ha='left', va='bottom',
             fontsize=7.0 * K, color='#8a8a8a')
    axc.legend(fontsize=7.0 * K, frameon=False, loc="lower right")
    axc.tick_params(labelsize=7.0 * K)
    fig_letter(fig, AX_F, 'f')
    savefig(fig, str(FIGS / "fig1_model_classification"))
    print("saved", FIGS / "fig1_model_classification.{png,pdf}")


if __name__ == "__main__":
    main()
