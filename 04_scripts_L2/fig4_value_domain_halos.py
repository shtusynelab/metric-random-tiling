# -*- coding: utf-8 -*-
"""fig4_value_domain_halos.py - main-text Fig 4 (v2, delivered version).

Layout (canvas 10.82x6.30 in, same frame as Fig 1/2/3; font sizes/line
widths written at the original pt and scaled uniformly by scale_fig with
K = 10.82/7.205 after rendering, returning to 6-8 pt when printed at
183 mm):
  row 1 a = three {patch | |F(q)|^2} pairs side by side ({1,2}/{1,2,3}/
       {4,5,6}) + two horizontal colorbars (left = tile area/mean, viridis;
       right = log10<|F|^2>, inferno); patch face coloring = area x display
       scale^2 with one pooled mean shared by the three blocks (colors must
       be comparable across blocks);
  row 2 b = ladder dual axis f_coh (sigma/mu, left) + I_halo/I_core (right,
       {1,2} not connected) /
       c = anisotropy merge (S_perp/S_parallel per-k-window points +
       window-mean squares, first window k=0.26 dropped -- it sits below the
       13x13 supercell fundamental shell 4pi/(sqrt3*d*13)=0.322 and carries
       no independent mode; 10 windows k=0.34-1.06; diamonds = r halo shape
       ratio, spots csv) /
       d = w_rms(R) single source (21x21 production ensemble across all R,
       scheme b, cliff removed);
  row 3 e = tension-h illustration (h=+/-0.15 real-tiling mirror pair: same
       physical scale, ring counts matched by circumscribed size r=3/26.0
       vs r=4/26.8; +/-0.15 edge-length strong mixing keeps the metric
       randomness visible) /
       f = m(h) + Delta s(h) dual axis /
       g = phason response f_coh(h) + I_halo/I_core(h).
Data: ../data/uniform/ (f_coh uses the *_ref columns, convention in SI S9;
w_rms single-source table wrms_curve_tor21.csv; e reads
raw/torus13_a123_h+/-0.15.npz).
Output: ../figs/fig4_value_domain_halos.{png,pdf}"""
import importlib
import math
import queue
import sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import paths_common as PC  # noqa: E402
sys.path.insert(0, str(PC.HC))
import pseudocrystal_lib as pl  # noqa: E402
from vd_suite2 import metric_library, ENSEMBLES  # noqa: E402
from vd_patch import (build_patch_graph, build_patch_template,  # noqa: E402
                      embed_config)
halo_anisotropy = importlib.import_module(  # noqa: E402
    PC.fname("P0_L1_halo_anisotropy.py")[:-3])
G_SQRT3 = halo_anisotropy.G_SQRT3  # noqa: E402
from l1_style import panel_label, savefig  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import PolyCollection  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402
from matplotlib.cm import ScalarMappable  # noqa: E402

DATA = PC.DATA
FIGS = PC.FIGURES
Q_MAX, Q_PIX = 10.5, 601
COL = {"b12": "#c0392b", "a123": "#9ecae1", "n124": "#e6811e",
       "a234": "#7fa8d9", "a345": "#5b8ec4", "a456": "#2f6db0",
       "a567": "#123f6b"}   # arithmetic domains: monotonic gradient in n
LBL = {"b12": r"{1,2}", "a123": r"{1,2,3}", "n124": r"{1,2,4}",
       "a234": r"{2,3,4}", "a345": r"{3,4,5}", "a456": r"{4,5,6}", "a567": r"{5,6,7}"}
SHOW = ["b12", "a123", "a456"]
EN = dict(ENSEMBLES)
ORDER = ["b12", "a123", "a234", "a345", "a456", "a567"]
# revision round: ladder extension rungs {6,7,8} and {7,8,9}
# (data/uniform/ladder_ext.csv). Panel b only; panel c keeps its six domains.
COL.update({"a678": "#0c2b4a", "a789": "#061a30"})
LBL.update({"a678": r"{6,7,8}", "a789": r"{7,8,9}"})
ORD_B = ORDER + ["a678", "a789"]
CMAP = plt.get_cmap("viridis")

# ---------------- layout constants (inches) ----------------
WIN, HIN = 10.82, 6.30
K = WIN / 7.205                     # font/linewidth compensation (uniform
                                    # scaling after render)
PW, MW, MG = 1.48, 1.44, 0.20      # patch width / map width / group gap
                                    # (compromise)
# real-space enlarged / reciprocal-space reduced (visual balance)
PGS = (0.45, 0.07, 0.07)           # patch-map gaps (first pair leaves room
                                   # for the map's q_y label)
AX0 = 0.52                          # row 1 content left edge
AY, AH = 4.50, 1.60                 # row 1 image band (bottom y / height)
BAR_Y = 4.10                        # colorbar band (two horizontal colorbars)
XB = (0.50, 4.16, 7.82)             # b/c/d column x (v2: right-axis label
                                    # clearance)
BW, BW3 = 2.66, 2.50                # b/c width / d width
BY, BH = 2.47, 1.43                 # row 2 y/height (top clears colorbar
                                    # ticks, bottom clears rotated labels)
EY, EH = 0.50, 1.45                 # row 3 y/height
RTE = XB[2] + BW3                   # right edge 10.32


def fx(x_in):
    return x_in / WIN


def fy(y_in):
    return y_in / HIN


def scale_fig(fig, k):
    """Globally scale all text/linewidths/markers by k after rendering
    (panel-internal original pt values unchanged)."""
    def sq(t):
        t.set_fontsize(t.get_fontsize() * k)
    for ax in fig.axes:
        for t in ax.texts:
            sq(t)
        for t in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
            sq(t)
        for lb in (ax.xaxis.label, ax.yaxis.label):
            sq(lb)
        for ln in ax.lines:
            ln.set_linewidth(ln.get_linewidth() * k)
            ln.set_markersize(ln.get_markersize() * k)
            ln.set_markeredgewidth(ln.get_markeredgewidth() * k)
        for col in ax.collections:
            lws = col.get_linewidths()
            if len(lws):
                col.set_linewidth(lws[0] * k)
        for pt in ax.patches:
            if hasattr(pt, 'set_linewidth'):
                pt.set_linewidth(pt.get_linewidth() * k)
        lg = ax.get_legend()
        if lg is not None:
            for t in lg.get_texts():
                sq(t)
            for h in lg.legend_handles:
                if hasattr(h, 'set_linewidth'):
                    h.set_linewidth(h.get_linewidth() * k)
                if hasattr(h, 'set_markersize'):
                    h.set_markersize(h.get_markersize() * k)
    for t in fig.texts:
        sq(t)


def fcol(df, base):
    return df[f"{base}_ref"] if f"{base}_ref" in df.columns else df[base]


def polygon_area(pts):
    x, y = zip(*pts)
    return 0.5 * abs(sum(x[i] * y[i + 1] - x[i + 1] * y[i]
                         for i in range(-1, len(x) - 1)))


def _hole_polygons(sol, lib):
    """Per-hole vertex coordinates (BFS reconstruction; moved over from the
    L2 fig3 script)."""
    data = {c: lib.weights[idx] for c, idx in sol.items()}
    pos = {(0, 0): (0.0, 0.0)}
    hole_verts = {}
    bfs = queue.Queue()
    bfs.put((0, 0))
    seen = {(0, 0)}
    while not bfs.empty():
        curr = bfs.get()
        if curr not in data:
            continue
        wx, wy = pos[curr]
        w = data[curr]
        verts = [(wx, wy)]
        vx, vy = float(wx), float(wy)
        for i in range(6):
            vx += float(w[i]) * math.cos(math.radians(i * 60))
            vy += float(w[i]) * math.sin(math.radians(i * 60))
            verts.append((vx, vy))
        hole_verts[curr] = verts
        for i in range(6):
            d = pl.DIRECTIONS[i]
            nb = (curr[0] + d[0], curr[1] + d[1])
            if nb in data and nb not in seen:
                n_w = data[nb]
                k = (i + 3) % 6
                ax_ = ay_ = 0.0
                for step in range(k + 1):
                    if step == k:
                        ax_n, ay_n = ax_, ay_
                    ax_ += float(n_w[step]) * math.cos(math.radians(step * 60))
                    ay_ += float(n_w[step]) * math.sin(math.radians(step * 60))
                dest = verts[i + 1]
                pos[nb] = (dest[0] - ax_n, dest[1] - ay_n)
                seen.add(nb)
                bfs.put(nb)
    return hole_verts


def patch_faces(A, n_rings, seed):
    """Real-space patch -> per-hole polygons + raw areas + all vertices
    (face-coloring convention)."""
    lib = metric_library(A)
    sol, _ = pl.solve_tessellation(n_rings, lib, periodic=False,
                                   rng=__import__("random").Random(seed))
    hv = _hole_polygons(sol, lib)
    polys = [np.asarray(v[1:], dtype=float) for v in hv.values()]
    areas = np.array([polygon_area(p) for p in polys])
    allp = np.concatenate(polys, axis=0)
    return polys, areas, allp


def main():
    fig = plt.figure(figsize=(WIN, HIN))

    # ---- a: three patch | |F(q)|^2 pairs side by side ----
    # each patch fits its own window; face-color value = area x display
    # scale^2 (each block scaled into the same display frame), the three
    # blocks share one pooled mean (colors must allow cross-block tile-size
    # comparison; absolute physical lengths are not drawn -- only ratios
    # are physical)
    pats = {name: patch_faces(EN[name], 6, 20261100) for name in SHOW}
    wa, disp = {}, {}
    for name, (polys, areas, allp) in pats.items():
        ccx = (allp[:, 0].min() + allp[:, 0].max()) / 2
        ccy = (allp[:, 1].min() + allp[:, 1].max()) / 2
        hx, hy = np.ptp(allp[:, 0]) / 2, np.ptp(allp[:, 1]) / 2
        wa[name] = (ccx, ccy, hx, hy)
        s = min(PW / (2 * hx), AH / (2 * hy))       # in/data unit (display scale)
        disp[name] = areas * s * s
    pooled = float(np.mean(np.concatenate(list(disp.values()))))
    lo = min(v.min() for v in disp.values()) / pooled
    hi = max(v.max() for v in disp.values()) / pooled
    clim = (lo, hi)
    im = None
    ax_a = None
    xs_a, xx = [], AX0
    for j in range(len(SHOW)):
        xs_a.append(xx)
        xx += PW + PGS[j] + MW + MG
    for j, name in enumerate(SHOW):
        x0 = xs_a[j]
        axp = fig.add_axes([fx(x0), fy(AY), fx(PW), fy(AH)])
        if ax_a is None:
            ax_a = axp
        polys, areas, _ = pats[name]
        pc = PolyCollection(polys, array=disp[name] / pooled, cmap=CMAP,
                            edgecolors="black", linewidths=0.8)
        pc.set_clim(*clim)
        axp.add_collection(pc)
        ccx, ccy, chx, chy = wa[name]
        axp.set_xlim(ccx - chx, ccx + chx)
        axp.set_ylim(ccy - chy, ccy + chy)
        axp.set_aspect("equal")
        axp.axis("off")
        # domain label moved below the patch (space above left free)
        fig.text(fx(x0 + PW / 2), fy(AY) - 0.012,
                 f"$\\{{{LBL[name][1:-1]}\\}}$", fontsize=7.0, ha="center",
                 va="top", color="#333333")
        # reciprocal-space |F(q)|^2 (separate colormap family from the
        # real-space face coloring)
        axm = fig.add_axes([fx(x0 + PW + PGS[j]), fy(AY), fx(MW), fy(MW)])
        f = DATA / f"meanF2_{name}.npy"
        if not f.exists():
            axm.text(0.5, 0.5, "pending", ha="center", va="center",
                     transform=axm.transAxes, fontsize=8, color="#999999")
            continue
        I = np.load(f)
        dq = 2 * Q_MAX / Q_PIX
        yy, xx_ = np.mgrid[0:Q_PIX, 0:Q_PIX]
        rr0 = np.hypot((xx_ - Q_PIX // 2) * dq, (yy - Q_PIX // 2) * dq)
        Im = I.astype(float).copy()
        Im[Im <= 0] = Im[Im > 0].min()
        Im = np.where(rr0 < 1.0, Im[rr0 >= 1.0].min(), Im)
        im = axm.imshow(np.log10(Im.T), origin="lower",
                        extent=[-Q_MAX, Q_MAX, -Q_MAX, Q_MAX],
                        cmap="inferno", aspect="equal")
        for kk in range(6):
            phi = np.radians(30 + 60 * kk)   # measured first-shell bright
            # spots sit at 30 deg + 60k (log intensities 2.32/2.77/3.43 vs
            # 2.03/2.12/1.80 for the 0 deg group)
            axm.plot(G_SQRT3 * np.cos(phi), G_SQRT3 * np.sin(phi), marker="o",
                     ms=4.4, mfc="none", mec="w", mew=0.7, ls="none")
            # circles enlarged (the central bright spot of {4,5,6} previously
            # blended with the circle)
        axm.set_xlim(-9.2, 9.2)
        axm.set_ylim(-9.2, 9.2)
        axm.set_xticks([-6, 0, 6])
        axm.set_yticks([-6, 0, 6])
        axm.tick_params(labelsize=6.5, length=2)
        if j == 0:
            axm.set_ylabel(r"$q_y$ ($1/\ell$)")
        else:
            axm.set_yticklabels([])
        axm.set_xlabel(r"$q_x$ ($1/\ell$)" if j == 1 else "")
    # two horizontal colorbars: left = tile area/mean (face-coloring ruler);
    # right = log10<|F|^2>
    cax1 = fig.add_axes([fx(AX0), fy(BAR_Y), fx(1.90), fy(0.11)])
    cb1 = fig.colorbar(ScalarMappable(Normalize(*clim), CMAP), cax=cax1,
                       orientation="horizontal")
    cb1.set_ticks([round(clim[0], 1), round(clim[1], 1)])
    cb1.ax.xaxis.set_ticks_position("bottom")
    cb1.ax.tick_params(labelsize=5.2, width=0.5, length=1.2, pad=1.0)
    fig.text(fx(AX0 + 1.90 + 0.08), fy(BAR_Y + 0.055), "tile area / mean",
             fontsize=6.0, ha="left", va="center", color="#333333")
    if im is not None:
        cax2 = fig.add_axes([fx(RTE - 1.90), fy(BAR_Y), fx(1.90), fy(0.11)])
        cb = fig.colorbar(im, cax=cax2, orientation="horizontal")
        cb.ax.xaxis.set_ticks_position("bottom")
        cb.ax.tick_params(labelsize=5.2, width=0.5, length=1.2, pad=1.0)
        fig.text(fx(RTE - 1.90 - 0.08), fy(BAR_Y + 0.055),
                 r"$\log_{10}\langle|F(\mathbf{q})|^2\rangle$", fontsize=6.0,
                 ha="right", va="center", color="#333333")



    # ---- data ----
    rows = {}
    for name, A in EN.items():
        f = DATA / f"spots_{name}.csv"
        if not f.exists():
            continue
        df = pd.read_csv(f)
        rows[name] = dict(disp=np.std(A) / np.mean(A),
                          fc=fcol(df, "f_coh").mean(),
                          lo=fcol(df, "fcoh_lo").min(),
                          hi=fcol(df, "fcoh_hi").max(),
                          hc=df.halo.sum() / df.core.sum(),
                          hc_lo=(float(fcol(df, "hc_lo").mean())
                                 if "hc_lo" in df.columns else None),
                          hc_hi=(float(fcol(df, "hc_hi").mean())
                                 if "hc_hi" in df.columns else None),
                          r=df.aniso.mean(),
                          r_lo=df.r_lo.min(), r_hi=df.r_hi.max())
    _ext = pd.read_csv(DATA / "ladder_ext.csv")
    for _tag, _A in (("a678", (6, 7, 8)), ("a789", (7, 8, 9))):
        _r = _ext[_ext.tag == _tag]
        if len(_r):
            _r = _r.iloc[0]
            rows[_tag] = dict(disp=float(np.std(_A) / np.mean(_A)),
                              fc=float(_r.f_coh), lo=float(_r.fcoh_lo),
                              hi=float(_r.fcoh_hi), hc=float(_r.halo_core),
                              hc_lo=float(_r.hc_lo) if "hc_lo" in _ext.columns else None,
                              hc_hi=float(_r.hc_hi) if "hc_hi" in _ext.columns else None,
                              r=float(_r.r), r_lo=float(_r.r), r_hi=float(_r.r))

    dsp = pd.read_csv(DATA / "displacement_spectra.csv")
    dsp = dsp[dsp.k_mid > 0.30]   # first window k=0.26 dropped (below the
    # 13x13 supercell fundamental shell 4pi/(sqrt3*d*13)=0.322, no
    # independent mode; also removes the outlier's pull on the mean)
    wrm = pd.read_csv(DATA / "wrms_curve_tor21.csv")  # single source (21x21)
    # spanning all R; scheme b
    hsn = pd.read_csv(DATA / "hscan_summary.csv")

    # ---- b: ladder dual axis -- f_coh (left) + halo/core (right) ----
    axb = fig.add_axes([fx(XB[0]), fy(BY), fx(BW), fy(BH)])
    axb.set_yscale("log")
    lad = sorted([n for n in ORD_B if n in rows and n != "b12"],
                 key=lambda n: rows[n]["disp"])
    axb.plot([rows[n]["disp"] for n in lad], [rows[n]["fc"] for n in lad],
             color=COL["a456"], lw=0.8, ls="--", zorder=1)
    for n in ORD_B:
        if n not in rows:
            continue
        rw = rows[n]
        axb.errorbar(rw["disp"], rw["fc"],
                     yerr=[[max(rw["fc"] - rw["lo"], 1e-8)],
                           [max(rw["hi"] - rw["fc"], 1e-8)]],
                     fmt="o", ms=4.2, mfc="none", mec=COL[n], mew=1.2,
                     ecolor=COL[n], elinewidth=0.8, capthick=0.8,
                     capsize=2.8, zorder=3)
    axb.set_xlabel(r"relative dispersion $\sigma/\mu$")
    axb.set_ylabel(r"$f_{\rm coh}=\langle|F\rangle|^2/H_{\rm ref}$")
    axb.tick_params(labelsize=6.5)
    axb2 = axb.twinx()
    hcl = sorted([n for n in ORD_B if n in rows and n != "b12"],
                 key=lambda n: rows[n]["disp"])   # {1,2} not connected into
    # the line
    axb2.plot([rows[n]["disp"] for n in hcl], [rows[n]["hc"] for n in hcl],
              color="#888888", lw=0.8, ls=":", zorder=1)
    for n in ORD_B:
        if n not in rows:
            continue
        rw = rows[n]
        if rw.get("hc_lo") and rw.get("hc_hi"):
            axb2.errorbar(rw["disp"], rw["hc"],
                          yerr=[[max(rw["hc"] - rw["hc_lo"], 1e-3)],
                                [max(rw["hc_hi"] - rw["hc"], 1e-3)]],
                          fmt="D", ms=3.0, mfc="none", mec=COL[n], mew=1.0,
                          ecolor=COL[n], elinewidth=0.8, capthick=0.8,
                          capsize=2.6, zorder=3)
        else:
            axb2.plot(rw["disp"], rw["hc"], "D", ms=3.2, mfc="none",
                      mec=COL[n], mew=1.0)
    axb2.set_yscale("log")
    axb2.set_yticks([1, 10], labels=["1", "10"])   # saves space
    axb2.set_ylabel(r"$I_{\rm halo}/I_{\rm core}$", labelpad=1)
    axb2.tick_params(labelsize=6.5)
    from matplotlib.lines import Line2D
    hnd_b = [Line2D([], [], marker="o", ls="none", ms=3.0, mfc="none", mew=1.2,
                    mec="#888888", label=r"$f_{\rm coh}$  (left)"),
             Line2D([], [], marker="D", ls="none", ms=3.2, mfc="none",
                    mec="#888888", mew=1.0,
                    label=r"$I_{\rm halo}/I_{\rm core}$  (right)")]
    # legend replaces inline text (gray = marker-shape proxies; the previous
    # inline text was crossed by the b12 error-bar verticals, measured 51 px
    # of ink overlap)
    axb.legend(handles=hnd_b, fontsize=5.6, frameon=False, loc="center left",
               bbox_to_anchor=(0.02, 0.60), handletextpad=0.35,
               labelspacing=0.3, borderaxespad=0)

    # ---- c: anisotropy merge (S_perp/S_parallel per-k + r) ----
    axc = fig.add_axes([fx(XB[1]), fy(BY), fx(BW), fy(BH)])
    rng = np.random.default_rng(0)
    for i, n in enumerate(ORDER):
        g = dsp[dsp.system == n]
        if len(g):
            axc.plot(i - 0.17 + rng.uniform(-0.055, 0.055, len(g)),
                     g.ratio, "o", ms=2.1, mfc=COL[n], mec="none",
                     alpha=0.40, zorder=2)
            axc.plot(i - 0.17, g.ratio.mean(), "s", ms=3.4, color=COL[n],
                     mec="black", mew=0.5, zorder=3)
        if n in rows:
            rw = rows[n]
            axc.errorbar(i + 0.17, rw["r"],
                         yerr=[[max(rw["r"] - rw["r_lo"], 0)],
                               [max(rw["r_hi"] - rw["r"], 0)]],
                         fmt="D", ms=3.4, color=COL[n], capsize=2, lw=1,
                         zorder=3)
    axc.axhline(1.0, color="#999999", lw=0.7, ls="--")
    axc.text(0.02, 1.0, "isotropic", transform=axc.get_yaxis_transform(),
             fontsize=6, color="#777777", va="bottom")
    axc.set_xticks(range(len(ORDER)))
    axc.set_xticklabels([LBL[n] for n in ORDER], fontsize=6, rotation=30)
    axc.set_ylim(0, 1.15)
    axc.set_ylabel(r"anisotropy  ($S_\perp/S_\parallel$ and $r$)")
    axc.tick_params(labelsize=6.5)
    from matplotlib.lines import Line2D
    hnd = [Line2D([], [], marker="o", ls="none", ms=3.0, mfc="#9aa0a6",
                  mec="none", alpha=0.5, label=r"per $k$-window"),
           Line2D([], [], marker="s", ls="none", ms=3.2, mfc="#bbbbbb",
                  mec="black", mew=0.5, label="mean"),
           Line2D([], [], marker="D", ls="none", ms=3.2, color="#555555",
                  label=r"$r$")]
    axc.legend(handles=hnd, fontsize=5.2, frameon=False, loc="upper right",
               bbox_to_anchor=(1.0, 0.74),   # moved down: the original spot
               # was crossed by the isotropic dashed line
               handletextpad=0.25, labelspacing=0.22, borderaxespad=0.15)

    # ---- d: w_rms(R) single source (21x21 production ensemble across all
    #      R; scheme b) ----
    axd = fig.add_axes([fx(XB[2]), fy(BY), fx(BW3), fy(BH)])
    for n in ("b12", "a123", "a456"):
        g = wrm[wrm.tag == n].sort_values("R")
        w = np.sqrt(g.wrms2_mean.values)
        axd.errorbar(g.R, w, yerr=g.wrms2_sem.values / (2 * w),
                     fmt="o", ms=3.0, lw=0.9, color=COL[n], capsize=2.6,
                     capthick=0.9, elinewidth=0.9, label=f"$\\{{{LBL[n]}\\}}$")
    # ln L guide dashed line removed on purpose: the three domains have
    # different logarithmic coefficients (w/lnL: 0.045/0.051/0.113) and a
    # shallow power law (nu ~ 0.14-0.24) fits equally well; a single guide
    # line would over-claim a logarithmic law; roughness shown by the data
    # points only.
    axd.set_ylim(0.14, 0.62)   # top margin clears the single-row legend (was
                               # overlapping points)
    axd.legend(fontsize=5.4, frameon=False, loc="upper center", ncol=3,
               handletextpad=0.25, columnspacing=0.9, borderaxespad=0.3)
    axd.set_xlabel(r"$R$ (rings)")
    axd.set_ylabel(r"$w_{\rm rms}$")
    axd.tick_params(labelsize=6.5)

    # ---- e: tension-h illustration (mirror patterns of metric randomness:
    #      at h=+/-0.15 the ensemble edge lengths are strongly mixed
    #      ({1:10/2:48/3:41}% vs {47/42/11}%), patterns show clearly
    #      irregular hexagons; the two patches share one physical scale,
    #      ring counts matched by circumscribed size (r=3/26.0 vs r=4/26.8,
    #      3% apart) -- left slightly larger, right slightly smaller, both
    #      keeping metric randomness) ----
    axe = fig.add_axes([fx(XB[0]), fy(EY), fx(BW), fy(EH)])
    axe.axis("off")
    mvals = {}
    YC, YTXT = 17.0, 33.4
    for hh, r, xcen in (("-0.15", 3, 16.14), ("+0.15", 4, 48.55)):
        dd = np.load(DATA / "raw" / f"torus13_a123_h{hh}.npz")
        cfgs = dd["cfgs"]
        ms = np.array([c.sum() for c in cfgs]) / 169.0
        mvals[hh] = float(ms.mean())
        ic = int(np.argmin(np.abs(ms - ms.mean())))   # representative config
        # closest to the ensemble mean
        Gp = build_patch_graph(13, 13, root_rings=r)
        Tp = build_patch_template(Gp)
        pts, cv = embed_config(Gp, Tp, cfgs[ic])
        polys = [pts[cv[c]] for c in Gp["cells"]]
        cx = (pts[:, 0].min() + pts[:, 0].max()) / 2
        cy = (pts[:, 1].min() + pts[:, 1].max()) / 2
        polys = [p - [cx, cy] + [xcen, YC] for p in polys]   # translate into
        # place at the same scale
        axe.add_collection(PolyCollection(polys, facecolors="#dce9f5",
                                          edgecolors="black", linewidths=0.8))
    axe.text(16.14, 2.9, "$h=-0.15$" + chr(10) + "$m=%.2f$" % mvals["-0.15"],
             fontsize=6, ha="center", va="top")
    axe.text(48.55, 2.9, "$h=+0.15$" + chr(10) + "$m=%.2f$" % mvals["+0.15"],
             fontsize=6, ha="center", va="top")
    axe.annotate("", xy=(34.4, YC), xytext=(29.9, YC),
                 arrowprops=dict(arrowstyle="->", color="#555555", lw=0.9))
    axe.text(32.35, YTXT,
             r"tension $h$ biases edge lengths:  $p(l)\propto e^{-hl}$",
             fontsize=6.2, ha="center", va="center")
    axe.set_xlim(0, 65.1)
    axe.set_ylim(0, 35.5)
    axe.set_aspect("equal")

    # ---- f: m(h) + Delta s(h) dual axis (was misdrawn on panel e) ----
    axf = fig.add_axes([fx(XB[1]), fy(EY), fx(BW), fy(EH)])
    d = hsn.sort_values("h")
    axf.plot(d.h, d.Sigma_l_per_F, "-o", ms=3.5, color="#2f6db0", lw=0.9)
    axf.set_xlabel(r"tension $h$")
    axf.set_ylabel(r"$m(h)=\langle\Sigma l\rangle_h/F$")
    axf.tick_params(labelsize=6.5)
    ax2 = axf.twinx()
    ax2.plot(d.h, d.Delta_s, "-s", ms=3.5, color="#c0392b", lw=0.9)
    ax2.axhline(0, color="#999999", lw=0.6, ls=":")
    ax2.set_ylabel(r"$\Delta s$ ($k_B$ per face)", color="#a03020", labelpad=1)
    ax2.tick_params(labelsize=6.5, colors="#a03020")

    # ---- g: phason response ----
    axg = fig.add_axes([fx(XB[2]), fy(EY), fx(BW3), fy(EH)])
    _fcv = np.maximum(fcol(d, "f_coh"), 1e-6)
    _hcv = np.maximum(d.halo_core, 1e-2)
    _fs = (fcol(d, "f_coh_spread").values if "f_coh_spread" in d.columns
           else np.zeros(len(d)))
    _hsp = (d.halo_core_spread.values if "halo_core_spread" in d.columns
            else np.zeros(len(d)))
    # bars: same-measure two-sampling difference, drawn as wide-capped grey
    # intervals ABOVE the markers so even sub-marker intervals read as ticks
    axg.errorbar(d.h, _fcv, yerr=[np.minimum(_fs, 0.85 * _fcv), _fs],
                 fmt="none", ecolor="#2f6db0", elinewidth=0.9, capsize=3.4,
                 capthick=0.9, zorder=3)
    axg.errorbar(d.h, _hcv, yerr=[np.minimum(_hsp, 0.85 * _hcv), _hsp],
                 fmt="none", ecolor="#c0392b", elinewidth=0.9, capsize=3.4,
                 capthick=0.9, zorder=3)
    axg.plot(d.h, _fcv, "-", color="#2f6db0", lw=0.9, zorder=2)
    axg.plot(d.h, _hcv, "-", color="#c0392b", lw=0.9, zorder=2)
    axg.plot(d.h, _fcv, "o", ms=3.6, mfc="none", mec="#2f6db0", mew=1.2,
             zorder=4, label=r"$f_{\rm coh}$")
    axg.plot(d.h, _hcv, "s", ms=3.6, mfc="none", mec="#c0392b", mew=1.2,
             zorder=4, label=r"$I_{\rm halo}/I_{\rm core}$")
    axg.set_yscale("log")
    axg.legend(fontsize=5.6, frameon=False)
    axg.set_xlabel(r"tension $h$")
    axg.set_ylabel("phason response", labelpad=1)
    axg.tick_params(labelsize=6.5)

    panel_label(ax_a, "a", dy=0.965)
    panel_label(axb, "b")
    panel_label(axc, "c")
    panel_label(axd, "d")
    panel_label(axe, "e")
    panel_label(axf, "f")
    panel_label(axg, "g")
    scale_fig(fig, K)
    savefig(fig, str(FIGS / "fig4_value_domain_halos"))

if __name__ == "__main__":
    main()
