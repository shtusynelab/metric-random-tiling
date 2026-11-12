# -*- coding: utf-8 -*-
"""vd_si_sigs.py -- SI S10.1/S10.2/Fig S7 regeneration (S9 exact-uniform
suite, six domains x 20 configurations).

S10.1 paired-density decay: per-domain ensemble g(r) (incircle clip + analytic
      disc normalisation), sliding-window RMS, far/mid ratio, the
      decay-to-noise radius, algebraic vs exponential form fit.
S10.2 edge-colour structure factor S_L(q): edge-midpoint sampling with
      (l - mean(l)) weights, radial average after masking local maxima,
      compared against the edge-colour permutation null (same mask); reports
      wing/valley aperture ratios.
Fig S7 halo orientation: per-configuration |F|^2 six-spot principal-axis
      offsets dtheta = theta_p (relative to ||G), and the axial lock-in angle
      of the ensemble-averaged |F|^2.
Outputs: <DATA>/pdf_decay_suite.csv / pdf_curves_suite.npz /
      slq_suite.csv / slq_profiles_suite.npz / halo_orient_suite.csv;
      <FIGS>/S9_pdf_slq.{png,pdf} / S10_halo_azimuth.{png,pdf}
"""
import math
import random
import sys
import time
import importlib
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import paths_common as PC  # noqa: E402

sys.path.insert(0, str(PC.HC))
from vd_patch import (build_patch_graph, build_patch_template,  # noqa: E402
                      embed_config, patch_bonds)
import pseudocrystal_lib as pl  # noqa: E402
_halo = importlib.import_module(PC.fname("P0_L1_halo_anisotropy.py")[:-3])  # noqa: E402
halo_anisotropy = _halo.halo_anisotropy
G_SQRT3 = _halo.G_SQRT3
_pdf3 = importlib.import_module(PC.fname("P0_M0_PDF_three_reference.py")[:-3])  # noqa: E402
pdf_2d, window_rms, rms_decay_length = (_pdf3.pdf_2d, _pdf3.window_rms,
                                        _pdf3.rms_decay_length)
fit_decay, clip_incircle, disk_radius = (_pdf3.fit_decay, _pdf3.clip_incircle,
                                         _pdf3.disk_radius)
from scipy.ndimage import maximum_filter  # noqa: E402

DATA = PC.DATA
RAW = DATA / "raw"
FIGS = PC.FIGURES
DOMAINS = ["b12", "a123", "a234", "a345", "a456", "a567"]
LAB = {"b12": "{1,2}", "a123": "{1,2,3}", "a234": "{2,3,4}", "a345": "{3,4,5}",
       "a456": "{4,5,6}", "a567": "{5,6,7}"}
COL = {"b12": "#c0392b", "a123": "#2166ac", "a234": "#4393c3",
       "a345": "#74add1", "a456": "#4575b4", "a567": "#313695"}
Q_MAX, Q_PIX = 10.5, 601            # main diffraction aperture (|F|^2, halo orientation)
QSL_MAX, QSL_PIX = 9.0, 301         # S_L(q) aperture (covers the first shell at 7.255)
R_MAX_PDF = 8.0

G = build_patch_graph(13, 13)
T = build_patch_template(G)
x_ideal = np.ones(G["E"], dtype=np.int64)
pts_id, cv_id = embed_config(G, T, x_ideal)
bonds = [(a, b) for a, b, e in patch_bonds(G, T, cv_id)]
bl_id = float(np.mean([np.hypot(*(pts_id[a] - pts_id[b])) for a, b in bonds]))
I_id, q1d_id = pl.structure_factor(pts_id / bl_id, q_max=Q_MAX, q_pixels=Q_PIX)


def spot_pixels_ideal(I, q1d, g=G_SQRT3):
    c = I.shape[0] // 2
    dq = q1d[1] - q1d[0]
    yy, xx = np.mgrid[0:I.shape[0], 0:I.shape[1]]
    rr = np.hypot((xx - c) * dq, (yy - c) * dq)
    ann = (rr >= 0.85 * g) & (rr <= 1.15 * g)
    iy0, ix0 = np.unravel_index(np.argmax(np.where(ann, I, 0.0)), I.shape)
    r0 = float(np.hypot((ix0 - c) * dq, (iy0 - c) * dq))
    phi0 = float(np.arctan2((iy0 - c) * dq, (ix0 - c) * dq))
    rad = int(0.35 / dq)
    out = []
    for kk in range(6):
        phi = phi0 + np.pi / 3 * kk
        ix = int(round(c + r0 * np.cos(phi) / dq))
        iy = int(round(c + r0 * np.sin(phi) / dq))
        sub = I[max(iy - rad, 0):iy + rad + 1, max(ix - rad, 0):ix + rad + 1]
        dy, dx = np.unravel_index(np.argmax(sub), sub.shape)
        out.append((kk, (max(ix - rad, 0) + dx - c) * dq,
                    (max(iy - rad, 0) + dy - c) * dq,
                    max(ix - rad, 0) + dx, max(iy - rad, 0) + dy))
    return out, c, dq


PIX, CC, DQQ = spot_pixels_ideal(I_id, q1d_id)
print("spots:", [(kk, round(gx, 2), round(gy, 2)) for kk, gx, gy, _, _ in PIX],
      flush=True)


def embed_norm(cfg):
    pts, _ = embed_config(G, T, cfg)
    bl = float(np.mean([np.hypot(*(pts[a] - pts[b])) for a, b in bonds]))
    return pts / bl, bl


def load_cfgs(tag):
    return np.load(RAW / f"torus13_{tag}.npz")["cfgs"].astype(np.int64)


# ---------------------------------------------------------------- S10.1 PDF
def run_pdf():
    rows, curves = [], {}
    for tag in DOMAINS:
        cfgs = load_cfgs(tag)
        gs, R_in = [], None
        for cfg in cfgs:
            pn, _bl = embed_norm(cfg)
            ctr, g, G2D, rho0, R = pdf_2d(pn, r_max=R_MAX_PDF)
            gs.append(g)
            if R_in is None:
                R_in = disk_radius(clip_incircle(pn))
        g = np.mean(gs, axis=0)
        wr, ws = window_rms(ctr, g)
        desc, param, r2 = fit_decay(np.column_stack([wr, ws]))
        rms_mid = float(ws[(wr > 2.0) & (wr < 3.0)].mean())
        rms_far = float(ws[(wr > 4.0) & (wr < 5.5)].mean())
        plateau = float(ws[(wr > 6.0) & (wr < 7.5)].mean())
        r_dec = rms_decay_length(wr, ws, floor=max(2.0 * plateau, 0.02))
        rows.append(dict(system=tag, domain=LAB[tag], n_config=len(cfgs),
                         R_incircle=R_in, rms_r2_3=rms_mid, rms_r4_55=rms_far,
                         rms_ratio_far_mid=rms_far / rms_mid,
                         rms_plateau_6_7_5=plateau, r_decay_to_noise=r_dec,
                         decay_form=desc, decay_param=param, decay_R2=r2))
        curves[tag] = np.column_stack([wr, ws])
        print(f"[PDF] {tag}: ratio {rms_far/rms_mid:.3f}  r_decay {r_dec}  "
              f"{desc} (R2={r2:.3f})  R_in={R_in:.1f}", flush=True)
    pd.DataFrame(rows).to_csv(DATA / "pdf_decay_suite.csv", index=False,
                              encoding="utf-8-sig")
    np.savez_compressed(DATA / "pdf_curves_suite.npz", **curves)
    return rows


# ---------------------------------------------------------------- S10.2 S_L(q)
def radial_masked(I, q1d, spot_centers, r_mask_q=0.6, n_bins=90):
    c = I.shape[0] // 2
    dq = q1d[1] - q1d[0]
    Y, X = np.ogrid[:I.shape[0], :I.shape[1]]
    R = np.sqrt((X - c) ** 2 + (Y - c) ** 2) * dq
    valid = R > 3 * dq
    for (sx, sy) in spot_centers:
        valid &= np.hypot((X - sx) * dq, (Y - sy) * dq) > r_mask_q
    bins = np.linspace(0.1, QSL_MAX, n_bins + 1)
    prof = np.full(n_bins, np.nan)
    for b, (lo, hi) in enumerate(zip(bins[:-1], bins[1:])):
        m = (R >= lo) & (R < hi) & valid
        if m.sum() > 8:
            prof[b] = I[m].mean()
    return 0.5 * (bins[:-1] + bins[1:]), prof


def find_spots(I, thr_frac=0.20):
    thr = I.max() * thr_frac
    loc = (I == maximum_filter(I, size=15)) & (I > thr)
    ys, xs = np.nonzero(loc)
    return list(zip(xs, ys))


def run_slq(n_perm=1, null_max_cfg=6):
    rows, profs = [], {}
    for tag in DOMAINS:
        cfgs = load_cfgs(tag)
        en, nu, en0 = [], [], []
        for ic, cfg in enumerate(cfgs):
            pn, bl = embed_norm(cfg)
            mid = np.array([0.5 * (pn[a] + pn[b]) for a, b in bonds])
            w = np.array([np.hypot(*(pn[a] - pn[b])) for a, b in bonds])
            wc = w - w.mean()
            I, q1d = pl.structure_factor(mid, q_max=QSL_MAX, q_pixels=QSL_PIX,
                                         weights=wc)
            spots = find_spots(I)
            rc, pr = radial_masked(I, q1d, spots)
            rc, pr0 = radial_masked(I, q1d, [])
            en.append(pr)
            en0.append(pr0)
            print(f"  [SLq] {tag} cfg {ic+1}/{len(cfgs)}", flush=True)
            rng = random.Random(9100 + 17 * ic)
            for _ in range(n_perm if ic < null_max_cfg else 0):
                wp = wc.copy()
                rng.shuffle(wp)
                I2, _ = pl.structure_factor(mid, q_max=QSL_MAX,
                                            q_pixels=QSL_PIX, weights=wp)
                _, pr2 = radial_masked(I2, q1d, spots)
                nu.append(pr2)
        en_m = np.nanmean(en, axis=0)
        nu_m = np.nanmean(nu, axis=0)
        e0_m = np.nanmean(en0, axis=0)
        wing = (rc >= 0.3) & (rc <= 2.0)
        valley = (rc >= 2.5) & (rc <= 5.5)
        rem = (rc >= 6.0) & (rc <= 8.0)
        rows.append(dict(system=tag, domain=LAB[tag], n_config=len(cfgs),
                         wing_ratio=float(e0_m[wing].mean() / nu_m[wing].mean()),
                         valley_ratio=float(en_m[valley].mean() / nu_m[valley].mean()),
                         remnant_ratio=float(en_m[rem].mean() / nu_m[rem].mean())))
        profs[f"{tag}|ens"] = np.column_stack([rc, en_m])
        profs[f"{tag}|ens0"] = np.column_stack([rc, e0_m])
        profs[f"{tag}|null"] = np.column_stack([rc, nu_m])
        print(f"[SLq] {tag}: wing {rows[-1]['wing_ratio']:.2f}  "
              f"valley {rows[-1]['valley_ratio']:.2f}  "
              f"remnant {rows[-1]['remnant_ratio']:.2f}", flush=True)
    pd.DataFrame(rows).to_csv(DATA / "slq_suite.csv", index=False,
                              encoding="utf-8-sig")
    np.savez_compressed(DATA / "slq_profiles_suite.npz", **profs)
    return rows


# ---------------------------------------------------------------- Fig S7 orient
def run_orient():
    rows, lock = [], []
    for tag in DOMAINS:
        cfgs = load_cfgs(tag)
        for ic, cfg in enumerate(cfgs):
            pn, _bl = embed_norm(cfg)
            I, q1d = pl.structure_factor(pn, q_max=Q_MAX, q_pixels=Q_PIX)
            print(f"  [ORIENT] {tag} cfg {ic+1}/{len(cfgs)}", flush=True)
            for kk, gx, gy, _ix, _iy in PIX:
                h = halo_anisotropy(I, q1d, gx, gy)
                rows.append(dict(cfg=ic, system=tag, spot=kk,
                                 phi_G=float(np.degrees(np.arctan2(gy, gx)) % 180),
                                 dtheta=float(h["theta_p"]),
                                 eig_ratio=float(h["eig_ratio"]),
                                 r=float(h["aniso"])))
        Im = np.load(DATA / f"meanF2_{tag}.npy")
        ts = [halo_anisotropy(Im, q1d, gx, gy)["theta_p"]
              for _kk, gx, gy, _ix, _iy in PIX]
        lock.append(dict(system=tag, domain=LAB[tag],
                         mean_abs_theta=float(np.mean(np.abs(ts))),
                         max_abs_theta=float(np.max(np.abs(ts)))))
        print(f"[ORIENT] {tag}: per-cfg dtheta sd "
              f"{np.std([r['dtheta'] for r in rows if r['system']==tag], ddof=1):.1f} deg"
              f" | ens-avg lock {lock[-1]['mean_abs_theta']:.1f} deg", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(DATA / "halo_orient_suite.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(lock).to_csv(DATA / "halo_orient_lock_suite.csv", index=False,
                              encoding="utf-8-sig")
    return df, lock


# ---------------------------------------------------------------- render
def render(pdf_rows, slq_rows, orient_df):
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullFormatter
    z = np.load(DATA / "pdf_curves_suite.npz")
    zp = np.load(DATA / "slq_profiles_suite.npz")

    fig = plt.figure(figsize=(3.35, 2.65))
    ax_a = fig.add_axes([0.135, 0.22, 0.33, 0.69])
    ax_b = fig.add_axes([0.665, 0.22, 0.30, 0.69])
    for tag in DOMAINS:
        arr = z[tag]
        ax_a.plot(arr[:, 0], arr[:, 1], lw=0.9, color=COL[tag],
                  label=f"$\{{{LAB[tag][1:-1]}\}}$")
    ax_a.set_yscale("log")
    ax_a.set_xlabel(r"$r$ (bond-length units)", labelpad=1.0, fontsize=6.5)
    ax_a.set_ylabel(r"$|g(r)-1|$ RMS", fontsize=6.5)
    ax_a.yaxis.set_label_coords(-0.24, 0.5)
    ax_a.set_xlim(0, 7)
    from matplotlib.ticker import NullFormatter
    ax_a.set_yticks([0.1, 0.2, 0.5, 1.0, 2.0])
    ax_a.set_yticklabels(["0.1", "0.2", "0.5", "1", "2"])
    ax_a.yaxis.set_minor_formatter(NullFormatter())
    ax_a.legend(fontsize=4.2, loc="upper right", borderaxespad=0.2,
                handlelength=1.0, labelspacing=0.16, ncol=2,
                columnspacing=0.5, framealpha=0.9)
    for tag in ("a123", "a456", "a567"):
        arr = zp[f"{tag}|ens"]
        ax_b.loglog(arr[:, 0], arr[:, 1], lw=0.9, color=COL[tag],
                    label=f"$\{{{LAB[tag][1:-1]}\}}$")
    arr = zp["a123|null"]
    ax_b.loglog(arr[:, 0], arr[:, 1], lw=0.9, color="#9a9a9a",
                label="permutation null")
    ax_b.set_yticks([2, 5, 10, 20, 50])
    ax_b.set_yticklabels(["2", "5", "10", "20", "50"])
    ax_b.yaxis.set_minor_formatter(NullFormatter())
    ax_b.set_xlabel(r"$q$ (bond-length$^{-1}$)", labelpad=1.0, fontsize=6.5)
    ax_b.set_ylabel(r"$S_L(q)$", fontsize=6.5)
    ax_b.yaxis.set_label_coords(-0.30, 0.5)
    ax_b.legend(fontsize=4.2, loc="lower left", borderaxespad=0.15,
                handlelength=1.0, labelspacing=0.16)
    for ax, s in ((ax_a, "a"), (ax_b, "b")):
        ax.tick_params(labelsize=5.2, length=2, pad=1.5)
        ax.text(-0.16, 1.03, s, transform=ax.transAxes, fontsize=7,
                fontweight="bold", va="bottom", ha="right")
    fig.savefig(FIGS / "S9_pdf_slq.png", dpi=300)
    fig.savefig(FIGS / "S9_pdf_slq.pdf")
    plt.close(fig)
    print("[OK] S9_pdf_slq", flush=True)

    d = orient_df
    lock = pd.read_csv(DATA / "halo_orient_lock_suite.csv")
    SIG = {"b12": 0.333, "a123": 0.408, "a234": 0.272, "a345": 0.204,
           "a456": 0.163, "a567": 0.136}
    fig = plt.figure(figsize=(3.35, 2.5))
    ax_a = fig.add_axes([0.12, 0.20, 0.375, 0.71])
    ax_b = fig.add_axes([0.625, 0.20, 0.335, 0.71])
    ax_a.axhline(0, color="#9a9a9a", lw=0.6, zorder=1)
    for tag in DOMAINS:
        g = d[d.system == tag]
        ax_a.plot(g.phi_G, g.dtheta, "o", ms=1.3, mfc=COL[tag], mec="none",
                  alpha=0.75, zorder=2, label=f"$\{{{LAB[tag][1:-1]}\}}$")
    bins = np.arange(0, 91, 15)
    mids, means = [], []
    for b0, b1 in zip(bins[:-1], bins[1:]):
        m = (d.phi_G >= b0) & (d.phi_G < b1)
        if m.sum():
            mids.append(0.5 * (b0 + b1))
            means.append(d.dtheta[m].mean())
    ax_a.plot(mids, means, "o-", ms=2.8, lw=0.9, color="#c0392b", zorder=3,
              label="azimuth bin mean")
    ax_a.set_xlabel("spot azimuth relative to $\|G$ (degrees)", labelpad=1.0, fontsize=6.5)
    ax_a.set_ylabel("$\Delta\theta$ (degrees)", fontsize=6.5)
    ax_a.yaxis.set_label_coords(-0.34, 0.5)
    ax_a.set_ylim(-100, 100)
    ax_a.legend(fontsize=4.2, loc="lower left", borderaxespad=0.15,
                handlelength=1.0, labelspacing=0.18, ncol=2,
                columnspacing=0.5, markerscale=2.0, framealpha=0.9)
    rng = np.random.default_rng(7)
    for tag in DOMAINS:
        g = d[d.system == tag]
        x = SIG[tag] + rng.uniform(-0.006, 0.006, len(g))
        ax_b.plot(x, g.dtheta, "o", ms=1.2, mfc=COL[tag], mec="none",
                  alpha=0.6, zorder=2)
    ax_b.axhline(0, color="#9a9a9a", lw=0.6, zorder=1)
    ax_b.plot([SIG[t] for t in DOMAINS], lock.mean_abs_theta, "s", ms=3.2,
              mfc="none", mec="#222222", mew=0.7, ls="none", zorder=4,
              label="ensemble mean |dtheta|")
    ax_b.set_xlim(0.42, 0.12)          # sigma/mu decreasing to the right
    ax_b.set_xlabel("relative dispersion $\sigma/\mu$", labelpad=1.0, fontsize=6.5)
    ax_b.set_ylim(-100, 100)
    ax_b.legend(fontsize=4.2, loc="lower left", borderaxespad=0.15,
                handlelength=1.0, framealpha=0.9)
    for ax, s in ((ax_a, "a"), (ax_b, "b")):
        ax.tick_params(labelsize=5.0, length=2, pad=1.5)
        ax.text(-0.17, 1.03, s, transform=ax.transAxes, fontsize=7,
                fontweight="bold", va="bottom", ha="right")
    fig.savefig(FIGS / "S10_halo_azimuth.png", dpi=300)
    fig.savefig(FIGS / "S10_halo_azimuth.pdf")
    plt.close(fig)
    sd0 = d[d.system.isin(["b12", "a123"])].dtheta.std(ddof=1)
    sd1 = d[d.system.isin(["a234", "a345", "a456", "a567"])].dtheta.std(ddof=1)
    print(f"[OK] S10_halo_azimuth (base-classes sd {sd0:.1f} deg, ladder sd {sd1:.1f} deg, "
          f"lock {lock.mean_abs_theta.min():.1f}-{lock.mean_abs_theta.max():.1f} deg)",
          flush=True)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    t0 = time.time()
    pdf_rows = run_pdf() if which in ("all", "pdf") else []
    print(f"--- pdf done {time.time()-t0:.0f}s", flush=True)
    slq_rows = run_slq() if which in ("all", "slq") else []
    print(f"--- slq done {time.time()-t0:.0f}s", flush=True)
    orient_df, lock = run_orient() if which in ("all", "orient") else (None, [])
    print(f"--- orient done {time.time()-t0:.0f}s", flush=True)
    if which == "all":
        render(pdf_rows, slq_rows, orient_df)
    print(f"done {time.time()-t0:.0f}s", flush=True)
