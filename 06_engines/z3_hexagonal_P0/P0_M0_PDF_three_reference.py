"""P0-M0 PDF G(r) three-reference analysis (criterion test of pillar 2).

Pillar 2: absence of long-range translational order -- "the pseudocrystal PDF
decays to the mean line within a few periods, with a measurable correlation
length". Three references: REF-C crystal / PC pseudocrystal / REF-A amorphous.

Method:
  - scattering units = the 3-coordinated network vertices (metal/linker sites,
    same convention as the FFT pipeline);
  - r in units of each system's mean bond length (comparable across the three
    references);
  - g(r): all-pairs distance histogram with equal-area disk approximation +
    circle-circle intersection area fraction correcting the finite-disk boundary;
  - G2D(r) = 2*pi*r*rho0*(g(r)-1);
  - envelope quantification: per-peak heights A_k (local maxima), power-law
    (log-log) vs exponential (semilog) fit comparison taking the better one,
    reporting the correlation length xi (defined by A = A1*exp(-r/xi)).

Criterion expectations:
  REF-C  oscillations do not decay (extend to the disk edge);
  PC     oscillations decay significantly within a few periods (measurable,
         finite correlation length);
  REF-A  nearly flat beyond 1-2 nearest-neighbor distances (only a broadened
         first peak).
"""
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from P0_M0_three_reference_data import load_all, load_all_ensembles

R_MAX, DR = 6.0, 0.02
BIN_EDGES = np.arange(0.0, R_MAX + DR, DR)
BIN_CTR = 0.5 * (BIN_EDGES[:-1] + BIN_EDGES[1:])


def disk_radius(pts):
    """Effective disk radius (convex-hull area equivalent)."""
    from scipy.spatial import ConvexHull
    hull = ConvexHull(pts)
    return math.sqrt(hull.volume / math.pi)   # 2D: volume = area


def circle_overlap_frac(rho_i, r, R):
    """Area fraction of circle (i, r) inside the disk (origin, R);
    |rho_i| + r <= R -> 1."""
    d = math.hypot(*rho_i)
    if d + r <= R:
        return 1.0
    if d >= R + r or d + R <= r:
        return 0.0
    # circle-circle intersection area (standard formula)
    r2 = R
    d2 = d
    if d2 < abs(r - r2):
        return min(1.0, (min(r, r2) / max(r, r2)) ** 2)
    a1 = r * r * math.acos((d2 * d2 + r * r - r2 * r2) / (2 * d2 * r))
    a2 = r2 * r2 * math.acos((d2 * d2 + r2 * r2 - r * r) / (2 * d2 * r2))
    a3 = 0.5 * math.sqrt(max(0.0, (-d2 + r + r2) * (d2 + r - r2)
                             * (d2 - r + r2) * (d2 + r + r2)))
    inter = a1 + a2 - a3
    return inter / (math.pi * r * r)


def clip_incircle(pts, shrink=0.99):
    """Convex-hull incircle clipping (same convention as the FFT pipeline):
    uniform density inside the circle makes the disk correction model exact."""
    from scipy.spatial import ConvexHull
    c = (pts.max(axis=0) + pts.min(axis=0)) / 2
    hull = ConvexHull(pts - c)
    r_in = np.inf
    for s1, s2 in hull.simplices:
        p1, p2 = pts[s1] - c, pts[s2] - c
        d = math.hypot(*(p2 - p1))
        if d > 0:
            r_in = min(r_in, abs(p1[0] * p2[1] - p2[0] * p1[1]) / d)
    r_in *= shrink
    keep = np.linalg.norm(pts - c, axis=1) <= r_in
    return pts[keep] - c


def pdf_2d(pts, r_max=R_MAX, dr=DR, use_incircle=True):
    """2D PDF: g(r) (boundary-corrected), G2D(r).

    use_incircle=True first clips to the convex-hull incircle (common convention
    for the three references, removing correction mismatch from convex-hull
    shape / edge density inhomogeneity).
    """
    if use_incircle:
        pts = clip_incircle(pts)
    N = len(pts)
    R = disk_radius(pts)
    rho0 = N / (math.pi * R * R)
    diff = pts[:, None, :] - pts[None, :, :]
    dist = np.sqrt((diff ** 2).sum(-1)).ravel()
    dist = dist[dist > 1e-9]
    hist, _ = np.histogram(dist, bins=np.arange(0.0, r_max + 2 * dr, dr))
    hist = hist.astype(float)
    # analytic normalization: sum_i rho0 * 2*pi*r*dr*f_i(r) (avoids an N x bin
    # double loop; N~300, bins~300, affordable)
    norm = np.zeros(len(hist))
    ctrs = (np.arange(len(hist)) + 0.5) * dr
    for i in range(N):
        for b, r in enumerate(ctrs):
            if r <= 0:
                continue
            f = circle_overlap_frac(pts[i], r, R)
            norm[b] += rho0 * 2 * math.pi * r * dr * f
    g = hist / norm
    ctr = (np.arange(len(hist)) + 0.5) * dr
    G2D = 2 * math.pi * rho0 * ctr * (g - 1.0)
    return ctr, g, G2D, rho0, R


def window_rms(r, g, half=0.25, step=0.1):
    """Sliding-window RMS of (g-1): robust measure of the oscillation amplitude
    (main criterion)."""
    out_r, out_s = [], []
    for r0 in np.arange(half + 0.4, r[-1], step):
        m = (r >= r0 - half) & (r <= r0 + half)
        if m.sum() > 10:
            out_r.append(r0)
            out_s.append(float(np.sqrt(np.mean((g[m] - 1) ** 2))))
    return np.array(out_r), np.array(out_s)


def rms_decay_length(r, s, floor=0.05):
    """First r where the oscillation RMS persistently drops below floor
    (engineering definition of decay to the mean line)."""
    below = s < floor
    for i in range(len(s)):
        if below[i] and (i + 2 >= len(s) or below[i + 1] and below[i + 2]):
            return r[i]
    return np.nan


def peak_envelope(r, g, r_min=0.7):
    """Per-local-maximum envelope (secondary quantification; shows bond-length
    splitting and shells)."""
    idx = [i for i in range(1, len(g) - 1)
           if g[i] >= g[i - 1] and g[i] > g[i + 1] and r[i] > r_min]
    # only collect peaks clearly above the mean line (threshold guards against
    # noise)
    peaks = []
    for i in idx:
        if g[i] > 1.05 and (not peaks or r[i] - peaks[-1][0] > 0.25):
            peaks.append((r[i], g[i]))
    return np.array(peaks)


def fit_decay(peaks):
    """Power-law vs exponential fit comparison. Returns (type, parameter, R^2);
    no peaks = the amorphous criterion itself."""
    if peaks is None or len(peaks) == 0:
        return "no peaks (amorphous)", np.nan, np.nan
    r, A = peaks[:, 0], peaks[:, 1]
    if len(r) < 3:
        return "insufficient", np.nan, np.nan
    # power law: lnA = -p*lnr + c
    ca = np.polyfit(np.log(r), np.log(A), 1)
    pa, ca0 = -ca[0], ca[1]
    ra = np.corrcoef(np.log(r), np.log(A))[0, 1] ** 2
    # exponential: lnA = -r/xi + c
    ce = np.polyfit(r, np.log(A), 1)
    xi = -1.0 / ce[0] if ce[0] < 0 else np.inf
    re = np.corrcoef(r, np.log(A))[0, 1] ** 2
    if ra >= re:
        return f"power p={pa:.2f}", pa, ra
    return f"exp xi={xi:.2f}", xi, re


def main():
    # ensemble averaging (M0 optional): REF-C 1 configuration / PC 5 / REF-A 3
    ens = load_all_ensembles()
    rows, summ = [], []
    curves, rms_curves, bands = {}, {}, {}
    for name, cfgs in ens.items():
        gs, G2Ds, rho0s, npts, Rs = [], [], [], [], []
        for pts, bonds, bl in cfgs:
            ctr, g, G2D, rho0, R = pdf_2d(pts)
            gs.append(g)
            G2Ds.append(G2D)
            rho0s.append(rho0)
            npts.append(len(pts))
            Rs.append(R)
        g = np.mean(gs, axis=0)
        g_std = np.std(gs, axis=0)
        G2D = np.mean(G2Ds, axis=0)
        rho0 = float(np.mean(rho0s))
        curves[name] = (ctr, g)
        bands[name] = (ctr, g_std)
        env = peak_envelope(ctr, g)
        desc, param, r2 = fit_decay(env)
        wr, ws = window_rms(ctr, g)
        rms_curves[name] = (wr, ws)
        rms_mid = float(ws[(wr > 2.0) & (wr < 3.0)].mean())
        rms_far = float(ws[(wr > 4.0) & (wr < 5.5)].mean())
        rms_desc, rms_p, rms_r2 = fit_decay(np.column_stack([wr, ws]))
        # single-configuration spread -> ensemble compression (main REF-A
        # validation)
        single_rms = []
        for gk in gs:
            wk, sk = window_rms(ctr, gk)
            single_rms.append(float(sk[(wk > 4.0) & (wk < 5.5)].mean()))
        print(f"[{name}] {len(cfgs)} configurations N~{np.mean(npts):.0f} "
              f"R_disk={np.mean(Rs):.2f}"
              f"(mean-bond-length units) rho0={rho0:.4f}")
        print(f"  envelope peaks r={np.round(env[:, 0], 2).tolist()}")
        print(f"  oscillation RMS: r in 2-3 {rms_mid:.3f} -> r in 4-5.5 {rms_far:.3f}"
              f"  ratio {rms_far / rms_mid:.3f}")
        print(f"  far-window RMS: single-config {np.mean(single_rms):.3f}"
              f"+/-{np.std(single_rms):.3f} -> ensemble {rms_far:.3f}"
              f" (compression ratio {rms_far / max(np.mean(single_rms), 1e-9):.2f})")
        print(f"  RMS decay fit {rms_desc} (R^2={rms_r2:.4f})")
        summ.append(dict(system=name, n_config=len(cfgs), N=int(np.mean(npts)),
                         rms_r2_3=rms_mid, rms_r4_55=rms_far,
                         rms_ratio_far_mid=rms_far / rms_mid if rms_mid else np.nan,
                         rms_single_mean=float(np.mean(single_rms)),
                         rms_single_std=float(np.std(single_rms)),
                         rms_decay_form=rms_desc, rms_decay_R2=rms_r2,
                         env_first_peaks=";".join(f"{r:.2f}" for r in env[:6, 0]) if len(env) else ""))
        for ri, gi, Gi, si in zip(ctr, g, G2D, g_std):
            rows.append(dict(r=ri, system=name, g=gi, G2D=Gi, g_std=si))

    df = pd.DataFrame(rows)
    piv = df.pivot(index='r', columns='system', values='g').reset_index()
    pivG = df.pivot(index='r', columns='system', values='G2D').reset_index()
    pivS = df.pivot(index='r', columns='system', values='g_std').reset_index()
    pivS = pivS.rename(columns={c: f'{c}_std' for c in pivS.columns if c != 'r'})
    out = piv.merge(pivG, on='r', suffixes=('_g', '_G2D')).merge(pivS, on='r')
    out.to_csv(ROOT / "P0_M0_PDF_three_reference.csv", index=False, encoding='utf-8-sig')
    pd.DataFrame(summ).to_csv(ROOT / "P0_M0_PDF_envelope_summary.csv", index=False,
                              encoding='utf-8-sig')

    # Plot: g(r) main panel + oscillation-RMS subpanel
    import matplotlib
    matplotlib.use('Agg')
    matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    matplotlib.rcParams['axes.unicode_minus'] = False
    import matplotlib.pyplot as plt
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(8.4, 7.2), height_ratios=[3, 2])
    colors = {'REF-C': 'tab:green', 'PC': 'tab:red', 'REF-A': 'tab:blue'}
    zh = {'REF-C': 'crystal', 'PC': 'pseudocrystal', 'REF-A': 'amorphous'}
    nbar = {'REF-C': 1, 'PC': len(ens['PC']), 'REF-A': len(ens['REF-A'])}
    for name, (ctr, g) in curves.items():
        tag = f" ({nbar[name]} configurations)" if nbar[name] > 1 else ""
        ax.plot(ctr, g, color=colors[name], lw=1.2,
                label=f"{name} ({zh[name]}){tag}")
        _, gs_band = bands[name]
        ax.fill_between(ctr, g - gs_band, g + gs_band, color=colors[name],
                        alpha=0.18, lw=0)
        wr, ws = rms_curves[name]
        ax2.semilogy(wr, ws, color=colors[name], lw=1.4, label=name)
    ax.axhline(1.0, color='gray', lw=0.6, ls=':')
    ax.axvline(1.0, color='k', lw=0.6, ls='--', alpha=0.3)
    ax.text(1.05, 0.4, "r=<bond length>", fontsize=8, alpha=0.6)
    ax.set_ylabel("g(r)")
    ax.set_title("Three-reference PDF g(r) (ensemble mean, band = inter-configuration "
                 "std): crystal oscillations do not decay / pseudocrystal decays "
                 "within a few periods / amorphous only a broadened bond peak")
    ax.set_xlim(0, R_MAX)
    ax.legend(fontsize=9)
    ax2.set_xlabel("r / <bond length>")
    ax2.set_ylabel("oscillation RMS of (g-1)")
    ax2.set_xlim(0, R_MAX)
    ax2.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(ROOT / "P0_M0_PDF_three_reference.png", dpi=200)
    print("Outputs: P0_M0_PDF_three_reference.csv / P0_M0_PDF_envelope_summary.csv / "
          ".png")


if __name__ == "__main__":
    main()
