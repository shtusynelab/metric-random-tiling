# -*- coding: utf-8 -*-
"""Step 13: three-panel method figure for the ln rho(W) solution steps.

(a) Geometry sketch: W=2, L=3 cylinder (one patch of the truncated honeycomb); one
    cut line highlighted with its 2W=4 mid edges = transfer state vector
    (a_1,a_2,b_1,b_2); arrows between columns = growth direction.
    Omega_open = 1^T T^L 1.
(b) Power-iteration convergence: x = iteration step n, y = rho_n = ||T v_n||_1
    (v_0 uniform); two curves (k=2 {1,2} W=5 -> 3.7604; k=3 {1,2,3} W=4 -> 12.2096);
    horizontal dashed = exact diagonalization values. This is the "solution":
    iterate to convergence; rho is the exact per-column growth factor as L -> inf.
(c) Spectrum tower: the 60 largest |lambda_i| of k=2 W=5; rho=3.7604 (tenfold
    translation-momentum degeneracy), secondary momentum towers, lambda=2 tower
    (124-fold) = the spectroscopic identity of the 2-army.
Output: TSD31212_rho_method.png/.pdf.
"""
import math
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Polygon as MplPolygon  # noqa: E402
import numpy as np  # noqa: E402

sys.stdout.reconfigure(encoding='utf-8')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import step7_tm as TM  # noqa: E402
from step3_dp_strips import build_31212_rect  # noqa: E402

BS = chr(92)
C_A, C_B = '#1f4e79', '#c0392b'
C_GUIDE = '#8a8a8a'
DOM = {2: '1,2', 3: '1,2,3', 4: '1,2,3,4'}


def dom_latex(k):
    return '$' + BS + '{' + DOM[k] + BS + '}$'


def build_T(W, k):
    tabs = TM.build_tables(W, k)
    TM.G.clear()
    TM.G.update(W=W, k=k, tabs=tabs)
    n = k ** (2 * W)
    T = np.zeros((n, n))
    for r in range(n):
        for (f, t, c) in TM.enum_row(r):
            T[t, f] += c
    return T


def panel_a(ax):
    """Geometry sketch: W=2 cylinder, three columns; cut line and state vector highlighted."""
    t = build_31212_rect(2, 3)
    ij, info, eqs = TM.extract_geometry(2, 3)
    # draw all faces
    for (keys, pts, ftype) in t.faces:
        fc = '#f2c14e' if ftype == 12 else '#cfe3f7'
        ax.add_patch(MplPolygon(pts, closed=True, facecolor=fc,
                                edgecolor='#9a9a9a', lw=0.5, zorder=1))
    # cut line: bucket1 (60 deg -> A'_i) and bucket2 (120 deg -> B'_{i-1}) of the j=1 column dodecagons
    cut_edges = []
    for fi, (i, j, bkt) in sorted(info.items()):
        if j != 1:
            continue
        for b in (1, 2):
            cut_edges.append((fi, bkt[b]))
    # edge geometry: face_edges order = pts order
    segs = []
    for fi, eid in cut_edges:
        pts = t.faces[fi][1]
        fe = t.face_edges[fi]
        for kk, (e, sgn) in enumerate(fe):
            if e == eid:
                segs.append((pts[kk], pts[(kk + 1) % len(pts)]))
                break
    for (p, q) in segs:
        ax.plot([p[0], q[0]], [p[1], q[1]], '-', lw=3.2, color=C_A,
                zorder=4, solid_capstyle='round')
    # state-vector annotations
    mid = [(0.5 * (p[0] + q[0]), 0.5 * (p[1] + q[1])) for (p, q) in segs]
    labels = ["$a_1$", "$a_2$", "$b_1$", "$b_2$"]
    for jj, ((mx, my), lab) in enumerate(
            zip(sorted(mid, key=lambda z: (z[1], z[0])), labels)):
        dx, dy = [(0.42, 0.30), (0.42, 0.30), (0.42, -0.52), (-0.95, -0.30)][jj]
        ax.annotate(lab, xy=(mx, my), xytext=(mx + dx, my + dy),
                    fontsize=8.5, color=C_A,
                    arrowprops=dict(arrowstyle='-', lw=0.6, color=C_A))
    ax.text(np.mean([m[0] for m in mid]) - 0.1,
            max(m[1] for m in mid) + 1.15,
            'cut state $s=(a_1,a_2,b_1,b_2)$', fontsize=7.5, color=C_A,
            ha='center')
    # growth-direction arrow (A2 direction)
    x0 = 2.2
    y0 = 0.4
    ax.annotate('', xy=(x0, y0 + 2.6), xytext=(x0, y0),
                arrowprops=dict(arrowstyle='-|>', lw=1.4, color='#333333'))
    ax.text(x0 + 0.12, y0 + 1.3, 'growth\n($L$ columns)', fontsize=7,
            color='#333333')
    ax.set_xlim(-0.9, 6.6)
    ax.set_ylim(-0.6, 5.4)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.text(0.03, 0.96, '(a)', transform=ax.transAxes, fontsize=9,
            fontweight='bold', va='top')
    ax.text(0.03, 0.86,
            'open cylinder: ' + r'$\Omega_{\rm open}=\mathbf{1}^{T}T^{L}'
            + r'\mathbf{1}$',
            transform=ax.transAxes, fontsize=7, va='top')


def panel_bc(ax_b, ax_c):
    """(b) power-iteration convergence; (c) spectrum tower."""
    cases = [(5, 2, C_A, 2), (4, 3, C_B, 3)]   # (W, k, color)
    for (W, k, c, _) in cases:
        T = build_T(W, k)
        ev = np.linalg.eigvals(T)
        absev = np.sort(np.abs(ev))[::-1]
        rho = absev[0]
        n = T.shape[0]
        v = np.ones(n) / n
        hist = []
        for it in range(1, 61):
            w = T @ v
            s = w.sum()
            hist.append(s)
            v = w / s
        ax_b.plot(range(1, 61), hist, '-', lw=1.3, color=c,
                  label=f'$k{{=}}{k}$ (' + dom_latex(k) + f'), $W{{=}}{W}$: '
                      f'$\\rho={rho:.4f}$')
        ax_b.axhline(rho, ls=(0, (5, 2.2)), lw=0.7, color=c, alpha=0.5,
                     zorder=1.5)
        print(f'k={k} W={W}: rho={rho:.8f} (60 iter, tail '
              f'{min(hist[-5:]):.6f}..{max(hist[-5:]):.6f})', flush=True)
        if (W, k) == (5, 2):
            m60 = absev[:60]
            ax_c.semilogy(range(1, 61), m60, 'o', ms=2.8, mfc=c, mec=c,
                          mew=0.0, zorder=3)
            ax_c.axhline(rho, ls=(0, (5, 2.2)), lw=0.9, color=c, alpha=0.8,
                         zorder=2)
            ax_c.text(1.2, rho * 1.04, rf'$\rho={rho:.3f}$', fontsize=7.5,
                      color=c, va='bottom')
            n2 = int(np.sum(np.abs(np.abs(ev) - 2.0) < 1e-6))
            i2 = int(np.argmax(np.abs(np.abs(ev) - 2.0) < 1e-6)) + 1
            ax_c.annotate(rf'$\lambda=2$ tower (${n2}\times$)',
                          xy=(i2, 2.0), xytext=(i2 * 0.52, 2.06),
                          fontsize=7, color='#555555',
                          arrowprops=dict(arrowstyle='-', lw=0.6,
                                          color='#999999'))
            ax_c.set_ylim(1.85, rho * 1.32)
    ax_b.set_xlabel('power-iteration step $n$', fontsize=7.5)
    ax_b.set_ylabel(r'$\rho_n=\|T v_n\|_1$', fontsize=7.5)
    ax_b.tick_params(labelsize=7)
    ax_b.legend(fontsize=6.5, loc='lower right', framealpha=0.92,
                labelspacing=0.3)
    ax_b.text(0.03, 0.96, '(b)', transform=ax_b.transAxes, fontsize=9,
              fontweight='bold', va='top')
    ax_b.text(0.97, 0.03, '60 steps shown; production: 600-2000 steps '
              'to rel.err $<10^{-10}$', transform=ax_b.transAxes,
              fontsize=5.8, color='#777777', ha='right')
    ax_c.set_xlabel('eigenvalue index $i$ (sorted by $|\\lambda|$, desc.)',
                    fontsize=7.5)
    ax_c.set_ylabel(r'$|\lambda_i(T)|$', fontsize=7.5)
    ax_c.tick_params(labelsize=7)
    ax_c.text(0.03, 0.96, '(c)', transform=ax_c.transAxes, fontsize=9,
              fontweight='bold', va='top')
    ax_c.text(0.97, 0.60,
              'PF: $\\Omega_{\\rm open}\\sim\\rho^L$\n'
              r'$\Rightarrow\ \lambda(W)=\ln\rho$',
              transform=ax_c.transAxes, fontsize=7.5, ha='right',
              va='center')


def main():
    fig = plt.figure(figsize=(9.6, 3.2), dpi=300)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.0, 1.0],
                          wspace=0.30, left=0.055, right=0.985,
                          top=0.96, bottom=0.14)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])
    panel_a(ax_a)
    panel_bc(ax_b, ax_c)
    for ext in ('.png', '.pdf'):
        fig.savefig(os.path.join(HERE, 'TSD31212_rho_method') + ext, dpi=300)
    print('written TSD31212_rho_method.png/.pdf')


if __name__ == '__main__':
    main()
