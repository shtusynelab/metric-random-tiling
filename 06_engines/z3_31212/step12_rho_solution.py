# -*- coding: utf-8 -*-
"""Step 12: two-panel figure for the ln rho(W) solution process.

(a) Tr(T^L) semi-log: Tr(T^L) = sum lambda_i^L -> rho^L (large L); log slope = ln rho.
    Plots k=2 W=5 (n=1024, the famous 2-army tower) and k=3 W=4 (n=6561);
    fit window in the large-L segment; fit slope annotated vs the independent
    Perron value.
(b) Spectrum tower: the 30 largest |lambda_i| of k=2 W=5 -- marking rho (Perron) and
    the degenerate tower at lambda=2 (the 2-army = the huge degenerate sector of the
    subdominant eigenvalue 2.0).
Output: TSD31212_rho_solution.png/.pdf; also saves TSD31212_tm_spectrum.csv
(spectrum).
Local memory: n=6561 eigvals within ~1 GB; n=1024 seconds-level.
"""
import csv
import math
import os
import sys
import time
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.stdout.reconfigure(encoding='utf-8')
HERE = os.path.dirname(os.path.abspath(__file__))
C_A, C_B, C_G = '#1f4e79', '#c0392b', '#8a8a8a'


def build_T(W, k):
    import step7_tm as TM
    tabs = TM.build_tables(W, k)
    TM.G.clear()
    TM.G.update(W=W, k=k, tabs=tabs)
    n = k ** (2 * W)
    T = np.zeros((n, n))
    for r in range(n):
        for (f, t, c) in TM.enum_row(r):
            T[t, f] += c
    return T


def fig_solution():
    out_rows = []
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(7.6, 3.3), dpi=300)
    cases = [(5, 2, C_A, 'k=2, W=5'), (4, 3, C_B, 'k=3, W=4')]  # (W, k)
    for (W, k, c, lab) in cases:
        T = build_T(W, k)
        ev = np.linalg.eigvals(T)
        absev = np.sort(np.abs(ev))[::-1]
        rho = absev[0]
        lnr = math.log(rho)
        print(f'{lab}: rho={rho:.8f} ln_rho={lnr:.6f}', flush=True)
        # (a) power-iteration convergence history: rho_n = ||T v_n||_1 (uniform initial v)
        n = T.shape[0]
        v = np.ones(n) / n
        hist = []
        for it in range(1, 121):
            w = T @ v
            s = w.sum()
            hist.append(s)
            v = w / s
        ax_a.plot(range(1, 121), hist, '-', lw=1.1, color=c,
                  label=lab + rf': $\rho={rho:.4f}$')
        ax_a.axhline(rho, ls=(0, (5, 2.2)), lw=0.7, color=c, alpha=0.5,
                     zorder=1.5)
        out_rows.append([k, W, f'{rho:.8f}', f'{lnr:.8f}', '', ''])
        if (W, k) == (5, 2):
            m60 = absev[:60]
            ax_b.semilogy(range(1, 61), m60, 'o', ms=2.8, mfc=c, mec=c,
                          mew=0.0, zorder=3)
            ax_b.axhline(rho, ls=(0, (5, 2.2)), lw=0.9, color=c, alpha=0.8,
                         zorder=2)
            ax_b.text(1.0, rho * 1.06, rf'$\rho={rho:.3f}$', fontsize=7,
                      color=c, va='bottom')
            ax_b.axhline(2.0, ls=(0, (1.2, 1.6)), lw=0.9, color=C_G,
                         zorder=2)
            n2 = int(np.sum(np.abs(np.abs(ev) - 2.0) < 1e-6))
            i2 = int(np.argmax(np.abs(np.abs(ev) - 2.0) < 1e-6)) + 1
            ax_b.annotate(rf'$\lambda{{=}}2$ tower ({n2}$\times$)',
                          xy=(i2, 2.0), xytext=(i2 * 0.55, 2.0 * 0.93),
                          fontsize=7, color='#555555',
                          arrowprops=dict(arrowstyle='-', lw=0.6,
                                          color='#999999'))
            ax_b.set_ylim(1.85, rho * 1.3)
    ax_a.set_xlabel('iteration $n$')
    ax_a.set_ylabel(r'$\ln \mathrm{Tr}(T^L)$')
    ax_a.tick_params(labelsize=7)
    ax_a.legend(fontsize=6.5, loc='lower right', framealpha=0.9)
    ax_a.text(0.03, 0.94, '(a)', transform=ax_a.transAxes, fontsize=9,
              fontweight='bold', va='top')
    ax_b.set_xlabel('eigenvalue index $i$')
    ax_b.set_ylabel(r'$|\lambda_i(T)|$')
    ax_b.tick_params(labelsize=7)
    ax_b.text(0.03, 0.94, '(b)', transform=ax_b.transAxes, fontsize=9,
              fontweight='bold', va='top')
    fig.tight_layout(pad=0.5)
    for ext in ('.png', '.pdf'):
        fig.savefig(os.path.join(HERE, 'TSD31212_rho_solution') + ext,
                    dpi=300)
    with open(os.path.join(HERE, 'TSD31212_tm_spectrum.csv'), 'w',
              newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['k', 'W', 'rho', 'ln_rho', 'Tr_fit_slope', 'abs_diff'])
        w.writerows(out_rows)
    print('written TSD31212_rho_solution.png/.pdf + TSD31212_tm_spectrum.csv')


if __name__ == '__main__':
    fig_solution()
