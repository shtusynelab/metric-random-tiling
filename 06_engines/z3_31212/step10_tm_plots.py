# -*- coding: utf-8 -*-
"""Step 10: 3.12.12 TM adjudication plots (two standalone panels).

(a) Width-scaling main panel: lambda_mid(W) = ln rho(W) vs W (k=2/3/4 series, linear
    fits); area law <=> lambda proportional to W (slope = ln k units per width increment
    -- not 3s per face).
    Reference shapes, text annotations only (no data mixed in; different conventions:
    the honeycomb uses the hexagon closure equations, 3.12.12 the truncated mid
    equations; numerically identical in the binary case): binary honeycomb perimeter-law
    shape (lambda flat, ln 2 horizontal line, from the z3_hexagonal_P0 binary honeycomb
    TM of the same value system) and the honeycomb ternary arithmetic family
    (s ~ 0.45-0.5, steep lambda proportional to W).
(b) Crossover observations: lambda(L) sequences for W=2/3 (k=3) and W=5 (k=2) +
    TM asymptotes (horizontal dashed) + 2-army horizontal line (ln 2 per column;
    W=5: ln 2).
Output: TSD31212_width_scaling.png/.pdf + TSD31212_crossover.png/.pdf.
"""
import csv
import math
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.stdout.reconfigure(encoding='utf-8')
HERE = os.path.dirname(os.path.abspath(__file__))
RHO = os.path.join(HERE, 'TSD31212_tm_rho.csv')
CROSS = os.path.join(HERE, 'TSD31212_tm_crossover.csv')
DEEP = os.path.join(HERE, 'TSD31212_tm_deep.csv')

C_K2, C_K3, C_K4 = '#1f4e79', '#c0392b', '#1e8449'
C_GUIDE = '#8a8a8a'

# Alphabet value domains (1-based manuscript convention; TM internally uses 0..k-1)
DOM = {2: '1,2', 3: '1,2,3', 4: '1,2,3,4'}


def dom_latex(k):
    return '$\\{' + DOM[k] + '\\}$'


def load_rho():
    d = {}
    with open(RHO, encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            d.setdefault(int(r['k']), {})[int(r['W'])] = float(r['ln_rho'])
    return {k: sorted(v.items()) for k, v in d.items()}


def load_cross():
    pts = []
    if os.path.exists(CROSS):
        with open(CROSS, encoding='utf-8-sig') as f:
            for r in csv.DictReader(f):
                if r.get('lambda_per_col'):
                    pts.append((int(r['W']), int(r['L']),
                                float(r['lambda_per_col']),
                                float(r['TM_ln_rho'])))
    return pts


def load_deep():
    pts = []
    if os.path.exists(DEEP):
        with open(DEEP, encoding='utf-8-sig') as f:
            for r in csv.DictReader(f):
                pts.append((int(r['k']), int(r['W']), int(r['L']),
                            float(r['lnOmega'])))
    return pts


def fig_width_scaling(rho):
    fig, ax = plt.subplots(figsize=(4.4, 3.4), dpi=300)
    fits = {}
    for k, c, mk in ((2, C_K2, 'o'), (3, C_K3, 's'), (4, C_K4, '^')):
        if k not in rho:
            continue
        Ws = [w for w, _ in rho[k]]
        lns = [v for _, v in rho[k]]
        ax.plot(Ws, lns, mk, ms=4.2, color=c, linestyle='none',
                label=f'$k{{=}}{k}$ (' + dom_latex(k) + ')', zorder=4)
        A = np.polyfit(Ws, lns, 1)
        fits[k] = A
        xx = np.array([1.5, max(Ws) + 0.5])
        ax.plot(xx, np.polyval(A, xx), '-', lw=0.9, color=c, alpha=0.7,
                zorder=3)
    # perimeter-law reference: lambda constant (horizontal line = the per-column ln 2 of
    # the binary 2-army of honeycomb/3.12.12; but the TM shows the 2-army is transient,
    # so the reference line sketches "what a perimeter law would look like": the lambda(2)
    # horizontal line of k=2)
    lam2 = rho[2][0][1]
    ax.axhline(lam2, ls=(0, (5, 2.2)), lw=0.8, color=C_GUIDE, zorder=1.2,
               label='perimeter law: $\\lambda$ flat')
    ax.set_xlabel('width $W$')
    ax.set_ylabel(r'$\lambda_{\rm mid}(W)=\ln\rho(W)$')
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=6.5, loc='upper left', framealpha=0.9,
              labelspacing=0.3)
    ax.set_xlim(1.5, 8.6)
    fig.tight_layout(pad=0.4)
    for ext in ('.png', '.pdf'):
        fig.savefig(os.path.join(HERE, 'TSD31212_width_scaling') + ext,
                    dpi=300)
    for k, A in sorted(fits.items()):
        print(f'  lambda(W) linear fit k={k}: slope={A[0]:.4f} '
              f'intercept={A[1]:.4f} => s_mid ~= {A[0]/3:.4f} per face '
              f'(asymptotic-slope convention)')
    print('written TSD31212_width_scaling.png/.pdf')


def fig_crossover(rho):
    cross = load_cross()
    deep = load_deep()
    fig, ax = plt.subplots(figsize=(4.6, 3.4), dpi=300)
    # W=2 (k=3): DP crossover chain
    p2 = [(L, lam) for (W, L, lam, tgt) in cross if W == 2]
    if p2:
        ax.plot([p[0] for p in p2], [p[1] for p in p2], 'o', ms=3.4,
                color=C_K3, label='$k{=}3$, $W{=}2$: torus DP', zorder=4)
        tgt2 = p2[0] and [t for (W, L, lam, t) in cross if W == 2][0]
        ax.axhline(tgt2, ls=(0, (5, 2.2)), lw=0.8, color=C_K3, alpha=0.8,
                   zorder=1.2)
    # W=3 (k=3): crossover chain (2-army -> oscillating climb)
    p3 = [(L, lam) for (W, L, lam, tgt) in cross if W == 3]
    if p3:
        ax.plot([p[0] for p in p3], [p[1] for p in p3], 's', ms=3.4,
                color=C_K4, label='$k{=}3$, $W{=}3$: torus DP', zorder=4)
        tgt3 = [t for (W, L, lam, t) in cross if W == 3][0]
        ax.axhline(tgt3, ls=(0, (5, 2.2)), lw=0.8, color=C_K4, alpha=0.8,
                   zorder=1.2)
    # W=5 (k=2): exact 2-army points + deep-chain points
    p5 = [(L, ln / L) for (k, W, L, ln) in deep if k == 2 and W == 5]
    if p5:
        ax.plot([p[0] for p in p5], [p[1] for p in p5], 'D', ms=3.4,
                color=C_K2, label='$k{=}2$, $W{=}5$: torus DP', zorder=4)
        ax.axhline(rho[2][4][1], ls=(0, (5, 2.2)), lw=0.8, color=C_K2,
                   alpha=0.8, zorder=1.2)
    # 2-army line (ln 2, 1 bit per column)
    ax.axhline(math.log(2), ls=(0, (1.2, 1.6)), lw=0.8, color=C_GUIDE,
               zorder=1.0, label='exact 2-army: $\\ln 2$ per col')
    ax.set_xlabel('torus length $L$')
    ax.set_ylabel(r'$\lambda_{\rm mid}(L)=\ln\Omega(L)-\ln\Omega(L{-}1)$ '
                  r'per column')
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=6.0, loc='center right', framealpha=0.92,
              labelspacing=0.3)
    ax.set_ylim(0.5, 3.4)
    fig.tight_layout(pad=0.4)
    for ext in ('.png', '.pdf'):
        fig.savefig(os.path.join(HERE, 'TSD31212_crossover') + ext, dpi=300)
    print('written TSD31212_crossover.png/.pdf')


def main():
    rho = load_rho()
    print('rho table:', {k: len(v) for k, v in rho.items()})
    fig_width_scaling(rho)
    fig_crossover(rho)


if __name__ == '__main__':
    main()
