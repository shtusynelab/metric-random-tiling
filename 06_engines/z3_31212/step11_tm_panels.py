# -*- coding: utf-8 -*-
"""Step 11: 3.12.12 TM interpretation panels -- two-column layout matching the
manuscript Fig 2c/d style.

Left column (cf. Fig 2c): three stacked axes (k=4 / k=3 / k=2 top to bottom); each
segment = the exact torus DP chain at fixed W (lnOmega vs L, solid points) plus the
TM asymptotic slope line (ln rho(W) * L, dashed); separated x axis with ticks
pointing down between segments, true per-segment y scales; W annotated inside each
segment.
Right column (cf. Fig 2d): per-face entropy lambda(W)/(3W) vs 1/W scatter + linear
extrapolation, intercept = s_inf (k=2/3/4); zero line; honeycomb ternary arithmetic
TM points (open reference markers, convention noted) can be toggled.
Data: TSD31212_tm_rho.csv (lambda), TSD31212_tm_crossover.csv / deep (exact chains),
TSDHC_tm_rho.csv (honeycomb reference); missing chains are merged automatically from
the step7_deep / step7_cross outputs.
Output: TSD31212_tm_panels.png/.pdf.
"""
import csv
import math
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.stdout.reconfigure(encoding='utf-8')
HERE = os.path.dirname(os.path.abspath(__file__))
HC = Path(__file__).resolve().parents[1] / "z3_hexagonal_P0"

C_K = {2: '#1f4e79', 3: '#c0392b', 4: '#1e8449'}
C_HC = '#e6811e'
C_GUIDE = '#8a8a8a'
SPAN_PAD = (0.08, 0.10)   # vertical padding fraction within each segment

# Alphabet value domains (1-based manuscript convention; TM internally uses 0..k-1)
DOM = {2: '1,2', 3: '1,2,3', 4: '1,2,3,4'}


def dom_latex(k):
    return '$\\{' + DOM[k] + '\\}$'


def load_rho():
    d = {}
    with open(os.path.join(HERE, 'TSD31212_tm_rho.csv'),
              encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            d.setdefault(int(r['k']), {})[int(r['W'])] = float(r['ln_rho'])
    return d


def load_chains():
    """Exact chains at fixed (k, W): {(k, W): [(L, lnOmega), ...]} (crossover + deep merged)."""
    ch = {}
    fp = os.path.join(HERE, 'TSD31212_tm_crossover.csv')
    if os.path.exists(fp):
        with open(fp, encoding='utf-8-sig') as f:
            for r in csv.DictReader(f):
                if not r.get('Omega_mid'):
                    continue
                ch.setdefault((3, int(r['W'])), []).append(
                    (int(r['L']), float(r['lnOmega'])))
    fp = os.path.join(HERE, 'TSD31212_tm_deep.csv')
    if os.path.exists(fp):
        with open(fp, encoding='utf-8-sig') as f:
            for r in csv.DictReader(f):
                ch.setdefault((int(r['k']), int(r['W'])), []).append(
                    (int(r['L']), float(r['lnOmega'])))
    fp = os.path.join(HERE, 'TSD31212_tm_dpval.csv')
    if os.path.exists(fp):
        with open(fp, encoding='utf-8-sig') as f:
            for r in csv.DictReader(f):
                if not r.get('Omega_mid'):
                    continue
                ch.setdefault((int(r['k']), int(r['W'])), []).append(
                    (int(r['L']), float(r['lnOmega'])))
    for kk in ch:
        ch[kk] = sorted(set(ch[kk]))
    return ch


def load_hc_ternary():
    out = {}
    fp = os.path.join(HC, 'TSDHC_tm_rho.csv')
    if os.path.exists(fp):
        with open(fp, encoding='utf-8-sig') as f:
            for r in csv.DictReader(f):
                if r.get('alphabet') == 'ternary':
                    out[int(r['Ly'])] = float(r['ln_rho'])
    return out


def fig_panels(rho, chains, hc_ter=None, show_hc=True):
    ks = [k for k in (4, 3, 2) if k in rho and rho[k]]
    fig = plt.figure(figsize=(5.4, 4.3), dpi=300)
    # layout (Fig 2c proportions): left column x in [0.115, 0.60], right column x in [0.685, 0.965]
    lx0, lxw = 0.125, 0.475
    ay0, ayh = 0.10, 0.86
    gh = ayh / len(ks)
    axes = []
    ytop = ay0 + ayh
    for k in ks:
        ax = fig.add_axes([lx0, ytop - gh, lxw, gh])
        axes.append((k, ax))
        ytop -= gh
    # left-column chain choice: three segments = three transient phases -- k=3 W=2 long
    # chain (L=29..57, already near the asymptote: lambda 1.6015 -> 1.6052 climbing toward
    # 1.6094, Perron-Frobenius limit observed directly); k=4 W=2 (still climbing,
    # lambda 1.85 -> 1.97 -> 2.140); k=2 W=5 (2-army phase, exact for L<=21, crossover
    # bursts at the commensurate point L=25)
    W_of = {2: 5, 3: 2, 4: 2}   # displayed (k, W) chains
    fits_txt = []
    for k, ax in axes:
        W = W_of.get(k)
        lam = rho[k].get(W)
        pts = chains.get((k, W), [])
        c = C_K[k]
        # TM asymptotic slope line (dashed, drawn from the origin; x range matches the chain)
        Lmax = max((p[0] for p in pts), default=8)
        if lam is not None:
            xr = np.array([0, Lmax * 1.18])
            ax.plot(xr, lam * xr, '--', lw=1.0, color=c, alpha=0.85, zorder=2)
        # exact chain
        if pts:
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            ax.plot(xs, ys, 'o', ms=3.4, mfc=c, mec=c, mew=0.0, zorder=3)
        yvals = [p[1] for p in pts] + ([lam * Lmax] if lam is not None else [])
        ylo = min(yvals)
        yhi = max(yvals)
        pad = 0.06 * (yhi - ylo) + 0.3
        ax.set_xlim(0, Lmax * 1.18)
        ax.set_ylim(ylo - pad, yhi + pad)
        ax.text(0.30, 0.70, f'$k{{=}}{k}$ (' + dom_latex(k) + f'), $W{{=}}{W}$', transform=ax.transAxes,
                fontsize=7.5, ha='left', va='bottom', color='#333333')
        if lam is not None:
            fits_txt.append((k, W, lam))
        ax.tick_params(labelsize=6.8, length=2.0, width=0.8, pad=1.8)
    # separated x axes: middle/top segments get ticks pointing up, bottom segment full;
    # x ticks in uniform 5-unit steps per segment range
    for idx, (k, ax) in enumerate(axes):
        W = W_of[k]
        Lmax = max((p[0] for p in chains.get((k, W), [])), default=8) * 1.18
        step = 5 if Lmax > 25 else 2
        ax.set_xticks(np.arange(0, Lmax + 0.1, step))
        if idx < len(axes) - 1:
            ax.tick_params(labelbottom=False, bottom=False, top=True,
                           direction='out', length=1.8)
        else:
            ax.tick_params(bottom=True, labelbottom=True, top=True,
                           direction='out', length=1.8)
    axes[-1][1].set_xlabel('$L$ (torus length)', fontsize=7.5, labelpad=1.2)
    fig.text(lx0 - 0.034, ay0 + 0.5 * ayh, r'$\ln\Omega$', fontsize=7.5,
             rotation=90, ha='center', va='center')
    # TM asymptote legend (one dashed proxy line in the left-column arith. segment)
    axes[0][1].plot([], [], '--', lw=1.0, color=C_GUIDE,
                    label='TM asymptote $\\ln\\rho(W)\\cdot L$')
    axes[0][1].legend(fontsize=6.2, handlelength=1.4, loc='lower right',
                      borderaxespad=0.3, labelspacing=0.24)
    # ---- right column: lambda/(3W) vs 1/W extrapolation ----
    ax_d = fig.add_axes([0.685, ay0, 0.275, ayh])
    for k in (2, 3, 4):
        if k not in rho:
            continue
        Ws = sorted(rho[k])
        xs = np.array([1.0 / w for w in Ws])
        ys = np.array([rho[k][w] / (3 * w) for w in Ws])
        c = C_K[k]
        ax_d.plot(xs, ys, 'o', ms=4.6, mfc=c, mec=c, mew=0.0, zorder=3,
                  label=f'$k{{=}}{k}$ (' + dom_latex(k) + ')')
        cf = np.polyfit(xs, ys, 1)
        xr = np.linspace(0, xs.max() * 1.08, 40)
        ax_d.plot(xr, cf[0] * xr + cf[1], color=c, lw=1.0,
                  ls=(0, (3, 1.6)), zorder=2)
        ax_d.plot([0], [cf[1]], 's', ms=6.0, mfc='white', mec=c, mew=0.9,
                  zorder=3)
        print(f'[QA] k={k}: per-face {np.round(ys, 4).tolist()} '
              f'-> s_inf = {cf[1]:.4f}', flush=True)
    if show_hc and hc_ter:
        Ws = sorted(hc_ter)
        xs = np.array([1.0 / w for w in Ws])
        ys = np.array([hc_ter[w] / w for w in Ws])   # honeycomb per pore = lambda/W
        ax_d.plot(xs, ys, '^', ms=4.2, mfc='none', mec=C_HC, mew=0.9,
                  linestyle='none', zorder=3,
                  label='honeycomb $k{=}3$ arith.')
        cf = np.polyfit(xs, ys, 1)
        xr = np.linspace(0, xs.max() * 1.08, 40)
        ax_d.plot(xr, cf[0] * xr + cf[1], color=C_HC, lw=0.9,
                  ls=(0, (2, 1.4)), zorder=2, alpha=0.8)
        ax_d.plot([0], [cf[1]], '^', ms=5.5, mfc='white', mec=C_HC, mew=0.9,
                  zorder=3)
        print(f'[QA] honeycomb k=3 arith: s_inf = {cf[1]:.4f}', flush=True)
    ax_d.axhline(0.0, color='#9a9a9a', lw=0.8, zorder=1)
    ax_d.text(0.03, 0.012, '$s_\\infty$', fontsize=6.5, color='#555555',
              ha='left', va='bottom')
    ax_d.set_xlabel('$1/W$', fontsize=7.5, labelpad=1.2)
    ax_d.set_ylabel(r'$\lambda(W)/(3W)$', fontsize=7.5, labelpad=3.0)
    ax_d.tick_params(labelsize=6.8, length=2.0, width=0.8, pad=1.8)
    ax_d.legend(fontsize=6.2, handlelength=1.0, loc='upper left',
                borderaxespad=0.3, labelspacing=0.26, handletextpad=0.4)
    for ext in ('.png', '.pdf'):
        fig.savefig(os.path.join(HERE, 'TSD31212_tm_panels') + ext, dpi=300)
    print('written TSD31212_tm_panels.png/.pdf')


def main():
    rho = load_rho()
    chains = load_chains()
    hc_ter = load_hc_ternary()
    print('chains:', {k: len(v) for k, v in sorted(chains.items())})
    print('hc ternary TM:', hc_ter)
    fig_panels(rho, chains, hc_ter)


if __name__ == '__main__':
    main()
