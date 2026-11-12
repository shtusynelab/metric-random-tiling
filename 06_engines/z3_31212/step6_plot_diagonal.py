# -*- coding: utf-8 -*-
"""Step 6: 3.12.12 diagonal family lnOmega vs N_F log-log plot (styled after the
manuscript Fig 1c panel).

Coordinates (same as Fig 1c): points = (N_F, lnOmega), both axes log =>
  slope 1 <=> lnOmega proportional to N_F (area law, nonzero per-face entropy);
  slope 1/2 <=> lnOmega proportional to sqrt(N_F) (perimeter law).
Series: binary diagonal L=1.. (DP, filled) + ternary diagonal L=1 (brute) / 2,3
(DP, filled) / 4,5,6 (Knuth, open). Reference lines: slope-1 and slope-1/2 gray
dashed (translated through anchor points, as in Fig 1c); series named in the legend.
End-segment local slopes printed per series (two-point and three-point fits).
Output: z3_31212/TSD31212_diagonal_loglog.png + .pdf (300 dpi).
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
BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DIAG = os.path.join(BASE, 'z3_31212', 'TSD31212_diagonal.csv')
OUT = os.path.join(BASE, 'z3_31212', 'TSD31212_diagonal_loglog')

C_BIN = '#1f4e79'    # binary: dark blue
C_TER = '#c0392b'    # ternary: brick red
C_GUIDE = '#8a8a8a'


def load_diagonal():
    """Aggregate diagonal-family data from multiple sources (dp_strips W=L rows,
    diagonal.csv, Knuth diagonal points, L=1 brute anchor).

    Returns (binary, ternary): elements (L, lnOmega, N_F, engine, mid, cut).
    """
    b, t = [], []
    # dp_strips: W=L rows (binary)
    with open(os.path.join(BASE, 'z3_31212', 'TSD31212_dp_strips.csv'),
              encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            if int(r['W']) == int(r['L']):
                L = int(r['L'])
                om, oc = int(r['Omega_mid']), int(r['Omega_cut'])
                b.append((L, math.log(om * oc), 3 * L * L, 'dp', om, oc))
    # diagonal.csv (step5 supplement)
    if os.path.exists(DIAG):
        with open(DIAG, encoding='utf-8-sig') as f:
            for r in csv.DictReader(f):
                L = int(r['L'])
                pt = (L, float(r['lnOmega']), float(r['N_F']), r['engine'],
                      int(r['Omega_mid']), int(r['Omega_cut']))
                (b if r['alphabet'] == '{1,2}' else t).append(pt)
    # ternary Knuth diagonal points (probe/production; L=1 brute anchor)
    t.append((1, math.log(243), 3.0, 'brute', 27, 9))
    for fn, tag in (('TSD31212_knuth_diagonal.csv', 'knuth'),):
        fp = os.path.join(BASE, 'z3_31212', fn)
        if os.path.exists(fp):
            with open(fp, encoding='utf-8-sig') as f:
                for r in csv.DictReader(f):
                    if not r.get('Omega'):
                        continue
                    L = int(r['L'])
                    om = float(r['Omega'])
                    t.append((L, math.log(om), 3 * L * L, tag, 0, 0))
    dedup = lambda pts: sorted({p[0]: p for p in pts}.values())
    return dedup(b), dedup(t)


def load_strips():
    """Narrow-strip counting points (fixed-W chains): binary = all dp_strips;
    ternary = DP (W=2 full series + W=3 L=2,4) + Knuth production
    (W=3 L=5,7,8,10 / W=4 L=3,5,7 / W=5 L=2,3; calibration/transpose rows and
    W >= 7 excluded).

    Returns (bin_strips, ter_strips): {W: [(N_F, lnOmega, engine), ...]}
    """
    bs, ts = {}, {}
    fp = os.path.join(BASE, 'z3_31212', 'TSD31212_dp_strips.csv')
    with open(fp, encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            W, L = int(r['W']), int(r['L'])
            bs.setdefault(W, []).append((3 * W * L, float(r['lnOmega']), 'dp'))
    fp = os.path.join(BASE, 'z3_31212', 'TSD31212_dp_ternary.csv')
    with open(fp, encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            W, L = int(r['W']), int(r['L'])
            ts.setdefault(W, []).append((3 * W * L, float(r['lnOmega_mid3'])
                                         + math.log(int(r['Omega_cut_ternary'])),
                                         'dp'))
    fp = os.path.join(BASE, 'z3_31212', 'TSD31212_knuth_ternary.csv')
    with open(fp, encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            if r.get('note') == 'calibration' or not r.get('Omega_est'):
                continue
            W, L = int(r['W']), int(r['L'])
            if W <= 2 or W > 5:
                continue
            if any(abs(x[0] - 3 * W * L) < 1e-9 for x in ts.get(W, [])):
                continue
            ts.setdefault(W, []).append((3 * W * L, math.log(float(r['Omega_est'])),
                                         'knuth'))
    for d in (bs, ts):
        for W in d:
            d[W].sort()
    return bs, ts


def local_slopes(pts):
    """End-segment local log-log slopes: adjacent two-point
    (slope = d ln(lnOmega)/d ln N_F) and last-two-point fit."""
    x = np.array([p[2] for p in pts])
    y = np.array([p[1] for p in pts])
    two = [(math.log(y[i + 1]) - math.log(y[i]))
           / (math.log(x[i + 1]) - math.log(x[i])) for i in range(len(x) - 1)]
    sl = np.polyfit(np.log(x[-2:]), np.log(y[-2:]), 1)[0]
    return two, sl


def main():
    b, t = load_diagonal()
    # binary L=1 anchor (brute full orbit 32; dp_strips has no W=L=1 row) + L=5
    # (from the step5 run; its CSV is written only after the binary loop ends, so
    # patched here; the CSV takes precedence once step5 output exists)
    if not any(p[0] == 1 for p in b):
        b.append((1, math.log(32), 3.0, 'brute', 8, 4))
    if not any(p[0] == 5 for p in b):
        b.append((5, math.log(6260832), 75.0, 'dp', 43478, 144))
    b.sort()
    print('binary diagonal:', [(p[0], f'{p[1]:.3f}') for p in b])
    print('ternary diagonal:', [(p[0], f'{p[1]:.3f}') for p in t])
    tb, sb = local_slopes(b)
    tt, st = local_slopes(t)
    print(f'binary two-pt slopes: {[f"{s:.3f}" for s in tb]}  tail-2pt: {sb:.3f}')
    print(f'ternary two-pt slopes: {[f"{s:.3f}" for s in tt]}  tail-2pt: {st:.3f}')
    # three-point fit (last 3 points) is more stable
    for name, pts in (('binary', b), ('ternary', t)):
        x = np.log([p[2] for p in pts[-3:]])
        y = np.log([p[1] for p in pts[-3:]])
        print(f'{name} tail-3 fit slope: {np.polyfit(x, y, 1)[0]:.3f}')

    fig, ax = plt.subplots(figsize=(4.6, 3.6), dpi=300)
    # narrow strips (fixed-W chains, thin lines, small points): binary W's fan out
    # (coefficient ~ 1/W, perimeter law); ternary W's stack (constant-coefficient
    # band, area law)
    bs, ts = load_strips()
    for W in sorted(bs):
        pts = bs[W]
        ax.plot([p[0] for p in pts], [p[1] for p in pts], '-', lw=0.6,
                color=C_BIN, alpha=0.45, zorder=2)
        ax.plot([p[0] for p in pts], [p[1] for p in pts], 'o', ms=2.0,
                color=C_BIN, alpha=0.75, linewidth=0, zorder=2)
    ax.plot([], [], 'o', ms=2.6, color=C_BIN, alpha=0.85,
            label='binary strips $W{=}2$-$4$')
    for W in sorted(ts):
        pts = ts[W]
        ax.plot([p[0] for p in pts], [p[1] for p in pts], '-', lw=0.6,
                color=C_TER, alpha=0.45, zorder=2)
        pe = [p for p in pts if p[2] == 'dp']
        pk = [p for p in pts if p[2] == 'knuth']
        if pe:
            ax.plot([p[0] for p in pe], [p[1] for p in pe], 'o', ms=2.4,
                    color=C_TER, linewidth=0, zorder=2.5)
        if pk:
            ax.plot([p[0] for p in pk], [p[1] for p in pk], 'o', ms=3.0,
                    mfc='none', mec=C_TER, mew=0.7, linestyle='none', zorder=2.5)
    ax.plot([], [], 'o', ms=2.6, color=C_TER,
            label='ternary strips $W{=}2$-$5$')
    # series: binary (all filled, DP)
    ax.plot([p[2] for p in b], [p[1] for p in b], 'o', ms=4.0, color=C_BIN,
            label='3.12.12 binary $\\{1,2\\}$', zorder=4)
    # ternary: L=1,2,3 filled (DP/brute); L>=4 open (Knuth)
    t_exact = [p for p in t if p[3] in ('dp', 'brute')]
    t_knuth = [p for p in t if p[3] not in ('dp', 'brute')]
    ax.plot([p[2] for p in t_exact], [p[1] for p in t_exact], 'o', ms=4.0,
            color=C_TER, label='3.12.12 ternary $\\{1,2,3\\}$', zorder=4)
    if t_knuth:
        ax.plot([p[2] for p in t_knuth], [p[1] for p in t_knuth], 'o', ms=4.6,
                mfc='none', mec=C_TER, mew=1.0, linestyle='none', zorder=5)
    # reference lines: slope-1 through the ternary tail point; slope-1/2 through the
    # binary tail point (Fig 1c-style translation)
    c_ext = t[-1][1] / t[-1][2]
    xx = np.array([3.0, 300.0])
    ax.plot(xx, c_ext * xx, ls=(0, (5, 2.2)), lw=0.8, color=C_GUIDE, zorder=1.2,
            label='slope 1: area law')
    c_sub = b[-1][1] / math.sqrt(b[-1][2])
    ax.plot(xx, c_sub * np.sqrt(xx), ls=(0, (1.2, 1.6)), lw=0.8, color=C_GUIDE,
            zorder=1.2, label='slope $\\frac{1}{2}$: perimeter law')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('$N_F$')
    ax.set_ylabel(r'$\ln\Omega$', labelpad=1.0)
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=6.2, loc='lower right', framealpha=0.9,
              labelspacing=0.3, handlelength=1.6)
    fig.tight_layout(pad=0.4)
    fig.savefig(OUT + '.png', dpi=300)
    fig.savefig(OUT + '.pdf')
    print(f'written {OUT}.png/.pdf')


if __name__ == '__main__':
    main()
