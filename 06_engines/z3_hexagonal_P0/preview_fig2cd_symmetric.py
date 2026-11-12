# -*- coding: utf-8 -*-
"""Final preview v2 of fig2c/d (canonical style + resonant/non-resonant
split + mixed model).

Left column, three stacked panels: lnOmega vs Lx (linear 0..150). One series
per Ly:
  resonant (Lx % Ly == 0, filled) -- asymptotic dashed line lambda(Ly)*Lx
  (winding fully open, hugging the line);
  non-resonant (open) -- sublaw dotted line ln k*(Lx+Ly+1) (winding closed,
  perimeter law exactly).
Right column: lambda(Ly)/Ly vs Ly, mixed global model s + c/Ly + A*q^Ly.
"""
import sys
from pathlib import Path

import csv
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = REPO_ROOT / "04_scripts_L2"
ROOT = HERE
sys.path.insert(0, str(SCRIPTS))
from l1_style import plt, savefig, DOUBLE_COL  # noqa: E402

ALPHA_C = {'binary': '#2166ac', 'arithmetic': '#c0392b',
           'nonarith': '#e6811e'}
A_TAG = {'binary': '$\\{1,2\\}$ (binary)',
         'arithmetic': '$\\{1,2,3\\}$ (arith.)',
         'nonarith': '$\\{1,2,4\\}$ (non-arith.)'}
W_COL = {'Ly2': '#762a83', 'Ly3': '#2166ac', 'Ly5': '#5aae61'}
W_MK = {'Ly2': 'o', 'Ly3': 's', 'Ly5': '^'}
W_LAB = {'Ly2': '$L_y=2$', 'Ly3': '$L_y=3$', 'Ly5': '$L_y=5$'}
LNK = {'binary': np.log(2), 'nonarith': np.log(3), 'arithmetic': np.log(3)}

TM_LAM = {'binary': {2: 0.89588, 3: 0.96657, 5: 1.32452},
          'nonarith': {2: 1.24245, 3: 1.38629, 5: 1.58305},
          'arithmetic': {2: 1.60944, 3: 2.01215, 5: 3.01488}}
MIXPAR = {'binary': (0.2094, 0.268, 0.320, 0.112, 0.572),
          'nonarith': (0.2415, 0.081, 1.029, 0.574),
          'arithmetic': (0.5436, 0.289, 1.607, 0.270)}

long = {}
for lf in (ROOT / 'TSDHC_long_trace.csv',
           ROOT / 'TSDHC_long_trace_arith5.csv'):
    if not lf.exists():
        continue
    with open(lf, encoding='utf-8-sig') as f:
        for row in csv.reader(f):
            if len(row) < 7 or not row[2].isdigit():
                continue
            key = (row[0].lstrip('\ufeff'), int(row[2]))
            long.setdefault(key, ([], []))
            long[key][0].append(int(row[3]))
            long[key][1].append(float(row[5]))

df3 = pd.read_csv(ROOT / 'P0_PM_torus_trace_shape_classes_Ly2Ly3.csv')
df5 = pd.read_csv(ROOT / 'P0_PM_torus_trace_shape_classes_Ly5.csv')


def series_split(gname, ly):
    d = df3[(df3.alphabet == gname) & (df3.Ly == ly)].sort_values('Lx')
    lx = d.Lx.values.astype(float)
    y = d.lnOmega.values.astype(float)
    if ly == 5:
        d5 = df5[(df5.alphabet == gname) & (df5.Ly == 5)].sort_values('Lx')
        lx = np.concatenate([lx, d5.Lx.values.astype(float)])
        y = np.concatenate([y, d5.lnOmega.values.astype(float)])
    key = (gname, ly)
    if key in long:
        lx2, y2 = long[key]
        o = np.argsort(lx2)
        lx = np.concatenate([lx, np.array(lx2)[o]])
        y = np.concatenate([y, np.array(y2)[o]])
    o = np.argsort(lx)
    return lx[o], y[o]


fig = plt.figure(figsize=(DOUBLE_COL, 6.9))
LX0, LW = 0.075, 0.53
GH = 0.255
YTOP = 0.965
SPANS = {'arithmetic': (0, 320.0), 'nonarith': (0, 225.0),
         'binary': (0, 215.0)}
YTK = {'arithmetic': [0, 100, 200, 300], 'nonarith': [0, 100, 200],
       'binary': [0, 100, 200]}

axes = {}
for gname in ('arithmetic', 'nonarith', 'binary'):
    ax = fig.add_axes([LX0, YTOP - GH, LW, GH])
    axes[gname] = ax
    YTOP -= GH + 0.042
    lnk = LNK[gname]
    for ly in (2, 3, 5):
        if ly not in TM_LAM[gname]:
            continue
        lx, y = series_split(gname, ly)
        c = W_COL[f'Ly{ly}']
        res = (lx % ly) == 0
        lam = TM_LAM[gname][ly]
        xr = np.array([0, 152])
        ax.plot(xr, lam * xr, '--', color=c, lw=1.0, zorder=2, alpha=0.8)
        ax.plot(xr, lnk * (xr + ly + 1), ':', color=c, lw=0.9, zorder=2,
                alpha=0.55)
        m = res
        ax.plot(lx[m], y[m], W_MK[f'Ly{ly}'], ms=3.0, mfc=c, mec=c,
                mew=0.0, zorder=3,
                label=W_LAB[f'Ly{ly}'] if gname == 'arithmetic' else None)
        ax.plot(lx[~m], y[~m], W_MK[f'Ly{ly}'], ms=3.0, mfc='white',
                mec=c, mew=0.7, zorder=3)
    lo, hi = SPANS[gname]
    ax.set_xlim(0, 152)
    ax.set_ylim(lo, hi)
    ax.set_xticks([0, 25, 50, 75, 100, 125, 150])
    ax.set_yticks(YTK[gname])
    ax.tick_params(labelsize=7, length=2.0, width=0.8, pad=1.8)
    if gname != 'binary':
        ax.tick_params(labelbottom=False, bottom=False)
    ax.text(0.965, 0.055, A_TAG[gname], transform=ax.transAxes,
            fontsize=7.5, ha='right', va='bottom', color='#333333')
    ax.text(0.03, 0.93, 'resonant (solid) / non-resonant (open)',
            transform=ax.transAxes, fontsize=5.4, va='top', color='#555555')
axes['arithmetic'].legend(fontsize=6.4, handlelength=1.0, ncol=3,
                          loc='upper left', borderaxespad=0.3,
                          labelspacing=0.24, handletextpad=0.4)
axes['binary'].set_xlabel(r'$L_x$', fontsize=7.5)
fig.text(0.018, 0.55, r'$\ln\Omega$', fontsize=7.5, rotation=90,
         ha='center', va='center')

axd = fig.add_axes([0.70, 0.42, 0.27, 0.42])
for gname in ('binary', 'nonarith', 'arithmetic'):
    c = ALPHA_C[gname]
    wss = sorted(TM_LAM[gname])
    axd.plot(wss, [TM_LAM[gname][w] / w for w in wss], 'o', ms=4.6,
             mfc=c, mec=c, mew=0.0, zorder=3,
             label='$\\{1,2\\}$' if gname == 'binary'
             else '$\\{1,2,3\\}$' if gname == 'arithmetic'
             else '$\\{1,2,4\\}$')
    xr = np.linspace(1.6, 10.4, 200)
    if gname == 'binary':
        s, cc, Ae, Ao, q = MIXPAR[gname]
        for A, lsty in ((Ae, '-'), (Ao, (0, (3, 1.5)))):
            axd.plot(xr, s + cc / xr + A * q ** xr, linestyle=lsty,
                     color=c, lw=0.9, zorder=2, alpha=0.7)
    else:
        s, cc, A, q = MIXPAR[gname]
        axd.plot(xr, s + cc / xr + A * q ** xr, '-', color=c, lw=0.9,
                 zorder=2, alpha=0.85)
axd.set_xlim(1.6, 10.4)
axd.set_ylim(0.1, 0.9)
axd.set_xticks([2, 4, 6, 8, 10])
axd.set_xlabel('$L_y$', fontsize=7.5)
axd.set_ylabel(r'$\lambda(L_y)/L_y$', fontsize=7.5, labelpad=2)
axd.tick_params(labelsize=7, length=2.0, width=0.8, pad=1.8)
axd.legend(fontsize=6.2, handlelength=1.0, loc='upper right',
           borderaxespad=0.3)
axd.text(0.05, 0.05, '$s + c/L_y + A\\,q^{L_y}$ (mixed global fit)',
         transform=axd.transAxes, fontsize=5.6, color='#555555')

savefig(fig, str(HERE / 'preview_fig2cd_symmetric'))
print('saved', flush=True)
