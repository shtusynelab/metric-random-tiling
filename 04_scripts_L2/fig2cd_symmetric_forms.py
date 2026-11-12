# -*- coding: utf-8 -*-
"""Canonical standalone version of Fig 2 panels c/d.

Sizes and format = the fig2 canonical panel_c slot 1:1 (3.819x2.606 in =
panel_c(fig, 0.425, 0.375, 0.53, 0.367) @ 7.205x7.1; internal relative
factors lx0=0.075/lxw=0.515/ay0=0.075/ayh=0.88/right column
[0.715,0.075,0.277,0.88] carried over; font sizes tick 7/A_TAG 8/legend 7/
axis labels 7.5, identical to the original).

c (left column, three tightly stacked strips): honeycomb three-alphabet
torus counts lnOmega vs Lx (x up to 150).
One series per width Ly (Ly = 2/3/5), split by Lx mod Ly into
resonant/non-resonant, each class subsampled at equal spacing
(target ~10 points/class: large-Ly resonant points are scarce, so the two
ratios are tuned separately to keep filled/open visually balanced):
  resonant (filled) -- asymptotic dashed line lnOmega = lambda(Ly)*Lx
  (lambda = TM Perron, winding sector open);
  non-resonant (open) -- sub-law dotted line lnOmega = ln k*(Lx+Ly+1);
  binary/non-arithmetic exact to machine precision (QA assert); arithmetic
  crossover ~10-20 columns, both branches merge onto the asymptote at
  large Lx.
  The fill/open protocol line is labeled once (top strip, below legend).
d (right column): lambda(Ly)/Ly vs Ly for all TM width points + mixed
global model lambda/Ly = s + c/Ly + A'*q^Ly (binary even/odd branches);
endpoint s values 0.544/0.242/0.209.

Data: shape-class CSVs (Lx = 2..20) + TSDHC_long_trace{,_arith5}.csv (long
chains; appended by multiprocessing, headerless with BOM-contaminated rows,
filtered by row[2].isdigit() and stripped of '\ufeff'; lnOmega = row[5]).
nonarith Ly=5 long chains extended to 100 columns (step12b_parallel.py,
server, 10 processes).

Caption (delivered, panel letters c/d):

c, Resonant/non-resonant splitting of the torus counting. ln Omega versus
L_x for the three honeycomb edge alphabets ({1,2,3} arithmetic,
{1,2,4} non-arithmetic, {1,2} binary; L_y = 2, 3, 5). Every series splits
by the transposition phase L_x mod L_y: resonant sizes (L_x = 0 mod L_y;
filled symbols) lie on the asymptotic lines ln Omega = lambda(L_y) L_x,
with lambda(L_y) the transfer-matrix Perron eigenvalue at width L_y, while
non-resonant sizes (open symbols) follow the perimeter law
ln Omega = ln k (L_x + L_y + 1) exactly, to machine precision (rms = 0),
for the binary and non-arithmetic alphabets (k = 2 and 3). In the
arithmetic alphabet the crossover is fast (10-20 columns) and both
branches merge onto the asymptote. The two branches are the winding-sector
off/on states of the transposition-symmetric form
ln Omega = a L_x L_y + b (L_x + L_y) + A L_x L_y (q^{L_x} + q^{L_y})/2,
of which the perimeter law is the a = 0 special case; the modulation term
is positive for the arithmetic and binary alphabets (channel-opening
excess) and negative for the non-arithmetic alphabet (sub-opening
deficit). Two-stage convergence: the apparent slope departs from the
sub-law value ln k and converges to the Perron asymptote - the metric
entropy of the honeycomb lives in the winding degrees of freedom, and the
exact perimeter law at non-resonant sizes is a precise statement of
entropy inaccessibility.
d, Width scaling of the per-column entropy current lambda(L_y)/L_y.
Symbols: transfer-matrix Perron eigenvalues (binary L_y = 2-9,
non-arithmetic 2-7, arithmetic 2-8; the arithmetic L_y = 8 point is the
3.12.12 interlocking check). Lines: mixed global fit
lambda/L_y = s + c/L_y + A' q^{L_y} (binary: even/odd branches,
A' = 0.320/0.112). Asymptotic entropy densities s = 0.544 (arithmetic),
0.242 (non-arithmetic), 0.209 (binary); model band +/-0.01.
"""
import sys
from pathlib import Path

import csv
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import paths_common as PC  # noqa: E402
from l1_style import plt, savefig, DOUBLE_COL  # noqa: E402

ROOT = PC.HC
FIGS = PC.FIGURES

ALPHA_C = {'binary': '#2166ac', 'arithmetic': '#c0392b',
           'nonarith': '#e6811e'}
A_TAG = {'binary': '$\\{1,2\\}$ (binary)',
         'arithmetic': '$\\{1,2,3\\}$ (arith.)',
         'nonarith': '$\\{1,2,4\\}$ (non-arith.)'}
W_COL = {'Ly2': '#762a83', 'Ly3': '#2166ac', 'Ly5': '#5aae61'}
W_MK = {'Ly2': 'o', 'Ly3': 's', 'Ly5': '^'}
W_LAB = {'Ly2': '$L_y=2$', 'Ly3': '$L_y=3$', 'Ly5': '$L_y=5$'}
LNK = {'binary': np.log(2), 'nonarith': np.log(3), 'arithmetic': np.log(3)}

# TM Perron lambda(Ly) full table (after spectral correction; the arithmetic
# W=8 point is the 3.12.12 interlocking check)
TM_LAM = {
    'binary': {2: 0.89588, 3: 0.96657, 4: 1.24245, 5: 1.32452, 6: 1.59197,
               7: 1.73615, 8: 1.99572, 9: 2.17424},
    'nonarith': {2: 1.24245, 3: 1.38629, 4: 1.50444, 5: 1.58305,
                 6: 1.78945, 7: 1.89771},
    'arithmetic': {2: 1.60944, 3: 2.01215, 4: 2.50211, 5: 3.01488,
                   6: 3.54934, 7: 4.09316, 8: 4.643},
}
# mixed global model: binary 5 parameters (s,c,A_even,A_odd,q);
# arith/nonarith 4 parameters (s,c,A,q)
MIXPAR = {'binary': (0.2094, 0.268, 0.320, 0.112, 0.572),
          'nonarith': (0.2415, 0.081, 1.029, 0.574),
          'arithmetic': (0.5436, 0.289, 1.607, 0.270)}
S_END = {'binary': '0.209', 'nonarith': '0.242', 'arithmetic': '0.544'}

# ---- data: shape-class CSVs (Lx<=20) + long-chain CSVs (Lx=21..150) ----
long = {}
for lf in (ROOT / 'TSDHC_long_trace.csv',
           ROOT / 'TSDHC_long_trace_arith5.csv'):
    with open(lf, encoding='utf-8-sig') as f:
        for row in csv.reader(f):
            if len(row) < 7 or not row[2].isdigit():
                continue
            key = (row[0].lstrip('\ufeff'), int(row[2]))
            long.setdefault(key, ([], []))
            long[key][0].append(int(row[3]))
            long[key][1].append(float(row[5]))

df3 = pd.read_csv(ROOT / PC.fname('P0_PM_torus_trace_shape_classes_Ly2Ly3.csv'))
df5 = pd.read_csv(ROOT / PC.fname('P0_PM_torus_trace_shape_classes_Ly5.csv'))

# ---- QA: non-resonant perimeter law bitwise exact (binary k=2 /
#      nonarith k=3, full data) ----
_k_of = {"binary": 2, "nonarith": 3}
for nm, k in _k_of.items():
    lnk = np.log(k)
    n_chk = 0
    for ly, dd in ((2, df3), (3, df3), (5, df5)):
        for _, r in dd[(dd.alphabet == nm) & (dd.Ly == ly)].iterrows():
            if int(r.Lx) % ly == 0:
                continue
            assert abs(float(r.lnOmega)
                       - lnk * (int(r.Lx) + ly + 1)) < 1e-6, \
                f"perimeter law broken: {nm} {int(r.Lx)}x{ly}"
            n_chk += 1
        for x, y in zip(*long.get((nm, ly), ([], []))):
            if x % ly == 0:
                continue
            assert abs(y - lnk * (x + ly + 1)) < 1e-6, \
                f"perimeter law broken: {nm} {x}x{ly}"
            n_chk += 1
    print(f"[QA] perimeter law ln k*(Lx+Ly+1) bitwise exact on {n_chk} "
          f"non-res points ({nm}, k={k}, Ly2+Ly3+Ly5, short+long)",
          flush=True)
# arithmetic non-resonant does not follow the perimeter law (fast crossover):
# print how close the large-Lx points sit to the asymptote
for ly in (2, 3, 5):
    pts = list(zip(*long.get(('arithmetic', ly), ([], []))))
    big = [(x, y, TM_LAM['arithmetic'][ly] * x) for x, y in pts
           if x % ly != 0 and x >= 60]
    if big:
        x, y, asy = big[-1]
        print(f"[QA] arith Ly={ly} non-res @Lx={x}: lnOmega={y:.2f} vs "
              f"asymptote {asy:.2f} (dev {y - asy:+.2f}); perimeter law "
              f"dev {y - LNK['arithmetic'] * (x + ly + 1):+.1f} -> no "
              f"bitwise perimeter law for arithmetic", flush=True)


def series_split(gname, ly, target=10):
    """Full data -> subsample the resonant/non-resonant classes each at
    equal spacing (target points per class).

    Large-Ly resonant points are scarce (Ly=5 has only 1 in 5), so the two
    ratios are tuned separately to keep filled/open visually balanced; the
    QA asserts run on the full data and are unaffected by subsampling."""
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
    lx, y = lx[o], y[o]
    res = (lx % ly) == 0
    keep = []
    for m in (res, ~res):
        idx = np.where(m)[0]
        if len(idx):
            keep.append(idx[::max(1, len(idx) // target)])
    sel = np.sort(np.concatenate(keep))
    return lx[sel], y[sel]


# ---- layout: fig2 panel_c slot 1:1 (3.819x2.606 in, internal factors ----
# ---- carried over)                                                 ----
FW, FH = 0.53 * 7.205, 0.367 * 7.1          # = 3.819 x 2.606 in
fig = plt.figure(figsize=(FW, FH))
LX0, LXW = 0.075, 0.515
AY0, AYH = 0.075, 0.88
GH = AYH / 3
YTOP = AY0 + AYH
SPANS = {'arithmetic': (0, 320.0), 'nonarith': (0, 225.0),
         'binary': (0, 215.0)}
YTK = {'arithmetic': [0, 100, 200, 300], 'nonarith': [0, 100, 200],
       'binary': [0, 100, 200]}
XTK = [0, 25, 50, 75, 100, 125, 150]
XMAX = 152

axes = {}
for gname in ('arithmetic', 'nonarith', 'binary'):
    ax = fig.add_axes([LX0, YTOP - GH, LXW, GH])
    axes[gname] = ax
    YTOP -= GH
    lnk = LNK[gname]
    for ly in (2, 3, 5):
        lx, y = series_split(gname, ly)
        c = W_COL[f'Ly{ly}']
        mk = W_MK[f'Ly{ly}']
        res = (lx % ly) == 0
        lam = TM_LAM[gname][ly]
        xr = np.array([0, XMAX])
        ax.plot(xr, lam * xr, ls=(0, (5, 2)), color=c, lw=1.5, zorder=2,
                alpha=0.85)
        ax.plot(xr, lnk * (xr + ly + 1), ':', color=c, lw=1.0, zorder=2,
                alpha=0.6)
        ax.plot(lx[res], y[res], mk, ms=5.0, mfc=c, mec=c, mew=0.0,
                zorder=3,
                label=W_LAB[f'Ly{ly}'] if gname == 'arithmetic' else None)
        ax.plot(lx[~res], y[~res], mk, ms=4.0, mfc='white', mec=c, mew=0.7,
                zorder=3)
    lo, hi = SPANS[gname]
    ax.set_xlim(0, XMAX)
    ax.set_ylim(lo, hi)
    ax.set_xticks(XTK)
    ax.set_yticks(YTK[gname])
    ax.tick_params(labelsize=7, length=2.0, width=0.8, pad=1.8)
    if gname == 'nonarith':
        # middle strip: draw the x ticks on the upper side only (upward)
        ax.tick_params(labelbottom=False, bottom=False, top=True,
                       direction='out', length=1.8)
    elif gname == 'binary':
        ax.tick_params(bottom=True, labelbottom=True, top=True,
                       direction='out', length=1.8)
    else:
        ax.tick_params(labelbottom=False, bottom=False, top=False)
    ax.text(0.975, 0.08, A_TAG[gname], transform=ax.transAxes,
            fontsize=8, ha='right', va='bottom', color='#333333')
axes['arithmetic'].legend(fontsize=7, handlelength=1.0, ncol=1,
                          loc='upper left', borderaxespad=0.3,
                          labelspacing=0.24, handletextpad=0.4,
                          columnspacing=0.8)
# fill/open protocol line labeled once: two short lines in the free
# upper-left area of the second strip (non-arith); a single full line would
# run into the Ly5 asymptote
axes['nonarith'].text(0.025, 0.965, 'resonant: filled symbols\n'
                      'non-resonant: open symbols',
                      transform=axes['nonarith'].transAxes,
                      fontsize=6.5, va='top', linespacing=1.35,
                      color='#555555')
axes['binary'].set_xlabel(r'$L_x$', fontsize=7.5, labelpad=1.2)
fig.text(0.016, AY0 + 0.44 * AYH, r'$\ln\Omega$', fontsize=7.5,
         rotation=90, ha='center', va='center')
fig.text(0.028, 0.995, 'c', fontsize=8, fontweight='bold',
         ha='right', va='top', color='#1a1a1a')

# ---- right column: lambda(Ly)/Ly all TM width points + mixed global model ----
axd = fig.add_axes([0.715, AY0, 0.277, AYH])
XR = np.linspace(1.6, 10.4, 250)
for gname in ('binary', 'nonarith', 'arithmetic'):
    c = ALPHA_C[gname]
    wss = sorted(TM_LAM[gname])
    axd.plot(wss, [TM_LAM[gname][w] / w for w in wss], 'o', ms=5.5,
             mfc=c, mec=c, mew=0.0, zorder=3,
             label='$\\{1,2\\}$' if gname == 'binary'
             else '$\\{1,2,3\\}$' if gname == 'arithmetic'
             else '$\\{1,2,4\\}$')
    if gname == 'binary':
        s, cc, Ae, Ao, q = MIXPAR[gname]
        for A, lsty in ((Ae, '-'), (Ao, (0, (3, 1.5)))):
            axd.plot(XR, s + cc / XR + A * q ** XR, linestyle=lsty,
                     color=c, lw=1.0, zorder=2, alpha=0.75)
        yend = s + cc / 10.4 + Ao * q ** 10.4
        axd.text(10.35, yend - 0.022, S_END[gname], color=c, fontsize=6.5,
                 ha='right', va='top')
    else:
        s, cc, A, q = MIXPAR[gname]
        axd.plot(XR, s + cc / XR + A * q ** XR, '-', color=c, lw=1.2,
                 zorder=2, alpha=0.85)
        yend = s + cc / 10.4 + A * q ** 10.4
        axd.text(10.35, yend + 0.018, S_END[gname], color=c, fontsize=6.5,
                 ha='right', va='bottom')
axd.set_xlim(1.6, 10.4)
axd.set_ylim(0.1, 0.9)
axd.set_xticks([2, 4, 6, 8, 10])
axd.set_yticks([0.2, 0.4, 0.6, 0.8])
axd.set_xlabel('$L_y$', fontsize=7.5, labelpad=1.2)
axd.set_ylabel(r'$\lambda/L_y$', fontsize=7.5, labelpad=3.5)
axd.tick_params(labelsize=7, length=2.0, width=0.8, pad=1.8)
axd.legend(fontsize=7, handlelength=1.0, ncol=1, loc='upper right',
           borderaxespad=0.3, labelspacing=0.24, handletextpad=0.4)
axd.text(0.05, 0.05, 'mixed global fit', transform=axd.transAxes,
         fontsize=6.5, color='#555555')
fig.text(0.715 - 0.012, 0.995, 'd', fontsize=8, fontweight='bold',
         ha='right', va='top', color='#1a1a1a')

savefig(fig, str(FIGS / 'fig2cd_symmetric_forms'))
print('saved -> figs/fig2cd_symmetric_forms.{png,pdf}', flush=True)
