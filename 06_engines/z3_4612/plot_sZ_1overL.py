# -*- coding: utf-8 -*-
"""4.6.12 (three-value alphabet) Knuth-count extrapolation of s_Z: data + 1/L fits.

Data (full-orbit calibre; RTH4612_knuth_estimate_series.csv;
SI S8.2 / Table S5):
  L=1: Omega = 2349            exact (brute force)      s = ln(Omega)/6
  L=2: Omega = 29 723 517      exact (parametric DFS)   s = ln(Omega)/24
  L=3: Omega = 6.56623e12      Knuth 160k  -> s = 0.5465 +/- 0.0005 (SI-quoted)
  L=4: Omega = 8.351e19        Knuth 2M server -> s = 0.4778 +/- 0.0001
N_F = 6 L^2 (12-gon faces on the L x L torus).

Fits: s_Z(L) = s_inf + a/L  (1/L family; locked after L=4 landed inside the
frameworks' prediction window 0.445-0.496, see the constraint-matrix-layer
results doc):
  A: fit on L=1..3  ->  s_inf = 0.163, predicts s(4) = 0.445 (documented window
     lower end 0.445 -> assertion below)
  B: fit on L=2..4  ->  s_inf = 0.232

Documented windows:
  1/L family       s_Z(inf) ~ 0.16-0.24  (RTH4612_constraint_matrix_layer_results_2026-10-01.md)
  four frameworks  s_Z(inf) ~ 0.16-0.48  (SI S8.2 v1.7; decay not converged, candidate)

Output: _temp/4612_extrapolation/RTH4612_sZ_1overL_extrapolation.png/.pdf
"""
import io
import math
import os
import sys
from pathlib import Path

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / '04_scripts_L2'))
try:
    import l1_style  # noqa: F401  (rcParams applied at import time)
except Exception as exc:
    print('l1_style not applied:', exc)

# ---------------------------------------------------------------- data
# (L, Omega, per-hole error on s, kind)
DATA = [
    (1, 2349.0,       None,   'exact'),
    (2, 29723517.0,   None,   'exact'),
    (3, 6.56623e12,   0.0005, 'knuth'),
    (4, 8.351e19,     0.0001, 'knuth'),
]

Ls = np.array([d[0] for d in DATA], dtype=float)
NF = 6.0 * Ls ** 2
lnOm = np.log(np.array([d[1] for d in DATA], dtype=float))
sZ = lnOm / NF
err = [d[2] for d in DATA]

# canonical SI S8.2 values: 1.29 / 0.717 / 0.5465 / 0.4778
assert round(sZ[0], 2) == 1.29
assert round(sZ[1], 3) == 0.717
assert round(sZ[2], 4) == 0.5465
assert round(sZ[3], 4) == 0.4778

# ---------------------------------------------------------------- fits
def fit_1_over_L(mask):
    x = 1.0 / Ls[mask]
    y = sZ[mask]
    slope, intercept = np.polyfit(x, y, 1)
    return intercept, slope

sinfA, aA = fit_1_over_L(Ls <= 3)
sinfB, aB = fit_1_over_L(Ls >= 2)
pred4 = sinfA + aA / 4.0

print('fit A (L=1..3): s_inf=%.4f  a=%.4f  ->  predicted s(L=4)=%.4f'
      '  (documented prediction-window lower end 0.445)' % (sinfA, aA, pred4))
print('fit B (L=2..4): s_inf=%.4f  a=%.4f' % (sinfB, aB))
assert abs(pred4 - 0.445) < 0.001, pred4
assert 0.16 < sinfA < 0.17 and 0.22 < sinfB < 0.25, (sinfA, sinfB)

# ---------------------------------------------------------------- plot
fig, ax = plt.subplots(figsize=(4.6, 3.4))

# documented extrapolation windows, drawn at the 1/L -> 0 intercept zone
XB = 0.13
ax.fill_between([0, XB], 0.16, 0.48, color='#d9d9d9', alpha=0.55,
                linewidth=0, zorder=0)
ax.fill_between([0, XB], 0.16, 0.24, color='#969696', alpha=0.45,
                linewidth=0, zorder=0)
ax.text(XB + 0.015, 0.310, 'four-framework window 0.16-0.48 (SI S8.2)',
        fontsize=5.8, color='#555555', va='center')
ax.text(XB + 0.015, 0.172, '1/L family 0.16-0.24',
        fontsize=5.8, color='#555555', va='center')

# reference carriers (context only: different alphabet/protocol, not ranked)
ax.axhline(0.4504, ls=':', lw=0.9, color='#8a8a8a', zorder=1,
           label='honeycomb 0.45 (different alphabet/protocol)')
ax.axhline(0.031, ls=':', lw=0.9, color='#8a8a8a', zorder=1,
           label='3.4.6.4 leak 0.031')

# 1/L fits, each drawn over the window it was fitted on
xfA = np.linspace(0.0, 1.0, 100)
xfB = np.linspace(0.0, 0.5, 100)
ax.plot(xfA, sinfA + aA * xfA, '--', color='#b2182b', lw=1.1, zorder=2,
        label='fit L=1-3:  $s_\\mathbb{Z}=%.3f+%.2f/L$' % (sinfA, aA))
ax.plot(xfB, sinfB + aB * xfB, '-.', color='#35978f', lw=1.1, zorder=2,
        label='fit L=2-4:  $s_\\mathbb{Z}=%.3f+%.2f/L$' % (sinfB, aB))
for s_inf, a, c in ((sinfA, aA, '#b2182b'), (sinfB, aB, '#35978f')):
    ax.plot([0], [s_inf], 's', ms=4, color=c, zorder=4)
    ax.annotate('%.3f' % s_inf, (0, s_inf), xytext=(6, -2.5),
                textcoords='offset points', fontsize=6, color=c)

# data: exact filled, Knuth open with (invisible) error bars
x_all = 1.0 / Ls
idx_ex = [i for i, d in enumerate(DATA) if d[3] == 'exact']
idx_kn = [i for i, d in enumerate(DATA) if d[3] == 'knuth']
ax.plot(x_all, sZ, '-', color='#9ecae1', lw=0.8, zorder=2)
ax.plot(x_all[idx_ex], sZ[idx_ex], 'o', ms=5, color='#2166ac',
        markeredgecolor='k', markeredgewidth=0.6, zorder=3,
        label='exact (L=1, 2)')
ax.errorbar(x_all[idx_kn], sZ[idx_kn], yerr=[err[i] for i in idx_kn],
            fmt='o', ms=5, markerfacecolor='white', markeredgecolor='#2166ac',
            markeredgewidth=1.0, elinewidth=0.9, capsize=1.5, zorder=3,
            label='Knuth est. (L=3, 4; error bars < marker)')

for x, y, L in zip(x_all, sZ, Ls):
    ax.annotate('L=%d' % L, (x, y), xytext=(0, -11),
                textcoords='offset points', ha='center', fontsize=6,
                color='#555555')

ax.set_xlim(-0.02, 1.06)
ax.set_ylim(0, 1.45)
ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
ax.set_xlabel('$1/L$')
ax.set_ylabel('$s_\\mathbb{Z}=\\ln\\Omega_\\mathbb{Z}/N_F$')
ax.legend(fontsize=6, loc='upper left', frameon=False)

ax.text(0.98, 0.055,
        '4.6.12, three-value alphabet {1,2,3}, $L\\times L$ torus, $N_F=6L^2$\n'
        'decay not converged: thick-entropy candidate; exact $L\\geq 5$ registered (open)',
        transform=ax.transAxes, fontsize=5.5, ha='right', va='bottom',
        color='#555555')

fig.tight_layout()
outdir = '_temp/4612_extrapolation'
os.makedirs(outdir, exist_ok=True)
out = os.path.join(outdir, 'RTH4612_sZ_1overL_extrapolation.png')
fig.savefig(out, dpi=300)
fig.savefig(out.replace('.png', '.pdf'))
print('saved', out)
print('s_Z series :', ' '.join('%.4f' % v for v in sZ))
print('local exponents d ln s / d ln L:',
      ' '.join('%.2f' % v for v in np.diff(np.log(sZ)) / np.diff(np.log(Ls))))
