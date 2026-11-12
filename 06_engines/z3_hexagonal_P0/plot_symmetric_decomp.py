# -*- coding: utf-8 -*-
"""Decomposition figure for the symmetric form: lnOmega =
a*Lx*Ly + b*(Lx+Ly) + A*Lx*Ly*q^Ly decomposed at the lambda(Ly) limit into
three components: area a*Ly + perimeter b + modulation A'*Ly*q^Ly.
Left: arithmetic (full-range residual <= 0.005); right: binary (even branch
<= 3e-4, odd branch 2e-2).
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = REPO_ROOT / "04_scripts_L2"
sys.path.insert(0, str(SCRIPTS))
from l1_style import plt, savefig, DOUBLE_COL  # noqa: E402

C_DATA = {'arith': '#c0392b', 'binary': '#2166ac'}
C_AREA = '#999999'
C_PERIM = '#c8a24b'
C_MOD = '#5f8a5f'

fig = plt.figure(figsize=(DOUBLE_COL, 2.9))
ax1 = fig.add_axes([0.08, 0.16, 0.38, 0.76])
ax2 = fig.add_axes([0.58, 0.16, 0.38, 0.76])

# ---- arithmetic ----
ws = np.linspace(1.6, 8.4, 300)
a, b, Ap, q = 0.5436, 0.289, 1.607, 0.270
wd = np.arange(2, 9)
lam = np.array([1.60944, 2.01215, 2.50211, 3.01488, 3.54934, 4.09316, 4.643])
ax1.plot(wd, lam, 'o', ms=5, mfc=C_DATA['arith'], mec=C_DATA['arith'],
         mew=0, zorder=4, label='TM $\\lambda(L_y)$')
ax1.plot(ws, a*ws, '-', color=C_AREA, lw=1.4, zorder=2,
         label='$a\\,L_y$ (area)')
ax1.plot(ws, a*ws + b, '--', color=C_PERIM, lw=1.4, zorder=2,
         label='$+\\,b$ (perimeter)')
ax1.plot(ws, a*ws + b + Ap*ws*q**ws, '-', color=C_MOD, lw=1.6, zorder=3,
         label='$+\\,A\' L_y q^{L_y}$ (modulation)')
ax1.set_xlim(1.4, 8.6)
ax1.set_ylim(0.9, 5.2)
ax1.set_xticks([2, 4, 6, 8])
ax1.set_xlabel('$L_y$', fontsize=7.5)
ax1.set_ylabel(r'$\lambda(L_y)$', fontsize=7.5)
ax1.legend(fontsize=6.0, handlelength=1.3, loc='upper left',
           borderaxespad=0.3, labelspacing=0.24)
ax1.text(0.97, 0.05, '{1,2,3}  rms = 0.004', transform=ax1.transAxes,
         fontsize=6.4, va='bottom', ha='right', color='#333333')
ax1.text(0.03, 0.035, 'a', transform=ax1.transAxes, fontsize=8,
         fontweight='bold', va='bottom')

# ---- binary (even / odd branches) ----
a2, b2, q2 = 0.2094, 0.268, 0.572
wd2 = np.arange(2, 10)
lamb = {2: 0.89588, 3: 0.96657, 4: 1.24245, 5: 1.32452, 6: 1.59197,
        7: 1.73615, 8: 1.99572, 9: 2.17424}
wd_e = np.array([2, 4, 6, 8]); ye = np.array([lamb[w] for w in wd_e])
wd_o = np.array([3, 5, 7, 9]); yo = np.array([lamb[w] for w in wd_o])
wsm = np.linspace(1.6, 9.4, 300)
ax2.plot(wsm, a2*wsm, '-', color=C_AREA, lw=1.4, zorder=2)
ax2.plot(wsm, a2*wsm + b2, '--', color=C_PERIM, lw=1.4, zorder=2)
Ae, Ao = 0.320, 0.112
ax2.plot(wsm, a2*wsm + b2 + Ae*wsm*q2**wsm, '-', color=C_MOD, lw=1.6,
         zorder=3, label='even-$L_y$ modulation')
ax2.plot(wsm, a2*wsm + b2 + Ao*wsm*q2**wsm, '--', color=C_MOD, lw=1.2,
         zorder=2, alpha=0.7, label='odd-$L_y$ modulation')
ax2.plot(wd_e, ye, 's', ms=4.6, mfc='white', mec=C_DATA['binary'], mew=0.9,
         zorder=4, label='TM $\\lambda(L_y)$ (even)')
ax2.plot(wd_o, yo, 'o', ms=4.6, mfc=C_DATA['binary'], mec=C_DATA['binary'],
         mew=0, zorder=4, label='(odd)')
ax2.set_xlim(1.4, 9.6)
ax2.set_ylim(0.35, 2.5)
ax2.set_xticks([2, 4, 6, 8])
ax2.set_xlabel('$L_y$', fontsize=7.5)
ax2.legend(fontsize=5.6, handlelength=1.3, loc='upper left',
           borderaxespad=0.3, labelspacing=0.20)
ax2.text(0.03, 0.035, 'b', transform=ax2.transAxes, fontsize=8,
         fontweight='bold', va='bottom')
ax2.text(0.97, 0.05, '{1,2}: even rms $\\leq$ 3e-4', transform=ax2.transAxes,
         fontsize=6.4, va='bottom', ha='right', color='#333333')

for ax in (ax1, ax2):
    ax.tick_params(labelsize=7, length=2.0, width=0.8, pad=1.8)

savefig(fig, str(HERE / "plot_symmetric_decomp"))
print('saved', flush=True)
