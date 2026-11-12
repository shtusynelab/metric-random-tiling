# -*- coding: utf-8 -*-
"""Check figure for rho(W) under the cyclic-symmetric form:
residual of lambda(W) - [a*W + b] against the two-point line fixed by
(Ly=2, Ly=3), semi-log.
The symmetric form (no modulation term) predicts identically zero residual;
the measured deviation = the modulation term, exponential in shape.
Arithmetic/binary: positive deviation (excess channels open); non-arith:
negative deviation (channels under-opened).
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = REPO_ROOT / "04_scripts_L2"
sys.path.insert(0, str(SCRIPTS))
from l1_style import plt, savefig, DOUBLE_COL  # noqa: E402

ALPHA_C = {'binary': '#2166ac', 'arithmetic': '#c0392b',
           'nonarith': '#e6811e'}

fig = plt.figure(figsize=(5.4, 2.8))
ax = fig.add_axes([0.13, 0.16, 0.84, 0.78])

# per system: line fixed by the two points Ly=2,3; residual = lambda(W) - (a*W + b)
lam = {
    'arithmetic': {2: 1.60944, 3: 2.01215, 4: 2.50211, 5: 3.01488,
                   6: 3.54934, 7: 4.09316, 8: 4.643},
    'binary': {2: 0.89588, 3: 0.96657, 4: 1.24245, 5: 1.32452, 6: 1.59197,
               7: 1.73615, 8: 1.99572, 9: 2.17424},
    'nonarith': {2: 1.24245, 3: 1.38629, 4: 1.50444, 5: 1.58305, 6: 1.78945,
                 7: 1.89771},
}
MK = {'binary': 'o', 'nonarith': '^', 'arithmetic': 's'}
LAB = {'binary': '$\\{1,2\\}$', 'nonarith': '$\\{1,2,4\\}$',
       'arithmetic': '$\\{1,2,3\\}$'}
for name in ('arithmetic', 'binary', 'nonarith'):
    d = lam[name]
    sl = d[3] - d[2]
    ic = d[2] - sl * 2
    ws = sorted(w for w in d if w >= 4)
    dev = [d[w] - (sl * w + ic) for w in ws]
    ax.plot(ws, dev, MK[name], ms=4.6, mfc=ALPHA_C[name],
            mec=ALPHA_C[name], mew=0.0, zorder=3, label=LAB[name])
    xr = np.array([min(ws), 9.4])
    if name == 'arithmetic':
        A = dev[-1] / 0.27 ** 8
        ax.plot(xr, A * 0.27 ** xr, color=ALPHA_C[name], lw=0.9,
                ls=(0, (3, 1.6)), zorder=2, alpha=0.8)
    elif name == 'nonarith':
        A = abs(dev[-1]) / 0.31 ** 7
        ax.plot(xr, -A * 0.31 ** xr, color=ALPHA_C[name], lw=0.9,
                ls=(0, (3, 1.6)), zorder=2, alpha=0.8)
ax.axhline(0.0, color='#9a9a9a', lw=0.8, zorder=1)
ax.text(9.15, 0.06, '$\\{1,2,3\\}$: +modulation', fontsize=6.0,
        color=ALPHA_C['arithmetic'], ha='right')
ax.text(9.15, -0.075, '$\\{1,2,4\\}$: $-$modulation (under-opened)',
        fontsize=6.0, color=ALPHA_C['nonarith'], ha='right')
ax.set_yscale('symlog', linthresh=0.05)
ax.set_xlim(3.4, 9.6)
ax.set_ylim(-0.9, 0.9)
ax.set_yticks([-0.6, -0.3, 0, 0.3, 0.6])
ax.set_xlabel('$L_y$', fontsize=7.5)
ax.set_ylabel(r'$\lambda(L_y) - [a\,L_y + b]$  (two-point line)',
              fontsize=7.2)
ax.tick_params(labelsize=7, length=2.0, width=0.8, pad=1.8)
ax.legend(fontsize=6.4, handlelength=1.0, loc='upper left',
          borderaxespad=0.3)

savefig(fig, str(HERE / "plot_symm_linear_dev"))
print('saved', flush=True)
