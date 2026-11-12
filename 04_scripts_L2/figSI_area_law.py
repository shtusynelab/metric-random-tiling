# -*- coding: utf-8 -*-
"""SI asset figure: 3.4.6.4 area law and slope crossover (two panels, single column).
a) s*L vs L, eight points (U-shaped transient) + registered asymptotic band
   + L=9 hit;
b) local log-log slope dln(lnOmega)/dln(n_F) vs L (chord slope, computed
   from the CSV): rises monotonically from 0.44 to 0.76 -> asymptote 1
   (area term bL^2 dominates); at small L the low-order terms (constant +
   perimeter) dominate with slope ~ 1/2, visualising the crossover.
Output: <FIGURES>/S6_area_law_3464.{png,pdf}
"""
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import paths_common as PC  # noqa: E402

FIGS = PC.FIGURES
from l1_style import plt, savefig, SINGLE_COL  # noqa: E402
from fig1_model_classification import panel_area_law_si  # noqa: E402

TR = PC.engine("z4_3464") / PC.fname("RTH_torus_area_law_8pts.csv")


def panel_local_slope(ax):
    tr = pd.read_csv(TR)
    d = tr[tr.type == 'count'].sort_values('L')
    L = d.L.values.astype(float)
    lnO = d.lnOmega.values.astype(float)
    # chord slope dln(lnOmega)/dln(n_F); n_F = 6L^2 => dln n_F = 2 dln L
    Lm, sl = [], []
    for i in range(len(L) - 1):
        sl.append((math.log(lnO[i + 1]) - math.log(lnO[i])) /
                  (2 * (math.log(L[i + 1]) - math.log(L[i]))))
        Lm.append(0.5 * (L[i] + L[i + 1]))
    ax.plot(Lm, sl, 'o', ms=4.2, lw=0.9, color='#2166ac', zorder=3)
    ax.axhline(1.0, color='#e6811e', lw=0.8, ls='--', zorder=2)
    ax.axhline(0.5, color='#9a9a9a', lw=0.7, ls=':', zorder=2)
    ax.annotate('slope 1: area term\n$bL^2$ dominates',
                xy=(0.97, 0.965), xycoords='axes fraction', fontsize=5.0,
                color='#a8681a', ha='right', va='top', linespacing=1.2)
    ax.annotate('slope $\\frac{1}{2}$:\nlow-order terms\ndominate',
                xy=(0.03, 0.56), xycoords='axes fraction', fontsize=5.0,
                color='#777777', ha='left', linespacing=1.2)
    ax.set_xlim(1.5, 9.5)
    ax.set_ylim(0.35, 1.14)
    ax.set_xlabel('$L$')
    ax.set_ylabel(r'local slope $\frac{d\ln\ln\Omega}{d\ln N_F}$')
    print('[QA] local slopes:',
          [round(v, 3) for v in sl], flush=True)


def main():
    fig = plt.figure(figsize=(SINGLE_COL, 4.6))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.0],
                          left=0.145, right=0.96, top=0.965, bottom=0.10,
                          hspace=0.48)
    panel_a = fig.add_subplot(gs[0, 0])
    panel_b = fig.add_subplot(gs[1, 0])
    panel_area_law_si(panel_a)
    panel_local_slope(panel_b)
    for ax, s in ((panel_a, "a"), (panel_b, "b")):
        ax.text(-0.14, 1.04, s, transform=ax.transAxes, fontsize=8,
                fontweight="bold", va="bottom", ha="right")
    # QA spot checks against the registered CSV values
    tr = pd.read_csv(TR)
    d = tr[tr.type == 'count']
    print('[QA] sL L=2/9:', float(d[d.L == 2].sL.iloc[0]),
          float(d[d.L == 9].sL.iloc[0]), flush=True)
    savefig(fig, str(FIGS / "S6_area_law_3464"))


if __name__ == "__main__":
    main()
