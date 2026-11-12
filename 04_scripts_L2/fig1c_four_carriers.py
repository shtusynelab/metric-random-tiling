# -*- coding: utf-8 -*-
"""fig1c_v4 - Fig 1c of the theory paper (four-carrier ln Omega-N_F readout).
HISTORICAL GENERATOR: superseded by fig1_model_classification.py (the
composite Fig 1); its output file has been retired and this script is kept
only as an archived generator.

Design (v4):
1. 4.6.12: the S7a convention -- L=1,2 filled (exact) / L=3,4 open (Knuth
   estimates); annotation switched to the local log-log slopes
   (0.57/0.67/0.77) to show the climb explicitly (the per-tile
   s_Z = 1.29/0.72/0.55/0.48 values match Table S5 digit-for-digit, but a
   decreasing readout is unintuitive).
2. Honeycomb: canonical fixed-width Ly=3 torus sequence restored (11 points;
   the square-torus family is left with only 4 points under the 3^(2L)
   state-space cap and its crossover needs ~20x20 to approach s_infinity;
   the square-torus points 2..5 are kept as a caption note).
3. 3.12.12: mid ternary W=2 cylinder series restored (13 points, exact; same
   fixed-width family as honeycomb Ly=3, labeled with the cylinder
   convention).
Caption discipline: each carrier sequence is chosen by exact-countability
reachability and annotated individually; for the unconverged climber
(4.6.12) the positive-entropy verdict is carried by the spectral/strict-
level results (SI S8.2).
Output: ../figs/fig1c_four_carriers.{png,pdf}"""
import math
import sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import paths_common as PC  # noqa: E402
from l1_style import plt, savefig  # noqa: E402

FIGS = PC.FIGURES


def main():
    fig, ax = plt.subplots(figsize=(4.4, 3.5))
    tr = pd.read_csv(PC.engine("z4_3464") /
                     PC.fname("RTH_torus_area_law_8pts.csv"))
    # 1. honeycomb {1,2,3}: canonical fixed-width Ly=3 torus sequence (11
    #    points; Lx<=6 prefix-block DFS, Lx>=7 trace method; extension
    #    points sampled every other one to balance point density)
    pm = pd.read_csv(PC.HC / PC.fname("P0_PM_torus_growth_forms.csv"))
    tse = pd.read_csv(PC.HC /
                      PC.fname("P0_PM_torus_trace_extension_Ly3.csv"))
    da = pm[(pm.alphabet == 'arithmetic') & (pm.Ly == 3)]
    de = tse[(tse.alphabet == 'arithmetic') & (tse.Ly == 3) &
             (tse.Lx % 2 == 0)]
    da = pd.concat([da, de], ignore_index=True).sort_values('F')
    ax.plot(da.F, da.lnOmega, 's', ms=3.6, color='#c0392b',
            label='honeycomb $\\{1,2,3\\}$ (torus $L_y$=3)', zorder=4)
    # square-torus family (2x2..5x5, exact) -- caption note only, not drawn
    sq = "(square: 2\u00d72=729, 3\u00d73=22 767, 4\u00d74=1 279 101, " \
         "5\u00d75=214 523 187)"
    # 3. 3.4.6.4 (square-torus nine points + scaling-law fit line)
    d = tr[tr.type == 'count']
    ax.plot(6 * d.L ** 2, d.lnOmega, '^', ms=4.2, color='#e6811e',
            label='3.4.6.4 $\\{1,2\\}$ ($L\\times L$)', zorder=4)
    L_arr = d.L.values.astype(float)
    coef = np.polyfit(L_arr, d.lnOmega.values.astype(float), 2)
    nf_fit = np.logspace(math.log10(24), math.log10(600), 100)
    ax.plot(nf_fit, np.polyval(coef, np.sqrt(nf_fit / 6.0)), '-', lw=1.1,
            color='#e6811e', label='3.4.6.4: fit $bL^2\\!+cL\\!+d$', zorder=2)
    # 4. 4.6.12 (S7a convention: {1,2,3} square torus; filled = exact L=1,2;
    #    open = Knuth L=3,4)
    data4 = [(1, 6, math.log(2349), 'exact'),
             (2, 24, math.log(29723517), 'exact'),
             (3, 54, math.log(6.56623e12), 'knuth'),
             (4, 96, math.log(8.351e19), 'knuth')]
    xs = [t[1] for t in data4]
    ys = [t[2] for t in data4]
    ax.plot(xs[:2], ys[:2], 'o', ms=4.2, color='#2166ac', zorder=4,
            label='4.6.12 $\\{1,2,3\\}$ (exact, $L$=1,2)')
    ax.plot(xs[2:], ys[2:], 'o', ms=4.2, mfc='white', mec='#2166ac',
            mew=1.0, zorder=4, label='4.6.12 (Knuth est., $L$=3,4)')
    ax.plot(xs, ys, '-', color='#2166ac', lw=0.8, zorder=3)
    # 5. 3.12.12 mid: ternary W=2 cylinder exact series (13 points,
    #    TSD31212_dp_ternary; N_F=3WL=6L; same fixed-width family as
    #    honeycomb Ly=3)
    d3 = pd.read_csv(PC.engine("z3_31212") / "TSD31212_dp_ternary.csv")
    d3 = d3[d3.W == 2].sort_values("L")
    ax.plot(6 * d3.L, d3.lnOmega_mid3, 'v', ms=3.8, color='#7b3294', zorder=4,
            label='3.12.12 mid $\\{1,2,3\\}$ (cylinder $W$=2)')
    # slope reference lines: slope 1 through the honeycomb last point;
    # slope 1/2 through the 3.4.6.4 first point
    c_ext = float(da.lnOmega.iloc[-1]) / float(da.F.iloc[-1])
    c_sub = float(d.lnOmega.iloc[0]) / math.sqrt(24.0)
    ax.plot([3.5 / c_ext, 75.0 / c_ext], [3.5, 75.0],
            ls=(0, (5, 2.2)), lw=0.8, color='#8a8a8a', zorder=1.2,
            label='slope 1: area law')
    ax.plot([3.6, min((75.0 / c_sub) ** 2, 720.0)],
            [c_sub * math.sqrt(3.6), 75.0],
            ls=(0, (1.2, 1.6)), lw=0.8, color='#8a8a8a', zorder=1.2,
            label='slope $\\frac{1}{2}$: perimeter law')
    ax.set_xlabel(r"$N_F$")
    ax.set_ylabel(r"$\ln\Omega$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(fontsize=5.4, frameon=False, loc="upper left")
    ax.tick_params(labelsize=6.5)
    fig.tight_layout()
    savefig(fig, str(FIGS / "fig1c_four_carriers"))
    print("saved", FIGS / "fig1c_four_carriers.{png,pdf}")
    print("caption note - honeycomb square family:", sq)


if __name__ == "__main__":
    main()
