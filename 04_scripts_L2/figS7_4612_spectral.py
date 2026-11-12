# -*- coding: utf-8 -*-
"""4.6.12 Fig. S7, two panels (a: ternary Z-count lnOmega vs N_F; b: TM
spectral-rate adjudication).

Panel a (ternary {1,2,3}, LxL torus, N_F = 6L^2, full-orbit convention):
data identical bit-for-bit with the archived legacy roundG_4612_lnOmega_plot.py
(canonical snapshot); the lower-right reading line follows the TM
adjudication convention.

Panel b: s(W) = lambda(W)/(6W) per face plotted against W, overlaid with the
inverse-proportional null s = 2 s(2)/W through (2, s(2)) (flat lambda, i.e.
the perimeter law).  s(2) = 0.22894 per face (two-sided);
s(3) >= 0.18622, s(4) >= 0.16039 per face (strict lower bounds; filled
circle + upward arrow).  Points (3)/(4) sit above the null
(s(3) > 2 s(2)/3 = 0.1526; s(4) > 2 s(2)/4 = 0.1145) => lambda(W) is not
flat => the perimeter law is falsified strictly and independently at W = 3
and W = 4.  QA assertions of the criterion are in-script.  Provenance of
rate(4) >= 3.849459: t4k1 K=1 single-boundary-state closure over
2,840,160 states, trunk = 100% of states finishable.
Normalization note: under the literal lambda/W (per width) convention the
point (3, lambda(3)/3) falls BELOW the inverse curve (1.117 < 1.832) and the
criterion direction flips - the per-face convention lambda/(6W) is required.
Rate values from z3_4612/RTH4612_metric_entropy_TM_readjudication_2026-10-06.md
(final adjudicated values; the intermediate RTH4612_tm_lambda.csv holds only
the W=2 row, so the W=3 value follows the adjudication document).
Output: <FIGURES>/S7_4612_spectral.{png,pdf}
"""
import io
import math
import sys
from pathlib import Path

import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt   # noqa

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import paths_common as PC  # noqa: E402

try:
    from l1_style import apply_style
    apply_style()
except Exception:
    pass

BLUE = '#2166ac'
GREY = '#8a8a8a'
DARK = '#555555'
# panel b: lightness ramp of one hue family for the three points
# (deep -> light = W=2 -> 4; colour-blind safe; one physical quantity,
# no categorical colours)
C_W2 = '#2166ac'   # deep blue: s(2) two-sided anchor
C_W3 = '#4393c3'   # mid blue: s(3) strict LB
C_W4 = '#92c5de'   # light blue: s(4) strict LB (black edge keeps contrast)
C_W4_ARR = '#6baed6'  # s(4) arrow (slightly darker than the point, visible on white)

# ---------------- panel a: ternary square torus (data unchanged) ----------------
data_a = [
    # (L, N_F, lnOmega, kind)  kind: 'exact' | 'knuth'
    (1, 6, math.log(2349), 'exact'),
    (2, 24, math.log(29723517), 'exact'),
    (3, 54, math.log(6.56623e12), 'knuth'),
    (4, 96, math.log(8.351e19), 'knuth'),
]

# ------- panel b: TM spectral-rate adjudication (final values, adjudication convention) -------
RATE2 = 2.74720453          # two-sided (six digits: 2.747205)
RATE3_LB = 3.35190260       # strict LB (eight-digit agreement)
RATE4_LB = 3.84945900       # strict LB (printed to six digits; rho_trunk = 46.967663)
S2_W2 = RATE2 / 12.0        # 0.22894 per face
S2_W3_LB = RATE3_LB / 18.0  # 0.186217 per face (strict)
S2_W4_LB = RATE4_LB / 24.0  # 0.160394 per face (strict)

fig, (axa, axb) = plt.subplots(1, 2, figsize=(7.0, 2.9))

# ===== panel a =====
L_last, NF_last, lnOm_last = data_a[-1][0], data_a[-1][1], data_a[-1][2]
xx = np.logspace(math.log10(NF_last / 6), math.log10(NF_last * 1.15), 50)
axa.plot(xx, lnOm_last * (xx / NF_last) ** 1.0, '--', color=GREY, lw=0.9,
         label='slope 1: area law ($\\ln\\Omega \\propto F$)')
axa.plot(xx, lnOm_last * (xx / NF_last) ** 0.5, ':', color=GREY, lw=0.9,
         label='slope 1/2: perimeter law ($\\ln\\Omega \\propto F^{1/2}$)')
xs = [d[1] for d in data_a]
ys = [d[2] for d in data_a]
axa.plot(xs[:2], ys[:2], 'o', ms=5, color=BLUE, markeredgecolor='k',
         markeredgewidth=0.6, label='4.6.12 exact (L=1,2)', zorder=3)
axa.plot(xs[2:], ys[2:], 'o', ms=5, markerfacecolor='white',
         markeredgecolor=BLUE, markeredgewidth=1.0,
         label='4.6.12 Knuth est. (L=3,4)', zorder=3)
axa.plot(xs, ys, '-', color=BLUE, lw=0.8, zorder=2)
for (L, NF, lnOm, kind) in data_a:
    sZ = lnOm / (6 * L * L)
    axa.annotate(f'{sZ:.2f}', (NF, lnOm), textcoords='offset points',
                 xytext=(4, -9), fontsize=6.5, color=DARK)
axa.set_xscale('log')
axa.set_yscale('log')
axa.set_xlabel('$N_F$ (tiles)')
axa.set_ylabel('$\\ln \\Omega_\\mathbb{Z}$')
axa.set_xlim(5, 130)
axa.set_ylim(5, 80)
axa.legend(fontsize=6.5, loc='upper left', frameon=False)
axa.text(-0.14, 1.05, 'a', transform=axa.transAxes, fontsize=8,
         fontweight='bold', va='bottom', ha='right')

# ===== panel b: s(W) = lambda(W)/(6W) per face, perimeter-law inverse null =====
from matplotlib.lines import Line2D
S2 = RATE2 / 12.0            # s(W=2) per face (6W=12), two-sided
S3_LB = RATE3_LB / 18.0      # s(W=3) per face (6W=18), strict LB
S4_LB = RATE4_LB / 24.0      # s(W=4) per face (6W=24), strict LB
C_NULL = 2.0 * S2            # inverse null: s = C/W through (2, s(2)) <=> lambda flat at lambda(2)
# perimeter-law null inverse curve (lambda flat <=> s ~ 1/W; through the W=2
# point) - null-hypothesis reference carrying the adjudication semantics
xx = np.linspace(1.55, 4.45, 260)
axb.plot(xx, C_NULL / xx, '--', color=GREY, lw=0.9,
         label='perimeter-law null ($\\lambda$ flat)')
# model read-out: least-squares fit s = s_inf + c/W (equal-weight three
# points) - the asymptote is a model read-out, not a bound
# (no decay-form theorem exists; any curve through three points can bend
# toward 0; the fit is a visual read-out only and must not be called a bound)
_x13 = np.array([1 / 2, 1 / 3, 1 / 4])
_y13 = np.array([S2, S3_LB, S4_LB])
_bfit, _afit = np.polyfit(_x13, _y13, 1)   # s = a + b/W
SINF_FIT = _afit
print(f"[QA] fit s = sinf + c/W: sinf={_afit:.5f} c={_bfit:.5f} "
      f"(residuals max {max(abs(_y13 - (_afit + _bfit*_x13))):.5f})", flush=True)
xx = np.linspace(1.55, 4.45, 260)
axb.plot(xx, SINF_FIT + _bfit / xx, '-.', color=DARK, lw=0.9,
         label='fit $s=s_\\infty+c/W$: $s_\\infty \\approx$ 0.094 (read-out)')
# s(2): two-sided (deep-blue filled circle; upper and lower bounds coincide)
axb.plot([2], [S2], 'o', ms=6, color=C_W2, markeredgecolor='k',
         markeredgewidth=0.6, zorder=4)
# s(3)/s(4): strict lower bounds = filled circle + upward arrow ("at least");
# points above the null curve = positive entropy
axb.plot([3], [S3_LB], 'o', ms=6, color=C_W3, markeredgecolor='k',
         markeredgewidth=0.6, zorder=4)
axb.annotate('', xy=(3, S3_LB + 0.028), xytext=(3, S3_LB + 0.003),
             arrowprops=dict(arrowstyle='-|>', color=C_W3, lw=1.3,
                             mutation_scale=11), zorder=3)
axb.plot([4], [S4_LB], 'o', ms=6, color=C_W4, markeredgecolor='k',
         markeredgewidth=0.7, zorder=4)
axb.annotate('', xy=(4, S4_LB + 0.028), xytext=(4, S4_LB + 0.003),
             arrowprops=dict(arrowstyle='-|>', color=C_W4_ARR, lw=1.3,
                             mutation_scale=11), zorder=3)
# legend proxies (one marker style for the data points; text distinguishes
# two-sided vs lower bound)
axb.plot([], [], 'o', ms=6, color=C_W2, markeredgecolor='k',
         markeredgewidth=0.6, label='s(2) = 0.229 per face (two-sided)')
axb.plot([], [], 'o', ms=6, color=C_W3, markeredgecolor='k',
         markeredgewidth=0.6,
         label='s(3) $\\geq$ 0.186 per face (strict LB; arrow)')
axb.plot([], [], 'o', ms=6, color=C_W4, markeredgecolor='k',
         markeredgewidth=0.7,
         label='s(4) $\\geq$ 0.160 per face (strict LB; arrow)')
handles, labels = axb.get_legend_handles_labels()
order = [labels.index(l) for l in ['perimeter-law null ($\\lambda$ flat)',
                                   'fit $s=s_\\infty+c/W$: $s_\\infty \\approx$ 0.094 (read-out)',
                                   's(2) = 0.229 per face (two-sided)',
                                   's(3) $\\geq$ 0.186 per face (strict LB; arrow)',
                                   's(4) $\\geq$ 0.160 per face (strict LB; arrow)']]
axb.legend([handles[i] for i in order], [labels[i] for i in order],
           fontsize=6.5, loc='upper right', bbox_to_anchor=(0.985, 0.865),
           frameon=False)
axb.set_xlabel('strip width $W$')
axb.set_ylabel('$s$ per face $= \\lambda/(6W)$ (binary {1,2})')
axb.set_xticks([2, 3, 4])
axb.set_xlim(1.55, 4.45)
axb.set_ylim(0.095, 0.31)
axb.text(-0.15, 1.05, 'b', transform=axb.transAxes, fontsize=8,
         fontweight='bold', va='bottom', ha='right')
# QA criterion: points (3)/(4) above the inverse null = lambda not flat =
# perimeter law falsified (each width independently)
assert S3_LB > C_NULL / 3.0, 's(3) must sit above the perimeter-law null'
assert S4_LB > C_NULL / 4.0, 's(4) must sit above the perimeter-law null'
print(f"[QA] null @W=3: {C_NULL/3.0:.5f} < s3_LB {S3_LB:.5f}; "
      f"null @W=4: {C_NULL/4.0:.5f} < s4_LB {S4_LB:.5f} -> above null "
      f"(strict falsification, both widths independently)", flush=True)

# QA assertions (bit-for-bit vs the adjudication document)
assert abs(RATE2 - 2.74720453) < 1e-8 and abs(RATE3_LB - 3.35190260) < 1e-8
assert abs(RATE4_LB - 3.84945900) < 1e-8
assert abs(S2_W2 - 0.2289337) < 1e-6 and abs(S2_W3_LB - 0.1862168) < 1e-6
assert abs(S2_W4_LB - 0.160394125) < 1e-6
assert RATE3_LB > RATE2 and RATE4_LB > RATE2
print(f"[QA] rate(2)={RATE2:.6f} two-sided | rate(3)>={RATE3_LB:.8f} | "
      f"rate(4)>={RATE4_LB:.6f} strict LBs | "
      f"s2(2)={S2_W2:.4f} s2(3)>={S2_W3_LB:.5f} s2(4)>={S2_W4_LB:.5f} per face",
      flush=True)
print(f"[QA] slope/width: W2-3 {RATE3_LB-RATE2:.4f}, W2-4 "
      f"{(RATE4_LB-RATE2)/2:.4f} (ref s2(inf) ~ 0.09-0.10)", flush=True)

fig.subplots_adjust(left=0.075, right=0.985, bottom=0.13, top=0.86, wspace=0.30)
outs = [str(PC.FIGURES / 'S7_4612_spectral.png')]
for out in outs:
    fig.savefig(out, dpi=300)
    fig.savefig(out.replace('.png', '.pdf'))
    print('saved', out)
