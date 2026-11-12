# -*- coding: utf-8 -*-
"""4.6.12 Fig. S7 two-panel figure (panel a: ternary Z-count lnOmega vs
N_F; panel b: binary width-scaling adjudication).

Panel a (ternary, alphabet {1,2,3} [0-based {0,1,2} in data/scripts,
shift-equivalent, unified to 1-based in the manuscript]; L x L torus,
N_F = 6L^2; full-orbit calibre -- see the header of
RTH4612_knuth_estimate_series.csv):
  L=1: Omega=2349 (exact, brute force)
  L=2: Omega=29 723 517 (exact, parametric DFS)
  L=3: Omega~6.57e12 (Knuth 160k, dln 0.0005)
  L=4: Omega~8.35e19 (Knuth 2M samples, dln 0.0084)
log-log axes; two guide lines (slope 1 = area law; slope 1/2 = perimeter
law) through the last point; exact points filled, Knuth estimates open.

Panel b (binary {1,2} [0-based {0,1} in scripts/CSVs], width-scaling
adjudication):
  exact anchors (frontier DP == blocked-dfs, bitwise):
    2x2=22 912 / 2x3=58 240 / 3x3=11 261 320 / 4x3=15 300 096
  W=2 frontier-DP series L=2..14 (RTH4612_binary_strips.csv)
  strip Knuth (corrected weight X = prod f; RTH4612_knuth_strips2.csv;
    in-range = hit >= 2 with no exact anchor at the same point)
  range-edge rows (hit=1): open grey diamonds
  verdict: s_01(inf) >= 0.21 (no decay across widths; contrast the
  perimeter-law 1/W guide anchored at (2, 0.30) => 0.12 at W=5)
Outputs: z3_4612/TTH4612_lnOmega_NF.png + z3_4612/S7_4612_lnOmega_Z.png/.pdf
"""
import csv
import io
import math
import sys
from pathlib import Path

import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt   # noqa

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / '04_scripts_L2'))
try:
    from l1_style import apply_style
    apply_style()
except Exception:
    pass

BLUE = '#2166ac'
GREY = '#8a8a8a'
DARK = '#555555'

# ---------------- panel a: ternary square torus ----------------
data_a = [
    # (L, N_F, lnOmega, kind)  kind: 'exact' | 'knuth'
    (1, 6, math.log(2349), 'exact'),
    (2, 24, math.log(29723517), 'exact'),
    (3, 54, math.log(6.56623e12), 'knuth'),
    (4, 96, math.log(8.351e19), 'knuth'),
]

# ---------------- panel b: binary width scaling ----------------
# exact anchors (frontier DP / blocked DFS, bitwise; includes the 2x3
# transpose crosscheck)
anchors_b = [
    # (W, L, s)
    (2, 2, 0.418309),
    (2, 3, 0.304787),
    (3, 3, 0.300683),
    (4, 3, 0.229769),
]
# W=2 frontier-DP series (all exact; includes the L=2,3 anchors,
# deduped in post-processing)
dp_w2 = []
with open('z3_4612/RTH4612_binary_strips.csv', encoding='utf-8') as f:
    for row in csv.DictReader(f):
        if row['W'] == '2' and row['s_01']:
            dp_w2.append((int(row['W']), int(row['L']), float(row['s_01'])))
dp_w2.sort(key=lambda t: t[1])
assert len(dp_w2) == 13, f'W=2 DP series expect 13 points, got {len(dp_w2)}'

# strip Knuth (hit >= 2 = in-range; hit = 1 = range edge;
# hit = 0 = zero-hit region, not drawn)
knuth_in, knuth_edge = [], []
exact_set = {(w, l) for w, l, _ in anchors_b} | {(w, l) for w, l, _ in dp_w2}
with open('z3_4612/RTH4612_knuth_strips2.csv', encoding='utf-8') as f:
    rows = [ln for ln in f if not ln.startswith('#')]
for row in csv.DictReader(rows):
    w, l, hit = int(row['W']), int(row['L']), int(row['hit'])
    s = float(row['s_per_face'])
    if hit == 0:
        continue
    if (w, l) in exact_set:
        continue  # exact anchor covers this point
    (knuth_edge if hit == 1 else knuth_in).append((w, l, s))


print(f'panel b: DP W=2 series {len(dp_w2)} pts, Knuth in-range {len(knuth_in)}, '
      f'range-edge {len(knuth_edge)}, exact anchors {len(anchors_b)}')
for w, l, s in knuth_in:
    print(f'  knuth  W={w} L={l}: s={s:.4f}')
for w, l, s in knuth_edge:
    print(f'  edge   W={w} L={l}: s={s:.4f}')

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
                 xytext=(4, -9), fontsize=6, color=DARK)
axa.set_xscale('log')
axa.set_yscale('log')
axa.set_xlabel('$N_F$ (tiles)')
axa.set_ylabel('$\\ln \\Omega_\\mathbb{Z}$')
axa.set_xlim(5, 130)
axa.set_ylim(5, 80)
axa.legend(fontsize=6, loc='upper left', frameon=False)
axa.text(0.97, 0.04,
         '4.6.12 {1,2,3}  $s_\\mathbb{Z}=\\ln\\Omega_\\mathbb{Z}/N_F$\n'
         'third carrier: binary width scaling gives $s_\\mathbb{Z}(\\infty)\\geq 0.21$ (panel b)',
         transform=axa.transAxes, fontsize=5.5, ha='right', va='bottom', color=DARK)
axa.text(0.985, 0.975, '(a)', transform=axa.transAxes, fontsize=8,
         fontweight='bold', va='top', ha='right')

# ===== panel b =====
# perimeter-law contrast guide: anchor (2, 0.30) => 0.12 at W=5
Wg = np.array([2.0, 5.0])
axb.plot(Wg, 0.30 * 2.0 / Wg, '--', color=GREY, lw=0.9,
         label='perimeter law ($s \\propto 1/W$)')
# adjudication lower bound
axb.axhline(0.21, color=BLUE, lw=0.9, linestyle=':',
            label='$s_{01}(\\infty) \\geq 0.21$')
# W=2 DP exact series (filled circles)
bw = [p[0] for p in dp_w2]
bs = [p[2] for p in dp_w2]
axb.plot(bw, bs, 'o', ms=3.5, color=BLUE, markeredgecolor='k', markeredgewidth=0.4,
         label='exact (frontier DP / blocked DFS)', zorder=3)
# remaining exact anchors (3x3, 4x3)
aw = [p[0] for p in anchors_b if (p[0], p[1]) not in {(2, 2), (2, 3)}]
as_ = [p[2] for p in anchors_b if (p[0], p[1]) not in {(2, 2), (2, 3)}]
axb.plot(aw, as_, 'o', ms=4.5, color=BLUE, markeredgecolor='k', markeredgewidth=0.6,
         zorder=4)
# Knuth in-range (open squares)
kw = [p[0] for p in knuth_in]
ks = [p[2] for p in knuth_in]
axb.plot(kw, ks, 's', ms=4, markerfacecolor='white', markeredgecolor=DARK,
         markeredgewidth=0.9, linestyle='none', label='strip Knuth (calibrated)', zorder=3)
# range edge (open grey diamonds)
ew = [p[0] for p in knuth_edge]
es = [p[2] for p in knuth_edge]
axb.plot(ew, es, 'D', ms=4, markerfacecolor='white', markeredgecolor=GREY,
         markeredgewidth=0.9, linestyle='none', label='range-edge (single hit)', zorder=3)
axb.set_xlabel('strip width $W$')
axb.set_ylabel('$s$ per tile (binary {1,2})')
axb.set_xticks([2, 3, 4, 5])
axb.set_xlim(1.6, 5.4)
axb.set_ylim(0.08, 0.46)
axb.legend(fontsize=6, loc='upper right', frameon=False)
axb.text(0.02, 0.98, '(b)', transform=axb.transAxes, fontsize=8,
         fontweight='bold', va='top')
axb.text(0.03, 0.05,
         'bands overlap at $W=2\\ldots5$:\nno $1/W$ decay\n(verdict: numerical determination)',
         transform=axb.transAxes, fontsize=5.5, ha='left', va='bottom', color=DARK)

fig.tight_layout()
outs = ['z3_4612/TTH4612_lnOmega_NF.png',
        'z3_4612/S7_4612_lnOmega_Z.png']
for out in outs:
    fig.savefig(out, dpi=300)
    fig.savefig(out.replace('.png', '.pdf'))
    print('saved', out)
