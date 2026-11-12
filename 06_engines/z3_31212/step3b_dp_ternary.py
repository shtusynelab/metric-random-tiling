# -*- coding: utf-8 -*-
"""Step 3b: ternary {0,1,2} counting (alphabet-dependence adjudication).

Background (Step 3 result): the binary mid subsystem's non-resonant branch is an
exact power of 2, Omega_mid = 2^{L+W+1} (verified for W=2,3,4) => lambda = ln 2
constant => s_mid = ln2/(3W) -> 0 (perimeter-law shape, binary zero entropy).

Caution: binary zero entropy does not settle the ternary case (the mid equations are
sum rules of the l0 + l1 = l3 + l4 type; propagating under binary, looser constraints
under ternary) => adjudicate ternary directly. Criteria:
  - if Omega_mid3(W, L) = 3^{L+W+1} (propagation type, as in the square-lattice
    theorem) => all-alphabet zero entropy settled;
  - if lambda_3 = lnOmega_3(L+1) - lnOmega_3(L) > ln 3 (width scaling) => ternary
    positive entropy => alphabet-dependent type.

Frontier DP as in step3 (vals=(0,1,2); states <= 3^{frontier} ~ 3^{3W+O(1)}, ~5e5 at
W=3, affordable; W=4 ternary only up to L<=5). Resonant (commensurate) sizes are
recorded separately and excluded from adjudication.
Output: z3_31212/TSD31212_dp_ternary.csv + stdout.
"""
import csv
import math
import os
import sys
import time
from itertools import product

sys.stdout.reconfigure(encoding='utf-8')
BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from step3_dp_strips import build_subsystems, frontier_dp  # noqa: E402

OUT_CSV = os.path.join(BASE, 'z3_31212', 'TSD31212_dp_ternary.csv')


def main():
    rows = []
    print('=== ternary mid/cut DP (non-resonant branch adjudication) ===')
    plans = [
        (2, [3, 5, 7, 9, 11, 13, 15]),          # W=2: odd L (non-resonant)
        (3, [2, 4, 5, 7, 8, 10]),               # W=3: L not congruent 0 mod 3
        (4, [3, 5, 7]),                         # W=4: odd L (small L, heavy states)
    ]
    for W, Ls in plans:
        prev = None               # (L, lnOmega): non-resonant series may have delta_L > 1, lambda/col must divide by delta_L
        for L in Ls:
            sub = build_subsystems(W, L)
            t0 = time.time()
            om3 = frontier_dp(sub['mid_keys'], sub['mid_keysets'], sub['mid_eqs'],
                              label=f'mid3 {W}x{L}', vals=(0, 1, 2))
            oc3 = frontier_dp(sub['cut_keys'], sub['cut_keysets'], sub['cut_eqs'],
                              label=f'cut3 {W}x{L}', vals=(0, 1, 2))
            dt = time.time() - t0
            ref = (L + W + 1) * math.log(3)
            if prev is not None:
                dl = L - prev[0]
                lam_col = (math.log(om3) - prev[1]) / dl
                s_col = lam_col / (3 * W)
            else:
                lam_col = s_col = float('nan')
            print(f'  W={W} L={L}: Omega_mid3={om3} cut3={oc3} '
                  f'ln_mid3={math.log(om3):.4f} (3^{{L+W+1}}={ref:.4f}, '
                  f'ratio={math.log(om3)/ref:.4f}) lambda_mid3/col={lam_col:.4f} '
                  f'(ln3={math.log(3):.4f}) s_mid3={s_col:.4f} [{dt:.1f}s]', flush=True)
            rows.append([W, L, om3, oc3, f'{math.log(om3):.6f}', f'{ref:.6f}',
                         f'{math.log(om3)/ref:.6f}',
                         f'{lam_col:.6f}' if prev is not None else '',
                         f'{s_col:.6f}' if prev is not None else '', f'{dt:.1f}'])
            prev = (L, math.log(om3))
    with open(OUT_CSV, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['W', 'L', 'Omega_mid_ternary', 'Omega_cut_ternary',
                    'lnOmega_mid3', 'ln3^(L+W+1)', 'ratio',
                    'lambda_mid3_per_col', 's_mid3_per_col', 'time_s'])
        w.writerows(rows)
    print(f'CSV written {OUT_CSV}')


if __name__ == '__main__':
    main()
