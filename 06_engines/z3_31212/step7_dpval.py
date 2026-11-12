# -*- coding: utf-8 -*-
"""Server-only driver (originally run on the internal cluster); pass explicit
arguments when used locally.

Step 7 validation: exact torus DP (long non-resonant chains) cross-checked against
the TM spectral radius.
Expected (if the TM is correct): the DP per-column increment lambda(W, L->inf)
-> ln rho(W):
  k=2: W=5 -> 1.325, W=6 -> 1.575 (W<=4 known = ln 2, small-width special case);
  k=3: W=3 -> 2.012, W=5 -> 3.015 (W=2 known 1.609).
Lightweight (mid DP states ~ k^(3W +/-)).
"""
import csv
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from step3_dp_strips import build_subsystems, frontier_dp  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   'TSD31212_tm_dpval.csv')
TM_LNRHO = {(2, 5): 1.324517, (2, 6): 1.575145, (3, 3): 2.012150,
            (3, 5): 3.014592, (2, 7): 1.736150, (3, 7): None}
JOBS = [(5, 5, 2, [7, 9, 11, 13]), (6, 6, 2, [5, 7, 11, 13]),
        (3, 3, 3, [5, 7, 8]), (7, 7, 2, [9, 11, 13])]


def main():
    rows = []
    for (W, L, kk, Ls) in JOBS:
        if kk == 2 and L > 6 and W > 6:
            continue
        sub = build_subsystems(W, L)
        vals = (0, 1) if kk == 2 else (0, 1, 2)
        prev_ln, prev_L = None, None
        for Lc in Ls:
            sub = build_subsystems(W, Lc)
            t0 = time.time()
            om = frontier_dp(sub['mid_keys'], sub['mid_keysets'],
                             sub['mid_eqs'], label=f'mid k{kk} {W}x{Lc}',
                             vals=vals)
            dt = time.time() - t0
            lam = (math.log(om) - prev_ln) / (Lc - prev_L) if prev_ln else None
            tgt = TM_LNRHO.get((kk, W))
            tgt_s = f'{tgt:.4f}' if tgt else '?'
            print(f'k={kk} W={W} L={Lc}: Omega_mid={om} ln={math.log(om):.4f} '
                  f'lambda/col={("%.4f" % lam) if lam else "n/a"} (TM ln rho={tgt_s}) '
                  f'[{dt:.0f}s]', flush=True)
            rows.append([kk, W, Lc, om, f'{math.log(om):.6f}',
                         f'{lam:.6f}' if lam else '', tgt_s, f'{dt:.0f}'])
            prev_ln, prev_L = math.log(om), Lc
            with open(OUT, 'a', newline='', encoding='utf-8-sig') as f:
                w = csv.writer(f)
                if f.tell() == 0:
                    w.writerow(['k', 'W', 'L', 'Omega_mid', 'lnOmega',
                                'lambda_per_col', 'TM_ln_rho', 'time_s'])
                w.writerows(rows[-1:])
    print('done')


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    main()
