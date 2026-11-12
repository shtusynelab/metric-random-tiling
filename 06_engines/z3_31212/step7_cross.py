# -*- coding: utf-8 -*-
"""Step 7 validation 2: direct crossover observation -- the W=2/3 torus mid DP
extended to large L; lambda(L) should rise from the 2-army (2^(L+W+1) transient
phase) and converge to the TM spectral radius:
  k=3 W=2: lambda(L=27)=1.598 still rising -> TM ln rho=1.6094 (L->inf limit);
  k=3 W=3: lambda(L<=11)=0.693 (exact 2-army) -> expected rise to TM ln rho=2.0122.
Perron-Frobenius theorem: lim (1/L) ln Omega_torus = ln rho(T) -- this script is a
direct observation of that theorem.
State count independent of L (W=2: 81, W=3: 729); DP is cheap.
"""
import csv
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from step3_dp_strips import build_subsystems, frontier_dp  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   'TSD31212_tm_crossover.csv')
JOBS = [(2, [29, 33, 37, 41, 45, 49, 53, 57], 1.609429),
        (3, [13, 15, 17, 19, 21, 23], 2.012150)]


def main():
    for (W, Ls, tgt) in JOBS:
        prev_ln, prev_L = None, None
        for Lc in Ls:
            sub = build_subsystems(W, Lc)
            t0 = time.time()
            om = frontier_dp(sub['mid_keys'], sub['mid_keysets'],
                             sub['mid_eqs'], label=f'mid3 {W}x{Lc}',
                             vals=(0, 1, 2))
            dt = time.time() - t0
            lam = (math.log(om) - prev_ln) / (Lc - prev_L) if prev_ln else None
            print(f'k=3 W={W} L={Lc}: Omega_mid={om} lambda/col='
                  f'{("%.4f" % lam) if lam else "n/a"} (TM {tgt:.4f}) [{dt:.0f}s]',
                  flush=True)
            with open(OUT, 'a', newline='', encoding='utf-8-sig') as f:
                w = csv.writer(f)
                if f.tell() == 0:
                    w.writerow(['W', 'L', 'Omega_mid', 'lnOmega',
                                'lambda_per_col', 'TM_ln_rho', 'time_s'])
                w.writerow([W, Lc, om, f'{math.log(om):.6f}',
                            f'{lam:.6f}' if lam else '', f'{tgt:.6f}',
                            f'{dt:.0f}'])
            prev_ln, prev_L = math.log(om), Lc
    print('done', flush=True)


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    main()
