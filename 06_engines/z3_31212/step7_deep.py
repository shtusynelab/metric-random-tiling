# -*- coding: utf-8 -*-
"""Server-only driver (originally run on the internal cluster); pass explicit
arguments when used locally.

Step 7 deep-chain fleet: mid DP for a single (W, L, k), one process per task.
usage: python3 step7_deep.py W L k
Appends one line to TSD31212_tm_deep.csv (O_APPEND).
"""
import csv
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from step3_dp_strips import build_subsystems, frontier_dp  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   'TSD31212_tm_deep.csv')


def main():
    W, Lc, kk = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    vals = (0, 1) if kk == 2 else (0, 1, 2)
    sub = build_subsystems(W, Lc)
    t0 = time.time()
    om = frontier_dp(sub['mid_keys'], sub['mid_keysets'], sub['mid_eqs'],
                     label=f'mid k{kk} {W}x{Lc}', vals=vals)
    dt = time.time() - t0
    ln = math.log(om)
    lam = ln / Lc
    with open(OUT, 'a', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        if f.tell() == 0:
            w.writerow(['k', 'W', 'L', 'Omega_mid', 'lnOmega',
                        'lnOmega_over_L', 'time_s'])
        w.writerow([kk, W, Lc, om, f'{ln:.6f}', f'{lam:.6f}', f'{dt:.0f}'])
    print(f'k={kk} W={W} L={Lc}: Omega={om} ln/L={lam:.4f} [{dt:.0f}s]',
          flush=True)


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    main()
