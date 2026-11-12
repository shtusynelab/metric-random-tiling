# -*- coding: utf-8 -*-
"""4.6.12 rectangular-torus (strip) series: binary {0,1} frontier DP.

Width-scaling data for the lambda(W) route:
  lambda(W) = lim_{L->inf} ln Omega(W x L) / L  (per-column entropy),
  s_01(inf) read from the W->inf trend of lambda(W)/(6W) (honeycomb
  fig-2 precedent: Ly=2/3 fine-torus series).

On a fine torus W x L (W << L) the BFS wavefront saturates to a line of
width ~ W * c classes, so W=2/3 stay within the fail-closed gate; W=4
probed last. Each (W, L) row lands in the CSV as it finishes; a row that
hits the gate is logged and skipped.

Crosscheck: DP (2,3)/(3,2) vs independent blocked-DFS anchors
(roundF_server_blocks_binary.py --Lx --Ly) -- bitwise.

Run: python3 roundE_4612_strip_series.py
Output: RTH4612_binary_strips.csv
"""
import csv
import math
import os
import sys
import time

from roundE_4612_frontier_dp import dp_count

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_CSV = os.path.join(HERE, 'RTH4612_binary_strips.csv')

SERIES = [(2, list(range(2, 15))),
          (3, [4, 5, 6]),          # 3x3 = square torus, taken from the main DP
          (4, [2, 3])]


def main():
    done = set()
    if os.path.exists(OUT_CSV):
        with open(OUT_CSV) as f:
            for row in csv.DictReader(f):
                done.add((int(row['W']), int(row['L'])))
    new = not os.path.exists(OUT_CSV)
    f = open(OUT_CSV, 'a', newline='')
    w = csv.writer(f)
    if new:
        w.writerow(['W', 'L', 'N', 'tree_width', 'peak_states', 'DP_count',
                    'Omega_01', 'lnOmega', 's_01', 'lnOmega_over_L', 'time_s'])
    for W, Ls in SERIES:
        for L in Ls:
            if (W, L) in done:
                continue
            t0 = time.time()
            try:
                dp, n_free, N, tw, peak, dt = dp_count(W, L, vals=(0, 1),
                                                       rank_check=False)
            except SystemExit:
                print(f'W={W} L={L}: GATED (peak states)', flush=True)
                w.writerow([W, L, W * L, '', '', 'GATED', '', '', '', '', ''])
                f.flush()
                continue
            omega = dp * (2 ** n_free)
            lam = math.log(omega) / L
            s = math.log(omega) / (6 * W * L)
            print(f'W={W} L={L}: N={N} tree_w={tw} peak={peak} '
                  f'Omega={omega} ln={math.log(omega):.4f} '
                  f'lam=ln/L={lam:.4f} s={s:.4f} [{dt:.0f}s]', flush=True)
            w.writerow([W, L, N, tw, peak, dp, omega,
                        f'{math.log(omega):.4f}', f'{s:.6f}',
                        f'{lam:.6f}', f'{dt:.0f}'])
            f.flush()
    f.close()
    print('STRIP SERIES DONE', flush=True)


if __name__ == '__main__':
    main()
