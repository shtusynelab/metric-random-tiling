# -*- coding: utf-8 -*-
"""Step 5: diagonal family (square tori W=L) lnOmega vs N_F log-log slope data supplement.

Fig 1c paradigm: fixed-W narrow-torus chains always have slope 1 on log-log
(perimeter law and positive entropy are shape-identical there); resolving 1/2 vs 1
requires the diagonal family (N_F = 3L^2):
  binary: lambda constant => lnOmega proportional to L proportional to sqrt(N_F)
  => slope -> 1/2 (perimeter law);
  ternary: positive entropy => area term proportional to N_F => slope -> 1.
Existing diagonal points (dp_strips.csv): binary L=1..4; ternary L=1 (brute 243).
This script adds: binary diagonal L=5..8 (DP) + ternary diagonal L=2,3 (DP; L>=4 via
the Knuth driver step4b_knuth_server.py). All diagonal sizes are commensurate
(resonant); the excess enters the lambda asymptotics without changing the
proportional-to-L / proportional-to-N_F shapes, same convention as the 4.8.8
diagonal family.
Output: z3_31212/TSD31212_diagonal.csv (append-only) + stdout.
"""
import csv
import math
import os
import sys
import time

sys.stdout.reconfigure(encoding='utf-8')
BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from step3_dp_strips import build_subsystems, frontier_dp  # noqa: E402

OUT = os.path.join(BASE, 'z3_31212', 'TSD31212_diagonal.csv')


def append(rows):
    exists = os.path.exists(OUT)
    with open(OUT, 'a', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(['L', 'alphabet', 'engine', 'Omega_mid', 'Omega_cut',
                        'Omega', 'lnOmega', 'N_F', 'time_s'])
        w.writerows(rows)


def main():
    rows = []
    # binary diagonal L=5..8
    for L in (5, 6, 7, 8):
        sub = build_subsystems(L, L)
        t0 = time.time()
        om = frontier_dp(sub['mid_keys'], sub['mid_keysets'], sub['mid_eqs'],
                         label=f'mid {L}x{L}')
        oc = frontier_dp(sub['cut_keys'], sub['cut_keysets'], sub['cut_eqs'],
                         label=f'cut {L}x{L}')
        dt = time.time() - t0
        print(f'binary L={L}: mid={om} cut={oc} Omega={om*oc} '
              f'ln={math.log(om*oc):.4f} N_F={3*L*L} [{dt:.0f}s]', flush=True)
        rows.append([L, '{1,2}', 'dp', om, oc, om * oc, f'{math.log(om*oc):.6f}',
                     3 * L * L, f'{dt:.0f}'])
    append(rows)
    # ternary diagonal L=2,3 (DP)
    rows = []
    for L in (2, 3):
        sub = build_subsystems(L, L)
        t0 = time.time()
        om = frontier_dp(sub['mid_keys'], sub['mid_keysets'], sub['mid_eqs'],
                         label=f'mid3 {L}x{L}', vals=(0, 1, 2))
        oc = frontier_dp(sub['cut_keys'], sub['cut_keysets'], sub['cut_eqs'],
                         label=f'cut3 {L}x{L}', vals=(0, 1, 2))
        dt = time.time() - t0
        print(f'ternary L={L}: mid={om} cut={oc} Omega={om*oc} '
              f'ln={math.log(om*oc):.4f} N_F={3*L*L} [{dt:.0f}s]', flush=True)
        rows.append([L, '{1,2,3}', 'dp', om, oc, om * oc, f'{math.log(om*oc):.6f}',
                     3 * L * L, f'{dt:.0f}'])
    append(rows)
    print('done')


if __name__ == '__main__':
    main()
