# -*- coding: utf-8 -*-
"""Step 2: exact binary {1,2} anchor counts.

Branch A established (dim_Q = L^2+3L+1 quadratic-extensive, Step 1) => by the
manuscript's own criterion, the Z-adjudication of 3.12.12 is a counting problem.
This script provides binary {1,2} (internal labels {0,1}, affine-shift invariant)
anchors: Omega_01(L) and s_01(L) = ln Omega/(3L^2) (per-face convention, F = 3L^2).

Engine and gates:
  G0 structural gate: L=1..3 n_cls = 5L^2, eqs = 4L^2, dim = L^2+3L+1
     (L=1 eqs=0: opposite dodecagon edges coincide and cancel; legal).
  X1 orbit adjudication: L=1 brute (all class variables, full orbit) ==
     2 x paramdfs (gauge) => the complement shift s -> 1-s acts freely, orbit
     factor x2 (independently derived + cross-checked against brute force,
     not copied from 4.6.12).
  L=2/3: paramdfs binary (gauge x2 = full orbit).
  Size-extrapolation preview: lnOmega/(3L^2) sequence (decay shape compared
  with 4.6.12).

Memory < 100 MB, single core, L=3 expected minutes. Output:
  z3_31212/TSD31212_binary_anchors.csv + stdout.
"""
import csv
import math
import os
import sys
import time

sys.stdout.reconfigure(encoding='utf-8')
BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib31212 as LB  # noqa: E402

OUT_CSV = os.path.join(BASE, 'z3_31212', 'TSD31212_binary_anchors.csv')


def append_csv(row):
    exists = os.path.exists(OUT_CSV)
    with open(OUT_CSV, 'a', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(['L', 'alphabet', 'engine', 'Omega', 'lnOmega',
                        's_per_face(ln/3L2)', 'nf', 'n_cls', 'F', 'time_s',
                        'note'])
        w.writerow(row)


def main():
    print('=== G0 structural gate (class system vs Step 1 RREF adjudication) ===')
    for L in (1, 2, 3):
        cls_val, eqs, all_cls, n_cls, dim, N, V, E, F = LB.build_reduced(L)
        assert n_cls == 5 * L * L, (L, n_cls)
        assert dim == L * L + 3 * L + 1, (L, dim)
        if L >= 2:
            assert len(eqs) == 4 * L * L, (L, len(eqs))
        print(f'  L={L}: n_cls={n_cls} eqs={len(eqs)} dim={dim} PASS')

    print('=== X1 orbit adjudication: L=1 brute vs paramdfs ===')
    b1, n_cls, N, dt_b = LB.count_brute(1, vals=(0, 1))
    cnt, nf, n_cls, N, dt_p = LB.count_param_dfs(1, vals=(0, 1))
    full2 = cnt * 2
    print(f'  brute (all class variables, full orbit) = {b1}  paramdfs gauge = {cnt} x2 = {full2} '
          f'[{dt_b:.1f}s / {dt_p:.1f}s]')
    assert b1 == full2, ('X1 FAIL orbit factor', b1, full2)
    s1 = math.log(b1) / 3
    print(f'  Omega_01(1x1) = {b1}  s_01 = ln/(3L^2) = {s1:.4f} PASS')
    append_csv([1, '{1,2}', 'brute', b1, f'{math.log(b1):.4f}', f'{s1:.6f}',
                0, n_cls, 3, f'{dt_b:.1f}', 'all-class-variable enumeration = full orbit'])
    append_csv([1, '{1,2}', 'paramdfs', full2, f'{math.log(full2):.4f}',
                f'{math.log(full2) / 3:.6f}', nf, n_cls, 3, f'{dt_p:.1f}',
                'gauge x2 (complement-shift orbit, adjudicated against L=1 brute)'])

    print('=== binary anchors L=2,3 ===')
    for L in (2, 3):
        cnt, nf, n_cls, N, dt = LB.count_param_dfs(L, vals=(0, 1))
        full = cnt * 2
        s = math.log(full) / (3 * L * L)
        print(f'  L={L}: gauge={cnt} x2 = {full}  ln={math.log(full):.4f}  '
              f's_01={s:.4f}  nf={nf}  [{dt:.1f}s]')
        append_csv([L, '{1,2}', 'paramdfs', full, f'{math.log(full):.4f}',
                    f'{s:.6f}', nf, n_cls, 3 * L * L, f'{dt:.1f}',
                    'gauge x2 (complement-shift orbit)'])
    print('ALL GATES PASS')


if __name__ == '__main__':
    main()
