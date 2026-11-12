# -*- coding: utf-8 -*-
"""Qualitative anchor of the pure metric layer for the two-value alphabet
({0,1} affine class) (2026-09-28).

Open issue: {4,6} is phase-separated with s ~ 0 at the arm-number layer
({222,223}; 0.0019 in the full P0 table), but that theorem relies on
arm-number out-degree counting (a vertex structure) and cannot be transferred
to the pure metric layer -- in the pure metric layer mixed tiles (e.g.
alternating (0,1,0,1,0,1)) satisfy both closure equations and exist legally.
The qualitative verdict for the two-value class (phase-separated / stripe
s=0 vs positive entropy) is provisionally read from the growth shape of this
size series; the final ruling requires TM / larger systems.

Scope: pure edge assignment + the two honeycomb closure equations (same as
edge_alphabet_affine_check, no arm-number constraints).
Assertion 1: Omega({0,1}) = Omega({4,6}) = Omega({0,2}) pointwise equal
(affine invariance: translation + rescaling).
Assertion 2: brute-force anchor at (2,2) cross-check.
Memory: DFS assigns edge by edge, never materializes a state table; peak <10 MB.
"""
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from edge_alphabet_affine_check import count_assignments, brute_count  # noqa: E402

BIN = [(0, 1), (4, 6), (0, 2)]      # same affine class, must be equal pointwise
REF = [(4, 5, 6)]                   # arithmetic ternary reference (3x3 = 22767)
SIZES = [(2, 2), (3, 2), (3, 3), (4, 2), (4, 3), (5, 2)]

if __name__ == '__main__':
    table = {}
    for alph in BIN + REF:
        row = []
        for (Lx, Ly) in SIZES:
            t0 = time.time()
            c = count_assignments(Lx, Ly, alph)
            row.append(c)
            print(f'  {alph} {Lx}x{Ly}: Omega={c}  ({time.time()-t0:.2f}s)', flush=True)
        table[alph] = row

    print()
    hdr = 'alphabet   | ' + ' | '.join(f'{s[0]}x{s[1]}' for s in SIZES)
    print(hdr)
    for alph, row in table.items():
        cells = []
        for c, (Lx, Ly) in zip(row, SIZES):
            F = Lx * Ly
            cells.append(f'{c:>10d}' if c > 0 else '         0')
        lnF = ' '.join(f'{math.log(c)/(Lx*Ly):.4f}' if c > 0 else '  n/a '
                       for c, (Lx, Ly) in zip(row, SIZES))
        print(f'{str(alph):>10} | ' + ' | '.join(cells))
        print(f'{"lnOmega/F":>10} | {lnF}')

    # assertion 1: affine classes equal pointwise
    ref_bin = table[(0, 1)]
    for alph in [(4, 6), (0, 2)]:
        assert table[alph] == ref_bin, f'affine class mismatch: {alph} {table[alph]} vs {ref_bin}'
    print('affine invariance (0,1) = (4,6) = (0,2): PASS')

    # assertion 2: brute-force anchor
    for alph in [(0, 1), (4, 5, 6)]:
        b = brute_count(2, 2, alph)
        assert b == table[alph][0], f'brute mismatch {alph}: {b} vs {table[alph][0]}'
        print(f'brute anchor {alph} (2x2): {b} == DFS PASS')

    print('ALL ASSERTS PASSED')
