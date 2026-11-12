# -*- coding: utf-8 -*-
"""
Step 1: dim_Q six-point adjudication (L = 1..6).

Background: the earlier "dim_Q = 11/19/29 linear in L" reading is mathematically
inconsistent (differences 8, 10 not constant at equal L); the sequence is exactly
L^2+3L+1 at L=2,3,4 (quadratic, extensive). This script recomputes the full L=1..6
sequence and adjudicates with exact Fraction RREF.

Method (reuses z4_3464/RTH_numerical_verification.py without modifying it):
  - geometry: RTH.build_31212(L) (truncated honeycomb L x L torus; Euler / face census /
    vertex valence / vertex figures (3,12,12) / all-ones reference solution asserted
    at build time).
  - dim_R = E - rank_Q(sqrt(3)), four-way cross-check (consistency asserted inside
    compute_dims): float SVD / mod-p double prime (F_p and F_{p^2}) / exact
    Q(sqrt(3)) QD Fraction elimination.
  - dim_Q = E - rank_Q (split-row matrix), three-way cross-check:
    (1) independent Fraction RREF (frac_rref_rank here; integer split rows,
        no float prescreen);
    (2) _rank_fraction of the compute_dims exact branch (split-row Fraction,
        consistency with mod p asserted internally);
    (3) RTH.rref_and_param (float prescreen + Fraction RREF, independent code path).
Acceptance gate: L=2,3,4 reproduce dim_Q = 11/19/29; dim_R = 3L^2+2 at all six points.
Criterion: L=5,6 = 41/55 confirms the quadratic-extensive branch (branch A);
      constant difference => linear (branch B, audit the RREF pipeline).

Memory estimate (<1 GB, minutes-scale runtime): at L=6 the split-row matrix is
<= 432 rows x 324 cols, ~1.4e5 Fraction objects (~15 MB), elimination
O(nr*nc*rank) ~ 3e7 Fraction operations.
"""
import sys, os, time, csv, math

sys.stdout.reconfigure(encoding='utf-8')
BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
Z464 = os.path.join(BASE, 'z4_3464')
sys.path.insert(0, Z464)
import RTH_numerical_verification as RTH  # noqa: E402
from fractions import Fraction as Fr  # noqa: E402


def frac_rref_rank(rows_int, ncol):
    """Independent Fraction RREF rank (integer-coefficient rows; independent of the
    suite's _rank_fraction / rref_and_param implementations)"""
    M = [[Fr(v) for v in r] for r in rows_int]
    nr = len(M)
    rank = 0
    for c in range(ncol):
        piv = next((i for i in range(rank, nr) if M[i][c] != 0), None)
        if piv is None:
            continue
        M[rank], M[piv] = M[piv], M[rank]
        pv = M[rank][c]
        M[rank] = [x / pv for x in M[rank]]
        for i in range(nr):
            if i != rank and M[i][c] != 0:
                f = M[i][c]
                M[i] = [u - f * v for u, v in zip(M[i], M[rank])]
        rank += 1
        if rank == nr:
            break
    return rank


def split_rows(t):
    """Integer-coefficient split-row matrix: per-face closure rows (xrow, yrow) split
    into the rational part (a) and the sqrt(3) part (b) (x2 scaled to integers).

    Direction semantics (confirmed from the L=1 row contents, counter-intuitive):
      mid edges (mid-segments of honeycomb edges, 3L^2 of them): directions = odd
        multiples of 30 deg (150+60m family)
        => cos = +/-sqrt(3)/2 (b part), sin = +/-1/2 (rational a part);
      cut edges (triangle-dodecagon shared edges, 6L^2 of them): directions on the
        60-deg lattice (0+60m family)
        => cos = +/-1, +/-1/2 (a part), sin = 0, +/-sqrt(3)/2 (b part).
    => natural block decomposition after splitting:
      cut block cut(xa+yb): cut-edge columns only (6L^2 variables); contains the
        triangle 2x2 equations (three edges equal, rank 2: triangle locking) +
        2 truncation-position equations per dodecagon.
      mid block mid(xb+ya): mid-edge columns only (3L^2 variables);
        2 mid-position equations per dodecagon.
    dim_Q = dim_cut + dim_mid (cross-checked against the full RREF).
    """
    xa = [[int(2 * x.a) for x in r] for r in t.xrows]
    xb = [[int(2 * x.b) for x in r] for r in t.xrows]
    ya = [[int(2 * y.a) for y in r] for r in t.yrows]
    yb = [[int(2 * y.b) for y in r] for r in t.yrows]
    cut = [r for r in (xa + yb) if any(v != 0 for v in r)]
    mid = [r for r in (xb + ya) if any(v != 0 for v in r)]
    return cut, mid


def main():
    out_csv = os.path.join(BASE, 'z3_31212', 'TSD31212_dimRQ_L1to6.csv')
    rows_csv = []
    q_seq, r_seq = [], []
    print('L  V   E   F   | dimR (4-way) | dimQ (3-way) | time')
    for L in (1, 2, 3, 4, 5, 6):
        t0 = time.time()
        t = RTH.build_31212(L)
        assert t.euler_ok(), f'L={L}: Euler check failed'
        assert (t.V, t.E, t.F) == (6 * L * L, 9 * L * L, 3 * L * L), (L, t.V, t.E, t.F)
        census = dict(sorted(t.face_census().items()))
        assert census == {3: 2 * L * L, 12: L * L}, (L, census)
        assert t.degs == [3], (L, t.degs)
        figs = t.vertex_figures()
        assert all(RTH.rot_norm(f) == (3, 12, 12) for f in figs), (L, figs)
        # dim_R / dim_Q: float SVD + mod-p double prime + exact Q(sqrt(3)) QD
        # (mutual consistency asserted inside compute_dims)
        dimR, dimQ, det = RTH.compute_dims(t.xrows, t.yrows, 3, t.E,
                                           exact_limit=400, label=f'31212 L={L}')
        assert dimR == 3 * L * L + 2, (L, dimR, det)
        # independent split-row Fraction RREF (no float prescreen): full system +
        # cut/mid block decomposition
        cut, mid = split_rows(t)
        rk_cut = frac_rref_rank(cut, t.E)
        rk_mid = frac_rref_rank(mid, t.E)
        dim_cut = 6 * L * L - rk_cut            # cut-edge subsystem (xa+yb; includes triangle locking)
        dim_mid = 3 * L * L - rk_mid            # mid-edge subsystem (xb+ya)
        dimQ_ind = dim_cut + dim_mid
        assert dimQ_ind == dimQ, (L, dimQ_ind, dimQ, det)
        # second independent path: float prescreen + Fraction RREF (parametrized)
        T, piv, free, dimQ_p2 = RTH.rref_and_param(t.xrows, t.yrows, t.E)
        assert dimQ_p2 == dimQ, (L, dimQ_p2, dimQ)
        q_seq.append(dimQ)
        r_seq.append(dimR)
        dt = time.time() - t0
        d1 = dimQ - q_seq[-2] if len(q_seq) > 1 else ''
        print(f'{L}  {t.V}  {t.E}  {t.F}  | dimR={dimR} {det} | dimQ={dimQ} '
              f'(ind {dimQ_ind} = cut {dim_cut} + mid {dim_mid}; p2 {dimQ_p2}) | {dt:.1f}s',
              flush=True)
        rows_csv.append([L, t.V, t.E, t.F, dimR,
                         det['dimR_float'], '|'.join(map(str, det['dimR_mod'])), det.get('dimR_exact', ''),
                         dimQ, det['dimQ_float'], '|'.join(map(str, det['dimQ_mod'])), det.get('dimQ_exact', ''),
                         dimQ_ind, dim_cut, dim_mid, dimQ_p2, d1, dt])
    # difference reading
    diffs = [q_seq[i + 1] - q_seq[i] for i in range(len(q_seq) - 1)]
    d2s = [diffs[i + 1] - diffs[i] for i in range(len(diffs) - 1)]
    print(f'dim_Q sequence L=1..6 = {q_seq}')
    print(f'first differences = {diffs}')
    print(f'second differences = {d2s}')
    quad = [L * L + 3 * L + 1 for L in range(1, 7)]
    print(f'L^2+3L+1 reference = {quad}')
    rows_csv.append(['', '', '', '', '', '', '', '', '', '', '', '', '',
                     '', '', '', f'first_diff={diffs};second_diff={d2s};quad_ref={quad}', ''])
    # acceptance gate and criterion
    assert q_seq[1:4] == [11, 19, 29], f'acceptance gate failed: L=2,3,4 dim_Q = {q_seq[1:4]} != [11,19,29]'
    if q_seq == quad:
        verdict = 'A: dim_Q = L^2+3L+1 quadratic-extensive (L=5,6 = 41/55 confirmed)'
    elif len(set(diffs)) == 1:
        verdict = f'B: constant difference {diffs[0]} => linear (audit the pipeline)'
    else:
        verdict = f'?: sequence {q_seq} neither L^2+3L+1 nor constant-difference; manual adjudication'
    print('verdict: ' + verdict, flush=True)
    rows_csv.append(['verdict', verdict, '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', ''])
    with open(out_csv, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['L', 'V', 'E', 'F', 'dimR', 'dimR_float', 'dimR_modp', 'dimR_exact',
                    'dimQ', 'dimQ_float', 'dimQ_modp', 'dimQ_exact', 'dimQ_indep_fracRREF',
                    'dimQ_cut_subsys', 'dimQ_mid_subsys', 'dimQ_rref_and_param',
                    'first_diff', 'runtime_s'])
        w.writerows(rows_csv)
    print(f'CSV written {out_csv}')


if __name__ == '__main__':
    main()
