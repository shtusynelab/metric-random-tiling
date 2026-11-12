# -*- coding: utf-8 -*-
"""4.6.12 two-value {0,1} exact anchors (brute + parametrized DFS, binary).

Purpose: ground-truth anchors for the binary frontier DP
(roundE_4612_frontier_dp.py). Constraint matrix is value-domain independent
(homogeneous linear); by the monotonicity lemma the {0,1} count equals the
count for ANY two-value alphabet {a,b}.

Conventions:
  - full orbit = gauge-fixed x 2 (complement shift x -> 1-x is a free
    solution orbit because A*1 = 0 for equiangular face-closure rows);
  - s_01 = ln(Omega_full)/(6N), same full-orbit convention as the ternary
    s_Z series (Table S11).

Engines:
  E1 brute (L=1): enumerate all class variables {0,1}^n_cls, check all eqs.
  E2 paramdfs (L=1,2,3): RREF parametrization; binary closure S in {0, Lp};
     partial-sum bounds with per-variable max 1.
Regression gates (ternary, same code path):
  G1: ternary L=1 paramdfs(gauge) x 3 == 2349 (exact, archived).
  G2: ternary L=2 paramdfs(gauge) x 3 == 29,723,517 (exact, archived).
Crosscheck:
  X1: binary L=1 brute == 2 x paramdfs(gauge)  [free-class orbit check].

Memory < 500 MB, single core. Imports rely on CWD (no sys.path setup).
Output: RTH4612_binary_anchors.csv (append) + stdout.
"""
import csv
import itertools
import math
import os
import sys
import time
from collections import defaultdict
from fractions import Fraction

from roundE_4612_zcount import build_reduced

OUT_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'RTH4612_binary_anchors.csv')


def rref_parametrize(n_vars, eqs):
    """RREF -> (free_cols, deps): pivot var value = (sum coef*x_free)/Lp."""
    rows = [[Fraction(eq.get(v, 0)) for v in range(n_vars)]
            for eq in eqs if any(eq.values())]
    rank = 0
    pivots = {}
    for col in range(n_vars):
        piv = next((i for i in range(rank, len(rows)) if rows[i][col] != 0),
                   None)
        if piv is None:
            continue
        rows[rank], rows[piv] = rows[piv], rows[rank]
        pv = rows[rank][col]
        if pv != 1:
            rows[rank] = [x / pv for x in rows[rank]]
        for i in range(len(rows)):
            if i != rank and rows[i][col] != 0:
                fac = rows[i][col]
                rows[i] = [a - fac * b for a, b in zip(rows[i], rows[rank])]
        pivots[col] = rank
        rank += 1
        if rank == len(rows):
            break
    free = [c for c in range(n_vars) if c not in pivots]
    deps = {}
    for col, ri in pivots.items():
        terms = [(fk, -rows[ri][fk]) for fk in free if rows[ri][fk] != 0]
        if not terms:
            continue
        den_lcm = 1
        for fk, c in terms:
            den_lcm = math.lcm(den_lcm, c.denominator)
        deps[col] = (den_lcm, [(fk, int(c * den_lcm)) for fk, c in terms])
    return free, deps


_CACHE = {}   # (Lx, Ly) -> (eqs, n_cls, N, free, deps); per-worker reuse


def count_param_dfs(Lx, Ly=None, vals=(0, 1, 2), block=None):
    """Parametrized DFS, value-domain generic.

    Closure: S == 0 (mod Lp) and 0 <= S <= kmax*Lp (kmax = max(vals)).
    Ternary -> S in {0, Lp, 2Lp}; binary -> S in {0, Lp}.
    Gauge: free[0] pinned to 0 (removes constant-shift zero mode);
    full orbit = count x len(vals)  (shift group acts freely).
    block=(b, prefix): pin the first b BFS-order parameters (blocked
    parallel runs); count covers that block only.
    Returns (count_gauge, nf, n_cls, N, dt).
    """
    if Ly is None:
        Ly = Lx
    kmax = max(vals)
    hit = _CACHE.get((Lx, Ly))
    if hit is None:
        cls_of, eqs, all_cls, n_cls, dim, N, V, E, F = build_reduced(Lx, Ly)
        free, deps = rref_parametrize(n_cls, eqs)
        hit = (eqs, n_cls, N, free, deps)
        _CACHE[(Lx, Ly)] = hit
    eqs, n_cls, N, free, deps = hit
    free_eff = free[1:] if free else []
    nf = len(free_eff)
    pos = {fk: i for i, fk in enumerate(free_eff)}
    dep_list = []                      # [p, [(pos, coef)], part, nleft]
    Lp_list = []
    for p, (Lp, terms) in deps.items():
        tt = [(pos[fk], c) for fk, c in terms if fk in pos]
        if not tt:                     # depends only on pinned col -> const 0
            continue
        dep_list.append([p, tt, 0, len(tt)])
        Lp_list.append(Lp)
    p_of_pos = defaultdict(list)
    for di, rec in enumerate(dep_list):
        for (pos_i, c) in rec[1]:
            p_of_pos[pos_i].append(di)
    order = []
    seen = set()
    for start in range(nf):
        if start in seen:
            continue
        queue = [start]
        seen.add(start)
        while queue:
            k = queue.pop(0)
            order.append(k)
            for di in p_of_pos.get(k, []):
                for (pos_i, c) in dep_list[di][1]:
                    if pos_i not in seen:
                        seen.add(pos_i)
                        queue.append(pos_i)
    assert len(order) == nf
    assign = [None] * nf
    bfix = {}
    if block is not None:
        for k, v in zip(order[:block[0]], block[1]):
            bfix[k] = v
    part = [rec[2] for rec in dep_list]
    nleft = [rec[3] for rec in dep_list]
    count = 0

    def dfs(d):
        nonlocal count
        if d == nf:
            count += 1
            return
        k = order[d]
        vs = (bfix[k],) if k in bfix else vals
        for v in vs:
            assign[k] = v
            ok = True
            touched = []
            for di in p_of_pos.get(k, []):
                coef = dict(dep_list[di][1])[k]
                part[di] += coef * v
                nleft[di] -= 1
                touched.append((di, coef))
            for (di, coef) in touched:
                Lp = Lp_list[di]
                if nleft[di] == 0:
                    S = part[di]
                    if S % Lp != 0 or not (0 <= S <= kmax * Lp):
                        ok = False
                        break
                else:
                    lo = part[di]
                    hi = part[di]
                    for (pos_i, c) in dep_list[di][1]:
                        if assign[pos_i] is None:
                            if c > 0:
                                hi += kmax * c
                            else:
                                lo += kmax * c
                    if hi < 0 or lo > kmax * Lp:
                        ok = False
                        break
            if ok:
                dfs(d + 1)
            for (di, coef) in touched:
                part[di] -= coef * v
                nleft[di] += 1
            assign[k] = None

    t0 = time.time()
    sys.setrecursionlimit(100000)
    dfs(0)
    return count, nf, n_cls, N, time.time() - t0


def count_brute(L, vals=(0, 1)):
    """Brute force over ALL class variables (true full-orbit, L=1 only)."""
    cls_of, eqs, all_cls, n_cls, dim, N, V, E, F = build_reduced(L, L)
    eq_list = [list(eq.items()) for eq in eqs]
    cnt = 0
    for x in itertools.product(vals, repeat=n_cls):
        if all(sum(co * x[c] for c, co in eqd) == 0 for eqd in eq_list):
            cnt += 1
    return cnt, n_cls, N


def append_csv(row):
    exists = os.path.exists(OUT_CSV)
    with open(OUT_CSV, 'a', newline='') as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(['L', 'alphabet', 'engine', 'Omega', 'lnOmega',
                        's_per_face', 'nf', 'n_cls', 'N', 'time_s'])
        w.writerow(row)


def main():
    t_start = time.time()
    print('=== regression gates (ternary, same code path) ===', flush=True)
    # G1: ternary L=1
    cnt, nf, n_cls, N, dt = count_param_dfs(1, vals=(0, 1, 2))
    full = cnt * 3
    print(f'G1 ternary L=1: gauge={cnt} x3 = {full} (expect 2349) '
          f'[{dt:.1f}s]', flush=True)
    assert full == 2349, ('G1 FAIL', full)
    append_csv([1, '{0,1,2}', 'paramdfs', full, f'{math.log(full):.4f}',
                f'{math.log(full) / 6:.6f}', nf, n_cls, N, f'{dt:.1f}'])
    # G2: ternary L=2
    cnt, nf, n_cls, N, dt = count_param_dfs(2, vals=(0, 1, 2))
    full = cnt * 3
    print(f'G2 ternary L=2: gauge={cnt} x3 = {full} '
          f'(expect 29723517) [{dt:.1f}s]', flush=True)
    assert full == 29723517, ('G2 FAIL', full)
    append_csv([2, '{0,1,2}', 'paramdfs', full, f'{math.log(full):.4f}',
                f'{math.log(full) / 12:.6f}', nf, n_cls, N, f'{dt:.1f}'])

    print('=== binary anchors ===', flush=True)
    # X1: L=1 brute vs paramdfs
    b1, n_cls, N = count_brute(1, vals=(0, 1))
    cnt, nf, n_cls, N, dt = count_param_dfs(1, vals=(0, 1))
    full2 = cnt * 2
    print(f'X1 binary L=1: brute={b1}  paramdfs gauge={cnt} x2 = {full2} '
          f'[{dt:.1f}s]', flush=True)
    assert b1 == full2, ('X1 FAIL', b1, full2)
    s1 = math.log(b1) / 6
    print(f'    Omega_01(1x1) = {b1}  s_01 = {s1:.4f}', flush=True)
    append_csv([1, '{0,1}', 'brute', b1, f'{math.log(b1):.4f}',
                f'{s1:.6f}', 0, n_cls, N, '0.0'])
    append_csv([1, '{0,1}', 'paramdfs', full2, f'{math.log(full2):.4f}',
                f'{math.log(full2) / 6:.6f}', nf, n_cls, N, f'{dt:.1f}'])
    # L=2 binary
    cnt, nf, n_cls, N, dt = count_param_dfs(2, vals=(0, 1))
    full = cnt * 2
    s2 = math.log(full) / 24
    print(f'binary L=2: gauge={cnt} x2 = {full}  ln={math.log(full):.4f}  '
          f's_01={s2:.4f}  [{dt:.1f}s]', flush=True)
    assert full <= 29723517, ('monotonicity violated', full)
    append_csv([2, '{0,1}', 'paramdfs', full, f'{math.log(full):.4f}',
                f'{s2:.6f}', nf, n_cls, N, f'{dt:.1f}'])
    # L=3 binary (heavier; single core)
    t0 = time.time()
    cnt, nf, n_cls, N, dt = count_param_dfs(3, vals=(0, 1))
    full = cnt * 2
    s3 = math.log(full) / 54
    print(f'binary L=3: gauge={cnt} x2 = {full}  ln={math.log(full):.4f}  '
          f's_01={s3:.4f}  [{dt:.1f}s]', flush=True)
    assert full <= 6_566_000_000_000, ('monotonicity vs ternary L=3', full)
    append_csv([3, '{0,1}', 'paramdfs', full, f'{math.log(full):.4f}',
                f'{s3:.6f}', nf, n_cls, N, f'{dt:.1f}'])
    print(f'ALL GATES PASS, total {time.time() - t_start:.1f}s', flush=True)


if __name__ == '__main__':
    main()
