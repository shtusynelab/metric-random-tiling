# -*- coding: utf-8 -*-
"""4.6.12 Z-count v4 (clean version): parametrized DFS + incremental
verification + partial-sum-bound pruning.

Method: Fraction RREF -> each pivot (dependent) variable is a (fractional)
linear combination of free parameters; parameters take {0,1,2} and pivot
values must land in {0,1,2} (non-integral fractions drop out
automatically). Gauge fixing: free column 0 pinned (removes the
constant-shift zero mode). DFS order clusters parameters by dependency
group (same-group parameters adjacent); incremental verification: a pivot
variable is checked once all of its parameters are assigned, partially
assigned ones are pruned with partial-sum interval bounds.

Validation: L=1 crosschecked against brute force 2349.
Memory: RREF 48 x 48 Fractions; DFS stack <= 20; < 100 MB, local machine.
"""
import math
import sys
import time
from collections import defaultdict
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from roundE_4612_zcount import build_reduced   # noqa

VALS = (0, 1, 2)


def rref_parametrize(n_vars, eqs):
    """Return (free_cols, deps): pivot variable p's value = the sum over
        deps[p] of (free_col, Fraction coefficient) products."""
    rows = [[Fraction(eq.get(v, 0)) for v in range(n_vars)]
            for eq in eqs if any(eq.values())]
    rank = 0
    pivots = {}
    for col in range(n_vars):
        piv = next((i for i in range(rank, len(rows)) if rows[i][col] != 0), None)
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
        # x_col = -sum_free r*x_free (the RREF row gives 0 = x_col + sum r*x_free)
        terms = [(fk, -rows[ri][fk]) for fk in free if rows[ri][fk] != 0]
        if not terms:
            continue
        den_lcm = 1
        for fk, c in terms:
            den_lcm = math.lcm(den_lcm, c.denominator)
        ints = [(fk, int(c * den_lcm)) for fk, c in terms]
        deps[col] = (den_lcm, ints)          # s_col = (sum coef*x)/L_p in {0,1,2}
    return free, deps


def count_param_dfs(L, gauge_fix=True, quiet=False, block=None):
    cls_of, eqs, all_cls, n_cls, dim, N, V, E, F = build_reduced(L, L)
    free, deps = rref_parametrize(n_cls, eqs)
    fixed = {}
    if gauge_fix and free:
        fixed[free[0]] = 0            # gauge: remove the constant-shift zero mode
        free_eff = free[1:]
    else:
        free_eff = free[:]
    nf = len(free_eff)
    pos = {fk: i for i, fk in enumerate(free_eff)}
    # pivot-variable dependencies (int-ized; pinned columns take value 0
    # => their contribution is identically 0)
    dep_list = []                      # [p, [(pos, int coef)], 0, n_terms]
    Lp_list = []
    for p, (Lp, terms) in deps.items():
        tt = [(pos[fk], c) for fk, c in terms if fk in pos]
        if len(tt) == 0:               # depends only on pinned columns => pivot identically 0, admissible
            continue
        dep_list.append([p, tt, 0, len(tt)])
        Lp_list.append(Lp)
    # DFS order: parameters clustered by the dependent variables they
    # first appear in (BFS on the dep graph)
    p_of_pos = defaultdict(list)
    for di, (p, tt, cp, nk) in enumerate(dep_list):
        for (pos_i, c) in tt:
            p_of_pos[pos_i].append(di)
    order = []
    seen_pos = set()
    for start in range(nf):
        if start in seen_pos:
            continue
        queue = [start]
        seen_pos.add(start)
        while queue:
            k = queue.pop(0)
            order.append(k)
            for di in p_of_pos.get(k, []):
                for (pos_i, c) in dep_list[di][1]:
                    if pos_i not in seen_pos:
                        seen_pos.add(pos_i)
                        queue.append(pos_i)
    assert len(order) == nf
    assign = [None] * nf
    Lp_of = Lp_list
    part = [cp for (p, tt, cp, nk) in dep_list]     # settled partial sum (int)
    nleft = [nk for (p, tt, cp, nk) in dep_list]
    count = 0
    bfix = {}                          # blocking: prefix parameters pinned
    if block is not None:
        for k, v in zip(order[:block[0]], block[1]):
            bfix[k] = v

    def dfs(d):
        nonlocal count
        if d == nf:
            count += 1
            return
        k = order[d]
        vs = (bfix[k],) if k in bfix else VALS
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
                tt = dep_list[di][1]
                Lp = Lp_of[di]
                if nleft[di] == 0:                 # closes at this step: S in {0, Lp, 2Lp}
                    S = part[di]
                    if S != 0 and S != Lp and S != 2 * Lp:
                        ok = False
                        break
                else:                              # int partial-sum bounds vs {0, Lp, 2Lp}
                    lo = part[di]
                    hi = part[di]
                    for (pos_i, c) in tt:
                        if assign[pos_i] is None:
                            if c > 0:
                                hi += 2 * c
                            else:
                                lo += 2 * c
                    if hi < 0 or lo > 2 * Lp:
                        ok = False
                        break
            if ok:
                dfs(d + 1)
            for (di, coef) in touched:
                part[di] -= coef * v
                nleft[di] += 1
            assign[k] = None
    t0 = time.time()
    sys.setrecursionlimit(10000)
    dfs(0)
    dt = time.time() - t0
    return count, nf, N, dt


if __name__ == '__main__':
    for L in [1, 2]:
        cnt, nf, N, dt = count_param_dfs(L)
        print(f'L={L}: free(gauge-fixed)={nf}  Omega={cnt}  '
              f's_Z={math.log(cnt)/(6*N):.4f}  ({dt:.1f}s)', flush=True)
