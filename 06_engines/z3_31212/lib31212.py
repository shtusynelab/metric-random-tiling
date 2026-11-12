# -*- coding: utf-8 -*-
"""3.12.12 split-row class system (shared kernel for binary anchors / width scaling).

Structure (derived from the Step 1 block decomposition, verified numerically in code):
  after row splitting the constraints decouple into two independent subsystems by
  edge type (direction semantics confirmed by inspecting the L=1 rows):
    mid block mid(xb+ya): mid-hexagon edges (dodecagon-dodecagon shared edges,
          3L^2 of them; directions = odd multiples of 30 deg => cos = +/-sqrt(3)/2
          (b part), sin = +/-1/2 (rational part));
          equations = 2 per dodecagon (mid positions), no a-priori rank shrink,
          mid edges are independent variables one by one.
          Step 1 numerics: subsystem dimension = L^2+2 (quadratic, extensive)
          => entropy carrier.
    cut block cut(xa+yb): truncation edges (triangle-dodecagon shared edges,
          6L^2 of them; directions on the 60-deg lattice => cos rational,
          sin = +/-sqrt(3)/2 (b part));
          the 3 edges of each triangle enter two rows each = all-equal constraint
          (rank 2: triangle locking) => the union collapses to 2L^2 classes;
          the triangle-own equations vanish after collapse (auto-absorbed);
          remaining equations = 2 truncation positions per dodecagon.
          Step 1 numerics: dimension = 3L-1 (linear, subextensive).
  class variables n_cls = 3L^2 (mid) + 2L^2 (triangle classes) = 5L^2;
  equations = 4L^2 (4 per dodecagon).
  dim_Q = 5L^2 - rank = L^2+3L+1 (Step 1 six-point adjudication) => rank = 4L^2-3L-1.

Orbit structure (independently derived, adjudicated against brute-force anchors):
  - constant shift: every split row has coefficient sum = 0 (a/b parts of the
    all-ones reference solution) => shifts are free directions of the solution
    space => counts are independent of alphabet position,
    Omega({1,2}) = Omega({0,1}) (internal labels {0,1}).
  - negation s -> -s (homogeneous linear) => on {0,1} the complement shift
    s -> 1-s acts freely, orbit factor x2 (same type as 4.6.12, but adjudicated
    independently against the L=1 brute force here, not copied).
"""
import math
import sys
import os
from collections import defaultdict
from fractions import Fraction

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(BASE, 'z4_3464'))
import RTH_numerical_verification as RTH  # noqa: E402

_CACHE = {}


def build_reduced(L):
    """3.12.12 L x L torus split-row class system.

    Returns (cls_val, eqs, all_cls, n_cls, dim, N, V, E, F):
      cls_val: eid -> class id (even edge: its own class; odd edge: the triangle class it belongs to)
      eqs: [dict{cls: int_coef}] (scaled x2 to integers, all-integer coefficients)
      dim: solution-space dimension verified by exact Fraction RREF rank
    """
    if L in _CACHE:
        return _CACHE[L]
    t = RTH.build_31212(L)
    N = L * L
    assert (t.V, t.E, t.F) == (6 * N, 9 * N, 3 * N)
    assert dict(sorted(t.face_census().items())) == {3: 2 * N, 12: N}
    size_of = [len(f[0]) for f in t.faces]
    # edge -> incident faces (to attach triangles to odd edges)
    edge_faces = defaultdict(list)
    for fi, f in enumerate(t.faces):
        for (eid, sgn) in t.face_edges[fi]:
            edge_faces[eid].append(fi)
    assert all(len(v) == 2 for v in edge_faces.values())
    # equation collapse: union of the 3 truncation edges of each triangle (mid-segment edges not collapsed)
    parent = list(range(t.E))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for fi, f in enumerate(t.faces):
        if size_of[fi] == 3:
            ids = [eid for (eid, s) in t.face_edges[fi]]
            assert len(ids) == 3
            union(ids[0], ids[1])
            union(ids[1], ids[2])
    cls_of = {}
    cls_val = {}
    for eid in range(t.E):
        r = find(eid)
        if r not in cls_of:
            cls_of[r] = len(cls_of)
        cls_val[eid] = cls_of[r]
    n_mid = len({cls_val[e] for e in range(t.E)
                 if size_of[[fi for fi in edge_faces[e]][0]] == 12
                 and size_of[[fi for fi in edge_faces[e]][1]] == 12})
    n_cut = len(cls_of) - n_mid
    n_cls = len(cls_of)
    assert n_cls == 5 * N, (n_cls, 5 * N)
    assert n_mid == 3 * N and n_cut == 2 * N, (n_mid, n_cut, N)
    # equations: 4 per dodecagon (even part 2 + odd part 2); triangle equations must vanish automatically
    eqs = []
    for fi, f in enumerate(t.faces):
        rows = [defaultdict(int) for _ in range(4)]
        for (eid, sgn) in t.face_edges[fi]:
            c = cls_val[eid]
            # the xrows/yrows coefficients already include the edge direction sign (sgn) --
            # use them directly, do not multiply by k again
            # (double-sign caution: masked at L=1 where all triangle sgn are -1, exposed at L=2)
            rows[0][c] += 2 * t.xrows[fi][eid].a   # cut block X: rational part of xrow (cut classes only)
            rows[1][c] += 2 * t.yrows[fi][eid].b   # cut block Y: sqrt(3) part of yrow (cut classes only)
            rows[2][c] += 2 * t.xrows[fi][eid].b   # mid block X: sqrt(3) part of xrow (mid classes only)
            rows[3][c] += 2 * t.yrows[fi][eid].a   # mid block Y: rational part of yrow (mid classes only)
        # pairing rationale: cut edges (60-deg lattice) have rational cos / sin = +/-sqrt(3)/2;
        # mid edges (odd multiples of 30 deg) have cos = +/-sqrt(3)/2 / rational sin
        # => (xrow.a, yrow.b) contains only cut columns, (xrow.b, yrow.a) only mid columns.
        # triangles (all cut edges): rows[2]/[3] vanish identically; rows[0]/[1] coefficient
        # sums vanish after collapse (60-deg direction family with pairwise 120-deg
        # directions, sum cos = sum sin = 0).
        if size_of[fi] == 3:
            # triangle (all cut edges, one class after collapse): rows[2]/[3] identically
            # empty; rows[0]/[1] coefficient sums vanish (sum cos = sum sin = 0;
            # defaultdict += 0 leaves zero-valued keys, so test by value)
            assert all(v == 0 for r in rows for v in r.values()), 'triangle equations did not vanish as expected'
            continue
        assert size_of[fi] == 12
        for r in rows:
            r2 = {k: v for k, v in r.items() if v != 0}
            if r2:
                eqs.append(dict(r2))
    # dimension verification (Fraction RREF)
    # all_cls must contain all class ids (0..n_cls-1): extracting from eqs would miss the
    # L=1 zero-equation case and free classes appearing in no equation (the latter are
    # legitimate {0,1}^dim degrees of freedom)
    all_cls = list(range(n_cls))
    mat = [[Fraction(eq.get(c, 0)) for c in all_cls] for eq in eqs]
    rk = 0
    for col in range(len(all_cls)):
        piv = next((i for i in range(rk, len(mat)) if mat[i][col] != 0), None)
        if piv is None:
            continue
        mat[rk], mat[piv] = mat[piv], mat[rk]
        pv = mat[rk][col]
        if pv != 1:
            mat[rk] = [x / pv for x in mat[rk]]
        for i in range(len(mat)):
            if i != rk and mat[i][col] != 0:
                fac = mat[i][col]
                mat[i] = [a - fac * b for a, b in zip(mat[i], mat[rk])]
        rk += 1
        if rk == len(mat):
            break
    dim = n_cls - rk
    _CACHE[L] = (cls_val, eqs, all_cls, n_cls, dim, N, t.V, t.E, t.F)
    return _CACHE[L]


def rref_parametrize(n_vars, eqs):
    """RREF -> (free_cols, deps): pivot variable value = (sum coef*x_free)/Lp (integer-scaled)."""
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
        terms = [(fk, -rows[ri][fk]) for fk in free if rows[ri][fk] != 0]
        if not terms:
            continue
        den_lcm = 1
        for fk, c in terms:
            den_lcm = math.lcm(den_lcm, c.denominator)
        deps[col] = (den_lcm, [(fk, int(c * den_lcm)) for fk, c in terms])
    return free, deps


def count_param_dfs(L, vals=(0, 1), block=None):
    """Parametrized DFS (binary/ternary): pin free[0]=0 to fix the gauge; full orbit = count * len(vals).

    block=(b, prefix): pin the first b parameters in BFS order (for blocked parallel runs).
    Returns (count_gauge, nf, n_cls, N, dt).
    """
    kmax = max(vals)
    hit = _CACHE.get(L)
    if hit is None:
        build_reduced(L)
        hit = _CACHE[L]
    cls_val, eqs, all_cls, n_cls, dim, N, V, E, F = hit
    free, deps = rref_parametrize(n_cls, eqs)
    free_eff = free[1:] if free else []
    nf = len(free_eff)
    pos = {fk: i for i, fk in enumerate(free_eff)}
    dep_list = []
    Lp_list = []
    for p, (Lp, terms) in deps.items():
        tt = [(pos[fk], c) for fk, c in terms if fk in pos]
        if not tt:
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

    import time
    t0 = time.time()
    sys.setrecursionlimit(100000)
    dfs(0)
    return count, nf, n_cls, N, time.time() - t0


def count_brute(L, vals=(0, 1)):
    """Brute-force enumeration over all class variables (true full orbit; used for L=1)."""
    import itertools
    import time
    cls_val, eqs, all_cls, n_cls, dim, N, V, E, F = build_reduced(L)
    idx = {c: i for i, c in enumerate(all_cls)}
    eq_list = [[(idx[c], co) for c, co in eq.items()] for eq in eqs]
    cnt = 0
    t0 = time.time()
    for x in itertools.product(vals, repeat=len(all_cls)):
        if all(sum(co * x[i] for i, co in eqd) == 0 for eqd in eq_list):
            cnt += 1
    return cnt, n_cls, N, time.time() - t0
