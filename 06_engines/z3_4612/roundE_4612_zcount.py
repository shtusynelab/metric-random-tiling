# -*- coding: utf-8 -*-
"""4.6.12 exact Z-count (reduced-system DFS).

Structure after the row-split reduction (analytic; verified in code):
  square (4-walk) opposite sides are equal => SD(e,f)=SD(e,f'),
  SH(e,u)=SH(e,w): SD/SH values attach to the honeycomb edge e;
  duodecagon (12-walk) opposite diagonals are equal (position i <-> i+6)
  => HD values attach to diagonal pairs.
  => equality contraction: 9N class variables (SD 3N + SH 3N + HD 3N).
Remaining equations (face-local):
  duodecagon: 2 SD balance equations (coefficients +-1, +-2; the alpha/beta
    rows of the PAIR_COS/SIN lookup tables cancel to zero on HD classes);
  hexagon: 2 rational equations (3 SH classes + 3 HD classes,
    coefficients in {0,+-1});
  square: equations fully absorbed by the equalities (coefficients cancel).
Count = {0,1,2}^(9N) satisfying all reduced equations (affine and
translation invariant; the {4,5,6} alphabet gives the same count).
Verification gates: number of classes = 9N; reduced-equation rank = 5N-4
(dim = 9N - rank = 4N+4, consistent with batch 1).

Memory budget: class variables <= 54 (L=3); DFS stack + equation partial
sums < 50 MB; local machine.
Output: Omega_Z(L) and s_Z(L) = ln Omega/(6N); thick/thin read.
"""
import io
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

# stdout encoding is ensured by PYTHONIOENCODING (do not re-wrap: the
# import chain closes the stream)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from roundD_4612_numeric import build_torus, extract_faces, PAIR_COS, PAIR_SIN  # noqa


def build_reduced(Lx, Ly):
    vid, edges, V, E, pts = build_torus(Lx, Ly)
    faces = extract_faces(edges)
    N = Lx * Ly
    assert V == 12 * N and E == 18 * N and len(faces) == 6 * N
    # edge -> incident face size pair => type
    edge_faces = defaultdict(list)
    for fi, f in enumerate(faces):
        for h in f:
            edge_faces[h[3]].append(fi)
    assert all(len(v) == 2 for v in edge_faces.values())
    size_of = [len(f) for f in faces]
    etype = {}
    for eid, (f1, f2) in edge_faces.items():
        s1, s2 = sorted((size_of[f1], size_of[f2]))
        etype[eid] = {(4, 12): 'SD', (4, 6): 'SH', (6, 12): 'HD'}[(s1, s2)]
    # equality pairing: square (0,2),(1,3); duodecagon (i, i+6)
    parent = list(range(E))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra
    n_pair = 0
    for f in faces:
        n = len(f)
        if n == 4:
            for i in (0, 1):
                union(f[i][3], f[(i + 2) % 4][3])
                n_pair += 1
        elif n == 12:
            pass   # duodecagon has no forced equality: HD diagonals satisfy
                   # sum relations (L3+L5=L9+L11 type, given by
                   # beta_cos - alpha_sin), not pairwise equality -- HD
                   # stays as independent variables
    cls_of = {}
    for eid in range(E):
        r = find(eid)
        if r not in cls_of:
            cls_of[r] = len(cls_of)
    n_cls = len(cls_of)
    cnt_by_type = defaultdict(int)
    for eid in range(E):
        cnt_by_type[etype[eid]] += 1
    n_cls_by_type = defaultdict(set)
    for eid in range(E):
        n_cls_by_type[etype[eid]].add(cls_of[find(eid)])
    # SD/SH after square contraction = 3N classes; HD (no equalities) =
    # 6N independent variables
    assert n_cls == 12 * N, (n_cls, 12 * N)
    assert len(n_cls_by_type['SD']) == 3 * N and len(n_cls_by_type['SH']) == 3 * N \
        and len(n_cls_by_type['HD']) == 6 * N, \
        {t: len(v) for t, v in n_cls_by_type.items()}
    # reduced equations: 4 rows per face (alpha_c, beta_c, alpha_s,
    # beta_s); coefficients summed per class, zeros dropped
    eqs = []
    for f in faces:
        rows = [defaultdict(int) for _ in range(4)]
        for (uu, vv, bdir, eid) in f:
            c = cls_of[find(eid)]
            ac, bc = PAIR_COS[bdir]
            as_, bs = PAIR_SIN[bdir]
            rows[0][c] += ac
            rows[1][c] += bc
            rows[2][c] += as_
            rows[3][c] += bs
        for r in rows:
            r = {k: v for k, v in r.items() if v != 0}
            if r:
                eqs.append(dict(r))
    # rank verification (Fraction elimination)
    from fractions import Fraction
    all_cls = sorted({c for eq in eqs for c in eq})
    rows_f = [[Fraction(eq.get(c, 0)) for c in all_cls] for eq in eqs]
    rk = 0
    mat = rows_f
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
    assert dim == 4 * N + 4, (dim, 4 * N + 4)
    if Lx >= 2:
        assert len(eqs) == 8 * N, (len(eqs), 8 * N)   # duodecagon 4N + hexagon 4N (degenerate merge at L=1)
    return cls_of, eqs, all_cls, n_cls, dim, N, V, E, len(faces)


def count_solutions(n_cls, eqs, all_cls, vals=(0, 1, 2)):
    """DFS + interval pruning. Close-early equation order: greedily pick
        the equation with the fewest unassigned variables."""
    cls_idx = {c: i for i, c in enumerate(all_cls)}
    n = len(all_cls)
    eqs_i = [([(cls_idx[c], co) for c, co in eq.items()]) for eq in eqs]
    # greedy variable order: repeatedly pick the equation with the fewest
    # undetermined variables and add its undetermined variables
    order = []
    placed = set()
    open_eqs = set(range(len(eqs_i)))
    while len(order) < n:
        best_eq, best_k = None, 10 ** 9
        for ei in open_eqs:
            k = sum(1 for c, _ in eqs_i[ei] if c not in placed)
            if k < best_k:
                best_k, best_eq = k, ei
        if best_eq is None:                       # remaining variables appear in no equation
            order.extend(c for c in range(n) if c not in placed)
            break
        for c, _ in eqs_i[best_eq]:
            if c not in placed:
                placed.add(c)
                order.append(c)
        open_eqs.discard(best_eq)
    pos_in_order = {c: i for i, c in enumerate(order)}
    # equation variables sorted by order; last = position of the
    # last-decided variable
    eq_data = []
    for eq in eqs_i:
        terms = sorted(((pos_in_order[c], co) for c, co in eq))
        eq_data.append(terms)
    # equations each variable participates in (to update after its order position)
    eqs_of_var = defaultdict(list)
    for ei, terms in enumerate(eq_data):
        for (pos, co) in terms:
            eqs_of_var[pos].append((ei, co))
    kvals = list(vals)
    partial = [0] * len(eq_data)
    remain = [len(t) for t in eq_data]
    assigned_set = set()
    count = 0

    def feasible_fast(ei, d):
        lo = hi = 0
        for (pos, co) in eq_data[ei]:
            if pos <= d:
                continue
            if co > 0:
                lo += co * kvals[0]
                hi += co * kvals[-1]
            else:
                lo += co * kvals[-1]
                hi += co * kvals[0]
        return lo <= -partial[ei] <= hi

    def dfs(d):
        nonlocal count
        if d == n:
            count += 1
            return
        for v in kvals:
            ok = True
            touched = []
            for (ei, co) in eqs_of_var[d]:
                partial[ei] += co * v
                touched.append((ei, co))       # store the coef pair: undo reuses this co
            for (ei, co) in touched:
                if remain[ei] == 1:               # closes at this step
                    if partial[ei] != 0:
                        ok = False
                        break
                elif not feasible_fast(ei, d):
                    ok = False
                    break
            if ok:
                for (ei, co) in touched:
                    remain[ei] -= 1
                assigned_set.add(d)
                dfs(d + 1)
                assigned_set.discard(d)
                for (ei, co) in touched:
                    remain[ei] += 1
            for (ei, co) in touched:
                partial[ei] -= co * v
            if not ok:
                continue

    t0 = time.time()
    sys.setrecursionlimit(10000)
    dfs(0)
    return count, time.time() - t0


if __name__ == '__main__':
    for L in [2, 3]:
        t0 = time.time()
        cls_of, eqs, all_cls, n_cls, dim, N, V, E, F = build_reduced(L, L)
        t1 = time.time()
        cnt, dt = count_solutions(n_cls, eqs, all_cls)
        sZ = math.log(cnt) / (6 * N) if cnt > 0 else float('nan')
        print(f"L={L}: N={N} classes={n_cls} eqs={len(eqs)} dim={dim} "
              f"(build {t1-t0:.1f}s)  Omega_Z={cnt}  ln={math.log(cnt):.3f}  "
              f"s_Z=ln/(6N)={sZ:.4f}  (count {dt:.1f}s)", flush=True)
        print(f"   upper bound dim*ln3/F = {dim*math.log(3)/(6*N):.4f}   "
              f"reference: s_metric(honeycomb)=0.4504", flush=True)
