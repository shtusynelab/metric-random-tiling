# -*- coding: utf-8 -*-
"""TM debug 1: locate the 18x discrepancy between Tr(T^L) and exact DP counts
(W=2, k=3).
All lightweight (<1 GB): 81-state matrix, 18-variable RREF, ~1e5 enumerations.
"""
import sys, os, math, itertools
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from collections import defaultdict
from fractions import Fraction
import numpy as np
import step7_tm as TM
from step3_dp_strips import build_subsystems

W, L = 2, 3

print('=== 1) k=2 / k=3 Tr(T^L) vs DP ===')
for k, dp3, dp5 in ((2, 64, 256), (3, 1017, 16041)):
    tabs = TM.build_tables(W, k)
    TM.G.clear()
    TM.G.update(W=W, k=k, tabs=tabs)
    n = k ** (2 * W)
    T = np.zeros((n, n))
    for r in range(n):
        for (f, t, c) in TM.enum_row(r):
            T[t, f] += c
    for Lc, dp in ((3, dp3), (5, dp5)):
        tr = round(np.trace(np.linalg.matrix_power(T, Lc)))
        print(f'k={k} L={Lc}: Tr(T^L)={tr}  DP={dp}  {"MATCH" if tr == dp else "MISMATCH x" + str(round(dp / max(tr, 1), 2))}')

print()
print('=== 2) equation-system cross-check: step7 extraction vs build_subsystems (normalized to coefficient multisets) ===')
sub = build_subsystems(W, L)
# build_subsystems: mid_eqs per dodecagon index (face order same as build_31212_rect)
ij, info, eqs = TM.extract_geometry(W, L)
# build_subsystems face order = t.faces order; step7 info keys fi use the same face order (OK)
ref = []
for fi in range(sub['n_dodec']):
    for d in sub['mid_eqs'][fi]:
        if d:
            ref.append(tuple(sorted(d.items())))
mine = []
for fi in sorted(info):
    ex, ey = eqs[fi]
    for d in (ex, ey):
        if d:
            mine.append(tuple(sorted(d.items())))
print(f'ref eqs={len(ref)}  mine eqs={len(mine)}')
# key spaces differ (mid_of vs raw eid) -> compare by coefficient multisets
from collections import Counter
cref = Counter(tuple(sorted((c for _, c in e))) for e in ref)
cmine = Counter(tuple(sorted((c for _, c in e))) for e in mine)
print('coefficient multiset distribution ref :', dict(cref))
print('coefficient multiset distribution mine:', dict(cmine))
# structural comparison: nonzero-key count per equation and its position pattern among the six keys
# ref key order unknown -> compare only the nonzero-key count distribution
nref = Counter(len(e) for e in ref)
nmine = Counter(len(e) for e in mine)
print('nonzero-key count distribution ref :', dict(sorted(nref.items())))
print('nonzero-key count distribution mine:', dict(sorted(nmine.items())))

print()
print('=== 3) RREF + enumeration directly on the step7 equations (independent of the transfer logic) ===')
eqs_list = []
keys = set()
for fi in sorted(info):
    ex, ey = eqs[fi]
    for d in (ex, ey):
        if d:
            eqs_list.append(dict(d))
            keys.update(d)
keys = sorted(keys)
mat = [[Fraction(eqs_list[r].get(c, 0)) for c in keys] for r in range(len(eqs_list))]
mat = [r for r in mat if any(v != 0 for v in r)]
rk = 0
piv = []
for col in range(len(keys)):
    p = next((i for i in range(rk, len(mat)) if mat[i][col] != 0), None)
    if p is None:
        continue
    mat[rk], mat[p] = mat[p], mat[rk]
    pv = mat[rk][col]
    mat[rk] = [x / pv for x in mat[rk]]
    for i in range(len(mat)):
        if i != rk and mat[i][col] != 0:
            f = mat[i][col]
            mat[i] = [a - f * b for a, b in zip(mat[i], mat[rk])]
    piv.append(col)
    rk += 1
    if rk == len(mat):
        break
free = [c for c in range(len(keys)) if c not in set(piv)]
print(f'keys={len(keys)} rank={rk} dim={len(keys)-rk}')
deps = []
for ri, col in enumerate(piv):
    deps.append([(j, mat[ri][j]) for j in free if mat[ri][j] != 0])
cnt = 0
for assign in itertools.product((0, 1, 2), repeat=len(free)):
    val = dict(zip(free, assign))
    ok = True
    for terms in deps:
        v = -sum(co * val[j] for j, co in terms)
        if v.denominator != 1 or not (0 <= v <= 2):
            ok = False
            break
    if ok:
        cnt += 1
print(f'Omega(step7-equation RREF enumeration, k=3, W=2, L=3) = {cnt}  (DP: 1017)')

print()
print('=== 4) same enumeration on the build_subsystems equations (self-check of the checker) ===')
eqs_list = []
keys = set()
for fi in range(sub['n_dodec']):
    for d in sub['mid_eqs'][fi]:
        if d:
            eqs_list.append(dict(d))
            keys.update(d)
keys = sorted(keys)
mat = [[Fraction(eqs_list[r].get(c, 0)) for c in keys] for r in range(len(eqs_list))]
mat = [r for r in mat if any(v != 0 for v in r)]
rk = 0
piv = []
for col in range(len(keys)):
    p = next((i for i in range(rk, len(mat)) if mat[i][col] != 0), None)
    if p is None:
        continue
    mat[rk], mat[p] = mat[p], mat[rk]
    pv = mat[rk][col]
    mat[rk] = [x / pv for x in mat[rk]]
    for i in range(len(mat)):
        if i != rk and mat[i][col] != 0:
            f = mat[i][col]
            mat[i] = [a - f * b for a, b in zip(mat[i], mat[rk])]
    piv.append(col)
    rk += 1
    if rk == len(mat):
        break
free = [c for c in range(len(keys)) if c not in set(piv)]
print(f'keys={len(keys)} rank={rk} dim={len(keys)-rk}')
deps = []
for ri, col in enumerate(piv):
    deps.append([(j, mat[ri][j]) for j in free if mat[ri][j] != 0])
cnt = 0
for assign in itertools.product((0, 1, 2), repeat=len(free)):
    val = dict(zip(free, assign))
    ok = True
    for terms in deps:
        v = -sum(co * val[j] for j, co in terms)
        if v.denominator != 1 or not (0 <= v <= 2):
            ok = False
            break
    if ok:
        cnt += 1
print(f'Omega(ref-equation RREF enumeration, k=3, W=2, L=3) = {cnt}  (DP: 1017)')
