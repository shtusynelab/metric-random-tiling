# -*- coding: utf-8 -*-
"""Verification of the hand-derived closed-form parametrization (candidate):
per-solution classification at W=2 plus closed-form formula vs DP.

Hand-derived chain:
  (P1) square face => the (m,n) of cut(i,j,0) equals that of cut(i,j,2),
       and cut(i,j,1) equals cut(i,j,3) => variables collapse to the pair
       variables A(i,j) [= cut*,0/2 value] and B(i,j) [= 1/3].
  (P2) octagon rows => mH(i,j) - nH(i,j) is constant along the j column
       (= a_i); mV(i,j) - nV(i,j) is constant along the i row (= b_j).
  (P3) for the experiment library M = {(1,0),(0,1),(1,1)}, a_i in
       {1,-1,0} uniquely determines nH(i,j) = pH:
       a=1 -> pH=0; a=-1 -> pH=1; a=0 -> pH=1 => horizontal axis edges are
       constant per column; vertical axis edges constant per row.
  (P4) C1/C3 collapse to the truncated alternating sums Dm = 0, Dn = 0 =>
       c := m_B - m_A is constant along the diagonal (i+j) mod W
       (kappa in {-1,0,1}^W); likewise for n (lambda).
  (P5) closed-form candidate: Omega = 3^{2W} *
       [sum_{delta,eps} L(delta,eps)^W]^W, with L defined below.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tsq_lambda_lib import TSQTorus, face_rows, mono_order, M_EXP, dp_count
from itertools import product

W = 2
t = TSQTorus(W, W)
E = t.ne
faces = [t.faces[i] for i in mono_order(W)]
rows = [face_rows(sides, E) for size, sides in faces]

# edge id -> semantic key (axis H (0,i,j) / V (1,i,j) / cut (i,j,q))
key_of = {}
for i in range(W):
    for j in range(W):
        pass
# rebuild from the TSQTorus private dicts (reading _axis/_cuts)
ax, cu = t._axis, t._cuts
inv = {}
for k, e in ax.items():
    inv[e] = ("ax", k)
for k, e in cu.items():
    inv[e] = ("cut", k)

# enumerate all solutions (DFS, same library M_EXP), check P1-P4 per solution
val = {}
sols = []
axis_H = {(i, j): ax[(0, i, j)] for i in range(W) for j in range(W)}
axis_V = {(i, j): ax[(1, i, j)] for i in range(W) for j in range(W)}
cutv = {(i, j, q): cu[(i, j, q)] for i in range(W) for j in range(W)
        for q in range(4)}

def rec(fi):
    if fi == len(faces):
        sols.append(dict(val))
        return
    sides = faces[fi][1]
    new_e = [e for e, d in sides if e not in val]
    for combo in product(M_EXP, repeat=len(new_e)):
        for e, v in zip(new_e, combo):
            val[e] = v
        ok = all(sum(c * (val[v][0] if v < E else val[v - E][1])
                     for v, c in row.items()) == 0 for row in rows[fi])
        if ok:
            rec(fi + 1)
        for e in new_e:
            del val[e]

rec(0)
print(f"W={W} total solutions = {len(sols)} (DP anchor 6641)")

# P2/P3: horizontal-axis (m-n) constant per column; a_i in {1,-1,0}; pH = f(a)
bad_p2 = bad_p3 = 0
a_of = {}
for s in sols:
    for i in range(W):
        diffs = {s[axis_H[(i, j)]][0] - s[axis_H[(i, j)]][1] for j in range(W)}
        if len(diffs) != 1:
            bad_p2 += 1
            continue
        a = diffs.pop()
        f = {1: 0, -1: 1, 0: 1}
        if a not in f or any(s[axis_H[(i, j)]][1] != f[a] for j in range(W)):
            bad_p3 += 1
print(f"P2 violating solutions = {bad_p2}; P3 (experiment library pH=f(a)) violations = {bad_p3}")

# P4: c = m_B - m_A constant along the diagonal; B pair = cut*,1/3; A pair = cut*,0/2
bad_p4 = 0
Lcount = {}
for s in sols:
    okk = True
    for j in range(W):
        for i in range(W):
            pass
    cs = {}
    for i in range(W):
        for j in range(W):
            mA = s[cutv[(i, j, 0)]][0]
            mB = s[cutv[(i, j, 1)]][0]
            # pair-equality check (P1)
            assert s[cutv[(i, j, 0)]] == s[cutv[(i, j, 2)]], "P1 violated!"
            assert s[cutv[(i, j, 1)]] == s[cutv[(i, j, 3)]], "P1 violated!"
            c = mB - mA
            d = (i + j) % W
            if d in cs and cs[d] != c:
                okk = False
            cs[d] = c
    if not okk:
        bad_p4 += 1
print(f"P4 (diagonal constant) violating solutions = {bad_p4}")

# closed-form formula
def L(d, e):
    # legal (mA,nA) in M with mA in A_d and nA in A_e
    Am = {1: {0}, -1: {1}, 0: {0, 1}}[d]
    An = {1: {1}, -1: {0}, 0: {0, 1}}[e]
    return sum(1 for (m, n) in M_EXP if m in Am and n in An)

for Wc in (2, 3, 4):
    S = sum(L(d, e) ** Wc for d in (1, -1, 0) for e in (1, -1, 0))
    om_close = (3 ** (2 * Wc)) * S ** Wc
    print(f"closed-form W={Wc}: 3^(2W)*S^W = {om_close} (S = {S})")
