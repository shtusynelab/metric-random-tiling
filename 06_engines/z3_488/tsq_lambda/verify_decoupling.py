# -*- coding: utf-8 -*-
"""Analytic route: per-solution verification of the octagon-row decoupling
reduction (W=2 full-solution testbed).

Hand-derived proposition (corrected version; the beta-chain indices are
fixed):
  (D) the 4 rows of each octagon are equivalent to:
      Dmh + Dmv = n[beta(i+1,j)] - n[beta(i,j+1)]
      Dmh - Dmv = n[alpha(i+1,j+1)] - n[alpha(i,j)]
      Dnh + Dnv = m[beta(i+1,j)] - m[beta(i,j+1)]
      Dnh - Dnv = m[alpha(i+1,j+1)] - m[alpha(i,j)]
    where Dmh = m_S - m_N, Dmv = m_E - m_W, Dnh = n_S - n_N, Dnv = n_E - n_W;
    alpha(i,j) := value of cut[(i,j,0)] (= cut[(i,j,2)], property P1);
    beta(i,j) := value of cut[(i,j,1)] (= cut[(i,j,3)]).
    Cut-edge assignment: SW = alpha(i,j), SE = beta(i+1,j),
    NE = alpha(i+1,j+1), NW = beta(i,j+1).
  (P) parity: Dmh +/- Dmv in {-2..2} and Dmh, Dmv in {-1,0,1} implies
      nalpha-diff == nbeta-diff (mod 2).
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tsq_lambda_lib import TSQTorus, face_rows, mono_order, M_EXP
from itertools import product

W = 2
t = TSQTorus(W, W)
E = t.ne
faces = [t.faces[i] for i in mono_order(W)]
frows = [face_rows(sides, E) for size, sides in faces]
ax, cu = t._axis, t._cuts
axis_H = {(i, j): ax[(0, i, j)] for i in range(W) for j in range(W)}
axis_V = {(i, j): ax[(1, i, j)] for i in range(W) for j in range(W)}
alpha_e = {(i, j): cu[(i, j, 0)] for i in range(W) for j in range(W)}
beta_e = {(i, j): cu[(i, j, 1)] for i in range(W) for j in range(W)}

val = {}
sols = []
n_D_bad = n_P_bad = 0

def rec(fi):
    global n_D_bad, n_P_bad
    if fi == len(faces):
        sols.append(dict(val))
        return
    sides = faces[fi][1]
    new_e = [e for e, d in sides if e not in val]
    for combo in product(M_EXP, repeat=len(new_e)):
        for e, v in zip(new_e, combo):
            val[e] = v
        ok = all(sum(c * (val[v][0] if v < E else val[v - E][1])
                     for v, c in row.items()) == 0 for row in frows[fi])
        if ok:
            rec(fi + 1)
        for e in new_e:
            del val[e]

rec(0)
print(f"W={W} total solutions = {len(sols)} (anchor 6641)")

for si, s in enumerate(sols):
    al = {p: s[alpha_e[p]] for p in alpha_e}
    be = {p: s[beta_e[p]] for p in beta_e}
    for i in range(W):
        for j in range(W):
            assert s[cu[(i, j, 2)]] == al[(i, j)], f"P1 violated in sol{si}"
            assert s[cu[(i, j, 3)]] == be[(i, j)], f"P1 (beta) violated in sol{si}"
    for i in range(W):
        for j in range(W):
            mS = s[axis_H[(i, j)]][0]; mN = s[axis_H[(i, (j + 1) % W)]][0]
            mW = s[axis_V[(i, j)]][0]; mE = s[axis_V[((i + 1) % W, j)]][0]
            nS = s[axis_H[(i, j)]][1]; nN = s[axis_H[(i, (j + 1) % W)]][1]
            nW = s[axis_V[(i, j)]][1]; nE = s[axis_V[((i + 1) % W, j)]][1]
            c1 = al[(i, j)][1] - al[((i + 1) % W, (j + 1) % W)][1]   # n alpha-diff
            c2 = al[(i, j)][0] - al[((i + 1) % W, (j + 1) % W)][0]   # m alpha-diff
            c3 = be[((i + 1) % W, j)][1] - be[(i, (j + 1) % W)][1]   # n beta-diff
            c4 = be[((i + 1) % W, j)][0] - be[(i, (j + 1) % W)][0]   # m beta-diff
            if (c1 + c3) % 2:
                n_P_bad += 1
                if n_P_bad <= 2:
                    print(f"(P) violation sol{si} oct({i},{j}): c1={c1} c3={c3} "
                          f"alphaSW={al[(i,j)]} alphaNE={al[((i+1)%W,(j+1)%W)]} "
                          f"betaSE={be[((i+1)%W,j)]} betaNW={be[(i,(j+1)%W)]}")
            if (mS - mN) != -(c1 + c3) // 2 or (mE - mW) != (c1 - c3) // 2 \
               or (nS - nN) != -(c2 + c4) or (nE - nW) != (c2 - c4):
                n_D_bad += 1
                if n_D_bad <= 2:
                    print(f"(D) violation sol{si} oct({i},{j}): "
                          f"Dmh={mS-mN} vs {-(c1+c3)//2}; "
                          f"Dmv={mE-mW} vs {(c1-c3)//2}; "
                          f"Dnh={nS-nN} vs {-(c2+c4)}; "
                          f"Dnv={nE-nW} vs {c4-c2}")
print(f"(P) parity violations = {n_P_bad}; (D) decoupling violations = {n_D_bad}")
if n_P_bad == 0 and n_D_bad == 0:
    print("decoupling propositions (D)(P) hold on all solutions - the reduced parametrization is valid")
