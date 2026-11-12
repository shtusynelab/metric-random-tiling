# -*- coding: utf-8 -*-
"""Analytic route: closed-loop verification of the reduced counting.

Reduced formula (with the verified decoupling/parity/loop properties):
  A solution <=> an (alpha,beta) table (3 values per cell) plus 4 groups of
  axis gauge constants, subject to:
  (P) per octagon, c1+c3 even (c1 = n_alpha(i,j) - n_alpha(i+1,j+1),
      c3 = n_beta(i+1,j) - n_beta(i,j+1));
  (D) axis differences: Dmh = -(c1+c3)/2, Dmv = (c1-c3)/2,
      Dnh = -(c2+c4), Dnv = c2-c4 (c2 = m_alpha(i,j) - m_alpha(i+1,j+1),
      c4 = m_beta(i+1,j) - m_beta(i,j+1));
  (L) per column sum_j Dmh = sum_j Dnh = 0; per row sum_i Dmv = sum_i Dnv = 0
      (loop closure);
  (W) walks: h-m(i,j+1) = h-m(i,j) - Dmh(i,j) etc., all staying in {0,1};
  (G) gauge: h-m(i,0) and v-m(i,0) are the free gauge choices - W h starts
      per column, W v starts per row, likewise h-n and v-n, each in {0,1};
      given (alpha,beta), each line has 0/1/2 legal starting points.
  Omega = sum over (alpha,beta) passing (P)(L) of prod_lines #legal starts.

Verification: at W=2 directly enumerate the 3^8 = 6561 (alpha,beta) tables;
the result must equal 6641.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from itertools import product

LIB = [(1, 0), (0, 1), (1, 1)]
W = 2

def line_walk_count(diff):
    """Joint walk: states in LIB (3 points, excluding (0,0)); diff is the
    sequence of step vectors (length W, whose vector sum must be 0). Counts
    the legal starting points such that s_{k+1} = s_k + d_k stays in LIB
    throughout."""
    if sum(v[0] for v in diff) != 0 or sum(v[1] for v in diff) != 0:
        return 0
    cnt = 0
    for start in LIB:
        cur = start
        ok = True
        for d in diff:
            cur = (cur[0] + d[0], cur[1] + d[1])
            if cur not in LIB:
                ok = False
                break
        if ok:
            cnt += 1
    return cnt

total = 0
n_parity_fail = n_loop_fail = 0
for alpha in product(LIB, repeat=W * W):
    for beta in product(LIB, repeat=W * W):
        A = {(i, j): alpha[i * W + j] for i in range(W) for j in range(W)}
        B = {(i, j): beta[i * W + j] for i in range(W) for j in range(W)}
        # difference fields
        ok = True
        hst = [[(0, 0)] * W for _ in range(W)]
        vst = [[(0, 0)] * W for _ in range(W)]
        for i in range(W):
            for j in range(W):
                c1 = A[(i, j)][1] - A[((i + 1) % W, (j + 1) % W)][1]
                c2 = A[(i, j)][0] - A[((i + 1) % W, (j + 1) % W)][0]
                c3 = B[((i + 1) % W, j)][1] - B[(i, (j + 1) % W)][1]
                c4 = B[((i + 1) % W, j)][0] - B[(i, (j + 1) % W)][0]
                if (c1 + c3) % 2:
                    ok = False
                    break
                # H line steps = -(Dmh, Dnh) = ((c1+c3)/2, c2+c4)
                # V line steps = +(Dmv, Dnv) = ((c1-c3)/2, c2-c4)
                hst[i][j] = ((c1 + c3) // 2, c2 + c4)
                vst[i][j] = ((c1 - c3) // 2, c2 - c4)
            if not ok:
                break
        if not ok:
            n_parity_fail += 1
            continue
        # loop closure + walk + gauge counting
        factor = 1
        # H-axis joint walk (per column i, steps hst over j); V axis (per
        # row j, steps vst over i)
        for i in range(W):
            k = line_walk_count([hst[i][j] for j in range(W)])
            if k == 0:
                factor = 0
                break
            factor *= k
        if factor:
            for j in range(W):
                k = line_walk_count([vst[i][j] for i in range(W)])
                if k == 0:
                    factor = 0
                    break
                factor *= k
        if factor == 0:
            n_loop_fail += 1
            continue
        total += factor

print(f"W={W} reduced count Omega_red = {total} (anchor 6641)")
print(f"tables rejected by parity = {n_parity_fail}; rejected by walk/loop = {n_loop_fail}")
print("MATCH" if total == 6641 else "MISMATCH")
