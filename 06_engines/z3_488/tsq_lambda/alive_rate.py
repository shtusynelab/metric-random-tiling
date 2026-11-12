# -*- coding: utf-8 -*-
"""Decay of the alive-table rate with W (data for the asymptotic
discrimination): W=2 exact; W=3 random sampling.

Alive table = an (alpha,beta) table passing parity (P) with all 4 joint-walk
families alive (H 4 columns, V 4 rows, gauge factor > 0).
Alive rate lambda(W) = #alive / 3^{2W^2}. If ln(1/lambda) ~ c*W^2 (fast
decay) => the surface layer carries no entropy; if ~ c*W => the surface
layer contributes O(W) entropy (micro-entropy). The exact W=2 rate 9.1% is
the anchor.
"""
import sys, random
from itertools import product

LIB = [(1, 0), (0, 1), (1, 1)]


def table_alive(A, B, W):
    hst = [[(0, 0)] * W for _ in range(W)]
    vst = [[(0, 0)] * W for _ in range(W)]
    for i in range(W):
        for j in range(W):
            c1 = A[(i, j)][1] - A[((i + 1) % W, (j + 1) % W)][1]
            c2 = A[(i, j)][0] - A[((i + 1) % W, (j + 1) % W)][0]
            c3 = B[((i + 1) % W, j)][1] - B[(i, (j + 1) % W)][1]
            c4 = B[((i + 1) % W, j)][0] - B[(i, (j + 1) % W)][0]
            if (c1 + c3) % 2:
                return 0
            hst[i][j] = ((c1 + c3) // 2, c2 + c4)
            vst[i][j] = ((c1 - c3) // 2, c2 - c4)

    def walk(steps):
        if sum(s[0] for s in steps) != 0 or sum(s[1] for s in steps) != 0:
            return 0
        cnt = 0
        for start in LIB:
            cur = start
            ok = True
            for d in steps:
                cur = (cur[0] + d[0], cur[1] + d[1])
                if cur not in LIB:
                    ok = False
                    break
            if ok:
                cnt += 1
        return cnt

    for i in range(W):
        if walk(hst[i]) == 0:
            return 0
    for j in range(W):
        if walk([vst[i][j] for i in range(W)]) == 0:
            return 0
    return 1


# W=2 exact (anchor)
W = 2
alive = total = 0
for alpha in product(LIB, repeat=W * W):
    for beta in product(LIB, repeat=W * W):
        A = {(i, j): alpha[i * W + j] for i in range(W) for j in range(W)}
        B = {(i, j): beta[i * W + j] for i in range(W) for j in range(W)}
        total += 1
        alive += table_alive(A, B, W)
lam2 = alive / total
print(f"W=2: alive tables {alive}/{total} = {lam2:.4%} (anchor 597/6561 = 9.10%)")

# W=3 sampling
rng = random.Random(20260929)
W = 3
N = 200000
alive3 = 0
for _ in range(N):
    A = {(i, j): LIB[rng.randrange(3)] for i in range(W) for j in range(W)}
    B = {(i, j): LIB[rng.randrange(3)] for i in range(W) for j in range(W)}
    alive3 += table_alive(A, B, W)
if alive3:
    print(f"W=3 sampling: {alive3}/{N} = {alive3/N:.5%} (+/-{1.96*(alive3/N*(1-alive3/N)/N)**.5:.5%})")
    print(f"  ln(1/lambda): W=2 = {__import__('math').log(1/lam2):.3f}; "
          f"W=3 = {__import__('math').log(1/(alive3/N)):.3f}")
else:
    print(f"W=3 sampling: 0/{N} alive => lambda(3) < 3/N upper bound ~= {3/N:.2%}")
    print(f"  ln(1/lambda) >= {__import__('math').log(N/3):.2f} (vs W=2's "
          f"{__import__('math').log(1/lam2):.2f})")
