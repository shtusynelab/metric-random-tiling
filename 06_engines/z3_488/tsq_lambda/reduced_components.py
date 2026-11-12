# -*- coding: utf-8 -*-
"""Component statistics of the reduced model (W=2, built on the
closed-loop-verified reduced_count framework). Provides data for designing
the two-stage counting (n-bit stage + m-bit factorized stage):
  1. parity pass rate of the (alpha,beta) tables;
  2. among parity-passing tables: pass rate of the H/V joint walks
     (all 4 families alive / partially alive / all dead);
  3. distribution of the per-table gauge factor (prod of legal start
     counts);
  4. degeneracy of the n-bit configurations: for the same
     (alpha_n, beta_n), the factor sum over different m-bit tables ->
     tests the separability claim that the m part factorizes into a
     column product x row product.
"""
import sys
from itertools import product

LIB = [(1, 0), (0, 1), (1, 1)]
W = 2


def line_walk_count_mn(hst_col):
    """H-column joint walk: returns the number of legal starting points
    (0..3). hst = [((c1+c3)//2, c2+c4) x W]."""
    if sum(s[0] for s in hst_col) != 0 or sum(s[1] for s in hst_col) != 0:
        return 0
    cnt = 0
    for start in LIB:
        cur = start
        ok = True
        for d in hst_col:
            cur = (cur[0] + d[0], cur[1] + d[1])
            if cur not in LIB:
                ok = False
                break
        if ok:
            cnt += 1
    return cnt


def line_walk_count_v(vst_row):
    """V-row joint walk: vst = [((c1-c3)//2, c2-c4) x W]."""
    if sum(s[0] for s in vst_row) != 0 or sum(s[1] for s in vst_row) != 0:
        return 0
    cnt = 0
    for start in LIB:
        cur = start
        ok = True
        for d in vst_row:
            cur = (cur[0] + d[0], cur[1] + d[1])
            if cur not in LIB:
                ok = False
                break
        if ok:
            cnt += 1
    return cnt


n_tab = n_par = 0
h_alive = v_alive = both_alive = 0
fac_dist = {}
nbit_factor = {}
for alpha in product(LIB, repeat=W * W):
    for beta in product(LIB, repeat=W * W):
        A = {(i, j): alpha[i * W + j] for i in range(W) for j in range(W)}
        B = {(i, j): beta[i * W + j] for i in range(W) for j in range(W)}
        n_tab += 1
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
                hst[i][j] = ((c1 + c3) // 2, c2 + c4)
                vst[i][j] = ((c1 - c3) // 2, c2 - c4)
            if not ok:
                break
        if not ok:
            n_par += 1
            continue
        hks = [line_walk_count_mn(hst[i]) for i in range(W)]
        vks = [line_walk_count_v(vst[j]) for j in range(W)]
        h0 = sum(1 for k in hks if k)
        v0 = sum(1 for k in vks if k)
        if h0 == W and v0 == W:
            both_alive += 1
        elif h0 == W:
            h_alive += 1
        elif v0 == W:
            v_alive += 1
        fac = 1
        for k in hks + vks:
            fac *= k
        if fac:
            fac_dist[fac] = fac_dist.get(fac, 0) + 1
        nb = (tuple(a[1] for a in alpha), tuple(b[1] for b in beta))
        nbit_factor[nb] = nbit_factor.get(nb, 0) + fac

print(f"total tables = {n_tab}; rejected by parity = {n_par}; passed = {n_tab - n_par}")
print(f"walks: H-alive and V-alive = {both_alive}; only H alive = {h_alive}; "
      f"only V alive = {v_alive}; rest (dead) = "
      f"{n_tab - n_par - both_alive - h_alive - v_alive}")
print(f"gauge factor distribution (alive tables): {dict(sorted(fac_dist.items()))}")
print(f"Omega_red (check) = {sum(k * v for k, v in fac_dist.items())}")
alive_nb = {k: v for k, v in nbit_factor.items() if v}
print(f"alive (alpha_n, beta_n) configurations = {len(alive_nb)} (with factor > 0)")
top = sorted(alive_nb.items(), key=lambda x: -x[1])[:6]
for k, v in top:
    print(f"  alpha_n={k[0]} beta_n={k[1]}: factor sum = {v}")
