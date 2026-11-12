# -*- coding: utf-8 -*-
"""Gate N1: construction and verification of the row transfer matrix T
(memory-safe version).
State = (upper-row cut-value segment up[xW], current H-column values
hv[xW]), dimension 3^{3W}.
T[(up,hv) -> (nr,hv')]: weight = number of legal V-row-walk starting points
(0 if the (P) parity condition fails / the V walk hits a dead branch / the
H update leaves the library).
W=2: dim=729, dense 4.3 MB locally; W=3: dim=19683 (sparse power iteration,
finite nnz).
Verification: W=2 trace(T^2) = 6641.
"""
import sys, math
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from itertools import product

LIB = [(1, 0), (0, 1), (1, 1)]
LIBSET = {(1, 0), (0, 1), (1, 1)}


def build_T_dense(W):
    # row segment = independent 3-way choice of (alpha,beta) per cell:
    # segment vector = [alpha(0), beta(0), alpha(1), beta(1), ...] (2W slots)
    rows_all = {}
    for i, s in enumerate(product(LIB, repeat=2 * W)):
        rows_all[s] = i
    hv_all = {}
    for i, s in enumerate(product(LIB, repeat=W)):
        hv_all[s] = i
    nseg, nhv = len(rows_all), len(hv_all)
    n = nseg * nhv
    T = np.zeros((n, n))
    for up in rows_all:
        up_i = rows_all[up]
        # interleaved order for up: up[2i] = alpha_u(i), up[2i+1] = beta_u(i); same for nr
        pw = np.zeros(nseg)
        ph = [None] * nseg
        for nr in rows_all:
            nr_i = rows_all[nr]
            hst = []
            vst = []
            ok = True
            for i in range(W):
                i1 = (i + 1) % W
                c1 = up[2 * i][1] - nr[2 * i1][1]      # n alpha chain (1,1)
                c2 = up[2 * i][0] - nr[2 * i1][0]
                c3 = up[2 * i1 + 1][1] - nr[2 * i + 1][1]   # n beta chain (-1,1)
                c4 = up[2 * i1 + 1][0] - nr[2 * i + 1][0]
                if (c1 + c3) % 2:
                    ok = False
                    break
                hst.append(((c1 + c3) // 2, c2 + c4))
                vst.append(((c1 - c3) // 2, c2 - c4))
            if not ok:
                continue
            if sum(s[0] for s in vst) != 0 or sum(s[1] for s in vst) != 0:
                continue
            cnt = 0
            for start in LIB:
                cur = start
                okk = True
                for d in vst:
                    cur = (cur[0] + d[0], cur[1] + d[1])
                    if cur not in LIBSET:
                        okk = False
                        break
                if okk:
                    cnt += 1
            pw[nr_i] = cnt
            ph[nr_i] = tuple(hst)
        for hv in hv_all:
            src = hv_all[hv] * nseg + up_i
            for nr in rows_all:
                nr_i = rows_all[nr]
                w = pw[nr_i]
                if w == 0:
                    continue
                hstep = ph[nr_i]
                hn = tuple((hv[i][0] + hstep[i][0], hv[i][1] + hstep[i][1])
                           for i in range(W))
                if any(x not in LIBSET for x in hn):
                    continue
                dst = hv_all[hn] * nseg + nr_i
                T[dst, src] += w
    return T


W = 2
n = 3 ** (3 * W)
est_mb = n * n * 8 / 1e6
assert est_mb < 200, f"W={W} dense {est_mb:.0f}MB exceeds the local budget"
T = build_T_dense(W)
print(f"W={W}: dim={n}, T memory {n*n*8/1e6:.1f}MB")
tr2 = np.trace(np.linalg.matrix_power(T, 2))
print(f"trace(T^2) = {tr2:.0f} (anchor 6641)")
assert abs(tr2 - 6641) < 0.5
ev = np.linalg.eigvals(T)
lam = max(abs(ev))
top = sorted([abs(x) for x in ev], reverse=True)[:5]
print(f"lambda_max = {lam:.6f}, ln(lambda)/(2W) = {math.log(lam)/(2*W):.6f}")
print(f"top-5 eigenvalue moduli: {[round(x,3) for x in top]}")
