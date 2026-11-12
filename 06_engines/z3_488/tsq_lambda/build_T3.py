# -*- coding: utf-8 -*-
"""W=3 row transfer matrix lambda_max: sparse construction + power
iteration.
dim = 3^{3W} = 19683; nonzero per row <= 3^{2W} = 729 => nnz <= 14M
(~170 MB). lambda_max -> s_asymptotic(3) = ln(lambda)/6, compared against
the exact s(3) = 0.7357 (ln Omega(3)/3 = 4.414 = ln lambda + finite-size
correction / 3).
"""
import sys, math, time
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from itertools import product
import scipy.sparse as sp

LIB = [(1, 0), (0, 1), (1, 1)]
LIBSET = {(1, 0), (0, 1), (1, 1)}
W = 3

t0 = time.time()
rows_all = {}
for i, s in enumerate(product(LIB, repeat=2 * W)):
    rows_all[s] = i
hv_all = {}
for i, s in enumerate(product(LIB, repeat=W)):
    hv_all[s] = i
nseg, nhv = len(rows_all), len(hv_all)
n = nseg * nhv
print(f"dim = {n}; building nnz...", flush=True)

# segment-pair precomputation: up -> [(nr_i, w, hstep)]
pair = {}
for up in rows_all:
    lst = []
    for nr in rows_all:
        hst = []
        vst = []
        ok = True
        for i in range(W):
            i1 = (i + 1) % W
            c1 = up[2*i][1] - nr[2*i1][1]
            c2 = up[2*i][0] - nr[2*i1][0]
            c3 = up[2*i1+1][1] - nr[2*i+1][1]
            c4 = up[2*i1+1][0] - nr[2*i+1][0]
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
        if cnt:
            lst.append((rows_all[nr], cnt, tuple(hst)))
    pair[rows_all[up]] = lst
print(f"segment-pair precomputation {time.time()-t0:.0f}s; "
      f"nnz upper bound ~= sum of len(lst)", flush=True)

# COO assembly
rr, cc, vv = [], [], []
for up_i, lst in pair.items():
    for hv in hv_all:
        src = hv_all[hv] * nseg + up_i
        base = hv_all[hv] * nseg
        for nr_i, cnt, hstep in lst:
            hn = tuple((hv[i][0] + hstep[i][0], hv[i][1] + hstep[i][1])
                       for i in range(W))
            if any(x not in LIBSET for x in hn):
                continue
            rr.append(base + nr_i)
            cc.append(src)
            vv.append(cnt)
T = sp.csr_matrix((vv, (rr, cc)), shape=(n, n))
print(f"T nnz = {T.nnz}, total build {time.time()-t0:.0f}s", flush=True)

# power iteration
v = np.random.default_rng(1).random(n)
v /= np.linalg.norm(v)
for it in range(200):
    w = T @ v
    nw = np.linalg.norm(w)
    if nw == 0:
        print("died"); break
    v = w / nw
    if it % 20 == 0 or it > 190:
        lam = float(v @ (T @ v))
        print(f"  iter {it}: lambda ~= {lam:.6f}", flush=True)
lam = float(v @ (T @ v))
s_asym = math.log(lam) / (2 * W)
print(f"W={W}: lambda_max = {lam:.6f}")
print(f"s_asymptotic(3) = ln(lambda)/(2W) = {s_asym:.6f} (exact s(3) = 0.735668; "
      f"ln Omega(3)/3 = 4.4136 = ln lambda + correction/3)")
