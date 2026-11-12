# -*- coding: utf-8 -*-
"""Server-only driver (originally run on the internal cluster).

W=4 exp: stochastic trace(T^4) estimate (Hutchinson +/-1 probes).

T = row transfer matrix (W=4, dim = 3^{12} = 531441); Omega(4) = trace(T^4).
Estimator: Girard/Hutchinson - E[z^T A z] = tr(A) for z in {-1,1}^n;
tr(T^4) = E[z . T^4 z]. Each probe costs 4 sparse mat-vecs.
Error bar = sample std / sqrt(probes); compared against the prediction
Omega(4) ~= 3^16 x 1.11 ~= 4.79e7.

Construction: sparse COO assembly of T (segment-pair precomputation =
3^8 x 3^8 = 43M segment pairs x walk checks).
Memory: COO (nnz) ~= nnz x 12 B; nnz upper bound = 531441 x 6561 (actually
much smaller due to the walk filter; expected in the hundreds of millions).
"""
import sys, math, time
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from itertools import product
import scipy.sparse as sp

LIB = [(1, 0), (0, 1), (1, 1)]
LIBSET = {(1, 0), (0, 1), (1, 1)}
W = 4

t0 = time.time()
rows_all = {}
for i, s in enumerate(product(LIB, repeat=2 * W)):
    rows_all[s] = i
hv_all = {}
for i, s in enumerate(product(LIB, repeat=W)):
    hv_all[s] = i
nseg, nhv = len(rows_all), len(hv_all)
n = nseg * nhv
print(f"dim = {n} ({nseg} segments x {nhv} H values); segment-pair "
      f"precomputation starting...", flush=True)

# segment pairs: up -> [(nr_i, w, hstep)]
pair = {}
t1 = time.time()
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
print(f"segment-pair precomputation done {time.time()-t1:.0f}s; "
      f"mean live nr per up = {sum(len(v) for v in pair.values())/nseg:.1f}",
      flush=True)

# COO assembly: generated per (up, hv)
t2 = time.time()
rr = np.zeros(400_000_000, dtype=np.int32)   # preallocating the upper bound 531441 x 6561 is too large; use list append instead
rr = []
cc = []
vv = []
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
print(f"COO assembly done {time.time()-t2:.0f}s; nnz = {len(vv)}", flush=True)
T = sp.csr_matrix((np.array(vv, dtype=np.float64),
                   (np.array(rr, dtype=np.int64), np.array(cc, dtype=np.int64))),
                  shape=(n, n))
del rr, cc, vv, pair
print(f"T nnz = {T.nnz}, total build {time.time()-t0:.0f}s", flush=True)

# Hutchinson: tr(T^4)
rng = np.random.default_rng(777)
PROBES = 600
est = []
t3 = time.time()
for k in range(PROBES):
    z = rng.choice(np.array([-1.0, 1.0]), size=n)
    y = z
    for _ in range(4):
        y = T @ y
    est.append(float(z @ y))
    if (k + 1) % 100 == 0:
        a = np.array(est)
        m, se = a.mean(), a.std(ddof=1) / math.sqrt(len(a))
        print(f"  probe {k+1}: tr ~= {m:.4g} +/- {se:.3g}", flush=True)
a = np.array(est)
m, se = a.mean(), a.std(ddof=1) / math.sqrt(len(a))
print(f"=== trace(T^4) = {m:.5g} +/- {se:.3g}  ({time.time()-t3:.0f}s, {PROBES} probes)",
      flush=True)
om4 = m
g4 = om4 / 3 ** 16
print(f"Omega(4) ~= {om4:.4g} = 3^16 x {g4:.3f} (predicted g(4) ~= 1.11)")
print(f"s(4) = lnOmega/(32) = {math.log(om4)/32:.4f} (measured chain 1.1001/0.7357/...)")
