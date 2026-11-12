# -*- coding: utf-8 -*-
"""compare_T3.py - elementwise comparison of the brute literal transcription
vs the v2-assembled T (int, W=3)."""
import sys
import time
import numpy as np
import scipy.sparse as sp
from itertools import product
sys.path.insert(0, ".")
from T_builder_v2 import LIBS, build_pairs, assemble_csr

lib = LIBS["int"]
LIBSET = set(lib)
W = 3
segs = {s: i for i, s in enumerate(product(lib, repeat=2 * W))}
hvs = {s: i for i, s in enumerate(product(lib, repeat=W))}
nseg, nhv = len(segs), len(hvs)
n = nseg * nhv


def transfer_weight(up, hv, nr):
    hst, vst = [], []
    for i in range(W):
        i1 = (i + 1) % W
        c1 = up[2 * i][1] - nr[2 * i1][1]
        c2 = up[2 * i][0] - nr[2 * i1][0]
        c3 = up[2 * i1 + 1][1] - nr[2 * i + 1][1]
        c4 = up[2 * i1 + 1][0] - nr[2 * i + 1][0]
        if (c1 + c3) % 2:
            return 0
        hst.append(((c1 + c3) // 2, c2 + c4))
        vst.append(((c1 - c3) // 2, c2 - c4))
    for c in range(W):
        t = (hv[c][0] + hst[c][0], hv[c][1] + hst[c][1])
        if t not in LIBSET:
            return 0
    if sum(s[0] for s in vst) != 0 or sum(s[1] for s in vst) != 0:
        return 0
    cnt = 0
    for start in lib:
        cur = start
        ok = True
        for d in vst:
            cur = (cur[0] + d[0], cur[1] + d[1])
            if cur not in LIBSET:
                ok = False
                break
        if ok:
            cnt += 1
    return cnt


t0 = time.time()
rr, cc, vv = [], [], []
for up in segs:
    for nr in segs:
        for hv in hvs:
            w = transfer_weight(up, hv, nr)
            if w:
                hstp = []
                for i in range(W):
                    i1 = (i + 1) % W
                    c1 = up[2 * i][1] - nr[2 * i1][1]
                    c2 = up[2 * i][0] - nr[2 * i1][0]
                    c3 = up[2 * i1 + 1][1] - nr[2 * i + 1][1]
                    c4 = up[2 * i1 + 1][0] - nr[2 * i + 1][0]
                    hstp.append(((c1 + c3) // 2, c2 + c4))
                hn = tuple((hv[c][0] + hstp[c][0], hv[c][1] + hstp[c][1])
                           for c in range(W))
                rr.append(hvs[hn] * nseg + segs[nr])
                cc.append(hvs[hv] * nseg + segs[up])
                vv.append(float(w))
Tb = sp.csr_matrix((np.array(vv), (np.array(rr), np.array(cc))), shape=(n, n))
Tb.sum_duplicates()
print(f"brute T nnz={Tb.nnz} ({time.time()-t0:.0f}s)", flush=True)

up_ids, nr_ids, w, H1, H2 = build_pairs(W, lib, verbose_every=0)
Tv = assemble_csr(W, lib, up_ids, nr_ids, w, H1, H2, verbose_every=0)
print(f"v2    T nnz={Tv.nnz}", flush=True)

D = (Tb - Tv).tocoo()
print(f"|Tb-Tv| nnz = {D.nnz}, max|diff| = {np.abs(D.data).max() if D.nnz else 0}")
# sample one disagreement
if D.nnz:
    k = 0
    r, c, x = D.row[k], D.col[k], D.data[k]
    print(f"sample: dst={r} src={c} diff={x}; brute[{r},{c}]={Tb[r,c]}, v2={Tv[r,c]}")
    up_i, hv_i = c % nseg, c // nseg
    nr_i, hn_i = r % nseg, r // nseg
    print(f"src: hv#{hv_i} up#{up_i};dst: hn#{hn_i} nr#{nr_i}")
