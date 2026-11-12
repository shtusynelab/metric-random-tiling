# -*- coding: utf-8 -*-
"""brute_W3.py - W=3 arbitration: pure-Python literal transcription of the
build_T.py / ref_T_W2.py semantics, independent of the vectorized
implementation. Decides whether the W=3 mismatch is a formalization problem
or a transcription bug.
Usage: python brute_W3.py exp|int
"""
import sys
import math
import numpy as np
import scipy.sparse as sp
from itertools import product

LIBS = {"exp": [(1, 0), (0, 1), (1, 1)], "int": [(1, 0), (2, 0), (3, 0)]}
lib = LIBS[sys.argv[1] if len(sys.argv) > 1 else "exp"]
LIBSET = set(lib)
W = 3

segs = {s: i for i, s in enumerate(product(lib, repeat=2 * W))}
hvs = {s: i for i, s in enumerate(product(lib, repeat=W))}
nseg, nhv = len(segs), len(hvs)
n = nseg * nhv
print(f"nseg={nseg} nhv={nhv} n={n}", flush=True)


def transfer_weight(up, hv, nr):
    """Literal transcription of ref_T_W2.py (identical variable names)."""
    hst = []
    vst = []
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
    hn = []
    for c in range(W):
        t = (hv[c][0] + hst[c][0], hv[c][1] + hst[c][1])
        if t not in LIBSET:
            return 0
        hn.append(t)
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


rr, cc, vv = [], [], []
inv_seg = {v: k for k, v in segs.items()}
t0 = __import__("time").time()
nedges = 0
for up in segs:
    for nr in segs:
        # transfer_weight contains the H check internally (depends on hv);
        # called per hv
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
                src = hvs[hv] * nseg + segs[up]
                dst = hvs[hn] * nseg + segs[nr]
                rr.append(dst)
                cc.append(src)
                vv.append(float(w))
                nedges += 1
    if segs[up] % 100 == 0:
        print(f"  up {segs[up]}/{nseg} edges {nedges} "
              f"({__import__('time').time()-t0:.0f}s)", flush=True)
T = sp.csr_matrix((np.array(vv), (np.array(rr), np.array(cc))), shape=(n, n))
T.sum_duplicates()
print(f"T nnz = {T.nnz} ({__import__('time').time()-t0:.0f}s)", flush=True)

# exact trace(T^3): block method
tot = 0.0
for c0 in range(0, n, 512):
    c1_ = min(n, c0 + 512)
    X = np.zeros((n, c1_ - c0))
    X[np.arange(c0, c1_), np.arange(c1_ - c0)] = 1.0
    for _ in range(3):
        X = T @ X
    tot += float(np.trace(X))
print(f"=== [{sys.argv[1] if len(sys.argv)>1 else 'exp'}] literal-transcription trace(T^3) = "
      f"{round(tot)} (exp anchor 563553 / int anchor 531441)", flush=True)
