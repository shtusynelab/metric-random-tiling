# -*- coding: utf-8 -*-
"""ref_T_W2.py - brute-force reference implementation of the W=2 row
transfer matrix (ground-truth baseline)."""
import numpy as np
from itertools import product as iterprod

LIB = [(1, 0), (0, 1), (1, 1)]
LIBSET = set(LIB)
W = 2

segs = {}
hvs = {}
for i, s in enumerate(iterprod(LIB, repeat=2 * W)):
    segs[s] = i
for i, s in enumerate(iterprod(LIB, repeat=W)):
    hvs[s] = i
inv_s = {v: k for k, v in segs.items()}
n = len(segs) * len(hvs)
T = np.zeros((n, n))

def transfer_weight(up, hv, nr):
    hst = []
    vst = []
    for i in range(W):
        i1 = (i + 1) % W
        c1 = up[2*i][1] - nr[2*i1][1]
        c2 = up[2*i][0] - nr[2*i1][0]
        c3 = up[2*i1+1][1] - nr[2*i+1][1]
        c4 = up[2*i1+1][0] - nr[2*i+1][0]
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
    for start in LIB:
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

for up in segs:
    for hv in hvs:
        src = hvs[hv] * len(segs) + segs[up]
        for nr in segs:
            w = transfer_weight(up, hv, nr)
            if w:
                hstp = []
                for i in range(W):
                    i1 = (i + 1) % W
                    c1 = up[2*i][1] - nr[2*i1][1]
                    c2 = up[2*i][0] - nr[2*i1][0]
                    c3 = up[2*i1+1][1] - nr[2*i+1][1]
                    c4 = up[2*i1+1][0] - nr[2*i+1][0]
                    hstp.append(((c1 + c3) // 2, c2 + c4))
                hn = tuple((hv[c][0] + hstp[c][0], hv[c][1] + hstp[c][1])
                           for c in range(W))
                dst = hvs[hn] * len(segs) + segs[nr]
                T[dst, src] += w

tr2 = float(np.trace(T @ T))
print(f"reference: dim = {n}, nnz = {int((T != 0).sum())}")
print(f"reference trace(T^2) = {tr2:.0f} (anchor 6641)")
assert abs(tr2 - 6641) < 0.5, "reference self-check mismatch - the physical definition needs re-examination"
print("reference self-consistent OK (ground-truth baseline established)")
np.save("ref_T_W2.npy", T)
print("saved ref_T_W2.npy")
