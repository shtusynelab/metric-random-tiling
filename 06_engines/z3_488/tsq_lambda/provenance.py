# -*- coding: utf-8 -*-
"""provenance.py - full provenance audit of the (dst=1, src=1) edge in the
brute matrix (int, W=3)."""
import sys
from itertools import product
sys.path.insert(0, ".")
from T_builder_v2 import LIBS

lib = LIBS["int"]
LIBSET = set(lib)
W = 3
segs = {s: i for i, s in enumerate(product(lib, repeat=2 * W))}
hvs = {s: i for i, s in enumerate(product(lib, repeat=W))}
nseg = len(segs)
inv_seg = {i: s for s, i in segs.items()}
inv_hv = {i: s for s, i in hvs.items()}


def transfer_weight(up, hv, nr, verbose=False):
    hst, vst = [], []
    for i in range(W):
        i1 = (i + 1) % W
        c1 = up[2 * i][1] - nr[2 * i1][1]
        c2 = up[2 * i][0] - nr[2 * i1][0]
        c3 = up[2 * i1 + 1][1] - nr[2 * i + 1][1]
        c4 = up[2 * i1 + 1][0] - nr[2 * i + 1][0]
        if verbose:
            print(f"    i={i}: c=({c1},{c2},{c3},{c4})")
        if (c1 + c3) % 2:
            return 0
        hst.append(((c1 + c3) // 2, c2 + c4))
        vst.append(((c1 - c3) // 2, c2 - c4))
    for c in range(W):
        t = (hv[c][0] + hst[c][0], hv[c][1] + hst[c][1])
        if verbose:
            print(f"    H[{c}]: {hv[c]}+{hst[c]} = {t} in_lib={t in LIBSET}")
        if t not in LIBSET:
            return 0
    if sum(s[0] for s in vst) != 0 or sum(s[1] for s in vst) != 0:
        if verbose:
            print("    closure violated")
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


up1 = inv_seg[1]
hv0 = inv_hv[0]
nr1 = inv_seg[1]
print(f"up#1 = {up1}")
print(f"hv#0 = {hv0}")
print(f"nr#1 = {nr1}")
w = transfer_weight(up1, hv0, nr1, verbose=True)
print(f"transfer_weight = {w}")
print(f"dst = hvs[hn]*nseg + segs[nr] is generated only when w > 0; hand-computed hn:")
hst = []
for i in range(W):
    i1 = (i + 1) % W
    c1 = up1[2 * i][1] - nr1[2 * i1][1]
    c2 = up1[2 * i][0] - nr1[2 * i1][0]
    c3 = up1[2 * i1 + 1][1] - nr1[2 * i + 1][1]
    c4 = up1[2 * i1 + 1][0] - nr1[2 * i + 1][0]
    hst.append(((c1 + c3) // 2, c2 + c4))
hn = tuple((hv0[c][0] + hst[c][0], hv0[c][1] + hst[c][1]) for c in range(W))
print(f"hn = {hn}, hn#id = {hvs.get(hn, 'NOT in the hvs dict!')}")
