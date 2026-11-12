# -*- coding: utf-8 -*-
"""blocked_multi.py -- multi-block chained DFS: the disk is cut column-wise
into k blocks, each block enumerated by pure DFS inside (with per-face
closure pruning), recording (left-interface state, right-interface state);
chained matching sums over the blocks.

Interface = {F0(a_b, .)} union {F2(a_b, .)} of the block-boundary column
a_b (the edges shared by the two adjacent blocks).
Path independence: a completely different implementation from the column DP
(recurrence / state merging); the within-block enumeration is a per-edge
DFS.
"""
import sys, time
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent))
import disk_lib as dl


def _iface_edges(cells, a_col):
    """Block-spanning shared edges of column a_col (both adjacent faces
    exist):
      F0(a_col,b): faces (a_col,b) and (a_col-1,b+1) both inside the disk;
      F2(a_col,j): faces (a_col,j+1) and (a_col-1,j+1) both inside the
      disk."""
    right = {b for (a, b) in cells if a == a_col}
    left = {b for (a, b) in cells if a == a_col - 1}
    out = []
    for b in right:
        if (a_col - 1, b + 1) in cells:
            out.append(("0", a_col, b))
    for j in {b - 1 for b in right} & {b - 1 for b in left}:
        out.append(("2", a_col, j))
    return sorted(set(out))


def _enum_block(edges_b, faces_b, alphabet, iface_l, iface_r):
    E = len(edges_b)
    eindex = {e: i for i, e in enumerate(edges_b)}
    order = sorted(range(E), key=lambda i: dl._strip_key(edges_b[i]))
    pos = {eindex[edges_b[order[t]]]: t for t in range(E)}
    triggers = {}
    for slots in faces_b:
        idxs = [eindex[s] for s in slots]
        triggers.setdefault(max(pos[i] for i in idxs), []).append(idxs)
    l_idx = [eindex[e] for e in iface_l]
    r_idx = [eindex[e] for e in iface_r]
    vals = [None] * E
    out = defaultdict(int)
    alpha = list(alphabet)
    nleaf = 0

    def dfs(t):
        nonlocal nleaf
        if t == E:
            out[(tuple(vals[i] for i in l_idx), tuple(vals[i] for i in r_idx))] += 1
            nleaf += 1
            return
        ei = order[t]
        tg = triggers.get(t)
        for w in alpha:
            vals[ei] = w
            if tg is not None:
                ok = True
                for idxs in tg:
                    i0, i1, i2, i3, i4, i5 = idxs
                    if (vals[i1] + vals[i2] != vals[i4] + vals[i5]
                            or vals[i0] + vals[i5] != vals[i2] + vals[i3]):
                        ok = False
                        break
                if not ok:
                    continue
            dfs(t + 1)

    old = sys.getrecursionlimit()
    sys.setrecursionlimit(max(old, E + 200))
    dfs(0)
    sys.setrecursionlimit(old)
    return out, nleaf


def count_blocked_multi(R, alphabet, cuts=None, verbose=False, tcap=None):
    """cuts = list of the final column a of each block (increasing; the last
    entry must be the maximal column). Default: about 2 columns per block."""
    cells = dl.cells_disk(R)
    cols = sorted({a for a, _ in cells})
    if cuts is None:
        cuts = cols[1::2]
        if cuts[-1] != cols[-1]:
            cuts.append(cols[-1])
    _, faces_all = dl.build_disk(R)
    faces_by_a = defaultdict(list)
    for s in faces_all:
        faces_by_a[s[0][1]].append(s)
    blocks = []
    prev = cols[0] - 1
    for c in cuts:
        cs = [a for a in cols if prev < a <= c]
        blocks.append(cs)
        prev = c
    # per-block enumeration
    cur = {(): 1}      # entry left interface = the empty tuple
    t0 = time.time()
    for bi, cs in enumerate(blocks):
        cset = set(cs)
        bfaces = [s for a in cs for s in faces_by_a[a]]
        bedges = sorted({e for s in bfaces for e in s})
        a_lo, a_hi = cs[0], cs[-1]
        iface_l = _iface_edges(cells, a_lo) if bi > 0 else []
        iface_r = _iface_edges(cells, a_hi + 1)
        bmap, nleaf = _enum_block(bedges, bfaces, alphabet, iface_l, iface_r)
        if verbose:
            print(f"  block {bi}: cols {a_lo}..{a_hi} cells={len(cs)} E={len(bedges)} "
                  f"iface={len(iface_l)}+{len(iface_r)} leaves={nleaf} "
                  f"states={len(bmap)} t={time.time()-t0:.1f}s", flush=True)
        nxt = {}
        for (ls, rs), c in bmap.items():
            base = cur.get(ls)
            if base is None:
                continue
            nxt[rs] = nxt.get(rs, 0) + base * c
        cur = nxt
        if tcap and time.time() - t0 > tcap:
            raise TimeoutError("blocked_multi time cap hit")
    total = cur.get((), 0)
    return total, len(blocks), time.time() - t0


if __name__ == "__main__":
    R = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    spec = sys.argv[2] if len(sys.argv) > 2 else "01"
    alpha = sorted({int(c) for c in spec})
    t0 = time.time()
    c, nb, dt = count_blocked_multi(R, alpha, verbose=True)
    print(f"blocked_multi R={R} alpha={spec}: {c} blocks={nb} ({time.time()-t0:.1f}s total)", flush=True)
