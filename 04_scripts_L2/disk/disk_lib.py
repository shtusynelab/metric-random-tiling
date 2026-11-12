# -*- coding: utf-8 -*-
"""disk_lib.py -- honeycomb metric tilings: exact counting library for
open-boundary disk patches (v2).

Three independent paths (disk):
  count_dfs          -- full DFS enumeration (small-R anchors)
  count_dp_generic   -- generic key-merged frontier DP (edge by edge,
                        live-set window; strip order)
  count_dp_column    -- column-recurrence DP (column state = the F0/F2
                        values of that column; per cell u = l4 - l1 then
                        l3 = l0 - u, l2 = l5 + u)
Torus cross-anchors:
  build_torus_mine / count_torus_dfs (compared bitwise against the
  build_torus + metric_hc pipeline)
  count_torus_column(Lx, Ly, k) -- column transfer-matrix trace (independent
  third path; exact torus counts for arbitrary Lx)

Geometry / conventions (same as build_torus / metric_hc):
  cell (a,b); three edge families
    F0(a,b) = A(a,b)-B(a,b)
    F1(a,b) = A(a+1,b)-B(a,b)
    F2(a,b) = A(a,b+1)-B(a,b)
  face (a,b) slot order = [F0(a,b), F1(a,b), F2(a+1,b-1), F0(a+1,b-1),
                           F1(a,b-1), F2(a,b-1)]
  two closure equations (unit-coefficient form of SI S1.2):
    eqA: l1+l2 = l4+l5;  eqB': l0+l5 = l2+l3
  (the constraint set is invariant under rotation/reversal of the slot
  order -- equivalent to any legal face walk order)
Disk = cube coords (x,y,z)=(a+b,-a,-b) with max-norm <= R: N = 3R(R+1)+1.
"""
from __future__ import annotations

from itertools import product


# ----------------------------------------------------------------------
# Structural building blocks
# ----------------------------------------------------------------------

def cells_disk(R):
    cells = set()
    for a in range(-2 * R, 2 * R + 1):
        for b in range(-2 * R, 2 * R + 1):
            if max(abs(a + b), abs(a), abs(b)) <= R:
                cells.add((a, b))
    return cells


def _slots_disk(a, b):
    return (("0", a, b), ("1", a, b), ("2", a + 1, b - 1),
            ("0", a + 1, b - 1), ("1", a, b - 1), ("2", a, b - 1))


def _slots_torus(a, b, Lx, Ly):
    def mk(k, x, y):
        return (k, x % Lx, y % Ly)
    return (mk("0", a, b), mk("1", a, b), mk("2", a + 1, b - 1),
            mk("0", a + 1, b - 1), mk("1", a, b - 1), mk("2", a, b - 1))


def build_disk(R):
    cells = cells_disk(R)
    edge_set, faces = set(), []
    for (a, b) in sorted(cells):
        slots = _slots_disk(a, b)
        faces.append(slots)
        edge_set.update(slots)
    edges = sorted(edge_set)
    return edges, faces


def build_torus_mine(Lx, Ly):
    edge_set, faces = set(), []
    for a in range(Lx):
        for b in range(Ly):
            slots = _slots_torus(a, b, Lx, Ly)
            assert len(set(slots)) == 6, f"degenerate face ({a},{b})"
            faces.append(slots)
            edge_set.update(slots)
    edges = sorted(edge_set)
    return edges, faces


# ----------------------------------------------------------------------
# Edge order: strip order (column-advancing: F0/F2(a,*) first, then F1(a,*))
# ----------------------------------------------------------------------

def _strip_key(e):
    kind, a, b = e
    return (2 * a + 1, 0, b) if kind == "1" else (2 * a, 0 if kind == "0" else 1, b)


def _strip_key_torus(e, Ly):
    kind, a, b = e
    # b in cyclic order (b in Z_Ly): numeric order mod Ly suffices
    return (2 * a + 1, 0, b % Ly) if kind == "1" else (2 * a, 0 if kind == "0" else 1, b % Ly)


def default_order_disk(edges):
    return sorted(range(len(edges)), key=lambda i: _strip_key(edges[i]))


def default_order_torus(edges, Ly):
    return sorted(range(len(edges)), key=lambda i: _strip_key_torus(edges[i], Ly))


# ----------------------------------------------------------------------
# Path 1: full DFS enumeration
# ----------------------------------------------------------------------

def count_dfs(edges, faces, alphabet, order=None):
    E = len(edges)
    eindex = {e: i for i, e in enumerate(edges)}
    if order is None:
        order = default_order_disk(edges)
    pos = {eindex[edges[order[t]]]: t for t in range(E)}
    triggers = {}
    for slots in faces:
        idxs = [eindex[s] for s in slots]          # indices into edges
        tmax = max(pos[i] for i in idxs)           # completion step (position)
        triggers.setdefault(tmax, []).append(idxs)
    vals = [None] * E
    alpha = list(alphabet)
    count = 0

    def dfs(t):
        nonlocal count
        if t == E:
            count += 1
            return
        ei = order[t]
        for w in alpha:
            vals[ei] = w
            ok = True
            for idxs in triggers.get(t, ()):
                i0, i1, i2, i3, i4, i5 = idxs
                if (vals[i1] + vals[i2] != vals[i4] + vals[i5]
                        or vals[i0] + vals[i5] != vals[i2] + vals[i3]):
                    ok = False
                    break
            if ok:
                dfs(t + 1)

    import sys
    old = sys.getrecursionlimit()
    sys.setrecursionlimit(max(old, E + 200))
    dfs(0)
    sys.setrecursionlimit(old)
    return count


# ----------------------------------------------------------------------
# Path 2: generic key-merged frontier DP
# ----------------------------------------------------------------------

def count_dp_generic(edges, faces, alphabet, order=None, stats=None):
    E = len(edges)
    if order is None:
        order = default_order_disk(edges)
    eindex = {e: i for i, e in enumerate(edges)}
    step_of = [0] * E
    for t in range(E):
        step_of[eindex[edges[order[t]]]] = t
    tdead = [0] * E
    trig_by_step = {}
    for slots in faces:
        idxs = [eindex[s] for s in slots]
        tmax = max(step_of[i] for i in idxs)
        trig_by_step.setdefault(tmax, []).append(idxs)
        for i in idxs:
            if tmax > tdead[i]:
                tdead[i] = tmax
    # live_after[t]
    live_after = []
    cur = []
    for t in range(E):
        cur.append(eindex[edges[order[t]]])
        cur = [x for x in cur if t < tdead[x]]
        live_after.append(sorted(cur))
    alpha = list(alphabet)
    dp = {(): 1}
    for t in range(E):
        ei = eindex[edges[order[t]]]
        live_next = live_after[t]
        live_prev = live_after[t - 1] if t > 0 else []
        idx_prev = {e: k for k, e in enumerate(live_prev)}
        # framing: origin of each live_next element
        # (new edge = -1, otherwise an index into live_prev)
        src = [-1 if e == ei else idx_prev[e] for e in live_next]
        trigs = []
        for idxs in trig_by_step.get(t, ()):
            trigs.append([-1 if i == ei else idx_prev[i] for i in idxs])
        nxt = {}
        nxt_get = nxt.get
        for state, cnt in dp.items():
            for w in alpha:
                ok = True
                for place in trigs:
                    fv = [w if p == -1 else state[p] for p in place]
                    if (fv[1] + fv[2] != fv[4] + fv[5]
                            or fv[0] + fv[5] != fv[2] + fv[3]):
                        ok = False
                        break
                if not ok:
                    continue
                key = tuple(w if s == -1 else state[s] for s in src)
                nxt[key] = nxt_get(key, 0) + cnt
        dp = nxt
        if stats is not None:
            stats.append(len(dp))
    return sum(dp.values())


# ----------------------------------------------------------------------
# Path 3: column-recurrence DP (disk)
# ----------------------------------------------------------------------

def _col_lists(cells, cols):
    A, C, B, sa = {}, {}, {}, {}
    for a in cols:
        sa[a] = sorted(b for (aa, b) in cells if aa == a)
        prev = sorted(b for (aa, b) in cells if aa == a - 1)
        A[a] = sorted(set(sa[a]) | {b - 1 for b in prev})
        C[a] = sorted({b - 1 for b in sa[a]} | {b - 1 for b in prev})
        B[a] = sorted(set(sa[a]) | {b - 1 for b in sa[a]})
    return A, C, B, sa


def count_dp_column(R, alphabet, stats=None):
    cells = cells_disk(R)
    cols = sorted({a for a, _ in cells})
    A, C, B, sa = _col_lists(cells, cols)
    alpha = sorted(alphabet)
    aset = set(alpha)
    # virtual column at last + 1 (all of its A/C entries are determined edges)
    cols_ext = cols + [cols[-1] + 1]
    A[cols[-1] + 1] = sorted({b - 1 for b in sa[cols[-1]]})
    C[cols[-1] + 1] = sorted({b - 1 for b in sa[cols[-1]]})

    # precompute the per-column transfer structure
    plans = []
    for ai, a in enumerate(cols):
        an = cols_ext[ai + 1]
        An, Cn = A[an], C[an]
        s = sa[a]
        posAn = {b - 1: An.index(b - 1) for b in s}
        posCn = {b - 1: Cn.index(b - 1) for b in s}
        posA = {b: A[a].index(b) for b in s}
        posC = {b - 1: C[a].index(b - 1) for b in s}
        Bl = B[a]
        posB4 = {b - 1: Bl.index(b - 1) for b in s}
        posB1 = {b: Bl.index(b) for b in s}
        detA = set(posAn.values())
        detC = set(posCn.values())
        freeA = [k for k in range(len(An)) if k not in detA]
        freeC = [k for k in range(len(Cn)) if k not in detC]
        # the cell to check for each B position j (when its value == b);
        # None = no check
        cell_at = [None] * len(Bl)
        for b in s:
            cell_at[posB1[b]] = b
        plans.append(dict(a=a, an=an, s=s, Bl=Bl, cell_at=cell_at, posA=posA,
                          posC=posC, posB4=posB4, posB1=posB1, posAn=posAn,
                          posCn=posCn, freeA=freeA, freeC=freeC,
                          nA=len(An), nC=len(Cn), nB=len(Bl)))

    # initial column entirely free
    nA0, nC0 = len(A[cols[0]]), len(C[cols[0]])
    dp = {}
    for combA in product(alpha, repeat=nA0):
        for combC in product(alpha, repeat=nC0):
            dp[(combA, combC)] = 1

    one = alpha  # value set of the free positions
    for pl in plans:
        nA, nC = pl["nA"], pl["nC"]
        freeA, freeC = pl["freeA"], pl["freeC"]
        nfree = len(freeA) + len(freeC)
        dd = dp
        dpn = {}
        for (sA, sC), cnt in dd.items():
            out = []  # (nA tuple, nC tuple) collector (recursive, pruned)

            def rec(j, Bvals):
                if j == pl["nB"]:
                    nAv = [None] * nA
                    nCv = [None] * nC
                    for b in pl["s"]:
                        u = Bvals[pl["posB4"][b - 1]] - Bvals[pl["posB1"][b]]
                        nAv[pl["posAn"][b - 1]] = sA[pl["posA"][b]] - u
                        nCv[pl["posCn"][b - 1]] = sC[pl["posC"][b - 1]] + u
                    # free-position enumeration
                    for fill in product(one, repeat=nfree):
                        for k, fk in enumerate(freeA):
                            nAv[fk] = fill[k]
                        for k, fk in enumerate(freeC):
                            nCv[fk] = fill[len(freeA) + k]
                        key = (tuple(nAv), tuple(nCv))
                        dpn[key] = dpn.get(key, 0) + cnt
                    return
                b = pl["cell_at"][j]
                for w in one:
                    if b is not None:
                        u = Bvals[pl["posB4"][b - 1]] - w
                        if (sA[pl["posA"][b]] - u) not in aset:
                            continue
                        if (sC[pl["posC"][b - 1]] + u) not in aset:
                            continue
                    Bvals.append(w)
                    rec(j + 1, Bvals)
                    Bvals.pop()

            rec(0, [])
        dp = dpn
        if stats is not None:
            stats.append(len(dp))
    return sum(dp.values())


# ----------------------------------------------------------------------
# Torus: DFS cross-anchor + column trace (third path)
# ----------------------------------------------------------------------

def count_torus_dfs(Lx, Ly, alphabet):
    edges, faces = build_torus_mine(Lx, Ly)
    return count_dfs(edges, faces, alphabet, order=default_order_torus(edges, Ly))


def count_torus_column(Lx, Ly, alphabet, stats=None):
    """Column transfer-matrix trace: state = (A-column Ly values, C-column
    Ly values); the B column is free; run Lx steps and close the loop."""
    alpha = sorted(alphabet)
    aset = set(alpha)
    # single-step transfer: (sA,sC) -> dict[(nA,nC)] = multiplicity
    def step_map():
        trans = {}
        for sA in product(alpha, repeat=Ly):
            for sC in product(alpha, repeat=Ly):
                out = {}
                for Bv in product(alpha, repeat=Ly):
                    # B indices by the numeric position in the cyclic order
                    # of b in Z_Ly (0..Ly-1)
                    nA = [None] * Ly
                    nC = [None] * Ly
                    ok = True
                    for b in range(Ly):
                        u = Bv[(b - 1) % Ly] - Bv[b]
                        va = sA[b] - u
                        vc = sC[(b - 1) % Ly] + u
                        if va not in aset or vc not in aset:
                            ok = False
                            break
                        # new A-column index = b-1, i.e. F0(an, b-1)
                        nA[(b - 1) % Ly] = va
                        nC[(b - 1) % Ly] = vc
                    if ok:
                        key = (tuple(nA), tuple(nC))
                        out[key] = out.get(key, 0) + 1
                trans[(sA, sC)] = out
        return trans

    trans = step_map()
    states = list(trans.keys())
    idx = {s: k for k, s in enumerate(states)}
    n = len(states)
    # dp vector: start from every state (initial weight 1)
    dp = [0] * n
    # compute Trace(T^Lx) = sum_s (T^Lx)[s, s]: vector iteration per s,
    # i.e. sum_s e_s T^Lx e_s
    total = 0
    cnt_states = []
    for s0 in states:
        vec = {idx[s0]: 1}
        for _ in range(Lx):
            nv = {}
            for si, c in vec.items():
                for tstate, mult in trans[states[si]].items():
                    nv[idx[tstate]] = nv.get(idx[tstate], 0) + c * mult
            vec = nv
        total += vec.get(idx[s0], 0)
        if stats is not None:
            cnt_states.append(len(vec))
    return total
