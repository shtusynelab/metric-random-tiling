# -*- coding: utf-8 -*-
"""
tsq_lib.py - geometry model library for the z=3 mixed-face 4.8.8 (truncated
square lattice), equiangular metric.

Construction: 4.8.8 = truncation of the square lattice. Each face of the
original square lattice becomes a regular octagon and each original vertex
becomes a square; the 4.8.8 vertices are the two truncation points on each
original lattice bond (2 per bond, distinct).
Two edge classes: axis segments (middle part of the original bond,
connecting octagon-octagon, directions 0/90 deg) and truncation edges
(connecting square-octagon, directions 45/135/225/315 deg, assigned by
quadrant). Vertices are 3-valent; torus Lq x Lr (L >= 2): V = 4F, E = 6F,
F_sq = F_oc = F.

Orientation convention: each face lists its directed edge directions CCW.
  Square (node (i,j), four quadrant truncation edges k = 0,2,4,6):
  direction (k+3) mod 8;
  octagon (cell (i,j)): its eight edge directions are exactly 0,1,...,7
  (x 45 deg).
  A shared edge appears with opposite directions in its two faces (same
  self-consistency scheme as the kagome library).

sqrt(2) handling: closure coefficients lie in {+/-1, +/-sqrt(2)/2}. For
counting (integer alphabet) each real equation
sum (a_i + b_i*sqrt(2)) s_i = 0 iff sum a_i s_i = 0 and sum b_i s_i = 0
(exact equivalence over the integers), hence the splitting into integer
rows: 2 rows per square, 4 rows per octagon. Real-space dimension (rank)
must use the float cos/sin rows (the split rows have spuriously high rank
over R and must not be used for dimension tables).
"""
from __future__ import annotations

import math
from itertools import product

# Hard budget gate (fail-closed): before materializing k^dim combinations in
# the kernel enumeration, check the count and reject outright if it exceeds
# the budget.
MAX_ENUM_STATES = 2 * 10 ** 8

import numpy as np

# Eight directions (multiples of 45 deg): real unit vector = ((a+b*sqrt(2))/2, (c+d*sqrt(2))/2)
DIR8 = [(2, 0, 0, 0), (0, 1, 0, 1), (0, 0, 2, 0), (0, -1, 0, 1),
        (-2, 0, 0, 0), (0, -1, 0, -1), (0, 0, -2, 0), (0, 1, 0, -1)]

# Node quadrants: q = 0 (NE), 1 (NW), 2 (SW), 3 (SE) -> truncation edge key k = 2q,
# connecting the two truncation points at bearings (2q, 2q+2) mod 8; a
# truncation edge exists iff its quadrant cell exists
QUAD_CELL = {0: (0, 0), 1: (-1, 0), 2: (-1, -1), 3: (0, -1)}   # relative to the node
QUAD_DIR = {0: 3, 1: 1, 2: 7, 3: 1}                            # direction family (undirected)


def _quad_of_corner(cx, cy):
    """Quadrant index of cell (cx,cy) at each of its corners: corner = (cx+dx, cy+dy)."""
    return {(0, 0): 3, (1, 0): 2, (1, 1): 1, (0, 1): 0}


# ---------------------------------------------------------------- torus

class TSQTorus:
    """4.8.8 torus Lq x Lr (numbers of original square-lattice cells, both
    >= 2).

    Attributes: nv=4F, ne=6F; faces = [(size, [(edge, dirk)...])] (CCW);
    rows_split = integer split rows (counting); rows_float = float rows
    (rank over R). Variable numbering = edge id 0..ne-1."""

    def __init__(self, Lq: int, Lr: int):
        assert Lq >= 2 and Lr >= 2, "degenerate torus (parallel edges)"
        self.Lq, self.Lr = Lq, Lr
        F = Lq * Lr
        self.F = F

        # 4.8.8 vertices: (et, i, j, s): truncation point s=0/1 of bond H(0)/V(1)
        # s=0 near the bond start (i,j), s=1 near the bond end - the two are
        # distinct vertices
        vid = {}
        for i in range(Lq):
            for j in range(Lr):
                for et in (0, 1):
                    for s in (0, 1):
                        vid[(et, i, j, s)] = len(vid)
        self.nv = len(vid)
        assert self.nv == 4 * F

        edges = {}

        def eid(key):
            if key not in edges:
                edges[key] = len(edges)
            return edges[key]

        axis = {}
        for i in range(Lq):
            for j in range(Lr):
                axis[(0, i, j)] = eid(("ax", 0, i, j))   # axis segment of H(i,j)
                axis[(1, i, j)] = eid(("ax", 1, i, j))   # axis segment of V(i,j)
        cuts = {}
        for i in range(Lq):
            for j in range(Lr):
                for q in range(4):
                    cuts[(i, j, q)] = eid(("cut", i, j, q))
        self.ne = len(edges)
        assert self.ne == 6 * F, f"E={self.ne} != 6F"
        self._vid, self._axis, self._cuts = vid, axis, cuts

        # the four bearing truncation points of node (i,j) (bearings 0/2/4/6 = E/N/W/S)
        def node_pts(i, j):
            return {0: vid[(0, i, j, 0)],          # west end of H(i,j) (east side of the node)
                    2: vid[(1, i, j, 0)],          # south end of V(i,j) (north side of the node)
                    4: vid[(0, (i - 1) % Lq, j, 1)],   # east end of H(i-1,j)
                    6: vid[(1, i, (j - 1) % Lr, 1)]}   # north end of V(i,j-1)
        self._node_pts = node_pts

        # faces: squares (one per node, 4 quadrant truncation edges, CCW:
        # E->N->W->S) + octagons (one per cell)
        faces = []
        for i in range(Lq):
            for j in range(Lr):
                pts = node_pts(i, j)
                sides = []
                for q in range(4):
                    k = 2 * q
                    sides.append((cuts[(i, j, q)], (k + 3) % 8))
                faces.append((4, sides))
        for i in range(Lq):
            for j in range(Lr):
                sides = [
                    (axis[(0, i, j)], 0),                      # south axis segment
                    (cuts[((i + 1) % Lq, j, 1)], 1),           # SE truncation (cell on the NW side of the node)
                    (axis[(1, (i + 1) % Lq, j)], 2),           # east axis segment
                    (cuts[((i + 1) % Lq, (j + 1) % Lr, 2)], 3),   # NE truncation (cell on the SW side)
                    (axis[(0, i, (j + 1) % Lr)], 4),           # north axis segment
                    (cuts[(i, (j + 1) % Lr, 3)], 5),           # NW truncation (cell on the SE side)
                    (axis[(1, i, j)], 6),                      # west axis segment
                    (cuts[(i, j, 0)], 7),                      # SW truncation (cell on the NE side)
                ]
                faces.append((8, sides))
        assert len(faces) == 2 * F

        # self-checks: degree 3, every edge in exactly 2 faces, direction
        # cycles closed, all-ones solution satisfies the rows
        deg = [0] * self.nv
        cover = {}
        for size, sides in faces:
            for e, d in sides:
                cover[e] = cover.get(e, 0) + 1
        for a, b, dd in self._edge_endpoints():
            deg[a] += 1
            deg[b] += 1
        assert all(x == 3 for x in deg), "4.8.8 vertex degree != 3"
        assert all(c == 2 for c in cover.values()) and len(cover) == self.ne, \
            "edge cover != 2"
        for size, sides in faces:
            s = [0, 0, 0, 0]
            for e, d in sides:
                a, b, c, dd = DIR8[d]
                s[0] += a; s[1] += b; s[2] += c; s[3] += dd
            assert s == [0, 0, 0, 0], "face direction cycle not closed"
        self.faces = faces
        self.rows_split = self._build_rows(split=True)
        self.rows_float = self._build_rows(split=False)

    def _edge_endpoints(self):
        """List of (a, b, dir_family) triples (for the degree self-check)."""
        Lq, Lr, vid, axis, cuts = self.Lq, self.Lr, self._vid, self._axis, \
            self._cuts
        out = []
        node_pts = self._node_pts
        for (et, i, j), e in axis.items():
            if et == 0:
                out.append((vid[(0, i, j, 0)], vid[(0, i, j, 1)], 0))
            else:
                out.append((vid[(1, i, j, 0)], vid[(1, i, j, 1)], 2))
        for (i, j, q), e in cuts.items():
            k = 2 * q
            pts = node_pts(i, j)
            out.append((pts[k], pts[(k + 2) % 8], QUAD_DIR[q]))
        return out

    def _build_rows(self, split):
        ncol = self.ne
        rows = []
        for size, sides in self.faces:
            if split:
                rx, rx2, ry, ry2 = ([0] * ncol for _ in range(4))
                for e, d in sides:
                    a, b, c, dd = DIR8[d]
                    rx[e] += a; rx2[e] += b; ry[e] += c; ry2[e] += dd
                for r in (rx, rx2, ry, ry2):
                    if any(r):
                        rows.append(r)
            else:
                fx, fy = [0.0] * ncol, [0.0] * ncol
                for e, d in sides:
                    a, b, c, dd = DIR8[d]
                    fx[e] += (a + b * math.sqrt(2)) / 2
                    fy[e] += (c + dd * math.sqrt(2)) / 2
                rows.append(fx)
                rows.append(fy)
        return rows


# ---------------------------------------------------------------- patches
# W x L unit-cell patch with open boundaries (staircase boundary): faces =
# all W*L octagons plus (W-1)(L-1) interior squares; truncation edges exist
# per quadrant (a truncation edge exists iff its quadrant cell exists);
# boundary axis segments / truncation edges belong to a single face.
# Euler: V - E + F = 1.

def tsq_patch(W, L):
    """Returns (faces, edge_count):
      faces = [(size, [(edge, dirk)...])] (CCW, row-major order: octagons
      row by row, then squares row by row - the scan order for the strip
      DP)."""
    vid = {}

    def vend(et, i, j, s):
        key = (et, i, j, s)
        if key not in vid:
            vid[key] = len(vid)
        return vid[key]

    edges = {}

    def eid(key):
        if key not in edges:
            edges[key] = len(edges)
        return edges[key]

    def node_pts(i, j):
        pts = {}
        if i < W: pts[0] = vend(0, i, j, 0)
        if j < L: pts[2] = vend(1, i, j, 0)
        if i > 0: pts[4] = vend(0, i - 1, j, 1)
        if j > 0: pts[6] = vend(1, i, j - 1, 1)
        return pts

    def quad_exists(i, j, q):
        dx, dy = QUAD_CELL[q]
        cx, cy = i + dx, j + dy
        return 0 <= cx < W and 0 <= cy < L

    axis = {}
    for i in range(W):          # H(i,j): x runs from i to i+1, W segments per row
        for j in range(L + 1):  # rows y = 0..L (including north/south boundaries)
            axis[(0, i, j)] = eid(("ax", 0, i, j))
    for i in range(W + 1):      # V(i,j): on the vertical line x=i, W+1 lines
        for j in range(L):      # segments y = 0..L-1
            axis[(1, i, j)] = eid(("ax", 1, i, j))
    cuts = {}
    for i in range(W + 1):
        for j in range(L + 1):
            for q in range(4):
                if quad_exists(i, j, q):
                    cuts[(i, j, q)] = eid(("cut", i, j, q))

    faces = []
    # octagons row by row
    for j in range(L):
        for i in range(W):
            sides = [
                (axis[(0, i, j)], 0),
                (cuts[(i + 1, j, 1)], 1),
                (axis[(1, i + 1, j)], 2),
                (cuts[(i + 1, j + 1, 2)], 3),
                (axis[(0, i, j + 1)], 4),
                (cuts[(i, j + 1, 3)], 5),
                (axis[(1, i, j)], 6),
                (cuts[(i, j, 0)], 7),
            ]
            faces.append((8, [s for s in sides]))
    # interior squares row by row
    for j in range(1, L):
        for i in range(1, W):
            sides = []
            for q in range(4):
                k = 2 * q
                sides.append((cuts[(i, j, q)], (k + 3) % 8))
            faces.append((4, sides))

    # self-checks: Euler + every edge in exactly 2 faces + direction closure
    n_vars = len(edges)
    cover = {}
    for size, sides in faces:
        s = [0, 0, 0, 0]
        for e, d in sides:
            cover[e] = cover.get(e, 0) + 1
            a, b, c, dd = DIR8[d]
            s[0] += a; s[1] += b; s[2] += c; s[3] += dd
        assert s == [0, 0, 0, 0], "patch face direction cycle not closed"
    assert set(cover.values()) <= {1, 2} and len(cover) == n_vars, \
        "patch edge cover invalid (boundary edges covered once, interior twice)"
    n_v = 2 * (W * (L + 1) + (W + 1) * L)   # 2 truncation points per original bond
    assert n_v - n_vars + len(faces) == 1, \
        f"patch Euler failed: {n_v} - {n_vars} + {len(faces)}"
    return faces, n_vars


def tsq_patch_rows(faces, n_vars):
    """Split rows (counting) and float rows (dimension) of a patch."""
    rows_i, rows_f = [], []
    for size, sides in faces:
        rx, rx2, ry, ry2 = ([0] * n_vars for _ in range(4))
        fx, fy = [0.0] * n_vars, [0.0] * n_vars
        for e, d in sides:
            a, b, c, dd = DIR8[d]
            rx[e] += a; rx2[e] += b; ry[e] += c; ry2[e] += dd
            fx[e] += (a + b * math.sqrt(2)) / 2
            fy[e] += (c + dd * math.sqrt(2)) / 2
        for r in (rx, rx2, ry, ry2):
            if any(r):
                rows_i.append(r)
        rows_f.append(fx)
        rows_f.append(fy)
    return rows_i, rows_f


# ---------------------------------------------------------------- counting

def kernel_count(rows, alphabet):
    """Exact count of kernel(rows) intersect alphabet^n for an integer-row
    system (Fraction elimination + integer basis via common denominator +
    numpy enumeration; same scheme as kagome_lib.kernel_count_int, copied
    standalone to avoid a cross-library dependency)."""
    from fractions import Fraction
    from itertools import product
    ncol = len(rows[0])
    mat = [[Fraction(x) for x in r] for r in rows]
    m = len(mat)
    pivots = []
    r = 0
    for c in range(ncol):
        p = None
        for rr in range(r, m):
            if mat[rr][c] != 0:
                p = rr
                break
        if p is None:
            continue
        mat[r], mat[p] = mat[p], mat[r]
        pv = mat[r][c]
        mat[r] = [x / pv for x in mat[r]]
        for rr in range(m):
            if rr != r and mat[rr][c]:
                f = mat[rr][c]
                mat[rr] = [x - f * y for x, y in zip(mat[rr], mat[r])]
        pivots.append(c)
        r += 1
        if r == m:
            break
    free = [c for c in range(ncol) if c not in set(pivots)]
    if not free:
        return 1 if all(all(x == 0 for x in row) for row in rows) else 0
    basis = []
    for fc in free:
        vec = [Fraction(0)] * ncol
        vec[fc] = Fraction(1)
        for i, pc in enumerate(pivots):
            vec[pc] = -mat[i][fc]
        basis.append(vec)
    D = 1
    for vec in basis:
        for x in vec:
            if x:
                D = D * x.denominator // math.gcd(D, x.denominator)
    B = np.array([[int(x * D) for x in vec] for vec in basis], dtype=np.int64)
    A = np.array(sorted(set(alphabet)), dtype=np.int64)
    if (len(A) ** len(free)) * B.shape[1] > MAX_ENUM_STATES:
        raise RuntimeError(
            f"kernel_count refusing to run: {len(A)}^{len(free)} x {B.shape[1]} columns "
            f"> budget {MAX_ENUM_STATES} (fail-closed budget gate)")
    combos = np.array(list(product(A, repeat=len(free))), dtype=np.int64)
    X = combos @ B
    ok = (X % D == 0)
    return int(np.count_nonzero((ok & np.isin(X // D, A)).all(axis=1)))


def strip_count(faces, n_vars, alphabet, max_states=None):
    """Exact strip counting for 4.8.8: frontier DP (faces must be given in
    scan order).

    Per-face constraint = the split-row matrix (4 x |sides|; rows =
    rational-x / sqrt2-x / rational-y / sqrt2-y). Returns
    (omega, peak_states)."""
    nh = len(faces)
    frows = []
    for size, sides in faces:
        R = np.zeros((4, len(sides)), dtype=np.int64)
        for t, (e, d) in enumerate(sides):
            a, b, c, dd = DIR8[d]
            R[0, t] = a; R[1, t] = b; R[2, t] = c; R[3, t] = dd
        frows.append((sides, R))
    hosts = {}
    for hi, (sides, R) in enumerate(frows):
        for e, d in sides:
            hosts.setdefault(e, []).append(hi)
    first = {v: min(x) for v, x in hosts.items()}
    last = {v: max(x) for v, x in hosts.items()}
    A = np.asarray(list(alphabet), dtype=np.int64)
    k = len(A)

    def pending(i):
        return sorted(v for v in first if first[v] <= i - 1 and last[v] >= i)

    def after(i):
        return sorted(v for v in first if first[v] <= i and last[v] >= i + 1)

    states = np.zeros((1, 0), dtype=np.int64)
    counts = [1]
    peak = 0
    for i in range(nh):
        sides, R = frows[i]
        before = pending(i)
        bidx = {v: j for j, v in enumerate(before)}
        new_e = [e for e, d in sides if first[e] == i]
        aft = after(i)
        carr = np.asarray(counts, dtype=np.int64)
        out_vals, out_cnts = [], []
        for combo in product(range(k), repeat=len(new_e)):
            newval = {e: int(A[c]) for e, c in zip(new_e, combo)}
            eq4 = np.zeros((4, len(states)), dtype=np.int64)
            for t, (e, d) in enumerate(sides):
                coefs = R[:, t]
                if e in newval:
                    eq4 += coefs[:, None] * newval[e]
                else:
                    eq4 += coefs[:, None] * states[:, bidx[e]]
            valid = (eq4 == 0).all(axis=0)
            if not valid.any():
                continue
            sub = states[valid]
            subc = carr[valid]
            cols = np.empty((len(sub), len(aft)), dtype=np.int64)
            for j, v in enumerate(aft):
                if v in newval:
                    cols[:, j] = newval[v]
                else:
                    cols[:, j] = sub[:, bidx[v]]
            out_vals.append(cols)
            out_cnts.append(subc)
        if out_vals:
            raw = np.vstack(out_vals)
            rawc = np.concatenate(out_cnts)
        else:
            raw = np.zeros((0, len(aft)), dtype=np.int64)
            rawc = np.zeros(0, dtype=np.int64)
        merged = {}
        merged_rows = []
        for row, c in zip(raw, rawc):
            key = row.tobytes()
            if key in merged:
                merged[key] += int(c)
            else:
                merged[key] = int(c)
                merged_rows.append(row)
        states = (np.vstack(merged_rows) if merged_rows
                  else np.zeros((0, len(aft)), dtype=np.int64))
        counts = list(merged.values())
        peak = max(peak, len(states))
        if max_states is not None and len(states) > max_states:
            raise RuntimeError(f"strip states {len(states)} > {max_states}")
    return sum(counts), peak


# ---------------------------------------------------------------- self-tests

def _selftest():
    for (Lq, Lr) in [(2, 2), (3, 2), (2, 3), (3, 3)]:
        hc = TSQTorus(Lq, Lr)
        # split-row system: the all-ones solution must satisfy it (per-face
        # direction cycle closure)
        for r in hc.rows_split:
            assert sum(r) == 0, "all-ones not a solution"
        # count >= 1 (all-ones); the Omega(2,2) vs small-patch TM cross-check
        # lives in the verification scripts
        print(f"torus {Lq}x{Lr}: F={hc.F} nv={hc.nv} ne={hc.ne} OK")
    for (W, L) in [(2, 2), (3, 2), (2, 3)]:
        faces, n_vars = tsq_patch(W, L)
        rows_i, rows_f = tsq_patch_rows(faces, n_vars)
        print(f"patch {W}x{L}: F={len(faces)} E={n_vars} rows={len(rows_i)} OK")


if __name__ == "__main__":
    _selftest()
