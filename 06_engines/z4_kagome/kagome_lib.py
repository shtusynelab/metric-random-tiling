# -*- coding: utf-8 -*-
"""
kagome_lib.py -- library for the z=4 kagome isogonal metric disorder model.

Model (kagome = medial graph of the honeycomb / truncation construction):
  kagome vertices = honeycomb bonds (undirected); kagome edges = "angle
  blocks between two adjacent bonds at a honeycomb vertex". Each kagome
  edge belongs to exactly one triangle (at a honeycomb vertex) and one
  hexagon (at a honeycomb face).
  V_kag = E_HC = 3F, E_kag = 3*V_HC = 6F, 4-valent,
  faces = triangles (2F) + hexagons (F).

Isogonal metric constraints (all linear):
  - the three sides of a triangle lie in 3 fixed direction families with
    a 60 deg interior angle => the three sides are forced equal, bound
    into a single variable t per honeycomb vertex (exact elimination,
    not an approximation);
  - hexagon side lengths = the t of the 6 perimeter triangles; interior
    angle 120 deg => 2 pair-sum-type closure equations per face (type
    t1+t2 = t4+t5, same family as the honeycomb).
  Reduced system: 2F variables (honeycomb vertices), 2F constraints
  (2 per hexagon) -- exactly square (critical).

Coordinate conventions: all geometry is represented by integer pairs on
the triangular-lattice basis (e1, e2) = ((1,0), (1/2, sqrt(3)/2))
(positions use doubled integers to avoid half-integer points); direction
unit vectors are written as (sqrt(3)/2 * a, 1/2 * c) for small integer
pairs (a, c) => after multiplying the closure equations by 2/sqrt(3) and
2 all coefficients are integers (exact arithmetic available).

Torus: hexagons (q, r) in [0,Lq) x [0,Lr), center c = q*g1 + r*g2 with
g1 = (1,1), g2 = (-1,2) in basis coordinates (60 deg apart in real space).
Vertices/bonds are normalized by "lexicographically smallest orbit
representative" (frozenset dedup is not allowed on the torus: parallel
edges would be wrongly merged).
"""
from __future__ import annotations

import math
from fractions import Fraction
from itertools import product

# hard budget gate (fail-closed): check before materializing k^dim combos
# in kernel enumeration; reject outright when over the limit
MAX_ENUM_STATES = 2 * 10 ** 8

import numpy as np

# ---------------------------------------------------------------- constants

# six bond directions (integer pairs in basis coordinates; unit vectors at
# 0/60/.../300 degrees in real space)
U = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]
G1 = (1, 1)     # u0 + u1
G2 = (-1, 2)    # u1 + u2

# integer representation of direction unit vectors: real = (sqrt(3)/2 * a, 1/2 * c)
AC_OF_U = {u: ((1, 1), (0, 2), (-1, 1), (-1, -1), (0, -2), (1, -1))[i]
           for i, u in enumerate(U)}


def _add(p, v, n=1):
    return (p[0] + n * v[0], p[1] + n * v[1])


# ---------------------------------------------------------------- torus construction

class KagomeTorus:
    """Kagome combinatorial structure + isogonal closure constraints on an
    Lq x Lr hexagonal torus.

    Attributes (honeycomb layer):
      nv, nb, nf      numbers of vertices/bonds/hexagon faces (= 2F, 3F, F)
      vertices        list of vertex ids
      vertex_pos      vid -> doubled-integer coordinates (basis coordinates)
      bonds           list of (va, vb) (bonds = kagome vertices)
      hex_corners     fid -> [vid x6] cyclic
    kagome layer:
      kv_deg          degree of each kagome vertex (= bond), always 4
      kedges          list of (hv, bi, bj) (hv = honeycomb vertex, bond pair bi<bj)
      kedge_dir       kedge -> (a, c) integer direction pair
      triangles       honeycomb vertex -> [kedge x3] (the triangle's sides)
      hex_kedges      fid -> [kedge x6] cyclic (hexagon side length = endpoint triangle t)
    Constraints:
      M_reduced       2F x 2F integer matrix (columns = honeycomb-vertex t)
      A_full          6F x 6F integer matrix (columns = kagome edge lengths;
                      4F triangle rows + 2F hexagon rows)
    """

    def __init__(self, Lq: int, Lr: int):
        # when Lq or Lr = 1 a hexagon shares an edge with itself (parallel
        # bonds); endpoint-pair dedup would wrongly merge them, so the
        # minimal non-degenerate torus = 2x2
        assert Lq >= 2 and Lr >= 2, "degenerate torus: parallel bonds"
        self.Lq, self.Lr = Lq, Lr
        F = Lq * Lr
        self.F = F

        # ---- vertex normalization: planar coordinates of corner slot
        # (q,r,k) are p = q*G1 + r*G2 + U[k]; take the orbit under the
        # period lattice L = <Lq*G1, Lr*G2>; among all valid representations
        # in the orbit ((x*G1 + y*G2 + U[m]) with (x,y) mod Lq/Lr) pick the
        # lexicographically smallest as the key.
        # (Do NOT normalize by "host hexagon": after wraparound the
        # combinatorial slots and the geometric orbit would mismatch.)
        def vkey(q, r, k):
            p = (q * G1[0] + r * G2[0] + U[k][0],
                 q * G1[1] + r * G2[1] + U[k][1])
            best = None
            for m in range(6):
                w = (p[0] - U[m][0], p[1] - U[m][1])
                # solve w = c1*G1 + c2*G2 (G1, G2 integral basis, det = 3)
                det = G1[0] * G2[1] - G1[1] * G2[0]
                c1n = w[0] * G2[1] - w[1] * G2[0]
                c2n = G1[0] * w[1] - G1[1] * w[0]
                if c1n % det or c2n % det:
                    continue
                c1, c2 = c1n // det, c2n // det
                key = (c1 % Lq, c2 % Lr, m)
                if best is None or key < best:
                    best = key
            assert best is not None, "corner orbit representation not found"
            return best

        vid_map = {}
        hex_corners = []
        slot_pos = []   # slot_pos[fid][k] = exact planar coordinates of corner k of that hexagon
        for q, r in product(range(Lq), range(Lr)):
            row = []
            prow = []
            for k in range(6):
                key = vkey(q, r, k)
                if key not in vid_map:
                    vid_map[key] = len(vid_map)
                row.append(vid_map[key])
                prow.append((q * G1[0] + r * G2[0] + U[k][0],
                             q * G1[1] + r * G2[1] + U[k][1]))
            hex_corners.append(row)
            slot_pos.append(prow)
        self.nv = len(vid_map)
        assert self.nv == 2 * F, f"V={self.nv} != 2F={2 * F}"
        self.hex_corners = hex_corners
        self.vertices = list(range(self.nv))

        # orbit representative positions (reference only; geometry always
        # uses exact slot coordinates)
        vpos = {}
        for (x, y, m), v in vid_map.items():
            p = _add(_add((0, 0), G1, x), G2, y)
            vpos[v] = (2 * (p[0] + U[m][0]), 2 * (p[1] + U[m][1]))
        self.vertex_pos = vpos

        # ---- bonds (= kagome vertices); direction = difference of adjacent
        # corner-slot planar coordinates (exact)
        bond_dir = {}   # (a, b) with a<b -> planar delta (from a toward b), in +/-U
        for fid, row in enumerate(hex_corners):
            for k in range(6):
                a, b = row[k], row[(k + 1) % 6]
                key = (min(a, b), max(a, b))
                d = (slot_pos[fid][(k + 1) % 6][0] - slot_pos[fid][k][0],
                     slot_pos[fid][(k + 1) % 6][1] - slot_pos[fid][k][1])
                assert d in U or _add(d, (0, 0), -1) in U, \
                    f"bond delta {d} not a unit vector"
                if a > b:
                    d = (-d[0], -d[1])
                if key in bond_dir:
                    assert bond_dir[key] == d, \
                        f"bond {key} direction inconsistent across hexagons"
                bond_dir[key] = d
        self.bonds = sorted(bond_dir)
        self.nb = len(self.bonds)
        assert self.nb == 3 * F, f"E_HC={self.nb} != 3F"
        deg = [0] * self.nv
        for a, b in self.bonds:
            deg[a] += 1
            deg[b] += 1
        assert all(d == 3 for d in deg), "honeycomb vertex degree != 3"
        self.bond_id = {bd: i for i, bd in enumerate(self.bonds)}
        # away[(v, bi)] = unit direction of bond bi at its honeycomb-vertex-v
        # end, pointing away from v
        away = {}
        for (a, b), d in bond_dir.items():
            bi = self.bond_id[(a, b)]
            away[(a, bi)] = d
            away[(b, bi)] = (-d[0], -d[1])

        # ---- kagome edges: adjacent bond pairs at each honeycomb vertex
        vertex_bonds = {v: [] for v in self.vertices}
        for bi, (a, b) in enumerate(self.bonds):
            vertex_bonds[a].append(bi)
            vertex_bonds[b].append(bi)
        kedge_id = {}
        triangles = {}
        for v in self.vertices:
            bs = sorted(vertex_bonds[v])
            assert len(bs) == 3
            tr = []
            for i, j in ((0, 1), (1, 2), (0, 2)):
                key = (v, bs[i], bs[j])
                kedge_id[key] = len(kedge_id)
                tr.append(key)
            triangles[v] = tr
        self.kedges = sorted(kedge_id, key=kedge_id.get)
        self.nek = len(self.kedges)
        assert self.nek == 6 * F, f"E_kag={self.nek} != 6F"
        self.triangles = triangles

        # kagome vertex degree = 4 (each bond joins 2 bond pairs at each end)
        kv_deg = [0] * self.nb
        for (v, bi, bj) in self.kedges:
            kv_deg[bi] += 1
            kv_deg[bj] += 1
        assert all(d == 4 for d in kv_deg), "kagome vertex degree != 4"
        self.kv_deg = kv_deg

        # the 6 kagome edges of each hexagon (= bond pairs at its corners)
        hex_kedges = []
        for fid, row in enumerate(hex_corners):
            ks = []
            for k in range(6):
                ba = self.bond_id[(min(row[k], row[(k - 1) % 6]),
                                      max(row[k], row[(k - 1) % 6]))]
                bb = self.bond_id[(min(row[k], row[(k + 1) % 6]),
                                      max(row[k], row[(k + 1) % 6]))]
                ks.append((row[k], min(ba, bb), max(ba, bb)))
            hex_kedges.append(ks)
        # each kagome edge belongs to exactly one hexagon
        cover = {}
        for ks in hex_kedges:
            for ke in ks:
                cover[ke] = cover.get(ke, 0) + 1
        assert all(c == 1 for c in cover.values()) and len(cover) == self.nek, \
            "kagome edge hexagon-cover != 1"
        self.hex_kedges = hex_kedges

        # ---- face cyclic orientation and directions (integer increments in
        # basis coordinates; real space (d0,d1) -> (d0 + d1/2, sqrt(3)*d1/2)).
        # Directions are taken directly from the CCW cyclic order of the face
        # boundary, not from sorting bond pairs (sorting indices are arbitrary;
        # signs would decouple from the cyclic order).
        def angle_key(d):
            # real-space azimuth sorting key (only for CCW sorting, never
            # enters equations)
            return math.atan2(3 ** 0.5 * d[1], 2 * d[0] + d[1])

        kid = {ke: i for i, ke in enumerate(self.kedges)}

        # triangle (at v): the three bond midpoints sorted CCW in azimuth
        # about v; edge i = mid(b_{i+1}) - mid(b_i) ~ away_{i+1} - away_i
        # (the difference itself is the direction)
        tri_oriented = {}
        for v in self.vertices:
            bs = sorted(vertex_bonds[v])
            order = sorted(bs, key=lambda b: angle_key(away[(v, b)]))
            sides = []
            for i in range(3):
                b0, b1 = order[i], order[(i + 1) % 3]
                d = (away[(v, b1)][0] - away[(v, b0)][0],
                     away[(v, b1)][1] - away[(v, b0)][1])
                ke = (v,) + tuple(sorted((b0, b1)))
                sides.append((kid[ke], d))
            tri_oriented[v] = sides
        self.tri_oriented = tri_oriented

        # hexagon (fid): kagome edge at corner k = [mid(k-1,k), mid(k,k+1)],
        # direction ~ cpos[k+1] - cpos[k-1] (increasing corner index k is CCW)
        hex_oriented = []
        for fid, row in enumerate(hex_corners):
            sides = []
            for k in range(6):
                d = (slot_pos[fid][(k + 1) % 6][0] - slot_pos[fid][(k - 1) % 6][0],
                     slot_pos[fid][(k + 1) % 6][1] - slot_pos[fid][(k - 1) % 6][1])
                sides.append((kid[hex_kedges[fid][k]], d))
            hex_oriented.append(sides)
        self.hex_oriented = hex_oriented

        # geometric self-checks: dot4(d,e) = 4 * (real-space dot product)
        def dot4(d, e):
            return (4 * d[0] * e[0] + 2 * d[0] * e[1]
                    + 2 * d[1] * e[0] + 4 * d[1] * e[1])

        # triangle: side sum zero (closure), equal lengths, adjacent edges
        # 60 deg apart (exterior angle 120 deg)
        for v, sides in tri_oriented.items():
            s0 = [d for _, d in sides]
            assert s0[0][0] + s0[1][0] + s0[2][0] == 0 and \
                   s0[0][1] + s0[1][1] + s0[2][1] == 0, "triangle not closed"
            n0 = dot4(s0[0], s0[0])
            assert all(dot4(d, d) == n0 for d in s0), "triangle sides unequal"
            for i in range(3):
                assert dot4(s0[i], s0[(i + 1) % 3]) == -n0 // 2, \
                    f"triangle interior angle != 60deg at hv {v}"

        # hexagon: adjacent edges 60 deg apart (interior angle 120 deg): dot = n/2
        for fid, sides in enumerate(hex_oriented):
            ns = [dot4(d, d) for _, d in sides]
            assert len(set(ns)) == 1 or True  # edge lengths free (metric disorder); no equal-length check
            for i in range(6):
                d1, d2 = sides[i][1], sides[(i + 1) % 6][1]
                assert 2 * dot4(d1, d2) == dot4(d2, d2), \
                    f"hexagon interior angle != 120deg at face {fid}"

        # ---- constraint matrices
        self.M_reduced = self._build_reduced()
        self.A_full = self._build_full()

    # ------------------------------------------------------------------
    def _build_reduced(self):
        """2F x 2F: rows = 2 closure equations per hexagon, columns =
        honeycomb-vertex t.

        The triangle t at corner k (a honeycomb vertex) of hexagon fid is
        exactly the k-th side length of the hexagon; that side's (CCW)
        direction = integer increment in basis coordinates. Rows =
        sum(d0*t) and sum(d1*t) (the common real-space factor is dropped;
        exact integers)."""
        ncol = self.nv
        rows = []
        for sides in self.hex_oriented:
            rx = [0] * ncol
            ry = [0] * ncol
            for ke, d in sides:
                v = self.kedges[ke][0]   # first element of kedge = honeycomb vertex at the corner
                rx[v] += d[0]
                ry[v] += d[1]
            rows.append(rx)
            rows.append(ry)
        assert len(rows) == 2 * self.F
        return rows

    def _build_full(self):
        """6F x 6F: columns = kagome edge lengths; rows = triangles (2 each)
        + hexagons (2 each)."""
        ncol = self.nek
        rows = []
        for v in self.vertices:
            rx = [0] * ncol
            ry = [0] * ncol
            for ke, d in self.tri_oriented[v]:
                rx[ke] += d[0]
                ry[ke] += d[1]
            rows.append(rx)
            rows.append(ry)
        for sides in self.hex_oriented:
            rx = [0] * ncol
            ry = [0] * ncol
            for ke, d in sides:
                rx[ke] += d[0]
                ry[ke] += d[1]
            rows.append(rx)
            rows.append(ry)
        assert len(rows) == 2 * (self.nv + self.F)
        return rows


# ---------------------------------------------------------------- linear algebra

def nullity_exact(rows):
    """Nullity (ncols - rank) of a set of integer row vectors, exact via
    Fraction. For small matrices."""
    ncol = len(rows[0])
    mat = [[Fraction(x) for x in r] for r in rows]
    rank = 0
    r = 0
    for c in range(ncol):
        piv = None
        for rr in range(r, len(mat)):
            if mat[rr][c] != 0:
                piv = rr
                break
        if piv is None:
            continue
        mat[r], mat[piv] = mat[piv], mat[r]
        pv = mat[r][c]
        mat[r] = [x / pv for x in mat[r]]
        for rr in range(len(mat)):
            if rr != r and mat[rr][c] != 0:
                f = mat[rr][c]
                mat[rr] = [x - f * y for x, y in zip(mat[rr], mat[r])]
        r += 1
        rank = r
        if r == len(mat):
            break
    return ncol - rank


def nullity_modp(rows, p=(1 << 31) - 1):
    """Nullity over a large prime field (= nullity over Q; error probability
    negligible)."""
    ncol = len(rows[0])
    mat = [[x % p for x in r] for r in rows]
    rank = 0
    for c in range(ncol):
        piv = None
        for rr in range(rank, len(mat)):
            if mat[rr][c] % p:
                piv = rr
                break
        if piv is None:
            continue
        mat[rank], mat[piv] = mat[piv], mat[rank]
        inv = pow(mat[rank][c], p - 2, p)
        mat[rank] = [x * inv % p for x in mat[rank]]
        for rr in range(len(mat)):
            if rr != rank and mat[rr][c]:
                f = mat[rr][c]
                mrow = mat[rank]
                mat[rr] = [(x - f * y) % p for x, y in zip(mat[rr], mrow)]
        rank += 1
        if rank == len(mat):
            break
    return ncol - rank


def nullity_float(rows):
    import numpy as np
    a = np.array(rows, dtype=float)
    return a.shape[1] - int(np.linalg.matrix_rank(a))


# ---------------------------------------------------------------- counting

def count_assignments_reduced(hc: KagomeTorus, alphabet):
    """Brute force: enumerate all alphabet assignments of the honeycomb
    vertex variables t on the torus, filtering per hexagon.

    Complexity k^(2F): only for anchors with F <= ~6."""
    n = hc.nv
    rows = hc.M_reduced
    k = len(alphabet)
    count = 0
    for assign in product(range(k), repeat=n):
        ok = True
        for rx, ry in zip(rows[::2], rows[1::2]):
            s1 = sum(rx[v] * assign[v] for v in range(n) if rx[v])
            if s1:
                ok = False
                break
            s2 = sum(ry[v] * assign[v] for v in range(n) if ry[v])
            if s2:
                ok = False
                break
        if ok:
            count += 1
    return count


def kernel_count_int(rows, alphabet):
    """Exact count of kernel vectors valued in alphabet^n (Fraction
    elimination for a kernel basis; clear denominators to an integer basis,
    then vectorized numpy
    enumeration over free coordinates). Requires dim on the order of
    1..15 so that k^dim stays affordable.

    Returns omega. Math: kernel basis v_j (Fraction); after clearing
    denominators by D, B_j = D*v_j in Z^n; solutions x = sum_j y_j v_j with
    y_j = free-coordinate values in the alphabet; in cleared form
    D*x = sum_j y_j B_j; check coordinate-wise that D divides and the
    quotient lies in the alphabet."""
    ncol = len(rows[0])
    mat = [[Fraction(x) for x in r] for r in rows]
    m = len(mat)
    pivot_cols = []
    r = 0
    for c in range(ncol):
        piv = None
        for rr in range(r, m):
            if mat[rr][c] != 0:
                piv = rr
                break
        if piv is None:
            continue
        mat[r], mat[piv] = mat[piv], mat[r]
        pv = mat[r][c]
        mat[r] = [x / pv for x in mat[r]]
        for rr in range(m):
            if rr != r and mat[rr][c] != 0:
                f = mat[rr][c]
                mat[rr] = [x - f * y for x, y in zip(mat[rr], mat[r])]
        pivot_cols.append(c)
        r += 1
        if r == m:
            break
    free_cols = [c for c in range(ncol) if c not in set(pivot_cols)]
    dim = len(free_cols)
    if dim == 0:
        return 1 if all(all(x == 0 for x in row) for row in rows) else 0

    # kernel basis (unit vectors on free coordinates, back-substituted)
    basis = []
    for fc in free_cols:
        vec = [Fraction(0)] * ncol
        vec[fc] = Fraction(1)
        for i, pc in enumerate(pivot_cols):
            vec[pc] = -mat[i][fc]
        basis.append(vec)
    # clear denominators to an integer basis: D = lcm of denominators
    D = 1
    for vec in basis:
        for x in vec:
            if x:
                D = D * x.denominator // math.gcd(D, x.denominator)
    B = np.array([[int(x * D) for x in vec] for vec in basis], dtype=np.int64)
    Amap = np.array(sorted(set(alphabet)), dtype=np.int64)

    if (len(Amap) ** dim) * B.shape[1] > MAX_ENUM_STATES:
        raise RuntimeError(
            f"kernel enumeration refused: {len(Amap)}^{dim} x {B.shape[1]} columns "
            f"> budget {MAX_ENUM_STATES} (fail-closed budget gate)")
    combos = np.array(list(product(Amap, repeat=dim)), dtype=np.int64)
    X = combos @ B                          # (k^dim, ncol) = D*x
    ok = (X % D == 0)
    Xd = X // D
    in_al = np.isin(Xd, Amap)
    return int(np.count_nonzero((ok & in_al).all(axis=1)))


# ------------------------------------------------------- open-boundary strip counting

def build_patch(W, L):
    """W x L open-boundary hexagonal patch (planar coordinates are the
    vertex identity; no wraparound).

    Returns (hex_vids, hex_pos, rows, n_vars):
      hex_vids[hi] = [vid x6] (increasing corner index k = CCW);
      hex_pos[hi]  = [planar coordinates x6];
      rows[hi]     = (rx, ry) closure-equation coefficients (dicts keyed by vid);
      n_vars       = number of vertices (t variables)."""
    verts = {}

    def vid(p):
        if p not in verts:
            verts[p] = len(verts)
        return verts[p]

    hex_vids, hex_pos, rows = [], [], []
    for q in range(W):
        for r in range(L):
            c = (q * G1[0] + r * G2[0], q * G1[1] + r * G2[1])
            pos = [_add(c, U[k]) for k in range(6)]
            corners = [vid(p) for p in pos]
            rx, ry = {}, {}
            for k in range(6):
                d = (pos[(k + 1) % 6][0] - pos[(k - 1) % 6][0],
                     pos[(k + 1) % 6][1] - pos[(k - 1) % 6][1])
                rx[corners[k]] = rx.get(corners[k], 0) + d[0]
                ry[corners[k]] = ry.get(corners[k], 0) + d[1]
            hex_vids.append(corners)
            hex_pos.append(pos)
            rows.append((rx, ry))
    return hex_vids, hex_pos, rows, len(verts)


def strip_count(W, L, alphabet, max_states=None):
    """Exact solution count of an open-boundary W x L strip: frontier DP
    (vectorized).

    Hexagons are processed one by one in row-major order (r increasing,
    q increasing within a row). Given the scan order, the set of vertices
    "already assigned and still having a future host" is determined
    uniquely by the geometry (independent of values), so a state = a value
    tuple on a fixed slot list; transitions are done in one shot for all
    states with numpy column ops: new-corner branches (<= k^2 cases)
    check the two closure equations and assemble the new slot matrix.
    Returns (omega, peak_states). Raises RuntimeError when max_states is
    exceeded."""
    hex_vids, hex_pos, rows, n_vars = build_patch(W, L)
    nh = len(hex_vids)
    # build_patch is q-major r-minor; switch the scan order to r-major
    # (front width ~ 2W)
    order = sorted(range(nh), key=lambda h: (h % L, h // L))
    pos_in_order = {h: i for i, h in enumerate(order)}

    hosts = {}
    for h, corners in enumerate(hex_vids):
        for v in corners:
            hosts.setdefault(v, []).append(pos_in_order[h])
    first = {v: min(x) for v, x in hosts.items()}   # first assignment step
    last = {v: max(x) for v, x in hosts.items()}    # last consumption step

    A = np.asarray(list(alphabet), dtype=np.int64)
    k = len(A)

    def pending(i):   # vertices assigned before step i and not yet fully consumed
        return sorted(v for v in first if first[v] <= i - 1 and last[v] >= i)

    def after(i):     # vertices still needed after step i
        return sorted(v for v in first if first[v] <= i and last[v] >= i + 1)

    states = np.zeros((1, 0), dtype=np.int64)
    counts = [1]                     # Python ints, keep exactness
    peak = 0
    for i, h in enumerate(order):
        corners = hex_vids[h]
        rx, ry = rows[h]
        before = pending(i)
        bidx = {v: j for j, v in enumerate(before)}
        new_corners = [v for v in corners if first[v] == i]
        aft = after(i)
        carr = np.asarray(counts, dtype=np.int64)
        out_vals, out_cnts = [], []
        for combo in product(range(k), repeat=len(new_corners)):
            eqx = np.zeros(len(states), dtype=np.int64)
            eqy = np.zeros(len(states), dtype=np.int64)
            for v in corners:
                c, d = rx[v], ry[v]
                if first[v] == i:
                    eqx += c * A[combo[new_corners.index(v)]]
                    eqy += d * A[combo[new_corners.index(v)]]
                else:
                    eqx += c * states[:, bidx[v]]
                    eqy += d * states[:, bidx[v]]
            valid = (eqx == 0) & (eqy == 0)
            if not valid.any():
                continue
            sub = states[valid]
            subc = carr[valid]
            cols = np.empty((len(sub), len(aft)), dtype=np.int64)
            for j, v in enumerate(aft):
                if first[v] == i:
                    cols[:, j] = A[combo[new_corners.index(v)]]
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
        # merge duplicate states (same after-slot values from different
        # new-corner combos), keeping counts exact
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
            raise RuntimeError(f"strip_count({W},{L}): states {len(states)} "
                               f"> max_states {max_states}")
    return sum(counts), peak


def patch_matrix(W, L):
    """Dense integer constraint matrix of the open-boundary patch (same
    geometry as strip_count, for kernel-enumeration cross-checks)."""
    hex_vids, hex_pos, rows, n_vars = build_patch(W, L)
    mat = []
    for rx, ry in rows:
        mrx = [0] * n_vars
        mry = [0] * n_vars
        for v, c in rx.items():
            mrx[v] += c
        for v, c in ry.items():
            mry[v] += c
        mat.append(mrx)
        mat.append(mry)
    return mat


# ---------------------------------------------------------------- self-check

def _selftest():
    for (Lq, Lr) in [(2, 2), (3, 2), (2, 3), (3, 3), (4, 2)]:
        hc = KagomeTorus(Lq, Lr)
        F = hc.F
        dim_r = nullity_exact(hc.M_reduced)
        dim_f = nullity_exact(hc.A_full)
        print(f"torus {Lq}x{Lr}: F={F} nv={hc.nv} nek={hc.nek} "
              f"dim(M)={dim_r} dim(A_full)={dim_f} "
              f"{'OK' if dim_r == dim_f else 'MISMATCH'}")
        assert dim_r == dim_f


if __name__ == "__main__":
    _selftest()
