# -*- coding: utf-8 -*-
"""metric_hc.py -- core library for metric random honeycomb tilings
(thermodynamic limit).

The torus graph construction here shares its lineage with
edge_alphabet_affine_check.build_honeycomb_torus in the package
(accepted anchors: Omega(3x3,{0,1}) = 518 etc.); it is copied here
independently to keep this project self-contained.
Face closure, two equations (integer form, equivalent to the SI S1.2
opposite-edge sum rules):
  eqA: L1+L2 = L4+L5
  eqB: 2L0+L1+L5 = 2L3+L2+L4
Alphabets use the paper-1 bases {1,2}/{1,2,3}/{1,2,4} (affine,
translation-invariant; counts identical to the 0-based alphabet).

Memory envelope: all structures are sparse; at F=2304 (48x48) E=6912 and
the kernel basis is an (F+2) x E Fraction matrix of a few tens of MB; the
peak budget of a single run of this module stays below 500 MB, safe locally.
"""
from __future__ import annotations

import math
import random
from fractions import Fraction
from itertools import product

import numpy as np


# ----------------------------------------------------------------------
# Torus graph (isomorphic to build_honeycomb_torus in the package)
# ----------------------------------------------------------------------

def build_torus(Lx: int, Ly: int):
    """Honeycomb torus: returns (edge_set, faces).
    edge_set: [(u,v)] with u/v = ('A'|'B', i, j);
    faces: [6 edge indices in face-walk order].
    """
    def A(i, j): return ("A", i % Lx, j % Ly)
    def B(i, j): return ("B", i % Lx, j % Ly)

    edge_set, edge_index = [], {}
    def add_edge(u, v):
        key = frozenset((u, v))
        if key not in edge_index:
            edge_index[key] = len(edge_set)
            edge_set.append((u, v))
        return edge_index[key]

    for i in range(Lx):
        for j in range(Ly):
            add_edge(A(i, j), B(i, j))
            add_edge(A(i + 1, j), B(i, j))
            add_edge(A(i, j + 1), B(i, j))

    vertices = sorted({v for e in edge_set for v in e})
    pos = {}
    for i in range(Lx):
        for j in range(Ly):
            pos[("A", i, j)] = (float(i), float(j))
            pos[("B", i, j)] = (i + 0.5, j + 0.5)
    adj = {v: [] for v in vertices}
    for ei, (u, v) in enumerate(edge_set):
        adj[u].append((v, ei))
        adj[v].append((u, ei))

    def angle_of(v, w):
        (x1, y1), (x2, y2) = pos[v], pos[w]
        dx, dy = x2 - x1, y2 - y1
        if dx > Lx / 2: dx -= Lx
        if dx < -Lx / 2: dx += Lx
        if dy > Ly / 2: dy -= Ly
        if dy < -Ly / 2: dy += Ly
        return math.atan2(dy, dx)

    rot = {v: sorted(adj[v], key=lambda t: angle_of(v, t[0])) for v in vertices}

    faces, visited = [], set()
    for u, v0 in edge_set:
        for start in ((u, v0), (v0, u)):
            if start in visited:
                continue
            walk, cur = [], start
            while cur not in visited:
                visited.add(cur)
                a, b = cur
                walk.append((a, b))
                order = rot[b]
                k = next(kk for kk, (w, _) in enumerate(order) if w == a)
                cur = (b, order[(k + 1) % len(order)][0])
            if len(walk) != 6:
                raise AssertionError("face walk length != 6")
            eidx = [edge_index[frozenset(e)] for e in walk]
            if len(set(eidx)) != 6:
                raise AssertionError("repeated edge in face")
            faces.append(eidx)

    V, E, F = len(vertices), len(edge_set), len(faces)
    assert V - E + F == 0 and F == Lx * Ly
    return edge_set, faces


def closure_rows(faces, E):
    """Sparse integer closure rows: [(dict{edge: coeff}, ...)], two
    opposite-edge sum rules per face.
    Uses the unit-coefficient form eqA: L1+L2=L4+L5, eqB': L0+L5=L2+L3
    (original SI S1.2 form).
    Note: the package script's eqB (2L0+L1+L5=2L3+L2+L4) has the same
    solution set as {eqA, eqB'}, but the coefficient 2 makes the integer
    lattice spanned by the rational kernel basis index-deficient (measured
    on the binary 3x3 case: index 2, 50% coverage); the unit-coefficient
    form has no such defect (see test_sampler_uniformity)."""
    rows = []
    for f in faces:
        rows.append({f[1]: 1, f[2]: 1, f[4]: -1, f[5]: -1})
        rows.append({f[0]: 1, f[5]: 1, f[2]: -1, f[3]: -1})
    return rows


# ----------------------------------------------------------------------
# Exact rational kernel basis (sparse Fraction Gaussian elimination)
# ----------------------------------------------------------------------

def kernel_zbasis(rows, E, n_cols=None):
    """Saturated Z-kernel basis (unimodular integer elimination, active-set
    algorithm). Returns (basis_int_arrays, rank).
    Integer elimination on the rows of [M^T | I_E]: Bezout combinations are
    taken only between active rows (unimodular; the row lattice is
    unchanged); once a column has its unique pivot row, that row is frozen;
    active rows stay zero on all processed columns => after elimination,
    the still-unfrozen rows have an all-zero M part, and their augmentation
    part is a complete basis of ker_Z(M) (no index loss; fixes the
    sublattice defect caused by integrising the rational kernel basis).
    """
    ncons = len(rows)
    width = ncons + E
    R = np.zeros((E, width), dtype=object)  # arbitrary precision: Bezout coefficients compound
    for ri, row in enumerate(rows):
        for c, v in row.items():
            R[c, ri] = v
    R[np.arange(E), ncons + np.arange(E)] = 1

    active = list(range(E))
    rank = 0
    for c in range(ncons):
        nz = [i for i in active if R[i, c] != 0]
        if not nz:
            continue
        while len(nz) > 1:
            order = sorted(nz, key=lambda i: abs(int(R[i, c])))
            i, j = order[0], order[1]
            a, b = int(R[i, c]), int(R[j, c])
            g, s, t = _extended_gcd(a, b)
            Ri = s * R[i] + t * R[j]                      # this column = g
            Rj = (a // g) * R[j] - (b // g) * R[i]        # this column = 0
            R[i], R[j] = Ri, Rj
            nz = [k for k in active if R[k, c] != 0]
        active.remove(nz[0])
        rank += 1
    basis = [R[k, ncons:].copy() for k in active]
    for v in basis:
        assert all(
            sum(int(cv) * int(v[c]) for c, cv in row.items()) == 0
            for row in rows
        ), "zbasis vector failed Mv=0"
    return basis, rank


def reduce_basis(vecs, sweeps=6):
    """Integer basis size reduction (lattice-preserving): pairwise
    size-reduction + content gcd division.
    v/g (g = gcd(v)) is still a kernel vector and the spanned lattice is
    unchanged (superset argument), so it is allowed.
    Returns a list of np.object arrays (exact integers, possibly beyond
    int64).
    Dot products vectorised with numpy (int64 fast path when entries are far
    below the int64 limits, else object)."""
    import math as _m
    V = []
    for v in vecs:
        arr = np.array([int(x) for x in v], dtype=object)
        g = 0
        for x in arr:
            g = _m.gcd(g, int(x))
        if g > 1:
            arr = arr // g
        V.append(arr)

    def dot(a, b):
        return int(np.dot(a, b))

    # int64 fast path: input entries < 2^20 and E < 2^14 => |dot| < 2^54,
    # no overflow; if entries grow past 2^35 during reduction, redo the
    # whole run on the exact object path (same result)
    E_len = len(V[0]) if V else 0
    use64 = E_len < (1 << 14) and \
        all(int(np.abs(v).max()) < (1 << 20) for v in V)
    if use64:
        V = [v.astype(np.int64) for v in V]
    LIM = 1 << 35

    for _ in range(sweeps):
        changed = False
        sq = [dot(v, v) for v in V]
        for i in range(len(V)):
            if sq[i] == 0:
                continue
            for j in range(len(V)):
                if i == j:
                    continue
                dij = dot(V[i], V[j])
                q = (2 * dij + sq[i]) // (2 * sq[i])   # nearest integer
                if q == 0:
                    continue
                cand = V[j] - q * V[i]
                if int(np.abs(cand).sum()) < int(np.abs(V[j]).sum()):
                    g = 0
                    for x in cand:
                        g = _m.gcd(g, int(x))
                    if g > 1:
                        cand = cand // g
                    V[j] = cand
                    sq[j] = dot(V[j], V[j])
                    changed = True
                    if use64 and int(np.abs(cand).max()) >= LIM:
                        return reduce_basis(V, sweeps=sweeps)
        if not changed:
            break
    return V


def _extended_gcd(a, b):
    """Extended Euclid: returns (g, s, t) with s*a + t*b = g."""
    old_r, r = a, b
    old_s, s = 1, 0
    old_t, t = 0, 1
    while r != 0:
        q = old_r // r
        old_r, r = r, old_r - q * r
        old_s, s = s, old_s - q * s
        old_t, t = t, old_t - q * t
    return old_r, old_s, old_t


def exact_kernel(rows, E, ncols_expected=None):
    """Rational kernel basis of the homogeneous linear system Mx=0.
    rows: sparse rows; E: number of columns. Returns (basis, rank); basis
    is a list of Fraction vectors.
    Column order = natural order; the pivot is the nonzero entry of smallest
    absolute value (small numbers help keep sparsity and low growth).
    """
    mat = [{c: Fraction(v) for c, v in r.items()} for r in rows]
    pivots = {}          # pivot_col -> row_id
    rank = 0
    for row_id, r in enumerate(mat):
        # reduce against existing pivots
        changed = True
        while changed:
            changed = False
            for c in list(r):
                if c not in r or c not in pivots or pivots[c] == row_id:
                    continue  # during snapshot iteration that column may
                    # already have been eliminated to 0 and popped
                pr = mat[pivots[c]]
                f = r[c] / pr[c]
                for cc, vv in pr.items():
                    nv = r.get(cc, Fraction(0)) - f * vv
                    if nv == 0:
                        r.pop(cc, None)
                    else:
                        r[cc] = nv
                changed = True
        nz = [c for c in r if r[c] != 0]
        if not nz:
            continue
        # pivot column: the column with the smallest absolute coefficient
        # in this row (+-1 preferred)
        pc = min(nz, key=lambda c: (abs(r[c]), c))
        pivots[pc] = row_id
        rank += 1
    free_cols = [c for c in range(E) if c not in pivots]
    basis = []
    for fc in free_cols:
        # set the other free columns explicitly to 0 (otherwise a pivot row
        # containing free columns cannot be resolved)
        x = {c: Fraction(0) for c in free_cols}
        x[fc] = Fraction(1)
        # back substitution: for each pivot row,
        # x[pivot] = -sum(r[c]*x[c]) / r[pivot]
        # must be iterated in dependency order (sparse; iterate directly to
        # a fixed point)
        for _ in range(E):
            updated = False
            for pc, rid in pivots.items():
                if pc in x:
                    continue
                r = mat[rid]
                if all(c == pc or c in x or r.get(c, 0) == 0 for c in r):
                    s = Fraction(0)
                    for c, v in r.items():
                        if c != pc:
                            s += v * x.get(c, 0)
                    x[pc] = -s / r[pc]
                    updated = True
            if not updated:
                break
        vec = [x.get(c, Fraction(0)) for c in range(E)]
        assert all(
            sum(v * vec[c] for c, v in row.items()) == 0 for row in mat
        ), "kernel vector failed Mv=0"
        basis.append(vec)
    return basis, rank


def primitive_integer(basis):
    """Fraction kernel basis -> list of primitive integer vectors (LCM
    scaling + gcd reduction)."""
    out = []
    for vec in basis:
        l = math.lcm(*(v.denominator for v in vec)) if vec else 1
        ints = [int(v * l) for v in vec]
        g = math.gcd(*[abs(t) for t in ints if t]) or 1
        out.append(np.array([t // g for t in ints], dtype=np.int64))
    return out


# ----------------------------------------------------------------------
# Exact counting (small-system verification; per-edge DFS + trigger
# pruning, same as count_assignments in the package)
# ----------------------------------------------------------------------

def count_solutions(Lx, Ly, alphabet):
    edge_set, faces = build_torus(Lx, Ly)
    E = len(edge_set)
    order, seen = [], set()
    for f in faces:
        for e in f:
            if e not in seen:
                seen.add(e)
                order.append(e)
    pos_in = {e: k for k, e in enumerate(order)}
    triggers = {}
    for f in faces:
        tA = tuple(pos_in[e] for e in (f[1], f[2], f[4], f[5]))
        triggers.setdefault(max(tA), []).append(("A", tA))
        tB = tuple(pos_in[e] for e in (f[0], f[1], f[5], f[3], f[2], f[4]))
        triggers.setdefault(max(tB), []).append(("B", tB))
    vals = [0] * E
    alpha = list(alphabet)
    count = 0

    def dfs(k):
        nonlocal count
        if k == E:
            count += 1
            return
        for w in alpha:
            vals[k] = w
            ok = True
            for typ, terms in triggers.get(k, ()):
                if typ == "A":
                    a, b, c, d = terms
                    if vals[a] + vals[b] != vals[c] + vals[d]:
                        ok = False
                        break
                else:
                    e0, e1, e5, e3, e2, e4 = terms
                    if 2 * vals[e0] + vals[e1] + vals[e5] != \
                       2 * vals[e3] + vals[e2] + vals[e4]:
                        ok = False
                        break
            if ok:
                dfs(k + 1)
    import sys
    old = sys.getrecursionlimit()
    sys.setrecursionlimit(max(old, E + 100))
    dfs(0)
    sys.setrecursionlimit(old)
    return count


# ----------------------------------------------------------------------
# MCMC uniform sampler (kernel-direction random walk + box-constraint
# rejection)
# ----------------------------------------------------------------------

def _cut_move_family(edge_set, Lx, Ly):
    """Cut / type-class move family: Lx type-1 cuts + Ly type-2 cuts + 3
    type-class vectors.

    All satisfy the closure equations (cuts are kernel vectors; type
    classes g(type(e)) satisfy both closure equations for any g), and at
    coprime sizes they are always "fully uniform" on the solution set =>
    the +- flip is always feasible.
    """
    E = len(edge_set)
    types = np.zeros(E, dtype=np.int64)
    for ei, e in enumerate(edge_set):
        (u, v) = e
        a = u if u[0] == "A" else v
        b = v if u[0] == "A" else u
        di = (a[1] - b[1]) % Lx
        dj = (a[2] - b[2]) % Ly
        types[ei] = 0 if (di == 0 and dj == 0) else \
            (1 if (di == 1 and dj == 0) else 2)
    keys = {}
    for ei, e in enumerate(edge_set):
        if types[ei] == 0:
            continue
        (u, v) = e
        a = u if u[0] == "A" else v
        keys.setdefault((int(types[ei]), a[1] if types[ei] == 1 else a[2]),
                        []).append(ei)
    moves = []
    for k in sorted(keys):
        v = np.zeros(E, dtype=np.int64)
        for ei in keys[k]:
            v[ei] = 1
        moves.append(v)
    for t in range(3):
        v = (types == t).astype(np.int64)
        moves.append(v)
    return np.array(moves, dtype=np.int64)


class MetricSampler:
    """Uniform sampling on {x in alphabet^E : Mx=0}.
    Moves: x -> x +- v_k (v_k an integer kernel vector); symmetric proposal
    + box-constraint rejection => exactly uniform (on each connected
    component; connectivity cross-checked on small systems).
    """

    def __init__(self, Lx, Ly, alphabet, seed=0, max_frac_E=4096,
                 basis=None, use_cut_moves=True, extra_moves=None):
        self.Lx, self.Ly = Lx, Ly
        self.alpha = tuple(alphabet)
        self.rng = random.Random(seed)
        self.edge_set, self.faces = build_torus(Lx, Ly)
        self.E = len(self.edge_set)
        self.F = Lx * Ly
        if basis is None:
            rows = closure_rows(self.faces, self.E)
            zb, rank = kernel_zbasis(rows, self.E)
            basis = reduce_basis(zb)
        self.basis = [np.asarray(v) for v in basis]
        self.dim = self.F + 2
        assert len(self.basis) == self.dim, \
            f"kernel dim {len(self.basis)} != F+2 = {self.dim}"
        # alphabet membership: contiguous alphabets ({1,2}/{1,2,3}) take the
        # box-constraint fast path; non-contiguous ones ({1,2,4}) must
        # reject missing values such as 3, via a LUT
        self._lo, self._hi = min(self.alpha), max(self.alpha)
        if set(self.alpha) == set(range(self._lo, self._hi + 1)):
            self._lut = None
        else:
            lut = np.zeros(self._hi + 1, dtype=bool)
            lut[np.asarray(self.alpha, dtype=np.int64)] = True
            self._lut = lut
        # int64 fast path: when entries cannot overflow, store the basis as
        # an int64 2D array and vectorise the moves
        mx = max(max(abs(int(x)) for x in v) for v in self.basis)
        if mx < (1 << 61):
            self.basis = np.array([[int(x) for x in v] for v in self.basis],
                                  dtype=np.int64)
        # max step R = hi-lo: non-contiguous alphabets ({1,2,4}) are not
        # connected under +-1 single steps (3 is banned; 2->4 needs step 2),
        # so the move set is extended to +-m*v_k, m in {1..R};
        # contiguous alphabets have R=1, bitwise identical to the old move set
        self._R = self._hi - self._lo
        # cut / type-class move family (ensemble-fragmentation fix):
        # at coprime sizes "every straight cut fully uniform + type-0
        # classes fully uniform" is a universal property of the solution
        # set (full enumeration 4x3..6x5: 100%; 12/24 sampled
        # configurations: 100%), so whole-cut flips are always feasible;
        # the basis vectors (mixed vectors) are not connected on the
        # feasible graph (4x3: 2 components, 6x4: 1480), and after adding
        # the cuts (Lx+Ly of them) and the 3 type-class vectors the graph
        # is fully connected in tests (4x3..6x5).
        self.extra_moves = ([np.asarray(v, dtype=np.int64)
                             for v in extra_moves] if extra_moves else [])
        self.cut_moves = None
        if use_cut_moves:
            self.cut_moves = _cut_move_family(self.edge_set, Lx, Ly)
            rows_chk = closure_rows(self.faces, self.E)
            for mv in self.cut_moves:
                for row in rows_chk:
                    assert sum(c * int(mv[k]) for k, c in row.items()) == 0,                         "cut move not in kernel"
            self.cut_moves = [self.cut_moves[i] for i in
                              range(len(self.cut_moves))]
        # initial solution: constant configurations l = c (c in the
        # alphabet) satisfy both closure equations
        self.x = np.full(self.E, self.alpha[0], dtype=np.int64)
        self._check()

    def _feasible(self, x):
        if np.any(x < self._lo) or np.any(x > self._hi):
            return False
        if self._lut is None:
            return True
        return bool(np.all(self._lut[x]))

    def _project_to_box(self, x, sweeps=64):
        """Repair x to a nearby feasible solution: coordinate descent over
        kernel vectors (each step walks along v_k up to the boundary)."""
        lo, hi = min(self.alpha), max(self.alpha)
        x = x.copy()
        for _ in range(sweeps):
            moved = False
            for v in self.basis:
                idx = np.nonzero(v)[0]
                if len(idx) == 0:
                    continue
                # largest feasible step
                with np.errstate(divide="ignore", invalid="ignore"):
                    steps = []
                    for i in idx:
                        if v[i] > 0:
                            steps.append((hi - x[i]) / v[i])
                        elif v[i] < 0:
                            steps.append((lo - x[i]) / v[i])
                if not steps:
                    continue
                t = math.floor(min(steps))
                if t != 0:
                    x = x + t * v
                    moved = True
            if not moved:
                break
        return x

    def _check(self):
        assert self._feasible(self.x), "initial solution out of box"
        rows = closure_rows(self.faces, self.E)
        for row in rows[:64]:  # spot check here; the exhaustive check lives
            # in the full verify() interface
            assert sum(v * self.x[c] for c, v in row.items()) == 0

    def verify(self):
        rows = closure_rows(self.faces, self.E)
        for row in rows:
            assert sum(v * self.x[c] for c, v in row.items()) == 0
        return True

    def sample(self, n_samples, thin=1):
        """Return n_samples configurations (one per thin steps), a list of
        np.int64 arrays.
        Proposal = basis vectors union cut family (cut flips + type
        classes), symmetric; box/membership constraint rejection =>
        exactly uniform on a connected component. The cut family is always
        feasible (full-uniformity theorem at coprime sizes), so the chain
        traverses the hypercube of cut values, fixing the fragmentation of
        the basis-vector graph."""
        out = []
        B = self.basis
        C = self.cut_moves
        X = self.extra_moves
        nb = len(B)
        nc = (0 if C is None else len(C))
        ntot = nb + nc + len(X)
        R = self._R
        for step in range(n_samples * thin):
            k = self.rng.randrange(ntot)
            if k < nb:
                m = self.rng.randrange(1, R + 1) if R > 1 else 1
                sgn = 1 if self.rng.getrandbits(1) else -1
                cand = self.x + (m * sgn) * B[k]
            elif k < nb + nc:
                sgn = 1 if self.rng.getrandbits(1) else -1
                cand = self.x + sgn * C[k - nb]
            else:
                sgn = 1 if self.rng.getrandbits(1) else -1
                cand = self.x + sgn * X[k - nb - nc]
            if self._feasible(cand):
                self.x = cand
            if (step + 1) % thin == 0:
                out.append(self.x.copy())
        return out

    # ---- geometry: BFS embedding (face-walk slot s <-> direction angle
    # 60 deg * s) ----
    def embed(self, lengths=None):
        """Planar vertex coordinates (BFS on the covering plane). Returns
        {v: (X,Y)}.
        Each edge takes its direction theta = 60 deg * s from the walk slot
        of one of its faces; the other face necessarily gives the reverse
        (3 slots apart). On non-contractible cycles the coordinate
        difference equals the winding vector of that configuration
        (configuration-dependent); local observables are unaffected."""
        if lengths is None:
            lengths = self.x
        edge_unit = {}
        for f in self.faces:
            for s, e in enumerate(f):
                if e not in edge_unit:
                    th = math.pi / 3.0 * s
                    edge_unit[e] = (math.cos(th), math.sin(th))
        adj = {}
        for ei, (u, v) in enumerate(self.edge_set):
            adj.setdefault(u, []).append((v, ei))
            adj.setdefault(v, []).append((u, ei))
        pos = {self.edge_set[0][0]: (0.0, 0.0)}
        stack = [self.edge_set[0][0]]
        while stack:
            u = stack.pop()
            for v, ei in adj[u]:
                if v in pos:
                    continue
                uu, vv = self.edge_set[ei]
                d = edge_unit[ei] if (uu == u and vv == v) else \
                    (-edge_unit[ei][0], -edge_unit[ei][1])
                pos[v] = (pos[u][0] + d[0] * lengths[ei],
                          pos[u][1] + d[1] * lengths[ei])
                stack.append(v)
        return pos
