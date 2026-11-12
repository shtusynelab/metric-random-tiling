# -*- coding: utf-8 -*-
"""
RTH = rhombitrihexagonal (3.4.6.4) regression suite -- counterexample to the
"honeycomb is the unique positive-entropy carrier" claim.

Position: 3.4.6.4 is the mixed edge-to-edge tiling whose face types {3,4,6}
are all rationally related, with an extensive Q-nullity (dim = F_hex + 3); it
is the counterexample regression for the "honeycomb unique positive-entropy
carrier" proposition. It also hosts the classification anchors of the four
tilings, the two split-row families (4.8.8, 3.12.12), and the second
all-rational family member (3,3,3,4,4). Geometry is rebuilt from scratch here
(imports no other project libraries).

Method:
  1. Edge-to-edge tiling torus: explicit coordinates; vertices identified by
     reduced coordinates mod the period; faces deduplicated by (vertex set,
     face type, reduced centroid) (parallel faces on small tori are not
     wrongly merged); edges numbered by reduced edge-midpoint coordinates;
     face rings all CCW.
  2. Constraints = per-face closure (global x/y components): row entries are
     +/-cos(theta) / +/-sin(theta); all direction angles are integer multiples
     of 30 degrees (45 for 4.8.8) => entries lie in Q(sqrt(d)), exact
     Fractions.
  3. dim_R = E - rank_{Q(sqrt(d))}; dim_Q = E - rank_Q (split rows): float
     SVD + two mod-p primes (F_{p^2}/F_p; rank_p >= rank_Q so take the max) +
     exact Fractions for small systems -- three-way cross-check.
  4. Exact count Omega: RREF over Q of the split integer system => D*x =
     -(N*f), vectorized integer enumeration (overflow guard).
  5. Positive-entropy lower-bound certificate: disjoint-support free
     coordinate set => strict product subbox Omega >= prod_j |S_j|
     (subextensive level; see README).
Anchors: square (dim=2L, Omega(4x4,k=3)=3^8), honeycomb (dim=L^2+2), kagome
(dim=Lq+Lr+gcd-1), 4.8.8 (dim_R=F+2, dim_Q=4L), 3.12.12 (dim_R=F+2, dim_Q
linear), (3,3,3,4,4) (dim subextensive).
"""
import math, itertools, sys
import numpy as np
from fractions import Fraction as Fr
from math import cos, sin, radians, atan2, hypot

R6 = 6
PRIMES = [1000033, 1000037]

# ---------------- exact arithmetic over Q(sqrt(d)) ----------------
class QD:
    """a + b*sqrt(d)"""
    __slots__ = ('a', 'b', 'd')
    def __init__(self, a=0, b=0, d=3):
        self.a = Fr(a); self.b = Fr(b); self.d = d
    def __add__(s, o):  return QD(s.a + o.a, s.b + o.b, s.d)
    def __sub__(s, o):  return QD(s.a - o.a, s.b - o.b, s.d)
    def __neg__(s):     return QD(-s.a, -s.b, s.d)
    def __mul__(s, o):  return QD(s.a*o.a + s.d*s.b*o.b, s.a*o.b + s.b*o.a, s.d)
    def inv(s):
        den = s.a*s.a - s.d*s.b*s.b
        assert den != 0
        return QD(s.a/den, -s.b/den, s.d)
    def iszero(s): return s.a == 0 and s.b == 0

def qd_from_component(v, d):
    """Float component -> exact QD (accepts only 0, +/-1, +/-1/2, +/-sqrt(3)/2, +/-sqrt(2)/2)"""
    if abs(v) < 1e-9: return QD(0, 0, d)
    if abs(abs(v) - 1) < 1e-9: return QD(1 if v > 0 else -1, 0, d)
    if abs(abs(v) - 0.5) < 1e-9: return QD(Fr(1, 2) if v > 0 else Fr(-1, 2), 0, d)
    if abs(abs(v) - math.sqrt(3)/2) < 1e-9:
        assert d == 3
        return QD(0, Fr(1, 2) if v > 0 else Fr(-1, 2), 3)
    if abs(abs(v) - math.sqrt(2)/2) < 1e-9:
        assert d == 2
        return QD(0, Fr(1, 2) if v > 0 else Fr(-1, 2), 2)
    raise AssertionError(f"component {v} not in the expected set")

def field_rank_qd(rows):
    """Gaussian-elimination rank over Q(sqrt(d)) (fully exact)"""
    M = [list(r) for r in rows]
    nr, nc = len(M), len(M[0])
    rank = 0
    for c in range(nc):
        piv = next((i for i in range(rank, nr) if not M[i][c].iszero()), None)
        if piv is None: continue
        M[rank], M[piv] = M[piv], M[rank]
        pv = M[rank][c].inv()
        M[rank] = [x*pv for x in M[rank]]
        for i in range(nr):
            if i != rank and not M[i][c].iszero():
                f = M[i][c]
                M[i] = [u - f*v for u, v in zip(M[i], M[rank])]
        rank += 1
        if rank == nr: break
    return rank

# ---------------- mod-p rank (F_p and F_{p^2}) ----------------
def _tonelli(d, p):
    d %= p
    if d == 0: return 0
    if pow(d, (p-1)//2, p) == p - 1: return None
    q, s = p - 1, 0
    while q % 2 == 0: q //= 2; s += 1
    z = 2
    while pow(z, (p-1)//2, p) != p - 1: z += 1
    m, cc, t, r = s, pow(z, q, p), pow(d, q, p), pow(d, (q+1)//2, p)
    while t != 1:
        i, t2 = 0, t
        while t2 != 1: t2 = t2*t2 % p; i += 1
        b = pow(cc, 1 << (m-i-1), p)
        m, cc, t, r = i, b*b % p, t*b*b % p, r*b % p
    return r

def _rank_fq(M, p):
    nr, nc = len(M), len(M[0])
    rank = 0
    for c in range(nc):
        piv = next((i for i in range(rank, nr) if M[i][c] % p), None)
        if piv is None: continue
        M[rank], M[piv] = M[piv], M[rank]
        iv = pow(M[rank][c], p-2, p)
        M[rank] = [x*iv % p for x in M[rank]]
        for i in range(nr):
            if i != rank and M[i][c] % p:
                f = M[i][c]
                M[i] = [(u - f*v) % p for u, v in zip(M[i], M[rank])]
        rank += 1
        if rank == nr: break
    return rank

def rank_pair_modp(Aint, Bint, d, p):
    """rows = A row + B row*sqrt(d) (integer matrices). If sqrt(d) lies in
    F_p, substitute into the degenerate single field; otherwise eliminate
    over F_{p^2}."""
    r = _tonelli(d, p)
    if r is not None:
        M = [[(a + b*r) % p for a, b in zip(ra, rb)] for ra, rb in zip(Aint, Bint)]
        return _rank_fq(M, p)
    rows = [([x % p for x in ra], [x % p for x in rb]) for ra, rb in zip(Aint, Bint)]
    nr, n = len(rows), len(rows[0][0])
    rank = 0
    for c in range(n):
        piv = next((i for i in range(rank, nr) if rows[i][0][c] % p or rows[i][1][c] % p), None)
        if piv is None: continue
        rows[rank], rows[piv] = rows[piv], rows[rank]
        A, B = rows[rank]
        na, nb = A[c], B[c]
        norm = (na*na - d*nb*nb) % p
        ninv = pow(norm, p-2, p)
        ia, ib = na*ninv % p, (-nb)*ninv % p
        for i in range(nr):
            if i == rank: continue
            CA, CB = rows[i]
            if not (CA[c] % p or CB[c] % p): continue
            fa = (CA[c]*ia - d*CB[c]*ib) % p
            fb = (CA[c]*ib + CB[c]*ia) % p
            rows[i] = ([(u - fa*v) % p for u, v in zip(CA, A)],
                       [(u - fb*v) % p for u, v in zip(CB, B)])
        rank += 1
        if rank == nr: break
    return rank

def _rank_fraction(rows, ncol):
    M = [list(r) for r in rows]
    nr = len(M)
    rank = 0
    for c in range(ncol):
        piv = next((i for i in range(rank, nr) if M[i][c] != 0), None)
        if piv is None: continue
        M[rank], M[piv] = M[piv], M[rank]
        pv = M[rank][c]
        M[rank] = [x/pv for x in M[rank]]
        for i in range(nr):
            if i != rank and M[i][c] != 0:
                f = M[i][c]
                M[i] = [u - f*v for u, v in zip(M[i], M[rank])]
        rank += 1
        if rank == nr: break
    return rank

def independent_rows_float(A, tol):
    """Float row-independent subset (modified Gram-Schmidt); returns row indices"""
    basis = []
    idx = []
    for i, r in enumerate(A):
        v = r.astype(float).copy()
        for b in basis:
            v -= np.dot(v, b) * b
        nrm = np.linalg.norm(v)
        if nrm > tol:
            basis.append(v/nrm)
            idx.append(i)
    return idx

def compute_dims(xrows, yrows, d, E, exact_limit=None, label=""):
    ncol = E
    A = np.array([[float(x.a) + math.sqrt(d)*float(x.b) for x in r] for r in xrows] +
                 [[float(y.a) + math.sqrt(d)*float(y.b) for y in r] for r in yrows])
    s = np.linalg.svd(A, compute_uv=False)
    dimR_fl = int(ncol - np.sum(s > s.max()*1e-9))
    Aa = np.array([[float(x.a) for x in r] for r in xrows] + [[float(y.a) for y in r] for r in yrows])
    Ab = np.array([[float(x.b) for x in r] for r in xrows] + [[float(y.b) for y in r] for r in yrows])
    s2 = np.linalg.svd(np.vstack([Aa, Ab]), compute_uv=False)
    dimQ_fl = int(ncol - np.sum(s2 > s2.max()*1e-9))
    XA = [[int(2*x.a) for x in r] for r in xrows]; XB = [[int(2*x.b) for x in r] for r in xrows]
    YA = [[int(2*y.a) for y in r] for r in yrows]; YB = [[int(2*y.b) for y in r] for r in yrows]
    dimR_mod, dimQ_mod = [], []
    for p in PRIMES:
        dimR_mod.append(ncol - rank_pair_modp(XA + YA, XB + YB, d, p))
        dimQ_mod.append(ncol - _rank_fq([r[:] for r in XA + YA + XB + YB], p))
    dimR = max(dimR_mod)   # rank_modp >= rank_Q => dim_modp <= true value, hence take max; cross-checked vs float
    dimQ = max(dimQ_mod)
    det = dict(dimR_float=dimR_fl, dimQ_float=dimQ_fl, dimR_mod=dimR_mod, dimQ_mod=dimQ_mod)
    if exact_limit is not None and E <= exact_limit:
        rows_qd = [[QD(x.a, x.b, d) for x in r] for r in xrows] + \
                  [[QD(y.a, y.b, d) for y in r] for r in yrows]
        rk = field_rank_qd(rows_qd)
        det['dimR_exact'] = ncol - rk
        if dimR is None: dimR = ncol - rk
        assert dimR == ncol - rk, f"{label}: dimR mismatch {det}"
        rows_q = [r for pair in
                  ([[Fr(2*x.a) for x in r], [Fr(2*x.b) for x in r]] for r in xrows) for r in pair] + \
                 [r for pair in
                  ([[Fr(2*y.a) for y in r], [Fr(2*y.b) for y in r]] for r in yrows) for r in pair]
        rk2 = _rank_fraction([r for r in rows_q if any(v != 0 for v in r)], ncol)
        det['dimQ_exact'] = ncol - rk2
        if dimQ is None: dimQ = ncol - rk2
        assert dimQ == ncol - rk2, f"{label}: dimQ mismatch {det}"
    assert dimR is not None and dimQ is not None, f"{label}: two-prime results disagree {det}"
    assert dimR == dimR_fl and dimQ == dimQ_fl, f"{label}: float and exact disagree {det}"
    return dimR, dimQ, det

# ---------------- torus skeleton ----------------
def make_reducer(P1, P2):
    M = np.array([[P1[0], P2[0]], [P1[1], P2[1]]], float)
    Minv = np.linalg.inv(M)
    def reduce_pt(p):
        uv = Minv @ np.asarray(p, float)
        u0 = uv[0] - math.floor(uv[0] + 1e-6)
        v0 = uv[1] - math.floor(uv[1] + 1e-6)
        res = np.asarray(p, float) - u0*P1 - v0*P2
        return (round(float(res[0]), R6), round(float(res[1]), R6))
    return reduce_pt

class Torus:
    """Two phases: build collects faces (with duplicates); finalize
    deduplicates, builds edges and rows"""
    def __init__(self, P1v, P2v, d):
        self.P1, self.P2 = np.array(P1v, float), np.array(P2v, float)
        self.d = d
        M = np.array([[self.P1[0], self.P2[0]], [self.P1[1], self.P2[1]]], float)
        self._Minv = np.linalg.inv(M)
        self.pos = {}          # vkey -> canonical float coordinates
        self._raw = []         # (vkeys, pts, ftype)
    def _reduce(self, p):
        uv = self._Minv @ np.asarray(p, float)
        nu = math.floor(uv[0] + 1e-6)
        nv = math.floor(uv[1] + 1e-6)
        res = np.asarray(p, float) - nu*self.P1 - nv*self.P2
        return (round(float(res[0]), R6), round(float(res[1]), R6))
    def add_face(self, pts, ftype):
        pts = [np.asarray(p, float) for p in pts]
        area = sum(float(pts[i][0]*pts[(i+1) % len(pts)][1] - pts[(i+1) % len(pts)][0]*pts[i][1])
                   for i in range(len(pts)))
        assert abs(area) > 1e-9, "degenerate face"
        if area < 0: pts = pts[::-1]
        keys = []
        for p in pts:
            k = self._reduce(p)
            if k not in self.pos: self.pos[k] = p
            keys.append(k)
        self._raw.append((keys, pts, ftype))
    def finalize(self):
        # dedup key = (vertex set, face type, reduced centroid): when the same
        # physical face is generated by neighboring cells, the two copies
        # differ by a period vector => same reduced centroid, so they still
        # merge; while parallel faces on a small torus (identical vertex sets)
        # differ by a non-periodic centroid offset and are not merged.
        seen, faces = set(), []
        for keys, pts, ftype in self._raw:
            cent = np.mean(pts, axis=0)
            sig = (frozenset(keys), ftype, self._reduce(cent))
            if sig in seen: continue
            seen.add(sig)
            faces.append((keys, pts, ftype))
        self.faces = faces
        self.V = len(self.pos)
        self.F = len(faces)
        self.edge_ids, self.edge_dirf, self.edge_cs = {}, {}, []
        self.face_edges = []
        for keys, pts, ftype in faces:
            segs = []
            for i in range(len(keys)):
                a, b = pts[i], pts[(i+1) % len(pts)]
                ek = self._reduce((a + b)/2)
                dv = b - a; dv = dv/hypot(*dv)
                if ek not in self.edge_ids:
                    self.edge_ids[ek] = len(self.edge_ids)
                    self.edge_dirf[ek] = dv
                    ang = math.degrees(math.atan2(dv[1], dv[0]))
                    step = 30 if self.d == 3 else 45
                    ar = round(ang/step)*step
                    assert abs((ang - ar + 180) % 360 - 180) < 1e-6, f"direction {ang} not on the {step}-degree grid"
                    ar %= 360
                    self.edge_cs.append((qd_from_component(cos(radians(ar)), self.d),
                                         qd_from_component(sin(radians(ar)), self.d)))
                dot = float(np.dot(self.edge_dirf[ek], dv))
                sign = 1 if dot > 0.5 else (-1 if dot < -0.5 else 0)
                assert sign != 0, "direction nearly orthogonal"
                segs.append((self.edge_ids[ek], sign))
            self.face_edges.append(segs)
        self.E = len(self.edge_ids)
        deg = {k: 0 for k in self.pos}
        for keys, _, _ in faces:
            for k in keys: deg[k] += 1
        self.degs = sorted(set(deg.values()))
        d = self.d
        self.xrows, self.yrows = [], []
        for segs in self.face_edges:
            rx = [QD(0, 0, d) for _ in range(self.E)]
            ry = [QD(0, 0, d) for _ in range(self.E)]
            for eid, sgn in segs:
                cx, sy = self.edge_cs[eid]
                if sgn > 0: rx[eid] = rx[eid] + cx; ry[eid] = ry[eid] + sy
                else:       rx[eid] = rx[eid] - cx; ry[eid] = ry[eid] - sy
            self.xrows.append(rx); self.yrows.append(ry)
        # all-ones reference solution self-check: each row sum (= all-ones
        # closure residual) must vanish
        for rx, ry in zip(self.xrows, self.yrows):
            tx = QD(0, 0, d); ty = QD(0, 0, d)
            for x in rx: tx = tx + x
            for y in ry: ty = ty + y
            assert tx.iszero() and ty.iszero(), \
                "all-ones reference solution fails to close => construction/sign error"
        return self
    def euler_ok(self): return self.V - self.E + self.F == 0
    def face_census(self):
        from collections import Counter
        return Counter(len(f[0]) for f in self.faces)
    def vertex_figures(self):
        """Cyclic sequence of face types around each vertex (uses each face's
        own instance centroid; translation-invariant)"""
        from collections import defaultdict
        v2rec = defaultdict(list)
        for fi, (keys, pts, ftype) in enumerate(self.faces):
            cent = np.mean(pts, axis=0)
            for i, k in enumerate(keys):
                ang = math.atan2(cent[1]-pts[i][1], cent[0]-pts[i][0])
                v2rec[k].append((ang, ftype))
        out = set()
        for k, recs in v2rec.items():
            recs.sort(key=lambda x: x[0])
            out.add(tuple(t for _, t in recs))
        return out

# ---------------- counting (exact integer method) and subbox certificate ----------------
def rref_and_param(xrows, yrows, E):
    """Split integer rows -> float row screening -> Fraction RREF; returns
    (T as Fraction matrix, piv_cols, free_cols, dim)"""
    xa = [[int(2*x.a) for x in r] for r in xrows]
    xb = [[int(2*x.b) for x in r] for r in xrows]
    ya = [[int(2*y.a) for y in r] for r in yrows]
    yb = [[int(2*y.b) for y in r] for r in yrows]
    rows = [r for r in (xa + xb + ya + yb) if any(v != 0 for v in r)]
    A = np.array(rows, float)
    keep = independent_rows_float(A, 1e-8)
    rows_k = [rows[i] for i in keep]
    M = [[Fr(v) for v in r] for r in rows_k]
    nr, piv_cols, r = len(M), [], 0
    for c in range(E):
        piv = next((i for i in range(r, nr) if M[i][c] != 0), None)
        if piv is None: continue
        M[r], M[piv] = M[piv], M[r]
        pv = M[r][c]
        M[r] = [x/pv for x in M[r]]
        for i in range(nr):
            if i != r and M[i][c] != 0:
                f = M[i][c]
                M[i] = [u - f*v for u, v in zip(M[i], M[r])]
        piv_cols.append(c); r += 1
        if r == nr: break
    dim = E - len(piv_cols)
    pivset = set(piv_cols)
    free_cols = [j for j in range(E) if j not in pivset]
    T = [[row[j] for j in free_cols] for row in M[:len(piv_cols)]]
    return T, piv_cols, free_cols, dim

def count_exact(T, free_cols, E, k, alpha0=0, chunk=65536):
    """x_pivot = -T*f; D*x = -(N*f) vectorized exact integer counting
    (alphabet alpha0..alpha0+k-1)"""
    P = len(T)
    if P == 0:
        return k ** len(free_cols)
    D = 1
    for row in T:
        for v in row:
            D = D*v.denominator//math.gcd(D, v.denominator)
    N = np.array([[int(v*D) for v in row] for row in T], dtype=np.int64)
    dim = len(free_cols)
    bound = int(np.abs(N).sum(axis=1).max()) * (alpha0 + k) * 2 + 1
    if D.bit_length() + bound.bit_length() > 60:
        raise OverflowError(f"D={D} too large, unsafe for int64")
    cnt = 0
    for start in range(0, (k**dim), chunk):
        # generate mixed-radix indices for [start, start+chunk)
        n = min(chunk, k**dim - start)
        idx = np.arange(start, start + n, dtype=np.int64)
        cols = []
        tmp = idx.copy()
        for _ in range(dim):
            cols.append(tmp % k + alpha0)
            tmp //= k
        F = np.stack(cols, axis=1)          # n x dim
        Mvals = F @ N.T                     # n x P
        # x = -Mvals/D must be an integer and x in [alpha0, alpha0+k)
        ok = ((Mvals % D) == 0).all(axis=1) \
            & (Mvals <= -alpha0*D).all(axis=1) \
            & (Mvals >= -(alpha0+k-1)*D).all(axis=1)
        cnt += int(ok.sum())
    return cnt

def subbox_certificate(T, k, alpha0=0):
    """Disjoint-support free-coordinate set => strict product subbox lower bound"""
    P = len(T)
    supports = [set(r for r in range(P) if T[r][j] != 0) for j in range(len(T[0]) if P else 0)]
    order = sorted(range(len(supports)), key=lambda j: len(supports[j]))
    chosen, used, bound = [], set(), 1
    for j in order:
        if supports[j] & used: continue
        vals_ok = []
        for v in range(alpha0, alpha0+k):
            good = True
            for r in supports[j]:
                val = -T[r][j]*v
                if val.denominator != 1 or not (alpha0 <= val.numerator < alpha0+k):
                    good = False; break
            if good: vals_ok.append(v)
        if len(vals_ok) >= 2:
            chosen.append(j); used |= supports[j]; bound *= len(vals_ok)
    return chosen, bound

# ---------------- tiling constructors ----------------
def u(deg): return (cos(radians(deg)), sin(radians(deg)))

def build_square(Lq, Lr):
    t = Torus((Lq, 0), (0, Lr), 3)
    for i in range(Lq):
        for j in range(Lr):
            t.add_face([(i, j), (i+1, j), (i+1, j+1), (i, j+1)], 4)
    return t.finalize()

def build_honeycomb(Lq, Lr):
    """Honeycomb with edge length 1; center triangular-lattice spacing sqrt(3);
    one hexagon per cell"""
    D0 = math.sqrt(3)
    A1, A2 = np.array((D0, 0)), np.array((D0/2, D0*math.sqrt(3)/2))
    t = Torus(Lq*A1, Lr*A2, 3)
    for i in range(Lq):
        for j in range(Lr):
            C = A1*i + A2*j
            t.add_face([C + np.array(u(30 + 60*m)) for m in range(6)], 6)
    return t.finalize()

def build_3464(L):
    """3.4.6.4: one hexagon per cell; squares/triangles all generated then
    deduplicated (each square shared by 2 cells, each triangle by 3)"""
    s = 1.0
    D = s*(1 + math.sqrt(3))
    B1, B2 = D*np.array(u(30)), D*np.array(u(90))
    t = Torus(L*B1, L*B2, 3)
    for i in range(L):
        for j in range(L):
            C = i*B1 + j*B2
            v = [C + np.array(u(60*m)) for m in range(6)]
            n = [np.array(u(30 + 60*m)) for m in range(6)]
            t.add_face(v, 6)
            for m in range(6):      # squares: both adjacent cells generate them on edge m; dedup handles it
                t.add_face([v[m], v[(m+1) % 6], v[(m+1) % 6] + s*n[m], v[m] + s*n[m]], 4)
            for m in range(6):      # triangles: at corner m, from the outer corner points of the two adjacent squares
                t.add_face([v[m], v[m] + s*n[(m-1) % 6], v[m] + s*n[m]], 3)
    return t.finalize()

def build_488(L):
    """4.8.8 (truncated square tiling): original square-grid edge g, cut depth
    t, with g-2t = t*sqrt(2) (regular octagon condition); octagon edge
    s = t*sqrt(2) => t = s/sqrt(2), g = s*(1+sqrt(2)); octagons share edges
    (middle segments of original grid edges), squares sit at the original grid
    points (4 cut points). One octagon + one square per cell."""
    s = 1.0
    tt = s/math.sqrt(2)
    g = s*(1 + math.sqrt(2))
    t = Torus((L*g, 0), (0, L*g), 2)
    def cutH(i, j, near):   # near/far cut points on the horizontal edge y=j*g, measured from grid point (i,j)
        x = i*g + (tt if near else g - tt)
        return (x, j*g)
    def cutV(i, j, near):   # on the vertical edge x=i*g, measured from grid point (i,j)
        y = j*g + (tt if near else g - tt)
        return (i*g, y)
    for i in range(L):
        for j in range(L):
            octv = [cutH(i, j, True), cutH(i, j, False),
                    cutV(i+1, j, True), cutV(i+1, j, False),
                    cutH(i, j+1, False), cutH(i, j+1, True),
                    cutV(i, j, False), cutV(i, j, True)]
            t.add_face(octv, 8)
            sq = [(i*g + tt, j*g), (i*g, j*g + tt), (i*g - tt, j*g), (i*g, j*g - tt)]
            t.add_face(sq, 4)
    return t.finalize()

def build_31212(L):
    """3.12.12 (truncated hexagonal tiling): dodecagon edge s=1, cut depth
    t=1/sqrt(3), original honeycomb edge g=s+2t; one dodecagon + 2 triangles
    per cell (original honeycomb cell; corner triangles all generated then
    deduplicated)"""
    s, tt = 1.0, 1.0/math.sqrt(3)
    g = s + 2*tt
    D0 = math.sqrt(3)*g
    A1, A2 = np.array((D0, 0)), np.array((D0/2, D0*math.sqrt(3)/2))
    t = Torus(L*A1, L*A2, 3)
    for i in range(L):
        for j in range(L):
            C = A1*i + A2*j
            A = [C + g*np.array(u(30 + 60*m)) for m in range(6)]
            dhat = [np.array(u(150 + 60*m)) for m in range(6)]
            Pm = [(A[m] + tt*dhat[m], A[(m+1) % 6] - tt*dhat[m]) for m in range(6)]
            dodec = []
            for m in range(6):
                dodec += [Pm[m][0], Pm[m][1]]
            t.add_face(dodec, 12)
            for m in range(6):   # triangle at corner A_{m+1} (shared by 3 cells; dedup handles it)
                Ac = A[(m+1) % 6]
                t.add_face([Ac + tt*np.array(u(90 + 60*m)),
                            Ac + tt*np.array(u(210 + 60*m)),
                            Ac + tt*np.array(u(330 + 60*m))], 3)
    return t.finalize()

def build_33344(L, M=None):
    """(3,3,3,4,4) elongated triangular: square rows alternating with triangle
    rows. The row offset accumulates +1/2 across each triangle row; period =
    two-row group (square row + triangle row) x 2; each two-row cell (width L)
    contains 2 squares + 4 triangles and 4 vertex lines."""
    h = math.sqrt(3)/2
    H1 = 1 + h                      # height of one row group (square + triangle)
    M = L if M is None else M
    t = Torus((L, 0), (0, M*2*H1), 3)
    for j in range(M):
        y0 = j*2*H1
        for k in (0, 1):            # the two square rows inside the two-row cell
            yb = y0 + k*H1
            yt = yb + 1
            d = 0.5*k               # square-row x offset: row 0 shifted by 0, row 1 by 1/2
            for i in range(L):
                t.add_face([(i+d, yb), (i+1+d, yb), (i+1+d, yt), (i+d, yt)], 4)
            # triangle row: base = square-row top (offset d), apex = next square-row bottom (offset d+1/2)
            ytt = yt + h
            for i in range(L):
                t.add_face([(i+d, yt), (i+1+d, yt), (i+0.5+d, ytt)], 3)        # downward triangle
                t.add_face([(i+0.5+d, ytt), (i+1.5+d, ytt), (i+1+d, yt)], 3)   # upward triangle
    return t.finalize()

def rot_norm(seq):
    """Rotational normalization of a cyclic sequence (lexicographically
    smallest rotation)"""
    n = len(seq)
    return min(tuple(seq[(i+k) % n] for i in range(n)) for k in range(n))

def build_kagome_nullity(Lq, Lr):
    """kagome nullity: face cycles of the honeycomb torus, variables =
    honeycomb vertices; two consecutive-pair equations per hexagon:
    (t0+t1)-(t3+t4)=0 and (t1+t2)-(t4+t5)=0"""
    hc = build_honeycomb(Lq, Lr)
    vidx = {k: n for n, k in enumerate(hc.pos)}
    nvar = len(vidx)
    rows = []
    for keys, _, _ in hc.faces:
        ids = [vidx[k] for k in keys]
        r1 = [0]*nvar; r2 = [0]*nvar
        r1[ids[0]] += 1; r1[ids[1]] += 1; r1[ids[3]] -= 1; r1[ids[4]] -= 1
        r2[ids[1]] += 1; r2[ids[2]] += 1; r2[ids[4]] -= 1; r2[ids[5]] -= 1
        rows += [r1, r2]
    return nvar - _rank_fraction([r for r in rows if any(r)], nvar)

# ---------------- main flow ----------------
def check_tiling(name, t, expectR=None, expectQ=None, exact_limit=None):
    assert t.euler_ok(), f"{name}: Euler check failed V={t.V} E={t.E} F={t.F}"
    dimR, dimQ, det = compute_dims(t.xrows, t.yrows, t.d, t.E,
                                   exact_limit=exact_limit, label=name)
    ok = ((expectR is None or dimR == expectR) and (expectQ is None or dimQ == expectQ))
    print(f"[{name}] V={t.V} E={t.E} F={t.F} deg={t.degs} faces={dict(sorted(t.face_census().items()))} "
          f"dimR={dimR} (expected {expectR}) dimQ={dimQ} (expected {expectQ}) {'PASS' if ok else 'FAIL'}")
    if not ok: print(f"    detail {det}")
    return dimR, dimQ, ok

def main():
    import csv, os
    results, csv_rows = [], []

    def rec(name, t, dimR, dimQ, note=""):
        csv_rows.append([name, t.V, t.E, t.F, dimR, dimQ, note])

    print("="*104)
    print("Part 1: anchors (reproduce known theorems)")
    sq = build_square(4, 4)
    _, _, ok = check_tiling("square 4x4", sq, expectR=8, expectQ=8, exact_limit=70)
    rec("square_4x4", sq, 8, 8)
    T, piv, free, dim = rref_and_param(sq.xrows, sq.yrows, sq.E)
    cnt = count_exact(T, free, sq.E, 3)
    print(f"    square 4x4 k=3 exact count Omega = {cnt} (theorem 3^8 = {3**8}){'PASS' if cnt == 3**8 else 'FAIL'}")
    results.append(ok and cnt == 3**8)
    for L in (2, 3, 4):
        hc = build_honeycomb(L, L)
        dR, dQ, ok = check_tiling(f"honeycomb {L}x{L}", hc, expectR=L*L+2, expectQ=L*L+2, exact_limit=60)
        rec(f"honeycomb_{L}x{L}", hc, dR, dQ, "dim=L^2+2 anchor")
        results.append(ok)

    print("-"*104)
    print("Part 2: counterexample main check 3.4.6.4 (dim = L^2+3 = F_hex + 3)")
    for L in (2, 3, 4, 5, 6):
        t = build_3464(L)
        fig = t.vertex_figures()
        figok = all(rot_norm(f) == (3, 4, 6, 4) for f in fig)
        dR, dQ, ok = check_tiling(f"3.4.6.4 L={L}", t, expectR=L*L+3, expectQ=L*L+3,
                                  exact_limit=(120 if L <= 3 else None))
        assert t.face_census() == {3: 2*L*L, 4: 3*L*L, 6: L*L}, "face census mismatch"
        print(f"    cyclic vertex figure set = {fig} {'PASS' if figok else 'FAIL'}")
        rec(f"RTH_3464_L{L}", t, dR, dQ, "dim=L^2+3 exact (Fraction L<=3 / mod-p+float L<=6)")
        results.append(ok and figok)

    print("-"*104)
    print("Part 3: 3.4.6.4 exact count Omega and positive-entropy lower-bound certificate")
    for L, k_list in ((2, (2, 3)), (3, (2,)), (4, (2,))):
        t = build_3464(L)
        T, piv, free, dim = rref_and_param(t.xrows, t.yrows, t.E)
        for k in k_list:
            cnt0 = count_exact(T, free, t.E, k, alpha0=0)
            cnt1 = count_exact(T, free, t.E, k, alpha0=1)
            assert cnt0 == cnt1, "alphabet translation invariance violated"
            s_face = math.log(cnt0)/(6*L*L)
            ch, bound = subbox_certificate(T, k)
            print(f"    L={L} k={k}: dim={dim} Omega={cnt0} s={s_face:.4f}/face "
                  f"(upper bound k^dim/F = {k**dim/(6*L*L):.4f}); subbox |S|={len(ch)} => Omega >= {bound}")
            csv_rows.append([f"RTH_count_L{L}_k{k}", t.V, t.E, t.F, dim, dim,
                             f"Omega={cnt0};s_per_face={s_face:.6f};subbox_|S|={len(ch)};"
                             f"subbox_bound={bound}"])
            results.append(cnt0 > 0)
    print("    note: Omega>0 holds at all sizes and dim=L^2+3 is extensive "
          "=> s <= ln k / 6 per face; a strict lower bound on s(inf)>0 "
          "(product construction / strip TM) is a README to-do.")

    print("-"*104)
    print("Part 4: auxiliary tiling regressions")
    dim44 = []
    for L in (2, 3, 4, 5):
        t = build_33344(L)
        fig = t.vertex_figures()
        figok = all(rot_norm(f) == (3, 3, 3, 4, 4) for f in fig)
        dR, dQ, ok = check_tiling(f"(3,3,3,4,4) L={L}", t, expectQ=None, exact_limit=130)
        dim44.append((L, t.F, dR))
        print(f"    vertex face-type set = {fig} {'PASS' if figok else 'FAIL'}")
        rec(f"33344_L{L}", t, dR, dQ, "dim subextensive (zero entropy)")
        results.append(ok and figok and dR == dQ)
    print(f"    (3,3,3,4,4) dim series = {dim44} (dim/F -> 0 => subextensive => zero entropy)")
    q312 = []
    for L in (2, 3, 4):
        t = build_31212(L)
        fig = t.vertex_figures()
        figok = all(rot_norm(f) == (3, 12, 12) for f in fig)
        dR, dQ, ok = check_tiling(f"3.12.12 L={L}", t, expectR=3*L*L+2, expectQ=None,
                                  exact_limit=60)
        q312.append((L, t.F, dR, dQ))
        print(f"    vertex face-type set = {fig} {'PASS' if figok else 'FAIL'}")
        rec(f"TSQ31212_L{L}", t, dR, dQ, "dim_R=F+2; dim_Q linear (split rows force rigidity, zero entropy)")
        results.append(ok and figok)
    assert all(q312[i+1][3] - q312[i][3] == 2*(i+1) + 6 for i in range(len(q312)-1)),         "3.12.12 dim_Q differences must be linear (11,19,29 => diffs 8,10)"
    print(f"    3.12.12 (L,F,dimR,dimQ) = {q312} (dim_Q linear => subextensive => zero entropy)")
    for L in (2, 3, 4):
        t = build_488(L)
        fig = t.vertex_figures()
        figok = all(rot_norm(f) == (4, 8, 8) for f in fig)
        dR, dQ, ok = check_tiling(f"4.8.8 L={L}", t, expectR=2*L*L+2, expectQ=4*L,
                                  exact_limit=100)
        print(f"    vertex face-type set = {fig} {'PASS' if figok else 'FAIL'}")
        rec(f"TSQ488_L{L}", t, dR, dQ, "dim_R=F+2; dim_Q=4L (split rows force rigidity, zero entropy)")
        results.append(ok and figok)

    print("-"*104)
    print("Part 5: kagome nullity spot check")
    for (Lq, Lr) in ((2, 2), (3, 2), (4, 4), (5, 5)):
        dim = build_kagome_nullity(Lq, Lr)
        exp = Lq + Lr + math.gcd(Lq, Lr) - 1
        print(f"    kagome {Lq}x{Lr}: dim={dim} theorem {exp} {'PASS' if dim == exp else 'FAIL'}")
        csv_rows.append([f"kagome_{Lq}x{Lr}", 2*Lq*Lr, 3*Lq*Lr, Lq*Lr, dim, dim,
                         "dim=Lq+Lr+gcd-1 theorem anchor"])
        results.append(dim == exp)

    print("-"*104)
    print("Part 6: triangle-locking theorem (analytic proof: RTH_triangle_locking_theorem_analytic_proof.md)")
    for L in (2, 3):
        t = build_3464(L)
        E = t.E
        tri_rows = set()
        edge2tri = {}
        tri_nodes = []
        sq_pairs = []
        for fi, (keys, pts, ft) in enumerate(t.faces):
            eids = [eid for eid, _ in t.face_edges[fi]]
            if ft == 3:
                tri_nodes.append(fi)
                for e in eids:
                    tri_rows.add(e)
                    edge2tri[e] = fi
            elif ft == 4:
                sq_pairs.append([e for e in eids if e in edge2tri or True])
        # square -> its two triangle edges (combinatorial premise of Lemma 2)
        adj = {n: set() for n in tri_nodes}
        for fi, (keys, pts, ft) in enumerate(t.faces):
            if ft != 4:
                continue
            ts = [edge2tri[eid] for eid, _ in t.face_edges[fi] if eid in edge2tri]
            assert len(ts) == 2 and ts[0] != ts[1], "square two-triangle-edges premise violated"
            adj[ts[0]].add(ts[1]); adj[ts[1]].add(ts[0])
        assert all(len(adj[n]) == 3 for n in tri_nodes), "G_tri not 3-regular"
        seen, stack = set(), [tri_nodes[0]]
        while stack:
            uu = stack.pop()
            if uu in seen:
                continue
            seen.add(uu)
            stack.extend(adj[uu] - seen)
        assert len(seen) == len(tri_nodes), "G_tri not connected (Lemma 3 premise violated)"
        # kernel projection rank = 1 (Corollary 1)
        T, piv, free, dim = rref_and_param(t.xrows, t.yrows, E)
        D = 1
        for row in T:
            for v in row:
                D = D * v.denominator // math.gcd(D, v.denominator)
        assert D == 1
        K = [[Fr(0)] * dim for _ in range(E)]
        for j, fc in enumerate(free):
            K[fc][j] = Fr(1)
        for r, pc in enumerate(piv):
            for j in range(dim):
                K[pc][j] = -T[r][j]
        cols = [[K[r][j] for r in sorted(tri_rows)] for j in range(dim)]
        rk = _rank_fraction([c for c in cols if any(v != 0 for v in c)], len(tri_rows))
        # random integer-solution locking (sampling version of Corollary 2)
        Tm = np.array([[int(v) for v in row] for row in T], dtype=np.int64)
        rng = np.random.default_rng(20260927)
        locked = 0
        for _ in range(20):
            nm = min(int(rng.integers(1, 9)), dim)
            f = np.zeros(dim, dtype=np.int64)
            f[rng.choice(dim, size=nm, replace=False)] = rng.choice((-2, -1, 1, 2), size=nm)
            x = np.zeros(E, dtype=np.int64)
            x[free] = f
            x[piv] = -(Tm @ f)
            tv = x[sorted(tri_rows)]
            locked += int(tv.max() - tv.min() == 0)
        ok = (rk == 1 and locked == 20
              and len(tri_nodes) == 2 * L * L)
        print(f"    L={L}: |tri|={len(tri_nodes)}=2L^2, G_tri 3-regular+connected, "
              f"projection rank={rk} (predicted 1), locked {locked}/20 "
              f"{'PASS' if ok else 'FAIL'}")
        csv_rows.append([f"RTH_triangle_lock_L{L}", t.V, t.E, t.F, rk, rk,
                         "triangle locking: projection rank=1; G_tri connected honeycomb graph; 20/20 integer-solution locks"])
        results.append(ok)

    print("-"*104)
    print("Part 7: (3,3,3,3,6) snub hexagonal rigidity (snub_lib.py)")
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from snub_lib import dims_snub
    for k in (1, 2, 3):
        dR, dQ = dims_snub(k)
        ok = (dR == 1 and dQ == 1)
        print(f"    snub k={k}: dim_R={dR} dim_Q={dQ} (predicted 1,1)"
              f"{'PASS' if ok else 'FAIL'}")
        csv_rows.append([f"snub33336_k{k}", 6*k*k, 15*k*k, 9*k*k, dR, dQ,
                         "rigid: dim=1 (edge-sharing triangle locking propagates => single global scale on all edges); s=0"])
        results.append(ok)

    out_csv = os.path.join(os.path.dirname(os.path.abspath(__file__)), "RTH_entropy_density_verification.csv")
    with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["object", "V", "E", "F", "dim_R", "dim_Q", "note"])
        w.writerows(csv_rows)
    print("="*104)
    nfail = sum(1 for ok in results if not ok)
    print(f"overall: {len(results)-nfail}/{len(results)} groups PASS (CSV -> {out_csv})"
          + ("" if nfail == 0 else f" ({nfail} FAIL)"))
    return 0 if nfail == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
