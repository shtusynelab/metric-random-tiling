# -*- coding: utf-8 -*-
"""
3.3.4.3.4 (snub square) metric rigidity theorem -- numerical acceptance
=====================================================
Analytic predictions (SNB_3343_metric_rigidity_theorem.md):
  Lemma 1  triangles pair up two by two (perfect matching on TT edges);
  Lemma 2  the solution space ~= locally constant functions on the
           rhombus-pair graph G' (across rectangles, opposite), dim =
           number of connected components;
  Lemma 3  the compound tiling = checkerboard (even cells rectangles /
           odd cells rhombi, rhombus orientation o = i%2 cuts the short
           diagonal), G' = the distance-2 graph on odd cells, exactly two
           components {i odd}, {i even};
  Theorem  every legal torus (N,M both even) has dim_R = dim_Q = 2, and
           the kernel = the two rhombus-class constant vectors.

This script: build the torus compound tiling -> direction assignment
(BFS + two-face consistency) -> face closure rows (exact over K=Q(sqrt(3)))
-> Gaussian elimination over K (K a subfield of R => rank_R = rank_K)
-> verify dim=2 and kernel basis = vI, vII.
"""
from fractions import Fraction as Fr
from collections import deque

# ---------- K = Q(sqrt(3)) arithmetic: (p,q) represents p + q*sqrt(3) ----------
def kzero(): return (Fr(0), Fr(0))
def kadd(a, b): return (a[0] + b[0], a[1] + b[1])
def kneg(a): return (-a[0], -a[1])
def kmul(a, b): return (a[0]*b[0] + 3*a[1]*b[1], a[0]*b[1] + a[1]*b[0])
def kinv(a):
    den = a[0]*a[0] - 3*a[1]*a[1]
    assert den != 0, "nonzero elements of K are invertible (3 is not a rational square)"
    return (a[0]/den, -a[1]/den)

HALF = Fr(1, 2)
# unit vector (cos, sin) of direction d (30 deg steps, mod 12), as K elements
def unit(d):
    d %= 12
    tbl = {
        0:  ((Fr(1), Fr(0)), (Fr(0), Fr(0))),
        1:  ((Fr(0), HALF),  (HALF, Fr(0))),
        2:  ((HALF, Fr(0)),  (Fr(0), HALF)),
        3:  ((Fr(0), Fr(0)), (Fr(1), Fr(0))),
        4:  ((-HALF, Fr(0)), (Fr(0), HALF)),
        5:  ((Fr(0), -HALF), (HALF, Fr(0))),
        6:  ((-Fr(1), Fr(0)),(Fr(0), Fr(0))),
        7:  ((Fr(0), -HALF), (-HALF, Fr(0))),
        8:  ((-HALF, Fr(0)), (Fr(0), -HALF)),
        9:  ((Fr(0), Fr(0)), (-Fr(1), Fr(0))),
        10: ((HALF, Fr(0)),  (Fr(0), -HALF)),
        11: ((Fr(0), HALF),  (-HALF, Fr(0))),
    }
    return tbl[d]

def build(N, M):
    """Compound checkerboard torus: even cells rectangles, odd cells rhombi;
    rhombus orientation o = i%2 cuts the short diagonal."""
    assert N % 2 == 0 and M % 2 == 0, "two-color orientation pattern requires N,M both even (translation lattice = <(2,0),(0,2)>)"
    E = {}; elist = []
    def newedge(t, i, j):
        k = (t, i % N, j % M)
        if k not in E:
            E[k] = len(elist); elist.append(k)
        return E[k]
    H = lambda i, j: newedge('H', i, j)
    V = lambda i, j: newedge('V', i, j)
    def D(i, j):
        assert (i + j) % 2 == 1
        return newedge('D', i, j)

    faces = []      # (ordered edge-id list, exterior corner turns [30 deg units], face type)
    rect_id, tri_id = [], []
    for i in range(N):
        for j in range(M):
            if (i + j) % 2 == 0:
                g = [H(i, j), V(i + 1, j), H(i, j + 1), V(i, j)]
                faces.append((g, [3, 3, 3, 3], 'R')); rect_id.append(len(faces) - 1)
            else:
                g = [H(i, j), V(i + 1, j), H(i, j + 1), V(i, j)]
                d = D(i, j)
                if i % 2 == 1:   # orientation A: short diagonal (i,j)-(i+1,j+1), 120 deg corners at both ends
                    faces.append(([g[0], g[1], d], [4, 4, 4], 'T'))
                    faces.append(([d, g[2], g[3]], [4, 4, 4], 'T'))
                else:            # orientation B: short diagonal (i+1,j)-(i,j+1)
                    faces.append(([g[0], d, g[3]], [4, 4, 4], 'T'))
                    faces.append(([g[1], g[2], d], [4, 4, 4], 'T'))
                tri_id += [len(faces) - 2, len(faces) - 1]

    Vn, En, Fn = N * M, len(elist), len(faces)
    assert Vn - En + Fn == 0, "Euler"
    assert En == 2 * N * M + N * M // 2, "E = 2.5V"
    assert sum(1 for k in elist if k[0] == 'D') == Vn // 2, "#TT = V/2"

    # structural asserts: each triangle exactly 1 TT edge; each H/V edge
    # borders one rectangle and one triangle; each D edge has triangles on both sides
    ftype = [f[2] for f in faces]
    occ = {}
    for fi, (e, _, _) in enumerate(faces):
        for pos, ed in enumerate(e):
            occ.setdefault(ed, []).append((fi, pos))
    for ed, lst in occ.items():
        assert len(lst) == 2, "each edge has exactly two faces"
        tys = sorted(ftype[fi] for fi, _ in lst)
        if elist[ed][0] == 'D':
            assert tys == ['T', 'T'], "both sides of a TT edge are triangles"
        else:
            assert tys == ['R', 'T'], "lattice edge = one rectangle + one triangle (TS)"
    for fi, (e, _, ty) in enumerate(faces):
        if ty == 'T':
            assert sum(1 for ed in e if elist[ed][0] == 'D') == 1, "each triangle has exactly 1 TT edge (Lemma 1)"

    # vertex degree = 5
    deg = {}
    def touch(x, y):
        deg[(x % N, y % M)] = deg.get((x % N, y % M), 0) + 1
    for (t, i, j), eid in E.items():
        if t == 'H': touch(i, j); touch(i + 1, j)
        elif t == 'V': touch(i, j); touch(i, j + 1)
        else:
            if i % 2 == 1: touch(i, j); touch(i + 1, j + 1)
            else: touch(i + 1, j); touch(i, j + 1)
    assert all(v == 5 for v in deg.values()) and len(deg) == Vn, "all vertex degrees 5"

    # direction assignment: exterior turn = (180 - interior)/30 deg; CCW walk;
    # propagate across edges to the adjacent face (oriented direction difference 6)
    dir_of = {}
    def walk(fi, pos, d):
        """Starting from the known direction d at (fi,pos), assign directions
        around the whole face; returns the newly assigned slots."""
        e, turns, _ = faces[fi]
        m = len(e)
        new = []
        c = d
        for off in range(m):
            k = (pos + off) % m
            if off > 0:
                c = (c + turns[k]) % 12   # corner k connects edge k-1 and edge k
            key = (fi, k)
            if key in dir_of:
                assert dir_of[key] == c, "same-face direction consistency"
            else:
                dir_of[key] = c
                new.append(key)
        assert (c + turns[pos]) % 12 == d, "face direction walk closes"
        return new
    dir_of[(0, 0)] = 0
    dq = deque([(0, 0)])
    while dq:
        fi, pos = dq.popleft()
        for key in walk(fi, pos, dir_of[(fi, pos)]):
            ed = faces[fi][0][key[1]]
            fi2, pos2 = [o for o in occ[ed] if o != key][0]
            d2 = (dir_of[key] + 6) % 12
            if (fi2, pos2) in dir_of:
                assert dir_of[(fi2, pos2)] == d2, "two-face direction difference = 180 deg"
            else:
                dir_of[(fi2, pos2)] = d2
                dq.append((fi2, pos2))
    assert len(dir_of) == sum(len(f[0]) for f in faces), "direction assignment covers all slots"
    for ed, lst in occ.items():
        assert (dir_of[lst[0]] - dir_of[lst[1]]) % 12 == 6, "two-face direction difference = 180 deg"

    # face closure rows (exact over K)
    rows = []
    for fi, (e, _, _) in enumerate(faces):
        rx = [kzero() for _ in range(En)]
        ry = [kzero() for _ in range(En)]
        for pos, ed in enumerate(e):
            cx, cy = unit(dir_of[(fi, pos)])
            rx[ed] = kadd(rx[ed], cx)
            ry[ed] = kadd(ry[ed], cy)
        rows.append(rx); rows.append(ry)

    # edge -> class of its rhombus pair (0: i odd; 1: i even)
    def rhombus_of(k):
        t, i, j = k
        if t == 'H':  return (i, j) if (i + j) % 2 == 1 else (i, j - 1)
        if t == 'V':  return (i, j) if (i + j) % 2 == 1 else (i - 1, j)
        return (i, j)
    vI   = [1 if rhombus_of(k)[0] % 2 == 1 else 0 for k in elist]
    vII  = [1 if rhombus_of(k)[0] % 2 == 0 else 0 for k in elist]
    return dict(N=N, M=M, V=Vn, E=En, F=Fn, rows=rows, vI=vI, vII=vII, elist=elist)

def kvec_dot(row, v):
    s = kzero()
    for c, x in enumerate(v):
        if x:
            s = kadd(s, kmul(row[c], (Fr(x), Fr(0))))
    return s

def rank_K(rows, ncol):
    Mx = [r[:] for r in rows]
    rank = 0
    for col in range(ncol):
        piv = None
        for r in range(rank, len(Mx)):
            if Mx[r][col] != kzero():
                piv = r; break
        if piv is None:
            continue
        Mx[rank], Mx[piv] = Mx[piv], Mx[rank]
        inv = kinv(Mx[rank][col])
        Mx[rank] = [kmul(x, inv) for x in Mx[rank]]
        for r in range(len(Mx)):
            if r != rank and Mx[r][col] != kzero():
                f = kneg(Mx[r][col])
                Mx[r] = [kadd(Mx[r][c], kmul(f, Mx[rank][c])) for c in range(ncol)]
        rank += 1
    return rank

def main():
    sizes = [(2, 2), (2, 4), (4, 4), (4, 6), (6, 6), (4, 8), (6, 8)]
    print(f"{'N':>2} {'M':>2} {'V':>4} {'E':>5} {'F':>5} {'rows':>5} {'rank':>5} "
          f"{'dim':>4} {'pred':>4} {'ker=vI,vII':>10} {'redun':>5}  verdict")
    allpass = True
    for N, M in sizes:
        s = build(N, M)
        E, V, rows = s['E'], s['V'], s['rows']
        rk = rank_K(rows, E)
        dim = E - rk
        # kernel basis check: vI, vII lie exactly in the kernel; disjoint
        # nonzero supports => linearly independent; dim=2 => they form a basis
        inI  = all(kvec_dot(r, s['vI'])  == kzero() for r in rows)
        inII = all(kvec_dot(r, s['vII']) == kzero() for r in rows)
        indep = any(s['vI']) and any(s['vII']) and all(
            not (a and b) for a, b in zip(s['vI'], s['vII']))
        ok = (dim == 2) and inI and inII and indep
        allpass &= ok
        red = len(rows) - rk
        print(f"{N:>2} {M:>2} {V:>4} {E:>5} {s['F']:>5} {len(rows):>5} {rk:>5} "
              f"{dim:>4} {'2':>4} {str(inI and inII and indep):>10} {red:>5}  "
              f"{'PASS' if ok else 'FAIL'}")
    print("\nnaive bookkeeping: rows=2F=3V, unknowns E=2.5V, naive net freedom"
          " = -0.5V (negative, cannot serve as a predictor);"
          " measured dim=2 => constraint redundancy = V/2 + 2")
    print("ALL PASS" if allpass else "\nFAIL present -- do not write up; debug first")

if __name__ == '__main__':
    main()
