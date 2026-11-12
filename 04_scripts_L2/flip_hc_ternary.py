# -*- coding: utf-8 -*-
"""
Flip lower-bound program (1): honeycomb {1,2,3} third-difference candidate
construction and verification.
Template theorem: compactly supported kernel mode v on top of the all-2 seat
=> completed flips => s >= rho * ln 2.

Pipeline:
  A. Torus dim anchor: L x L closed honeycomb system dim_ker = L^2 + 2
     (L = 2..5, registered values)
  B. Window compact-mode existence: kernel dim D1 of (closure rows + zero
     rows on boundary edges); D1 > 0 means a compactly supported kernel
     mode exists
  C. Construction: a = Delta_o0 Delta_o1 Delta_o2 f (f = single site), then
     solve the opposite-edge difference system for v and inspect entries
  D. Flip certificate: all-2 seat + v; per-face closure recheck + alphabet
     check + patch bounding box + rho
  E. Negative control: square-lattice propagation system window kernel
     dim = 0 (theorem-level zero entropy, pipeline check)
"""
import json
import math
import os
import sys
from fractions import Fraction

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import paths_common as PC  # noqa: E402

OUT_DIR = os.path.join(str(PC.DATA_ROOT), 'lower_bounds')

# ---------------- honeycomb graph ----------------
# hexagon (i,j) axial coords (basis u=30deg, w=90deg); edge k=0..5 has
# direction k*60deg and outward normal k*60deg-90deg
# slot (i,j,k) and (i+ox[k], j+oy[k], (k+3)%6) are the same physical edge
OFF = {0: (0, -1), 1: (1, -1), 2: (1, 0), 3: (0, 1), 4: (-1, 1), 5: (-1, 0)}


def hc_torus(L):
    """torus honeycomb: returns (n_edges, closure rows).
    Closure rows: r1: v1+v2-v4-v5, r2: v0+v5-v2-v3"""
    slot2edge = {}
    nxt = 0
    for i in range(L):
        for j in range(L):
            for k in range(6):
                if (i, j, k) in slot2edge:
                    continue
                p = ((i + OFF[k][0]) % L, (j + OFF[k][1]) % L, (k + 3) % 6)
                slot2edge[(i, j, k)] = nxt
                slot2edge[p] = nxt
                nxt += 1
    rows = []
    for i in range(L):
        for j in range(L):
            g = lambda k: slot2edge[(i, j, k)]
            rows.append([(g(1), 1), (g(2), 1), (g(4), -1), (g(5), -1)])
            rows.append([(g(0), 1), (g(5), 1), (g(2), -1), (g(3), -1)])
    return nxt, rows


def hc_window(W):
    """open-boundary W x W window: returns (n_edges, closure rows,
    boundary edge list, slot2edge)"""
    inside = lambda i, j: 0 <= i < W and 0 <= j < W
    slot2edge = {}
    kinds = []  # 'int' interior edge / 'bnd' window boundary edge
    for i in range(W):
        for j in range(W):
            for k in range(6):
                if (i, j, k) in slot2edge:
                    continue
                pi, pj = i + OFF[k][0], j + OFF[k][1]
                e = len(kinds)
                slot2edge[(i, j, k)] = e
                if inside(pi, pj):
                    slot2edge[(pi, pj, (k + 3) % 6)] = e
                    kinds.append('int')
                else:
                    kinds.append('bnd')
    rows = []
    for i in range(W):
        for j in range(W):
            g = lambda k: slot2edge[(i, j, k)]
            rows.append([(g(1), 1), (g(2), 1), (g(4), -1), (g(5), -1)])
            rows.append([(g(0), 1), (g(5), 1), (g(2), -1), (g(3), -1)])
    bnd = [e for e, t in enumerate(kinds) if t == 'bnd']
    return len(kinds), rows, bnd, slot2edge


# ---------------- linear algebra (exact Fraction) ----------------
def kernel_dim(n_cols, rows):
    A = [[Fraction(0)] * n_cols for _ in rows]
    for r, row in enumerate(rows):
        for c, v in row:
            A[r][c] += v
    pivots = []
    r = 0
    for c in range(n_cols):
        piv = None
        for rr in range(r, len(A)):
            if A[rr][c] != 0:
                piv = rr
                break
        if piv is None:
            continue
        A[r], A[piv] = A[piv], A[r]
        pv = A[r][c]
        A[r] = [x / pv for x in A[r]]
        for rr in range(len(A)):
            if rr != r and A[rr][c] != 0:
                f = A[rr][c]
                A[rr] = [a - f * b for a, b in zip(A[rr], A[r])]
        pivots.append(c)
        r += 1
    return n_cols - len(pivots)


def solve_aug(n_cols, rows):
    """rows: [( [(col,coef),...], rhs )...];
    returns (particular solution or None, consistent, solution space dim)"""
    m = len(rows)
    A = [[Fraction(0)] * (n_cols + 1) for _ in range(m)]
    for r, (cs, b) in enumerate(rows):
        for c, v in cs:
            A[r][c] += v
        A[r][n_cols] = b
    pivots = []
    r = 0
    for c in range(n_cols):
        piv = None
        for rr in range(r, m):
            if A[rr][c] != 0:
                piv = rr
                break
        if piv is None:
            continue
        A[r], A[piv] = A[piv], A[r]
        pv = A[r][c]
        A[r] = [x / pv for x in A[r]]
        for rr in range(m):
            if rr != r and A[rr][c] != 0:
                f = A[rr][c]
                A[rr] = [a - f * b for a, b in zip(A[rr], A[r])]
        pivots.append(c)
        r += 1
    for rr in range(r, m):
        if A[rr][n_cols] != 0 and all(A[rr][c] == 0 for c in range(n_cols)):
            return None, False, 0
    x = [Fraction(0)] * n_cols
    for ri, pc in enumerate(pivots):
        x[pc] = A[ri][n_cols]
    return x, True, n_cols - len(pivots)


# ---------------- C: third-difference a-field + opposite-edge system for v --
def third_diff_a(origin):
    """a = Delta_o0 Delta_o1 Delta_o2 f, f = delta_origin;
    directions along the three stripe families"""
    o0, o1, o2 = (0, -1), (1, -1), (1, 0)

    def delta(dd, g):
        h = {}
        for (i, j), v in g.items():
            h[(i, j)] = h.get((i, j), 0) + v
            q = (i - dd[0], j - dd[1])
            h[q] = h.get(q, 0) - v
        return {k: v for k, v in h.items() if v != 0}

    g1 = delta(o2, {origin: 1})
    g2 = delta(o1, g1)
    return delta(o0, g2)


def build_mode(W, a):
    """solve the opposite-edge difference system on the window:
    v0-v3=-a, v1-v4=+a, v2-v5=-a per face; boundary edges v=0"""
    n_edges, _, bnd, slot2edge = hc_window(W)
    rows = []
    for i in range(W):
        for j in range(W):
            g = lambda k: slot2edge[(i, j, k)]
            ah = Fraction(a.get((i, j), 0))
            rows.append(([(g(0), 1), (g(3), -1)], -ah))
            rows.append(([(g(1), 1), (g(4), -1)], ah))
            rows.append(([(g(2), 1), (g(5), -1)], -ah))
    for e in bnd:
        rows.append(([(e, 1)], Fraction(0)))
    v, ok, dim = solve_aug(n_edges, rows)
    return v, ok, dim, bnd, slot2edge


def closure_ok(v, W, slot2edge):
    for i in range(W):
        for j in range(W):
            g = lambda k: v[slot2edge[(i, j, k)]]
            if g(1) + g(2) != g(4) + g(5):
                return False, (i, j, 'r1')
            if g(0) + g(5) != g(2) + g(3):
                return False, (i, j, 'r2')
    return True, None


# ---------------- D: flip certificate ----------------
def flip_certificate(W, v, slot2edge):
    """all-2 seat + v: recheck per-face closure / alphabet / boundary
    consistency; returns patch info and constants"""
    v_int = [int(x) for x in v]  # verified to be all integers
    supp = [e for e, x in enumerate(v_int) if x != 0]
    # patch = all faces incident to the directed support edges
    slot_face = {}
    for i in range(W):
        for j in range(W):
            for k in range(6):
                slot_face[(i, j, k)] = (i, j)
    P = set()
    supp_slots = []
    for (i, j, k), e in slot2edge.items():
        if e in set(supp):
            supp_slots.append(((i, j, k), e))
    for s, e in supp_slots:
        P.add(slot_face[s])
        # the far side of a support edge lies inside the window (guaranteed
        # by v=0 on boundary edges); find the partner face
        (i, j, k) = s
        p = (i + OFF[k][0], j + OFF[k][1], (k + 3) % 6)
        P.add(slot_face[p])
    # boundary consistency: patch boundary edges (exactly one side in P)
    # must carry v=0
    face_in_P = P
    for (i, j, k), e in slot2edge.items():
        pass  # (edge->faces map redone below)
    edge_faces = {}
    for i in range(W):
        for j in range(W):
            for k in range(6):
                e = slot2edge[(i, j, k)]
                edge_faces.setdefault(e, set()).add((i, j))
    boundary_bad = []
    for e, fs in edge_faces.items():
        n_in = sum(1 for f in fs if f in face_in_P)
        if n_in == 1 and v_int[e] != 0:
            boundary_bad.append((e, v_int[e]))
    assert not boundary_bad, "boundary edges with nonzero v: %s" % boundary_bad[:5]
    # double completion: per-face closure + alphabet
    for f in P:
        for k in range(6):
            val_p = 2 + v_int[slot2edge[(f[0], f[1], k)]]
            val_m = 2 - v_int[slot2edge[(f[0], f[1], k)]]
            assert val_p in (1, 2, 3) and val_m in (1, 2, 3)
    for sign in (+1, -1):
        for f in P:
            g = lambda k: 2 + sign * v_int[slot2edge[(f[0], f[1], k)]]
            assert g(1) + g(2) == g(4) + g(5)
            assert g(0) + g(5) == g(2) + g(3)
    assert any(x != 0 for x in v_int)
    # bounding box (axial coords) and rho
    ii = [f[0] for f in P]
    jj = [f[1] for f in P]
    d1 = max(ii) - min(ii) + 1
    d2 = max(jj) - min(jj) + 1
    rho = Fraction(1, d1 * d2)
    s_lb = math.log(2) / (d1 * d2)
    return {
        'n_faces_patch': len(P), 'd1': d1, 'd2': d2,
        'rho': float(rho), 's_lower_bound': s_lb,
        'support_edges': len(supp), 'P': sorted(P),
        'v_by_slot': {(i, j, k): v_int[slot2edge[(i, j, k)]]
                      for (i, j, k), e in slot2edge.items() if v_int[e] != 0},
    }


# ---------------- E: square-lattice negative control ----------------
def square_window_kernel(W):
    """square lattice: per cell vN=vS, vE=vW; outer boundary edges v=0;
    expected kernel dim 0"""
    slot2edge = {}
    kinds = []
    inside = lambda i, j: 0 <= i < W and 0 <= j < W
    PART = {0: (0, 1, 2), 1: (1, 0, 3), 2: (0, -1, 0), 3: (-1, 0, 1)}
    for i in range(W):
        for j in range(W):
            for k in range(4):
                if (i, j, k) in slot2edge:
                    continue
                di, dj, k2 = PART[k]
                e = len(kinds)
                slot2edge[(i, j, k)] = e
                if inside(i + di, j + dj):
                    slot2edge[(i + di, j + dj, k2)] = e
                    kinds.append('int')
                else:
                    kinds.append('bnd')
    rows = []
    for i in range(W):
        for j in range(W):
            g = lambda k: slot2edge[(i, j, k)]
            rows.append([(g(0), 1), (g(2), -1)])
            rows.append([(g(1), 1), (g(3), -1)])
    for e, t in enumerate(kinds):
        if t == 'bnd':
            rows.append([(e, 1)])
    return kernel_dim(len(kinds), rows)


# ---------------- main pipeline ----------------
def main():
    out = {}

    # A. torus dim anchor
    anchors = {}
    for L in (2, 3, 4, 5):
        n_edges, rows = hc_torus(L)
        d = kernel_dim(n_edges, rows)
        anchors['L=%d' % L] = {'dim': d, 'expect': L * L + 2, 'pass': d == L * L + 2}
    out['A_torus_dim_anchor'] = anchors

    # B. window compact-mode existence
    exist = {}
    for W in (8, 9):
        n_edges, rows, bnd, _ = hc_window(W)
        rows_full = list(rows) + [[(e, 1)] for e in bnd]
        d1 = kernel_dim(n_edges, rows_full)
        exist['W=%d' % W] = {'n_edges': n_edges, 'kernel_dim_compact_modes': d1}
    out['B_window_compact_modes'] = exist

    # C. construction: W=9, f placed at (4,4)
    W = 9
    a = third_diff_a((4, 4))
    out['C_a_field'] = {'%d,%d' % k: v for k, v in sorted(a.items())}
    v, ok, dim, bnd, slot2edge = build_mode(W, a)
    out['C_solve'] = {'consistent': ok, 'solution_space_dim': dim}
    if ok:
        ints = all(x.denominator == 1 for x in v)
        mx = max(abs(int(x)) for x in v) if ints else None
        out['C_entries'] = {'all_integers': ints, 'max_abs': mx}
        okc, where = closure_ok(v, W, slot2edge)
        out['C_closure_recheck'] = {'pass': okc, 'fail_at': where}

    # D. flip certificate
    if ok and ints:
        cert = flip_certificate(W, v, slot2edge)
        cert.pop('v_by_slot')
        cert.pop('P')
        out['D_flip_certificate'] = cert

    # E. square-lattice negative control
    out['E_square_control'] = {'W=8': square_window_kernel(8), 'W=10': square_window_kernel(10)}

    print(json.dumps(out, ensure_ascii=False, indent=1))
    with open(os.path.join(OUT_DIR, 'flip_hc_ternary_result.json'), 'w', encoding='utf-8') as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)

    # mode entry table (cited by the manuscript): nonzero slot -> v
    if ok and ints:
        tbl = {}
        for (i, j, k), e in slot2edge.items():
            x = int(v[e])
            if x:
                tbl['(%d,%d,%d)' % (i, j, k)] = x
        with open(os.path.join(OUT_DIR, 'mode_slots.json'), 'w', encoding='utf-8') as fh:
            json.dump(tbl, fh, ensure_ascii=False, indent=1)
        print('nonzero slots:', len(tbl))


if __name__ == '__main__':
    main()
