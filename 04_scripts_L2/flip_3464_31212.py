# -*- coding: utf-8 -*-
"""Flip lower-bound program (2): compactly supported kernel mode search and
flip certificates for 3.4.6.4 and the 3.12.12 mid carrier.
Method: torus construction (reuses RTH_numerical_verification.build_3464 /
lib31212.build_reduced) + zero forcing outside the region (= window compact
mode) + Fraction kernel basis + {0,+-1} mode extraction + exact per-face
flip verification.
"""
import importlib
import json
import math
import os
import sys
from collections import Counter
from fractions import Fraction

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import paths_common as PC  # noqa: E402

OUT_DIR = os.path.join(str(PC.DATA_ROOT), 'lower_bounds')

sys.path.insert(0, str(PC.engine("z4_3464")))
sys.path.insert(0, str(PC.engine("z3_31212")))

RTH = importlib.import_module(PC.fname("RTH_numerical_verification"))  # noqa: E402
import lib31212  # noqa: E402


# ---------------- linear algebra ----------------
def kernel_basis(n_cols, rows):
    """rows: list of (list of (col, coef)). Returns (rank, basis [list of
    list[Fraction]])."""
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
    free = [c for c in range(n_cols) if c not in pivots]
    basis = []
    for fc in free:
        v = [Fraction(0)] * n_cols
        v[fc] = Fraction(1)
        for ri, pc in enumerate(pivots):
            v[pc] = -A[ri][fc]
        basis.append(v)
    return len(pivots), basis


def to_int_vec(vf):
    """Fraction vector -> integer primal vector (clear denominators / divide
    out gcd); returns None for the all-zero vector."""
    from math import gcd
    dens = [x.denominator for x in vf if x != 0]
    if not dens:
        return None
    L = 1
    for d in dens:
        L = L * d // math.gcd(L, d)
    iv = [int(x * L) for x in vf]
    g = 0
    for x in iv:
        g = math.gcd(g, abs(x))
    if g == 0:
        return None
    return [x // g for x in iv]


def classify(vec):
    """integer vector -> (max_abs, kind): kind in {'zero','nonneg','nonpos','mixed'}"""
    nz = [x for x in vec if x != 0]
    if not nz:
        return 0, 'zero'
    mx = max(abs(x) for x in nz)
    if all(x > 0 for x in nz):
        return mx, 'nonneg'
    if all(x < 0 for x in nz):
        return mx, 'nonpos'
    return mx, 'mixed'


def find_unit_modes(basis, max_combo=2, coeff=2, allow_mixed=True):
    """search the small-coefficient combinations of the kernel basis for
    vectors with entries in {0,+-1}. Returns [(vec, kind, desc)].
    kind: nonneg/nonpos (binary usable) or mixed (ternary usable)."""
    out = []
    ints = []
    for b in basis:
        iv = to_int_vec(b)
        if iv:
            ints.append(iv)
    for idx, iv in enumerate(ints):
        mx, kind = classify(iv)
        if mx == 1 and (kind in ('nonneg', 'nonpos') or allow_mixed):
            out.append((iv, kind, 'basis[%d]' % idx))
    if out or max_combo < 2:
        return out
    # combinations only when needed; coefficients +/-1..coeff
    import itertools
    n = len(ints)
    for (i, j) in itertools.combinations(range(n), 2):
        for ci in (1, -1):
            for cj in (1, -1):
                iv = [ci * ints[i][t] + cj * ints[j][t] for t in range(len(ints[i]))]
                mx, kind = classify(iv)
                if kind in ('nonneg', 'nonpos') and mx == 1:
                    out.append((iv, 'c%d*d%d%+d*c%d*d%d' % (ci, i, cj, j)))
                    if len(out) >= 8:
                        return out
    return out


# ---------------- 3.4.6.4 ----------------
def run_3464(L=6, region=(1, 4)):
    t = RTH.build_3464(L)
    assert t.face_census() == Counter({3: 2 * L * L, 4: 3 * L * L, 6: L * L})
    E = t.E
    # face centroid -> cell (i,j)
    B1 = t.P1 * 1.0
    B2 = t.P2 * 1.0
    det = B1[0] * B2[1] - B1[1] * B2[0]
    Minv = [[B2[1] / det, -B2[0] / det], [-B1[1] / det, B1[0] / det]]

    def cell_of(pts):
        c = sum(pts) / len(pts)
        a = (Minv[0][0] * c[0] + Minv[0][1] * c[1]) * L
        b = (Minv[1][0] * c[0] + Minv[1][1] * c[1]) * L
        return (int(round(a)), int(round(b)))

    face_coord = [cell_of(pts) for (_, pts, _) in t.faces]  # continuous cell coords
    # faces inside the region (geometric box)
    lo, hi = region
    region_faces = set(fi for fi in range(len(t.faces))
                       if lo <= face_coord[fi][0] <= hi and lo <= face_coord[fi][1] <= hi)
    edge_faces = [[] for _ in range(E)]
    for fi, segs in enumerate(t.face_edges):
        for eid, s in segs:
            edge_faces[eid].append(fi)

    # integer split rows (xa, xb, ya, yb) x4 per face
    rows = []
    for rx, ry in zip(t.xrows, t.yrows):
        rows.append([(e, int(2 * x.a)) for e, x in enumerate(rx) if x.a != 0])
        rows.append([(e, int(2 * x.b)) for e, x in enumerate(rx) if x.b != 0])
        rows.append([(e, int(2 * y.a)) for e, y in enumerate(ry) if y.a != 0])
        rows.append([(e, int(2 * y.b)) for e, y in enumerate(ry) if y.b != 0])
    # zero forcing outside the region
    n_zero = 0
    for e in range(E):
        if not any(fi in region_faces for fi in edge_faces[e]):
            rows.append([(e, 1)])
            n_zero += 1
    rank, basis = kernel_basis(E, rows)
    dim = E - rank
    res = {'L': L, 'E': E, 'n_region_faces': len(region_faces),
           'n_zero_rows': n_zero, 'kernel_dim': dim}
    diag = []
    ints_all = [to_int_vec(b) for b in basis]
    for iv in ints_all:
        if iv:
            mx, kind = classify(iv)
            diag.append({'kind': kind, 'max_abs': mx,
                         'support': sum(1 for x in iv if x != 0)})
    res['basis_diag'] = diag
    modes = find_unit_modes(basis)
    res['n_unit_modes_found'] = len(modes)
    vec = None
    if modes:
        vec, kind, desc = modes[0]
        res['mode'] = {'desc': desc, 'kind': kind,
                       'max_abs': max(abs(x) for x in vec),
                       'support_edges': sum(1 for x in vec if x != 0)}
        res['flip'] = flip_3464(t, face_coord, edge_faces, vec)
    return res, (t, face_coord, edge_faces, vec)


def flip_3464(t, face_coord, edge_faces, mode):
    """flip certificate: single-sign modes use the binary {1,2} (base 1/2);
    mixed-sign modes use the ternary {1,2,3} (base 2)"""
    _, kind = classify([x for x in mode if x])
    if kind == 'nonneg':
        base, alpha = 1, (1, 2)
    elif kind == 'nonpos':
        base, alpha = 2, (1, 2)
    else:
        base, alpha = 2, (1, 2, 3)
    support = [e for e, x in enumerate(mode) if x != 0]
    P = sorted(set(fi for e in support for fi in edge_faces[e]))
    for s in (+1, -1):
        val = [base + s * x for x in mode]
        for fi, (rx, ry) in enumerate(zip(t.xrows, t.yrows)):
            ax = bx = ay = by = 0
            for eid, _sg in t.face_edges[fi]:
                c = val[eid]  # rx already carries the direction sign; do not
                # multiply by the in-face walk sign again
                ax += rx[eid].a * c
                bx += rx[eid].b * c
                ay += ry[eid].a * c
                by += ry[eid].b * c
            assert ax == 0 and bx == 0 and ay == 0 and by == 0, ('closure fail', fi, s)
        assert all(alpha[0] <= x <= alpha[-1] for x in val), 'alphabet fail'
    # patch cell bounding box (integer span on continuous coords)
    coords = [face_coord[fi] for fi in P]
    w1 = max(c[0] for c in coords) - min(c[0] for c in coords)
    w2 = max(c[1] for c in coords) - min(c[1] for c in coords)
    d1 = max(1, int(math.ceil(w1 - 1e-9)))
    d2 = max(1, int(math.ceil(w2 - 1e-9)))
    census = Counter(len(t.faces[fi][0]) for fi in P)
    per_face = 6 if len(alpha) == 2 else 6
    return {'base': base, 'alphabet': list(alpha), 'patch_faces': len(P),
            'patch_census': {'hex': census.get(6, 0), 'square': census.get(4, 0),
                             'tri': census.get(3, 0)},
            'support_edges': len(support), 'd1': d1, 'd2': d2,
            'rho_per_cell': 1.0 / (d1 * d2),
            's_lower_bound_per_face': math.log(2) / (per_face * d1 * d2)}


# ---------------- 3.12.12 ----------------
def run_31212(L=4, region=(1, 2)):
    cls_val, eqs, all_cls, n_cls, dim_ref, Nn, V, E, F = lib31212.build_reduced(L)
    # class position: midpoint of any member edge
    t = RTH.build_31212(L)
    eid2pos = {}
    for ek, eid in t.edge_ids.items():
        eid2pos[eid] = (float(ek[0]), float(ek[1]))
    cls_pos = {}
    for eid, cl in cls_val.items():
        if cl not in cls_pos:
            cls_pos[cl] = eid2pos[eid]
    A1 = t.P1 * 1.0
    A2 = t.P2 * 1.0
    det = A1[0] * A2[1] - A1[1] * A2[0]
    Minv = [[A2[1] / det, -A2[0] / det], [-A1[1] / det, A1[0] / det]]

    def cell_of(p):
        a = (Minv[0][0] * p[0] + Minv[0][1] * p[1]) * L
        b = (Minv[1][0] * p[0] + Minv[1][1] * p[1]) * L
        return (int(round(a)), int(round(b)))

    cls_cell = {cl: cell_of(cls_pos[cl]) for cl in all_cls}
    lo, hi = region
    region_cls = set(cl for cl in all_cls
                     if lo <= cls_cell[cl][0] <= hi and lo <= cls_cell[cl][1] <= hi)
    rows = []
    for eq in eqs:
        rows.append(sorted(eq.items()))
    for cl in all_cls:
        if cl not in region_cls:
            rows.append([(cl, 1)])
    rank, basis = kernel_basis(n_cls, rows)
    dim = n_cls - rank  # region kernel = compact-mode space; a different
    # quantity from the full-torus kernel dim_ref, not equal
    res = {'L': L, 'n_cls': n_cls, 'kernel_dim_full': dim_ref,
           'kernel_dim_region': dim, 'n_region_cls': len(region_cls)}
    if dim > 0:
        modes = find_unit_modes(basis)
        res['n_unit_modes_found'] = len(modes)
        if modes:
            vec, kind, desc = modes[0]
            res['mode'] = {'desc': desc, 'kind': kind,
                           'max_abs': max(abs(x) for x in vec),
                           'support': sum(1 for x in vec if x != 0)}
            res['mode_vec'] = list(vec)
            res['flip'] = flip_31212(eqs, all_cls, vec, cls_pos, Minv, L)
    return res


def flip_31212(eqs, all_cls, mode, cls_pos, Minv, L):
    """flip certificate: seat all c +- mode; try ternary {1,2,3} base 2
    first, then binary if the mode is single-sign"""
    outs = []
    for base, alpha in ((2, (1, 2, 3)), (1, (1, 2)), (2, (1, 2))):
        ok = True
        for s in (+1, -1):
            val = {cl: base + s * mode[cl] for cl in all_cls}
            if not all(a <= val[cl] <= b for cl in all_cls for a, b in [(alpha[0], alpha[-1])]):
                ok = False
                break
            for eq in eqs:
                if sum(c * val[cl] for cl, c in eq.items()) != 0:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            outs.append({'base': base, 'alphabet': list(alpha)})
    support = [cl for cl in all_cls if mode[cl] != 0]
    coords = []
    for cl in support:
        p = cls_pos[cl]
        coords.append(((Minv[0][0] * p[0] + Minv[0][1] * p[1]) * L,
                       (Minv[1][0] * p[0] + Minv[1][1] * p[1]) * L))
    w1 = max(c[0] for c in coords) - min(c[0] for c in coords)
    w2 = max(c[1] for c in coords) - min(c[1] for c in coords)
    d1 = max(1, int(math.ceil(w1 - 1e-9)))
    d2 = max(1, int(math.ceil(w2 - 1e-9)))
    return {'alphabets_legal': outs, 'support_classes': len(support),
            'd1': d1, 'd2': d2, 'rho_per_cell': 1.0 / (d1 * d2),
            's_lower_bound_per_face': math.log(2) / (3 * d1 * d2),
            'mode_kind': classify([mode[cl] for cl in support])[1]}


if __name__ == '__main__':
    out = {}
    print('=== 3.4.6.4 L=6 region[1,4] ===', flush=True)
    r3464, _ctx = run_3464(L=6, region=(1, 4))
    out['rth3464'] = r3464
    print(json.dumps(r3464, ensure_ascii=False, indent=1))
    print('=== 3.12.12 L=4 region[1,2] ===', flush=True)
    r31212 = run_31212(L=5, region=(1, 3))
    out['tsd31212'] = r31212
    print(json.dumps(r31212, ensure_ascii=False, indent=1))
    with open(os.path.join(OUT_DIR, 'flip_3464_31212_result.json'), 'w', encoding='utf-8') as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
    print('saved result json')
