# -*- coding: utf-8 -*-
"""Flip lower-bound program (3): 4.8.8 Lambda alphabet (M_exp) one-sided
test + 4.6.12 ternary flip search.
Shared: torus construction + zero forcing outside the region + Fraction
kernel basis + {0,+-1} / difference-set modes + exact double-completion
verification.
"""
import importlib
import json
import math
import os
import sys
from collections import defaultdict
from fractions import Fraction

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import paths_common as PC  # noqa: E402

OUT_DIR = os.path.join(str(PC.DATA_ROOT), 'lower_bounds')

sys.path.insert(0, str(PC.engine("z4_3464")))
sys.path.insert(0, str(PC.engine("z3_31212")))
sys.path.insert(0, str(PC.engine("z3_4612")))

RTH = importlib.import_module(PC.fname("RTH_numerical_verification"))  # noqa: E402
from roundE_4612_frontier_dp import build_face_system  # noqa: E402
from roundD_4612_numeric import build_torus as build_torus_4612, extract_faces  # noqa: E402


def kernel_basis(n_cols, rows):
    A = [[Fraction(0)] * n_cols for _ in rows]
    for r, row in enumerate(rows):
        for c, v in row:
            A[r][c] += v
    pivots = []
    r = 0
    for c in range(n_cols):
        piv = next((rr for rr in range(r, len(A)) if A[rr][c] != 0), None)
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
            v[pc] = -A[ri][pc if False else fc]
        basis.append(v)
    return len(pivots), basis


def to_int_vec(vf):
    from math import gcd
    dens = [x.denominator for x in vf if x != 0]
    if not dens:
        return None
    Lc = 1
    for d in dens:
        Lc = Lc * d // math.gcd(Lc, d)
    iv = [int(x * Lc) for x in vf]
    g = 0
    for x in iv:
        g = math.gcd(g, abs(x))
    return [x // g for x in iv] if g else None


def classify(vec):
    nz = [x for x in vec if x != 0]
    if not nz:
        return 0, 'zero'
    mx = max(abs(x) for x in nz)
    if all(x > 0 for x in nz):
        return mx, 'nonneg'
    if all(x < 0 for x in nz):
        return mx, 'nonpos'
    return mx, 'mixed'


# ================= 4.8.8 Lambda =================
M_EXP = {(1, 0), (0, 1), (1, 1)}


def run_488lam(W=6, region=(1, 4)):
    t = RTH.build_488(W)
    E = t.E
    g = 1.0 * (1 + math.sqrt(2))          # cell distance for s=1

    def cell_xy(p):
        return (p[0] / g, p[1] / g)

    # edge midpoints (canonical coords) and edge -> faces
    eid_mid = {}
    for ek, eid in t.edge_ids.items():
        eid_mid[eid] = (float(ek[0]), float(ek[1]))
    edge_faces = [[] for _ in range(E)]
    for fi, segs in enumerate(t.face_edges):
        for eid, s in segs:
            edge_faces[eid].append(fi)

    # integer rows for the Lambda -> Z^2 split: each real projection row
    # -> 2 rows (variables = (m_e, n_e) interleaved as 2e/2e+1)
    rows = []
    for rx, ry in zip(t.xrows, t.yrows):
        for row in (rx, ry):
            r1 = []
            r2 = []
            for e, sg in [(eid, 1) for eid in range(E)]:
                pass
            break
        break
    # the block above is an indentation guard; the real construction follows:
    rows = []

    def rows_of(row_qd):
        r1 = []
        r2 = []
        for e in range(E):
            q = row_qd[e]
            a = q.a
            b = q.b
            if a != 0 or b != 0:
                # rational part: 2a*m + b*n ; sqrt(2)/2 part: a*n + b*m
                # (multiplied by 2 to clear)
                if 2 * a != 0:
                    r1.append((2 * e, 4 * a))
                if b != 0:
                    r1.append((2 * e + 1, 4 * b))
                if a != 0:
                    r2.append((2 * e + 1, 4 * a))
                if 2 * b != 0:
                    r2.append((2 * e, 4 * b))
        if r1:
            rows.append(r1)
        if r2:
            rows.append(r2)

    for rx, ry in zip(t.xrows, t.yrows):
        rows_of(rx)
        rows_of(ry)

    # zero forcing outside the region
    lo, hi = region
    region_edges = set(e for e in range(E)
                       if lo <= cell_xy(eid_mid[e])[0] <= hi
                       and lo <= cell_xy(eid_mid[e])[1] <= hi)
    n_zero = 0
    for e in range(E):
        if e not in region_edges:
            rows.append([(2 * e, 1)])
            rows.append([(2 * e + 1, 1)])
            n_zero += 2
    rank, basis = kernel_basis(2 * E, rows)
    dim = 2 * E - rank
    res = {'W': W, 'E': E, 'n_vars': 2 * E, 'kernel_dim_region': dim,
           'n_region_edges': len(region_edges), 'n_zero_rows': n_zero}
    if dim == 0:
        return res

    # mode extraction: per-edge pair (dm, dn); find {0,+-1} modes whose
    # entry pairs lie in D (relative to some seat)
    cands = []
    ints_all = [to_int_vec(b) for b in basis]
    for iv in ints_all:
        if iv:
            pairs = [(iv[2 * e], iv[2 * e + 1]) for e in range(E)]
            nz = [p for p in pairs if p != (0, 0)]
            mx = max(max(abs(a), abs(b)) for a, b in nz) if nz else 0
            cands.append((iv, pairs, mx))
    res['basis_diag'] = [{'max_abs': mx,
                          'support': sum(1 for p in ps if p != (0, 0))}
                         for (_, ps, mx) in cands]
    def try_pairs(pairs, desc):
        for base in ((1, 0), (0, 1), (1, 1)):
            if all(p == (0, 0) or (base[0] + p[0], base[1] + p[1]) in M_EXP
                   for p in pairs):
                res['mode'] = {'base': list(base), 'desc': desc,
                               'support': sum(1 for p in pairs if p != (0, 0))}
                res['flip'] = flip_488(t, edge_faces, pairs, base)
                return True
        return False

    for iv, pairs, mx in cands:
        if mx != 1:
            continue
        if try_pairs(pairs, 'basis'):
            return res
    # pairwise combinations with +/-1
    n = len(cands)
    for i in range(n):
        for j in range(i + 1, n):
            for ci in (1, -1):
                for cj in (1, -1):
                    iv = [ci * cands[i][0][t2] + cj * cands[j][0][t2]
                          for t2 in range(2 * E)]
                    pairs = [(iv[2 * e], iv[2 * e + 1]) for e in range(E)]
                    mx = max(max(abs(a), abs(b)) for a, b in pairs
                             if (a, b) != (0, 0)) if any(pairs) else 0
                    if mx != 1:
                        continue
                    if try_pairs(pairs, 'combo d%d+d%d' % (i, j)):
                        return res
    res['mode'] = None
    return res


def flip_488(t, edge_faces, pairs, base):
    """double completion: t=0: base; t=1: base+pairs. Exact verification
    against the 4 integer rows per face + M_exp check"""
    support = [e for e, p in enumerate(pairs) if p != (0, 0)]
    P = sorted(set(fi for e in support for fi in edge_faces[e]))
    for s in (0, 1):
        val = {}
        for e in range(t.E):
            m = base[0] + s * pairs[e][0]
            n = base[1] + s * pairs[e][1]
            assert (m, n) in M_EXP, ('alphabet', e, (m, n))
            val[e] = (m, n)
        for fi, (rx, ry) in enumerate(zip(t.xrows, t.yrows)):
            for row in (rx, ry):
                s_r = s2 = Fraction(0)
                for eid, sg in t.face_edges[fi]:
                    q = row[eid]
                    m, n = val[eid]
                    # q.(m+n*sqrt2/2)*sg: rational part = q.a*m + q.b*n/2;
                    # sqrt2 coefficient = q.a*n + q.b*m
                    s_r += sg * (q.a * m + q.b * n / 2)
                    s2 += sg * (q.a * n + q.b * m)
                assert s_r == 0 and s2 == 0, ('closure', fi)
    # bounding box (cell coords)
    eid_mid = {eid: (float(ek[0]), float(ek[1])) for ek, eid in t.edge_ids.items()}
    g = 1.0 * (1 + math.sqrt(2))
    cs = [(eid_mid[e][0] / g, eid_mid[e][1] / g) for e in support]
    w1 = max(c[0] for c in cs) - min(c[0] for c in cs)
    w2 = max(c[1] for c in cs) - min(c[1] for c in cs)
    d1 = max(1, int(math.ceil(w1 - 1e-9)))
    d2 = max(1, int(math.ceil(w2 - 1e-9)))
    return {'base': list(base), 'patch_faces': len(P), 'support_edges': len(support),
            'd1': d1, 'd2': d2, 'rho_per_cell': 1.0 / (d1 * d2),
            's_lower_bound_per_face': math.log(2) / (2 * d1 * d2)}


# ================= 4.6.12 =================
def run_4612(Lx=5, Ly=5, region=(1, 3)):
    # anchor: the small system with rank_check=True asserts dim = 4N+4
    # internally (registered value)
    _ = build_face_system(2, 2, rank_check=True)
    face_eqs, n_cls, n_free, N, dim_none, all_eq_cls, face_nbrs, edge_cls =         build_face_system(Lx, Ly, rank_check=False)
    vid, edges, V, E, pts = build_torus_4612(Lx, Ly)
    # class position = midpoint of any member edge; use vid
    # (fundamental-domain residuals, consistent per vertex) not pts
    # (arbitrary copies)
    T1 = (2.0, math.sqrt(3.0))
    det = T1[0] * T1[1] - T1[1] * 1.0          # det of [[2,1],[0,sqrt3]] = 2*sqrt3
    Minv = [[T1[1] / det, -1.0 / det], [0.0, T1[0] / det]]

    def cell_xy(p):
        # vid holds fundamental-domain residuals (unit cell T1=(2,0),
        # T2=(1,sqrt3)); reduce mod the period after the shear
        a = (Minv[0][0] * p[0] + Minv[0][1] * p[1]) % Lx
        b = (Minv[1][0] * p[0] + Minv[1][1] * p[1]) % Ly
        return (a, b)

    cls_mid = {}
    for i, (u, v, b) in enumerate(edges):
        c = edge_cls[i]
        if c not in cls_mid:
            cls_mid[c] = ((u[0] + v[0]) / 2, (u[1] + v[1]) / 2)
    lo, hi = region
    region_cls = set(cl for cl in range(n_cls)
                     if cl in cls_mid
                     and lo <= cell_xy(cls_mid[cl])[0] <= hi
                     and lo <= cell_xy(cls_mid[cl])[1] <= hi)
    rows = []
    for eqs_f in face_eqs:
        for eq in eqs_f:
            rows.append(sorted(eq.items()))
    n_zero = 0
    for cl in range(n_cls):
        if cl not in region_cls:
            rows.append([(cl, 1)])
            n_zero += 1
    rank, basis = kernel_basis(n_cls, rows)
    dim = n_cls - rank
    res = {'Lx': Lx, 'Ly': Ly, 'N': N, 'n_cls': n_cls,
           'kernel_dim_region': dim, 'n_region_cls': len(region_cls)}
    if dim == 0:
        return res
    cands = []
    for b in basis:
        iv = to_int_vec(b)
        if iv:
            cands.append(iv)
    res['basis_diag'] = [{'kind': classify(iv)[1], 'max_abs': classify(iv)[0],
                          'support': sum(1 for x in iv if x != 0)} for iv in cands]
    for iv in cands:
        mx, kind = classify(iv)
        if mx == 1:
            res['mode'] = {'kind': kind, 'support': sum(1 for x in iv if x != 0)}
            res['flip'] = flip_4612(face_eqs, all_cls_of(n_cls), iv,
                                    cls_mid, cell_xy)
            return res
    res['mode'] = None
    return res


def union_(parent, a, b):
    ra, rb = a, b
    while parent[ra] != ra:
        ra = parent[ra]
    while parent[rb] != rb:
        rb = parent[rb]
    if ra != rb:
        parent[rb] = ra


def all_cls_of(n_cls):
    return list(range(n_cls))


def flip_4612(face_eqs, all_cls, mode, cls_mid, cell_xy):
    outs = []
    for base, alpha in ((2, (1, 2, 3)), (1, (1, 2)), (2, (1, 2))):
        ok = True
        for s in (1, -1):
            val = {cl: base + s * mode[cl] for cl in all_cls}
            if any(not (alpha[0] <= val[cl] <= alpha[-1]) for cl in all_cls):
                ok = False
                break
            for eqs_f in face_eqs:
                for eq in eqs_f:
                    if sum(c * val[cl] for cl, c in eq.items()) != 0:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                break
        if ok:
            outs.append({'base': base, 'alphabet': list(alpha)})
    support = [cl for cl in all_cls if mode[cl] != 0]
    coords = [cell_xy(cls_mid[cl]) for cl in support]
    w1 = max(c[0] for c in coords) - min(c[0] for c in coords)
    w2 = max(c[1] for c in coords) - min(c[1] for c in coords)
    d1 = max(1, int(math.ceil(w1 - 1e-9)))
    d2 = max(1, int(math.ceil(w2 - 1e-9)))
    # 4.6.12 has 6 faces per cell (3 squares + 2 hexagons + 1 dodecagon),
    # not the /3 copy of the 3.12.12 count
    return {'alphabets_legal': outs, 'support_classes': len(support),
            'd1': d1, 'd2': d2, 'rho_per_cell': 1.0 / (d1 * d2),
            's_lower_bound_per_face': math.log(2) / (6 * d1 * d2),
            'mode_kind': classify([mode[cl] for cl in support])[1]}


if __name__ == '__main__':
    out = {}
    print('=== 4.8.8 Lambda W=6 region[1,4] ===', flush=True)
    out['tsq488lam'] = run_488lam(W=6, region=(1, 4))
    print(json.dumps(out['tsq488lam'], ensure_ascii=False, indent=1))
    print('=== 4.6.12 Lx=Ly=5 region[1,3] ===', flush=True)
    out['rth4612'] = run_4612(Lx=6, Ly=6, region=(1, 4))
    print(json.dumps(out['rth4612'], ensure_ascii=False, indent=1))
    with open(os.path.join(OUT_DIR, 'flip_488lam_4612_result.json'), 'w', encoding='utf-8') as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
    print('saved')
