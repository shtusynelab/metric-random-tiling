# -*- coding: utf-8 -*-
"""Real-metric certificate figures for the three positive-entropy carriers:
3.4.6.4 / 3.12.12 / 4.6.12.
Replay the search to recover the mode vector -> generic equiangular BFS
real-metric layout -> render the double completion (+v / -v).
"""
import importlib
import json
import math
_json = json
import os
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import paths_common as PC  # noqa: E402

OUT = PC.FIGURES / 'assets_certificates'
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(PC.engine('z4_3464')))
sys.path.insert(0, str(PC.engine('z3_31212')))
sys.path.insert(0, str(PC.engine('z3_4612')))

RTH = importlib.import_module(PC.fname("RTH_numerical_verification"))  # noqa: E402
import lib31212  # noqa: E402
from roundE_4612_frontier_dp import build_face_system  # noqa: E402
from roundD_4612_numeric import build_torus as bt4612, extract_faces  # noqa: E402
from flip_3464_31212 import run_3464  # noqa: E402

U = lambda deg: (math.cos(math.radians(deg)), math.sin(math.radians(deg)))


def u_add(p, ell, d):
    return (p[0] + ell * d[0], p[1] + ell * d[1])


def norm_int(basis_vec):
    from math import gcd
    dens = [x.denominator for x in basis_vec if x != 0]
    if not dens:
        return None
    Lc = 1
    for d in dens:
        Lc = Lc * d // math.gcd(Lc, d)
    iv = [int(x * Lc) for x in basis_vec]
    gg = 0
    for x in iv:
        gg = math.gcd(gg, abs(x))
    if gg == 0:
        return None
    iv = [x // gg for x in iv]
    nz = [x for x in iv if x]
    return iv if nz and max(abs(x) for x in nz) == 1 else None


def kernel_basis(n_cols, rows):
    from fractions import Fraction
    A = [[Fraction(0)] * n_cols for _ in rows]
    for r, row in enumerate(rows):
        for c, v in row:
            A[r][c] += Fraction(v)
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
                fac = A[rr][c]
                A[rr] = [a - fac * b for a, b in zip(A[rr], A[r])]
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
    return basis


# ---------------- generic equiangular BFS layout ----------------
def layout(face_slots, face_nbrs, edge_val, root, include):
    placed = {}
    slots = face_slots[root]
    vs = [(0.0, 0.0)]
    for (eid, d) in slots:
        vs.append(u_add(vs[-1], edge_val[eid], U(d)))
    assert math.hypot(vs[-1][0] - vs[0][0], vs[-1][1] - vs[0][1]) < 1e-6, 'root face not closed'
    placed[root] = vs
    queue = [root]
    while queue:
        f = queue.pop(0)
        fv = placed[f]
        n = len(face_slots[f])
        for j, (eid, d) in enumerate(face_slots[f]):
            for nf in face_nbrs[eid]:
                if nf == f or nf in placed or nf not in include:
                    continue
                m = [k for k, (e2, d2) in enumerate(face_slots[nf]) if e2 == eid][0]
                nn = len(face_slots[nf])
                nv = [None] * nn
                nv[m] = fv[(j + 1) % n]
                for tt in range(nn):
                    k = (m + tt) % nn
                    e2, d2 = face_slots[nf][k]
                    nv[(k + 1) % nn] = u_add(nv[k], edge_val[e2], U(d2))
                assert math.hypot(nv[m][0] - fv[(j + 1) % n][0],
                                  nv[m][1] - fv[(j + 1) % n][1]) < 1e-6, ('glue', nf)
                placed[nf] = nv + [nv[0]]
                queue.append(nf)
    return placed


def render(fname_out, placed_p, placed_m, face_vals, patch, face_slots=None):
    """face_vals: fid -> {+1: [values], -1: [values]} (in slot order)"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Polygon as MplPolygon
    COL = {1: '#1f77b4', 2: '#8c8c8c', 3: '#d62728'}
    CTX = '#c6c6c6'
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 5.6))
    for ax, sign, tag in ((axes[0], +1, 'completion = base 2 + v'),
                          (axes[1], -1, 'completion = base 2 - v')):
        for f in sorted(placed_p):
            fv = placed_p[f] if sign > 0 else placed_m[f]
            fill = '#ffffff'
            ax.add_patch(MplPolygon(fv[:-1], closed=True, facecolor=fill,
                                    edgecolor='none', lw=0, zorder=1))

        labeled = set()
        drawn = set()
        for f in sorted(placed_p):
            fv = (placed_p if sign > 0 else placed_m)[f]
            vals = face_vals.get(f, {}).get(sign, [2] * (len(fv) - 1))
            n = len(fv) - 1
            for k in range(n):
                val = vals[k]
                a, b = fv[k], fv[k + 1]
                ek = (round(a[0], 4), round(a[1], 4), round(b[0], 4), round(b[1], 4))
                if ek in drawn:
                    continue
                drawn.add(ek)
                ax.plot([a[0], b[0]], [a[1], b[1]], color=COL[val], lw=2.6,
                        solid_capstyle='butt', zorder=3)
                mk = (round((a[0] + b[0]) / 2, 4), round((a[1] + b[1]) / 2, 4))
                if val != 2 and mk not in labeled:
                    labeled.add(mk)
                    tx, ty = mk
                    ax.text(tx, ty, str(val), color=COL[val],
                            fontsize=8, fontweight='bold', ha='center', va='center',
                            zorder=5,
                            bbox=dict(boxstyle='round,pad=0.12', fc='white',
                                      ec='none', alpha=0.85))
        ax.text(0.02, 0.98, tag, transform=ax.transAxes, fontsize=10,
                va='top', ha='left', family='monospace')
        ax.set_aspect('equal')
        ax.axis('off')
        xs = [p for fv in placed_p.values() for p in fv]
        ax.set_xlim(min(p[0] for p in xs) - 1, max(p[0] for p in xs) + 1)
        ax.set_ylim(min(p[1] for p in xs) - 1, max(p[1] for p in xs) + 1)
    if face_slots is not None:
        pcnt = defaultdict(int)
        for f2 in patch:
            for (e2, d2) in face_slots[f2]:
                pcnt[e2] += 1
        bnd = set(e for e, c in pcnt.items() if c == 1)
        for f in sorted(patch):
            fv = (placed_p if sign > 0 else placed_m)[f]
            n = len(fv) - 1
            for k, (e2, d2) in enumerate(face_slots[f]):
                if e2 in bnd:
                    a, b = fv[k], fv[k + 1]
                    ax.plot([a[0], b[0]], [a[1], b[1]], color='#000000',
                            lw=3.2, solid_capstyle='butt', zorder=4)
    handles = [Line2D([0], [0], color=COL[x], lw=3) for x in (1, 2, 3)]
    axes[1].legend(handles, ['edge value 1', 'edge value 2', 'edge value 3'],
                   loc='lower left', fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(fname_out, dpi=150)
    plt.close(fig)
    print('saved:', fname_out)


def mx_my(a, b):
    return (a[0] + b[0]) / 2, (a[1] + b[1]) / 2


def my_off(a, b):
    return mx_my(a, b)


# ---------------- 3.4.6.4 ----------------
def do_3464():
    res, (t, face_coord, edge_faces, vec) = run_3464(L=6, region=(1, 4))
    assert vec is not None and res['flip']['alphabet'] == [1, 2, 3]
    mode = list(vec)
    assert res['mode']['support_edges'] == sum(1 for x in mode if x) == 24
    face_slots, face_nbrs = {}, defaultdict(list)
    for fi, segs in enumerate(t.face_edges):
        pts = t.faces[fi][1]
        n = len(pts)
        slots = []
        for k in range(n):
            eid, sg = segs[k]
            a, b = pts[k], pts[(k + 1) % n]
            d = math.degrees(math.atan2(b[1] - a[1], b[0] - a[0]))
            slots.append((eid, d))
            face_nbrs[eid].append(fi)
        face_slots[fi] = slots
    support = [e for e, x in enumerate(mode) if x]
    patch = sorted(set(fi for e in support for fi in edge_faces[e]))
    ring = set()
    for fi in patch:
        for (eid, sg) in t.face_edges[fi]:
            for f2 in edge_faces[eid]:
                if f2 not in patch:
                    ring.add(f2)
    vmap = defaultdict(list)
    for fi2 in range(len(t.faces)):
        for kk in t.faces[fi2][0]:
            vmap[kk].append(fi2)
    include = set(patch) | ring
    for _ in range(2):
        grown = set()
        for fi2 in include:
            for kk in t.faces[fi2][0]:
                for f3 in vmap[kk]:
                    if f3 not in include:
                        grown.add(f3)
        include |= grown
    edge_base = {e: 2 for e in range(t.E)}
    face_vals = {}
    for fi in include:
        n = len(t.face_edges[fi])
        face_vals[fi] = {+1: [2 + mode[t.face_edges[fi][k][0]] for k in range(n)],
                         -1: [2 - mode[t.face_edges[fi][k][0]] for k in range(n)]}
    placed_p = layout(face_slots, face_nbrs,
                      {e: 2 + mode[e] for e in range(t.E)}, min(patch), include)
    placed_m = layout(face_slots, face_nbrs,
                      {e: 2 - mode[e] for e in range(t.E)}, min(patch), include)
    face_vals = {}
    for f in patch:
        n = len(t.face_edges[f])
        face_vals[f] = {+1: [2 + mode[t.face_edges[f][k][0]] for k in range(n)],
                        -1: [2 - mode[t.face_edges[f][k][0]] for k in range(n)]}
    render(str(OUT / 'cert_3464.png'), placed_p, placed_m, face_vals,
           set(patch), face_slots)
    return res


# ---------------- 3.12.12 ----------------
def do_31212():
    """Single flower pattern: the 12 honeycomb-flower support edges mapped to
    3.12.12 mid-section edge classes via (torus-reduced) midpoints."""
    L = 7
    t = RTH.build_31212(L)
    cls_val, eqs, all_cls, n_cls, dim_ref, N, V, E, F = lib31212.build_reduced(L)
    g = 1.0 + 2.0 / math.sqrt(3.0)
    D0 = math.sqrt(3.0) * g
    A1 = (D0, 0.0)
    A2 = (D0 * 0.5, D0 * math.sqrt(3.0) / 2.0)

    def hc_center(i, j):
        return (i * A1[0] + j * A2[0], i * A1[1] + j * A2[1])

    CENTRAL = (3, 3)
    AX = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]
    fs = {CENTRAL} | {(3 + d[0], 3 + d[1]) for d in AX}
    # 12 honeycomb-flower support edges (absolute midpoints, same Cartesian frame)
    orig = {}
    for c1 in fs:
        for d in AX:
            c2 = (c1[0] + d[0], c1[1] + d[1])
            if c2 in fs:
                pair = tuple(sorted([c1, c2]))
                if pair not in orig:
                    sgn = -1 if CENTRAL in pair else +1
                    m1 = hc_center(*c1)
                    m2 = hc_center(*c2)
                    key = (round((m1[0] + m2[0]) / 2, 6), round((m1[1] + m2[1]) / 2, 6))
                    orig[key] = (sgn, pair)
    assert len(orig) == 12, len(orig)
    # reduced-midpoint index of the 3.12.12 edges
    eid_by_mid = {}
    for ek, eid in t.edge_ids.items():
        eid_by_mid[(round(float(ek[0]), 6), round(float(ek[1]), 6))] = eid
    # matching: honeycomb midpoints -> torus-reduced coordinates
    mode = [0] * n_cls
    matched = 0
    P1 = t.P1 * 1.0
    P2 = t.P2 * 1.0
    detP = P1[0] * P2[1] - P1[1] * P2[0]
    MinvP = [[P2[1] / detP, -P2[0] / detP], [-P1[1] / detP, P1[0] / detP]]
    for key, (sgn, cpair) in orig.items():
        # key is an absolute coordinate; reduce to the fundamental domain
        a = MinvP[0][0] * key[0] + MinvP[0][1] * key[1]
        b = MinvP[1][0] * key[0] + MinvP[1][1] * key[1]
        au = a - math.floor(a + 1e-9)
        av = b - math.floor(b + 1e-9)
        pr = (au * P1[0] + av * P2[0], au * P1[1] + av * P2[1])
        k2 = (round(pr[0], 6), round(pr[1], 6))
        if k2 in eid_by_mid:
            eid = eid_by_mid[k2]
            cl = cls_val[eid]
            mode[cl] = sgn
            matched += 1
    assert matched == 12, ('matched count', matched)
    for eq in eqs:
        assert sum(c * mode[cl] for cl, c in eq.items()) == 0, 'equation not satisfied'
    sup = [cl for cl in all_cls if mode[cl]]
    assert len(sup) == 12
    eid_faces = defaultdict(list)
    for fi, segs in enumerate(t.face_edges):
        for (eid, sg) in segs:
            eid_faces[eid].append(fi)
    patch = sorted(set(fi for eid, cl in cls_val.items() if cl in set(sup)
                       for fi in eid_faces[eid]))
    assert len(patch) == 7, ('patch face count', len(patch))
    ring = set()
    for fi in patch:
        for (eid, sg) in t.face_edges[fi]:
            for f2 in eid_faces[eid]:
                if f2 not in patch:
                    ring.add(f2)
    include = set(patch) | ring
    face_slots, face_nbrs = {}, defaultdict(list)
    for fi, segs in enumerate(t.face_edges):
        pts = t.faces[fi][1]
        n = len(pts)
        slots = []
        for k in range(n):
            eid, sg = segs[k]
            a, b = pts[k], pts[(k + 1) % n]
            d = math.degrees(math.atan2(b[1] - a[1], b[0] - a[0]))
            slots.append((eid, d))
            face_nbrs[eid].append(fi)
        face_slots[fi] = slots
    placed_p = layout(face_slots, face_nbrs,
                      {eid: 2 + mode[cls_val[eid]] for eid in eid_faces},
                      min(patch), include)
    placed_m = layout(face_slots, face_nbrs,
                      {eid: 2 - mode[cls_val[eid]] for eid in eid_faces},
                      min(patch), include)
    face_vals = {}
    for fi in include:
        n = len(t.face_edges[fi])
        face_vals[fi] = {+1: [2 + mode[cls_val[t.face_edges[fi][k][0]]] for k in range(n)],
                         -1: [2 - mode[cls_val[t.face_edges[fi][k][0]]] for k in range(n)]}
    render(str(OUT / 'cert_31212.png'), placed_p, placed_m, face_vals,
           set(patch), face_slots)
    return {'patch_faces': len(patch), 'support_classes': 12,
            'patch_census': {'12': sum(1 for f in patch if len(t.faces[f][0]) == 12)},
            'matched': matched}


# ---------------- 4.6.12 ----------------
def do_4612():
    Lx, Ly = 6, 6
    face_eqs, n_cls, n_free, N, dim_none, all_eq_cls, face_nbrs_sys, edge_cls =         build_face_system(Lx, Ly, rank_check=False)
    vid, edges, V, E, pts = bt4612(Lx, Ly)
    faces = extract_faces(edges)
    T1 = (2.0, math.sqrt(3.0))
    det = T1[0] * T1[1] - T1[1] * 1.0
    Minv = [[T1[1] / det, -1.0 / det], [0.0, T1[0] / det]]
    cls_mid = {}
    for i, (u, v, b) in enumerate(edges):
        c = edge_cls[i]
        if c not in cls_mid:
            cls_mid[c] = ((u[0] + v[0]) / 2, (u[1] + v[1]) / 2)

    def cell_of(p):
        return ((Minv[0][0] * p[0] + Minv[0][1] * p[1]) % Lx,
                (Minv[1][0] * p[0] + Minv[1][1] * p[1]) % Ly)

    region_cls = set(cl for cl in range(n_cls)
                     if cl in cls_mid
                     and 1 <= cell_of(cls_mid[cl])[0] <= 4
                     and 1 <= cell_of(cls_mid[cl])[1] <= 4)
    rows = []
    for eqs_f in face_eqs:
        for eq in eqs_f:
            rows.append(sorted(eq.items()))
    for cl in range(n_cls):
        if cl not in region_cls:
            rows.append([(cl, 1)])
    basis = kernel_basis(n_cls, rows)
    mode = None
    dim = n_cls - len(basis)
    for b in basis:
        iv = norm_int(b)
        if iv is not None:
            mode = iv
            break
    assert mode is not None
    support_set = set(cl for cl in range(n_cls) if mode[cl])
    eid_faces = defaultdict(list)
    for fi, f in enumerate(faces):
        for h in f:
            eid_faces[h[3]].append(fi)
    support = [e for e in range(E) if edge_cls[e] in support_set]
    patch = sorted(set(fi for e in support for fi in eid_faces[e]))
    ring = set()
    for fi in patch:
        for h in faces[fi]:
            for f2 in eid_faces[h[3]]:
                if f2 not in patch:
                    ring.add(f2)
    vmap = defaultdict(list)
    for fi2, f in enumerate(faces):
        for h in f:
            vmap[h[0]].append(fi2)
            vmap[h[1]].append(fi2)
    include = set(patch) | ring
    for _ in range(2):
        grown = set()
        for fi2 in include:
            for h in faces[fi2]:
                for f3 in vmap[h[0]] + vmap[h[1]]:
                    if f3 not in include:
                        grown.add(f3)
        include |= grown
    face_vals = {}
    for fi in include:
        n = len(faces[fi])
        face_vals[fi] = {+1: [2 + mode[edge_cls[h[3]]] for h in faces[fi]],
                         -1: [2 - mode[edge_cls[h[3]]] for h in faces[fi]]}
    include_vset = set()
    for fi in include:
        for h in faces[fi]:
            include_vset.add(h[0])
            include_vset.add(h[1])
    # vertex BFS coordinates: restricted to the include vertex set
    # (topological disk, no wrap-around contamination)
    adjv = defaultdict(list)
    for i, (u, v, b) in enumerate(edges):
        adjv[u].append((v, b))
        adjv[v].append((u, (b + 6) % 12))
    rootv = faces[patch[0]][0][0]
    vpos = {rootv: (0.0, 0.0)}
    q = [rootv]
    while q:
        x = q.pop(0)
        for (y, b) in adjv[x]:
            if y in vpos or y not in include_vset:
                continue
            vpos[y] = u_add(vpos[x], 1.0, U(30 * b))
            q.append(y)
    # face slot directions (consistent with the vpos coordinates)
    face_slots, face_nbrs = {}, defaultdict(list)
    for fi in include:
        f = faces[fi]
        slots = []
        for k, (uu, vv, bdir, eid) in enumerate(f):
            pu, pv = vpos[uu], vpos[vv]
            d = math.degrees(math.atan2(pv[1] - pu[1], pv[0] - pu[0]))
            slots.append((eid, d))
            face_nbrs[eid].append(fi)
        face_slots[fi] = slots
    placed_p = layout(face_slots, face_nbrs,
                      {eid: 2 + mode[edge_cls[eid]] for eid in range(E)},
                      min(patch), include)
    placed_m = layout(face_slots, face_nbrs,
                      {eid: 2 - mode[edge_cls[eid]] for eid in range(E)},
                      min(patch), include)
    face_vals = {}
    for f in patch:
        n = len(faces[f])
        face_vals[f] = {+1: [2 + mode[edge_cls[h[3]]] for h in faces[f]],
                        -1: [2 - mode[edge_cls[h[3]]] for h in faces[f]]}
    render(str(OUT / 'cert_4612.png'), placed_p, placed_m, face_vals,
           set(patch), face_slots)
    return {'patch_faces': len(patch), 'dim_region': dim,
            'patch_census': dict(sorted(json.loads(json.dumps(
                {str(sz): sum(1 for f2 in patch if len(faces[f2]) == int(sz))
                 for sz in (4, 6, 12)})).items())), 'support_classes': len(support_set)}


if __name__ == '__main__':
    print('=== 3.4.6.4 ===', flush=True)
    r1 = do_3464()
    print('  flip:', json.dumps(r1['flip'], ensure_ascii=False), flush=True)
    print('=== 3.12.12 ===', flush=True)
    r2 = do_31212()
    print(' ', json.dumps(r2, ensure_ascii=False), flush=True)
    print('=== 4.6.12 ===', flush=True)
    r3 = do_4612()
    print(' ', json.dumps(r3, ensure_ascii=False), flush=True)
