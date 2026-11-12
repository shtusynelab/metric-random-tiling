# -*- coding: utf-8 -*-
"""gen_metric_patches.py - 3.12.12 / 4.6.12 metric-disorder pattern rendering
(area coloring).
Method: face closure of the equiangular tiling = 2 linear equations per face
(directions fixed, coefficients cos/sin) => the solution is
x = 1 + sum_i c_i k_i (k_i = integer-kernel basis vectors; basis computed
exactly by Fraction-RREF).
Each face is walked edge by edge from its own frame (edge length = solution
value) giving a closed polygon; faces are glued by reflection across shared
edges (a shared edge takes one common value => consistency); vertices are
deduplicated by position.  Coloring: only the target polygon class is
colored by area (viridis); the rest are left white with outlines.
Line attributes: all edges uniform thick black, consistent with the 3.4.6.4
assets (non-target faces = light grey #eeeeee fill; LW calibration inside
render).
Target class: the type with the largest relative area variation (3.12.12
triangles are locked => dodecagons).
Output: <FIGURES>/assets_carriers/31212_metric.png, 4612_metric.png"""
import math
import sys
from fractions import Fraction
from pathlib import Path
from collections import deque, defaultdict
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import paths_common as PC  # noqa: E402

OUT = PC.FIGURES / "assets_carriers"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(PC.engine("z3_31212")))
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Polygon  # noqa: E402


# ---------------- linear algebra (Fraction-RREF kernel basis) ----------------
def nullspace_basis(rows, n):
    A = [[Fraction(int(v)) for v in r] for r in rows]
    m = len(A)
    piv = []
    r = 0
    for c in range(n):
        p = next((i for i in range(r, m) if A[i][c] != 0), None)
        if p is None:
            continue
        A[r], A[p] = A[p], A[r]
        inv = Fraction(1, 1) / A[r][c]
        A[r] = [v * inv for v in A[r]]
        for i in range(m):
            if i != r and A[i][c] != 0:
                f = A[i][c]
                A[i] = [a - f * b for a, b in zip(A[i], A[r])]
        piv.append(c)
        r += 1
        if r == m:
            break
    free = [c for c in range(n) if c not in piv]
    basis = []
    for fc in free:
        v = [Fraction(0)] * n
        v[fc] = Fraction(1)
        for i, pc in enumerate(piv):
            v[pc] = -A[i][fc]
        basis.append(v)
    return basis


def rand_field(K, rng, sig=0.22):
    """Gaussian random field: equal-weight N(0,1) superposition of the full
    kernel basis, amplitude calibrated by std.
    The positivity constraint compresses sigma to at most ~0.3 at n~150
    edges by the max statistic; this is a physical boundary of the ensemble,
    not a sampling artefact (independent +/-50% noise necessarily flips some
    edges negative)."""
    c = rng.normal(0, 1, len(K))
    d = c @ K
    sd = d.std()
    if sd <= 0:
        return None
    return 1.0 + d * (sig / sd)


# ---------------- generic: build edge table + closure rows from the ideal face list ----------------
# exact integer split of 2cos/2sin: value = R + S*sqrt(3) (theta a multiple of 30 deg)
COS2 = {0: (2, 0), 30: (0, 1), 60: (1, 0), 90: (0, 0), 120: (-1, 0),
        150: (0, -1), 180: (-2, 0), 210: (0, -1), 240: (-1, 0),
        270: (0, 0), 300: (1, 0), 330: (0, 1)}
SIN2 = {0: (0, 0), 30: (1, 0), 60: (0, 1), 90: (2, 0), 120: (0, 1),
        150: (1, 0), 180: (0, 0), 210: (-1, 0), 240: (0, -1),
        270: (-2, 0), 300: (0, -1), 330: (-1, 0)}


def build_edge_system(faces_pts):
    """Exact integer closure rows: 4 rows per face (2cos/2sin each split
    into R and S parts).
    All directions theta are multiples of 30 deg => exact rows, exact
    kernel space, bit-exact closure."""
    eid_of, edges = {}, []
    for P in faces_pts:
        k = len(P)
        for j in range(k):
            a = tuple(np.round(P[j], 6))
            b = tuple(np.round(P[(j + 1) % k], 6))
            key = tuple(sorted([a, b]))
            if key not in eid_of:
                eid_of[key] = len(edges)
                edges.append(key)
    n = len(edges)
    rows = []
    for P in faces_pts:
        k = len(P)
        rc = [0] * n
        sc = [0] * n
        rs = [0] * n
        ss = [0] * n
        for j in range(k):
            a = tuple(np.round(P[j], 6))
            b = tuple(np.round(P[(j + 1) % k], 6))
            key = tuple(sorted([a, b]))
            e = eid_of[key]
            d = np.array(b, dtype=float) - np.array(a, dtype=float)
            theta = math.degrees(math.atan2(d[1], d[0])) % 360
            idd = int(round(theta / 30.0))
            assert abs(theta - 30 * idd) < 1e-3, (theta, d)
            idx = idd % 12
            # coefficient = this face's traversal direction (no key sign
            # needed; the opposite face naturally takes the reversed direction)
            Rc, Sc = COS2[30 * idx]
            Rs, Ss = SIN2[30 * idx]
            rc[e] += Rc
            sc[e] += Sc
            rs[e] += Rs
            ss[e] += Ss
        rows.append([Fraction(v) for v in rc])
        rows.append([Fraction(v) for v in sc])
        rows.append([Fraction(v) for v in rs])
        rows.append([Fraction(v) for v in ss])
    return eid_of, edges, rows


def metric_faces(faces_pts, eid_of, edges, x):
    """Walk each face edge by edge with edge values x (own-frame polygon),
    then glue across shared edges by BFS into a global patch.
    Shared-edge detection: two ideal polygons share the same (rounded)
    vertex pair."""
    from collections import defaultdict, deque
    # per-face own-frame polygon
    local = []
    for P in faces_pts:
        k = len(P)
        pts = [np.zeros(2)]
        for j in range(k):
            a = tuple(np.round(P[j], 6))
            b = tuple(np.round(P[(j + 1) % k], 6))
            key = tuple(sorted([a, b]))
            e = eid_of[key]
            d = P[(j + 1) % k] - P[j]
            u = d / np.linalg.norm(d)
            pts.append(pts[-1] + x[e] * u)
        assert np.allclose(pts[k], pts[0], atol=1e-4), "face not closed"
        local.append(np.array(pts[:k]))
    # face adjacency (shared edge -> (fi, idx_j, fj, idx_j'))
    e2f = defaultdict(list)
    for fi, P in enumerate(faces_pts):
        k = len(P)
        for j in range(k):
            a = tuple(np.round(P[j], 6))
            b = tuple(np.round(P[(j + 1) % k], 6))
            key = tuple(sorted([a, b]))
            e2f[key].append((fi, j))
    adj = defaultdict(list)
    for key, lst in e2f.items():
        if len(lst) == 2:
            (fi, ji), (fj, jj) = lst
            adj[fi].append((ji, fj, jj))
            adj[fj].append((jj, fi, ji))
    # BFS gluing: root = 0
    placed = {0: local[0]}
    order = deque([0])
    while order:
        fi = order.popleft()
        for (ji, fj, jj) in adj[fi]:
            if fj in placed:
                continue
            # both ends of the shared edge as placed in fi
            pa = placed[fi][ji]
            pb = placed[fi][(ji + 1) % len(faces_pts[fi])]
            v0, v1 = local[fj][jj], local[fj][(jj + 1) % len(faces_pts[fj])]
            # adjacent faces traverse the shared edge in opposite directions:
            # B's v0 end <-> A's pb end
            e_loc, e_glob = v1 - v0, pa - pb
            nn = float(np.dot(e_loc, e_loc))
            cosT = float(np.dot(e_loc, e_glob)) / nn
            sinT = float((e_loc[0] * e_glob[1] - e_loc[1] * e_glob[0]) / nn)
            R = np.array([[cosT, -sinT], [sinT, cosT]])
            placed[fj] = (local[fj] - v0) @ R.T + pb
            order.append(fj)
    assert len(placed) == len(faces_pts), f"unplaced faces {len(placed)}"
    return [placed[fi] for fi in range(len(faces_pts))]


def face_area(P):
    x, y = P[:, 0], P[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


# ---------------- ideal face lists of the two networks ----------------
def faces_31212(ncell=3):
    import step3_dp_strips as s3
    t = s3.build_31212_rect(ncell, ncell)
    out = []
    for f in t.faces:
        P = np.asarray(f[1], dtype=float)
        # canonical CCW + dedupe adjacent repeated points
        if face_area(P) > 0 and np.cross(P[1] - P[0], P[2] - P[0]) < 0:
            P = P[::-1]
        out.append((P, f[2]))
    return out


def faces_4612(n=3):
    s = 1.0
    R12 = s / (2 * math.sin(math.pi / 12))
    ap12 = s / (2 * math.tan(math.pi / 12))
    d0 = 2 * ap12 + s
    a1 = np.array([d0, 0.0])
    a2 = np.array([d0 / 2, d0 * math.sqrt(3) / 2])
    cells = [(m, n_) for m in range(n) for n_ in range(n)]
    centers = {c: c[0] * a1 + c[1] * a2 for c in cells}

    def reg(cx, cy, R, k, phi0):
        return np.array([[cx + R * math.cos(phi0 + 2 * math.pi * t / k),
                          cy + R * math.sin(phi0 + 2 * math.pi * t / k)]
                         for t in range(k)])

    out = []
    for c in cells:
        cx, cy = centers[c]
        out.append((reg(cx, cy, R12, 12, math.radians(15)), 12))
    sq_done = set()
    for m in range(n):
        for n_ in range(n):
            for d in [(1, 0), (0, 1), (-1, 1)]:
                p, q = (m, n_), (m + d[0], n_ + d[1])
                if p not in centers or q not in centers:
                    continue
                key = tuple(sorted([p, q]))
                if key in sq_done:
                    continue
                sq_done.add(key)
                mid = (centers[p] + centers[q]) / 2
                ang = math.atan2(centers[q][1] - centers[p][1],
                                 centers[q][0] - centers[p][0])
                out.append((reg(mid[0], mid[1], s / math.sqrt(2), 4,
                                ang - math.radians(45)), 4))
    for m in range(n):
        for n_ in range(n):
            for trio in ([(m, n_), (m + 1, n_), (m, n_ + 1)],
                         [(m + 1, n_), (m, n_ + 1), (m + 1, n_ + 1)]):
                if not all(t in centers for t in trio):
                    continue
                cen = np.mean([centers[t] for t in trio], axis=0)
                th = math.degrees(math.atan2(centers[trio[0]][1] - cen[1],
                                             centers[trio[0]][0] - cen[0]))
                out.append((reg(cen[0], cen[1], s, 6,
                                math.radians((th - 30.0) % 60.0)), 6))
    return out


# ---------------- rendering ----------------
def render(faces_pts, types, target_type, tag, seed=7, scale=1):
    eid_of, edges, rows = build_edge_system([P for P, _ in faces_pts])
    basis = nullspace_basis(rows, len(edges))
    print(f"[{tag}] edges={len(edges)} kernel dim={len(basis)}")
    # face-adjacency pairs (shared ideal edge): used for scoring
    e2f = defaultdict(list)
    for fi, (P_, _) in enumerate(faces_pts):
        k = len(P_)
        for j in range(k):
            a = tuple(np.round(P_[j], 6))
            b = tuple(np.round(P_[(j + 1) % k], 6))
            e2f[tuple(sorted([a, b]))].append(fi)
    adj_pairs = [(fs[0], fs[1]) for fs in e2f.values() if len(fs) == 2]

    def adj_roughness(areas):
        return float(np.mean([abs(math.log(areas[i] / areas[j]))
                              for i, j in adj_pairs
                              if areas[i] > 0 and areas[j] > 0]))

    K = np.array([[float(v) for v in b] for b in basis if max(abs(v) for v in b) <= 3])
    rng = np.random.default_rng(seed)
    best, best_score = None, -1
    for _ in range(600):
        x = rand_field(K, rng, sig=0.22)
        if x is None or x.min() <= 0.10:
            continue
        x = x / x.mean() * 1.2
        if x.min() < 0.15 or x.max() > 6.0:
            continue
        mf = metric_faces([P for P, _ in faces_pts], eid_of, edges, x)
        areas = np.array([face_area(P) for P in mf])
        tsel = [i for i, t in enumerate(types) if t == target_type]
        a = areas[tsel]
        # main term = adjacency roughness (per-cell randomness); moderate
        # area dispersion; penalize bimodal large contrast
        contrast = math.log(x.max() / x.min())
        score = (2.0 * adj_roughness(areas) + 0.6 * (a.std() / a.mean())
                 - 2.0 * max(0.0, contrast - 1.6))
        if score > best_score:
            best_score, best = score, (x, mf)
    if best is None:
        # fallback: one more round with a wider value band
        for _ in range(2000):
            x0 = rand_field(K, rng, sig=0.30)
            if x0 is None or x0.min() <= 0.05:
                continue
            x = x0 / x0.mean() * 1.2
            if x.min() < 0.06 or x.max() > 12.0:
                continue
            mf = metric_faces([P for P, _ in faces_pts], eid_of, edges, x)
            areas = np.array([face_area(P) for P in mf])
            tsel = [i for i, t in enumerate(types) if t == target_type]
            a = areas[tsel]
            contrast = math.log(x.max() / x.min())
            score = (2.0 * adj_roughness(areas) + 0.6 * (a.std() / a.mean())
                     - 2.0 * max(0.0, contrast - 1.6))
            if score > best_score:
                best_score, best = score, (x, mf)
    assert best is not None, "no valid solution in constraints band"
    x, mf = best
    print(f"[{tag}] score {best_score:.3f}; "
          f"edge range [{x.min():.2f},{x.max():.2f}]", flush=True)
    areas = np.array([face_area(P) for P in mf])
    tsel = [i for i, t in enumerate(types) if t == target_type]
    a = areas[tsel]
    lo, hi = a.min(), a.max()
    cmap = plt.get_cmap('viridis')
    fig, ax = plt.subplots(figsize=(3.4, 3.4))
    # uniform line attributes: all edges thick black; calibration = visible
    # linewidth after scaling the asset to the layout panel
    # (1.831 in @300 dpi, x0.511), matching the 3.4.6.4 asset (core 6 px @ 641)
    LW = 2.2
    for i, P in enumerate(mf):
        if types[i] == target_type:
            cval = (face_area(P) - lo) / max(hi - lo, 1e-9)
            ax.add_patch(Polygon(P, closed=True, facecolor=cmap(cval),
                                 edgecolor='black', lw=LW, zorder=3))
        else:
            ax.add_patch(Polygon(P, closed=True, facecolor='#eeeeee',
                                 edgecolor='black', lw=LW, zorder=2))
    allp = np.vstack(mf)
    cx = 0.5 * (allp[:, 0].min() + allp[:, 0].max())
    cy = 0.5 * (allp[:, 1].min() + allp[:, 1].max())
    side = 0.62 * min(np.ptp(allp[:, 0]), np.ptp(allp[:, 1]))   # central square crop window
    ax.set_xlim(cx - side / 2, cx + side / 2)
    ax.set_ylim(cy - side / 2, cy + side / 2)
    ax.set_aspect('equal')
    ax.axis('off')
    fig.tight_layout(pad=0.05)
    p = OUT / f"{tag}_metric.png"
    fig.savefig(p, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print("saved", p)


if __name__ == "__main__":
    # 3.12.12: triangle rows are area-locked => target class = dodecagon
    f31212 = faces_31212(6)
    types31212 = [t for _, t in f31212]
    print("3.12.12 face census:", {t: types31212.count(t) for t in set(types31212)})
    render(f31212, types31212, 12, "31212", seed=11)
    # 4.6.12: pick the face type with the largest relative area variance
    # (default dodecagon, adjustable)
    f4612 = faces_4612(6)
    types4612 = [t for _, t in f4612]
    print("4.6.12 face census:", {t: types4612.count(t) for t in set(types4612)})
    render(f4612, types4612, 12, "4612", seed=23)
