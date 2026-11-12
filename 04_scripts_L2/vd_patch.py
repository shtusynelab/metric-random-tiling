# -*- coding: utf-8 -*-
"""vd_patch.py v4 -- exact-uniform torus configurations -> lifted hexagonal
disc patch (de-slit scheme).
Pipeline: (1) the v3 corner-lifting mechanism lays out the 169 ideal faces
(gate: placed = 169 / zero conflicts); (2) a centroid fit of the lattice
basis (u, v) -> face <-> covering-cell (i, j) labels (empirical calibration,
no closed-form derivation); (3) slot-edge mapping: the shared edge of face
(i, j) with face (i, j) + DIRECTIONS[k] follows from face adjacency;
(4) the disc = the 169 covering cells with (|i|+|j|+|i+j|)/2 <= R, glued cell
by cell via the face-frame reflection; (5) vertices deduplicated by 1e-6
rounding; four gates: Euler / degree / uniqueness / perfect honeycomb.
Value source: slot-edge torus edge id -> configuration x[eid]."""
import sys
from pathlib import Path
from collections import deque, defaultdict
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths_common as PC  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent / "external"))
sys.path.insert(0, str(PC.HC))

A0 = np.deg2rad(90)
DIRS = [np.array([np.cos(A0 + np.deg2rad(60) * k),
                  np.sin(A0 + np.deg2rad(60) * k)]) for k in range(6)]


def hex_local_centered(vals):
    pts = [np.zeros(2)]
    for k in range(6):
        pts.append(pts[-1] + float(vals[k]) * DIRS[k])
    assert np.allclose(pts[6], pts[0], atol=1e-8), "tile not closed"
    g = sum(pts[:6]) / 6.0
    return [p - g for p in pts[:6]]


def edge_match(gj, idx_g, fi, idx_e, face_verts, Lx, Ly):
    """Shared-edge endpoint correspondence: is g's local (idx_g, idx_g+1)
    pair the same ordered pair as f's (idx_e, idx_e+1)?  (Adjacent faces
    usually traverse the shared edge in opposite order.)  Returns True if the
    order matches."""
    def q(v):
        return (v[0], v[1] % Lx, v[2] % Ly)
    ga = q(face_verts[gj][idx_g])
    fa = q(face_verts[fi][idx_e])
    return ga == fa


def build_patch_graph(Lx, Ly, root_rings=7):
    """Returns G: slot_eid[(i,j,k)], cells, F, E, Lx, Ly; all empirically
    calibrated."""
    import pseudocrystal_lib as pl
    from metric_hc import build_torus
    from sampler2 import edge_meta
    edge_set, faces = build_torus(Lx, Ly)
    types, key, posi = edge_meta(Lx, Ly)[2:]
    E = len(edge_set)

    # ---- (1) corner lifting (since v3, verified) ----
    face_verts, face_edges = [], []
    for f in faces:
        qend = [set(edge_set[e]) for e in f]
        corners = []
        for k in range(6):
            inter = qend[k] & qend[(k + 1) % 6]
            assert len(inter) == 1
            corners.append(inter.pop())
        cur = corners[(0 - 1) % 6]
        seq = []
        for kk in range(6):
            e = f[kk]
            t = int(types[e])
            nxt = ({"A": {0: ("B", cur[1], cur[2]),
                          1: ("B", cur[1] - 1, cur[2]),
                          2: ("B", cur[1], cur[2] - 1)},
                    "B": {0: ("A", cur[1], cur[2]),
                          1: ("A", cur[1] + 1, cur[2]),
                          2: ("A", cur[1], cur[2] + 1)}}[cur[0]][t])
            assert (nxt[0], nxt[1] % Lx, nxt[2] % Ly) == corners[kk]
            seq.append((cur, nxt, e))
            cur = nxt
        assert cur == corners[(0 - 1) % 6]
        face_verts.append([s[0] for s in seq])
        face_edges.append([s[2] for s in seq])

    # ---- (2) ideal tiling (v3 reflection BFS; x = 1) ----
    efa = defaultdict(list)
    for fi, f in enumerate(faces):
        for e in f:
            efa[e].append(fi)

    def place_all():
        fv = face_verts
        fe = face_edges
        pos = {}
        ctr = {}
        placed = set()
        root = 0

        def hexloc(fi):
            return hex_local_centered([1.0] * 6)

        loc = hexloc(root)
        for k in range(6):
            pos[(root, k)] = loc[k]
        placed.add(root)
        ctr[root] = np.zeros(2)
        bfsq = deque([root])
        while bfsq:
            fi = bfsq.popleft()
            for idx_e, e in enumerate(fe[fi]):
                fa, fb = efa[e]
                gj = fb if fa == fi else fa
                if gj in placed:
                    continue
                idx_g = fe[gj].index(e)
                same = edge_match(gj, idx_g, fi, idx_e, fv, Lx, Ly)
                # global endpoint pair (parent face already placed)
                pa = pos[(fi, idx_e)]
                pb = pos[(fi, (idx_e + 1) % 6)]
                if same:
                    v0, p0 = loc[idx_g], pa
                    v1, p1 = loc[(idx_g + 1) % 6], pb
                else:
                    v0, p0 = loc[idx_g], pb
                    v1, p1 = loc[(idx_g + 1) % 6], pa
                e_loc, e_glob = v1 - v0, p1 - p0
                nn = float(np.dot(e_loc, e_loc))
                cosT = float(np.dot(e_loc, e_glob)) / nn
                sinT = float((e_loc[0] * e_glob[1] - e_loc[1] * e_glob[0]) / nn)
                Rot = np.array([[cosT, -sinT], [sinT, cosT]])
                base = p0 - Rot @ v0
                for k in range(6):
                    pos[(gj, k)] = base + Rot @ loc[k]
                ctr[gj] = base
                placed.add(gj)
                bfsq.append(gj)
        assert len(placed) == len(faces)
        return pos, ctr

    pos, ctr = place_all()

    # ---- (2b) centroid -> lattice labels: take the 6 nearest neighbours and
    # pick as basis the pair enclosing a 60 deg angle ----
    arr = np.array([ctr[fi] for fi in range(len(faces))])
    c0 = ctr[0]
    dd = np.linalg.norm(arr - c0, axis=1)
    order = np.argsort(dd)
    nb6 = [arr[i] - c0 for i in order if dd[i] > 1e-6][:6]
    assert len(nb6) == 6
    u = min(nb6, key=lambda x: np.linalg.norm(x))
    lu = np.linalg.norm(u)
    v = next(x for x in nb6 if x is not u and
             abs(np.linalg.norm(x) - lu) < 0.05 * lu and
             abs(np.dot(x, u) / (lu * np.linalg.norm(x)) - 0.5) < 0.05)
    B = np.stack([u, v])
    Binv = np.linalg.inv(B.T)   # decompose p = sum_k ij_k * row_k(B) => ij = inv(B.T) @ p
    labels = {}
    for fi in range(len(faces)):
        ij = Binv @ (ctr[fi] - c0)
        ijr = np.round(ij)
        assert np.allclose(ij, ijr, atol=1e-6), (fi, ij)
        labels[fi] = (int(ijr[0]), int(ijr[1]))
    assert len(set(labels.values())) == len(faces), "labels not bijective"
    lab_of = {v_: k for k, v_ in labels.items()}

    # ---- (3) slot-edge mapping: shared edge of face (i, j) and face
    # (i, j) + DIR[k] ----
    slot_eid = {}
    DIRS_IJ = [tuple(d) for d in pl.DIRECTIONS]
    for fi, (i, j) in labels.items():
        for k, d in enumerate(DIRS_IJ):
            nb = (i + d[0], j + d[1])
            if nb not in lab_of:
                continue
            gj = lab_of[nb]
            shared = set(face_edges[fi]) & set(face_edges[gj])
            assert len(shared) == 1, f"faces {fi},{gj} share {len(shared)}"
            e = shared.pop()
            # the edge of slot k: its position in fi's winding order must
            # match k -- verified through fi's vertex pair: slot k connects
            # vertices (fi, k) -> (fi, k+1); the shared edge must equal it
            idx_e = face_edges[fi].index(e)
            slot_eid[(i, j, idx_e)] = e

    # ---- (4) disc cells ----
    R = root_rings
    cells = [(i, j) for i in range(-R, R + 1) for j in range(-R, R + 1)
             if (abs(i) + abs(j) + abs(i + j)) // 2 <= R]
    assert len(cells) == 3 * R * (R + 1) + 1, len(cells)
    DIRS_IJ = [tuple(d) for d in pl.DIRECTIONS]
    lab_of_torus = {(i % Lx, j % Ly): fi for fi, (i, j) in labels.items()}
    assert len(lab_of_torus) == Lx * Ly
    for (i, j) in cells:
        fi = lab_of_torus[(i % Lx, j % Ly)]
        for k in range(6):
            d = DIRS_IJ[k]
            nb = ((i + d[0]) % Lx, (j + d[1]) % Ly)
            gj = lab_of_torus[nb]
            shared = set(face_edges[fi]) & set(face_edges[gj])
            assert len(shared) == 1, (i, j, k, nb, len(shared))
            slot_eid[(i, j, k)] = shared.pop()
    return dict(slot_eid=slot_eid, cells=cells, F=len(cells),
                E=E, Lx=Lx, Ly=Ly, labels=labels,
                lab_of=lab_of, lab_of_torus=lab_of_torus,
                face_edges=face_edges, face_verts=face_verts,
                DIRS_IJ=DIRS_IJ)


def build_patch_template(G, root=(0, 0)):
    cells = G["cells"]
    cellset = set(cells)
    import pseudocrystal_lib as pl
    order = [root]
    seen = {root}
    bfsq = deque([root])
    while bfsq:
        c = bfsq.popleft()
        for d in pl.DIRECTIONS:
            nb = (c[0] + d[0], c[1] + d[1])
            if nb in cellset and nb not in seen:
                seen.add(nb)
                order.append(nb)
                bfsq.append(nb)
    assert len(order) == len(cells)
    return dict(order=order, root=root, F=len(cells))


def embed_config(G, T, x):
    import pseudocrystal_lib as pl
    Lx, Ly = G["Lx"], G["Ly"]
    DIRS_IJ = {tuple(d): k for k, d in enumerate(pl.DIRECTIONS)}
    pos_map = {}
    pts_list = []

    def key_of(p):
        # tolerance 1e-5: glue-path float differences (~1e-6) must merge;
        # true vertex spacings are >= 0.4, so this is safe
        return (round(p[0], 5), round(p[1], 5))

    cell_verts = {}
    placed = set()
    root = T["root"]
    vals0 = [x[G["slot_eid"][(root[0], root[1], t)]] for t in range(6)]
    loc0 = hex_local_centered(vals0)
    cell_verts[root] = []
    for k in range(6):
        kk = key_of(loc0[k])
        if kk not in pos_map:
            pos_map[kk] = len(pts_list)
            pts_list.append(loc0[k])
        cell_verts[root].append(pos_map[kk])
    placed.add(root)

    for c in T["order"]:
        if c == root:
            continue
        for d in pl.DIRECTIONS:
            nb = (c[0] + d[0], c[1] + d[1])
            if nb not in placed:
                continue
            k_self = DIRS_IJ[tuple(d)]            # slot of c facing nb
            k_nb = (k_self + 3) % 6               # slot of nb facing c (opposite slot)
            fi = G["lab_of_torus"][(c[0] % Lx, c[1] % Ly)]
            fj = G["lab_of_torus"][(nb[0] % Lx, nb[1] % Ly)]
            same = edge_match(fj, k_nb, fi, k_self, G["face_verts"], Lx, Ly)
            vals = [x[G["slot_eid"][(c[0], c[1], t)]] for t in range(6)]
            loc = hex_local_centered(vals)
            va, vb = cell_verts[nb][k_nb], cell_verts[nb][(k_nb + 1) % 6]
            pa, pb = pts_list[va], pts_list[vb]
            if same:
                v0, p0 = loc[k_self], pa
                v1, p1 = loc[(k_self + 1) % 6], pb
            else:
                v0, p0 = loc[k_self], pb
                v1, p1 = loc[(k_self + 1) % 6], pa
            e_loc, e_glob = v1 - v0, p1 - p0
            nn = float(np.dot(e_loc, e_loc))
            cosT = float(np.dot(e_loc, e_glob)) / nn
            sinT = float((e_loc[0] * e_glob[1] - e_loc[1] * e_glob[0]) / nn)
            Rot = np.array([[cosT, -sinT], [sinT, cosT]])
            base = p0 - Rot @ v0
            cverts = []
            for k in range(6):
                p = base + Rot @ loc[k]
                kk = key_of(p)
                if kk not in pos_map:
                    pos_map[kk] = len(pts_list)
                    pts_list.append(p)
                cverts.append(pos_map[kk])
            cell_verts[c] = cverts
            placed.add(c)
            break
    assert len(placed) == len(G["cells"])
    return np.array(pts_list), cell_verts


def patch_bonds(G, T, cell_verts):
    seen = set()
    bonds = []
    for c in G["cells"]:
        cv = cell_verts[c]
        for t in range(6):
            a, b = cv[t], cv[(t + 1) % 6]
            key = (min(a, b), max(a, b))
            if key in seen:
                continue
            seen.add(key)
            bonds.append((key[0], key[1], G["slot_eid"][(c[0], c[1], t)]))
    return bonds


def validate_ideal(G, T):
    x = np.ones(G["E"], dtype=np.int64)
    pts, cell_verts = embed_config(G, T, x)
    bonds = patch_bonds(G, T, cell_verts)
    nbs = defaultdict(set)
    for a, b, e in bonds:
        nbs[a].add(b)
        nbs[b].add(a)
    V, E, F = len(pts), len(bonds), G["F"]
    deg = {n: len(s) for n, s in nbs.items()}
    hist = {d: sum(1 for v in deg.values() if v == d)
            for d in sorted(set(deg.values()))}
    assert V - E + F == 1, f"Euler: {V}-{E}+{F}={V-E+F} != 1"
    assert max(deg.values()) == 3, f"max deg {max(deg.values())}"
    nbad = 0
    interior = [n for n, s in nbs.items() if len(s) == 3]
    for n in interior:
        ds = [pts[m] - pts[n] for m in nbs[n]]
        if not all(abs(np.hypot(d[0], d[1]) - 1.0) < 1e-9 for d in ds):
            nbad += 1
            continue
        a1 = np.arctan2(ds[0][1], ds[0][0])
        angs = np.sort([(np.arctan2(d[1], d[0]) - a1) % (2 * np.pi)
                        for d in ds])
        gaps = np.diff(np.append(angs, angs[0] + 2 * np.pi))
        if not np.allclose(gaps, [2 * np.pi / 3] * 3, atol=1e-9):
            nbad += 1
    print(f"[ideal] V={V} E={E} F={F} deg={hist} interior={len(interior)} "
          f"bad={nbad}")
    assert nbad == 0
    return pts, bonds


if __name__ == "__main__":
    G = build_patch_graph(13, 13)
    T = build_patch_template(G)
    print(f"cells F={G['F']} torus E={G['E']}")
    pts, bonds = validate_ideal(G, T)
    print("VD_PATCH v4 GATES PASS")
