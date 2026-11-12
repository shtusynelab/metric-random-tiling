# -*- coding: utf-8 -*-
"""
kagome_real_demo.py -- demo of random solutions with real (R) metric on the
kagome lattice + library dependency check.

Comparison with 4.8.8:
  - 4.8.8: the constraint matrix is rectangular (underdetermined) with an
    extensive R dimension F+2; "alive or dead" is carried by alphabet
    arithmetic (Z: rank-deficient rows die / Lambda: rank 2 lives /
    R: lives).
  - kagome: equilateral binding => the constraints are exactly square,
    2F x 2F (critical); the zero set of det Mhat is a 1-dimensional zero
    locus in k space (measure zero) => the grid hits only O(L) momenta
    => dim ker = Lq+Lr+gcd-1, sub-extensive -- R and Z are BOTH
    sub-extensive => infiniteness of the tile library is irrelevant; the
    death cause is geometric (spectral zeros), not the alphabet.
  - Kernel = three families of strip modes (rows/columns/anti-diagonals)
    => random R solutions = interwoven strip modulations, in visual
    contrast with the face-wise independent chaos of 4.8.8.

Outputs:
  z4_kagome/KAG_real_metric_random_solution_demo.png (three panels:
    a equilateral reference / b,c random R solutions)
  _temp/kagome_dim_library.csv (dimension formula check + finite-library
    counts)
"""
import os
import sys
import math
import time
from itertools import product

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "z4_kagome"))
import kagome_lib as K  # noqa: E402


# ------------------------------------------------------- 1. torus dimension table

def dim_table():
    rows_csv = ["Lq,Lr,F,dim_exact,dim_formula,omega_k2,omega_k3,"
                "k2_over_2dim,k3_over_3dim"]
    print(f"{'Lq,Lr':>7} {'dim':>4} {'formula':>8} {'Om(k=2)':>10} "
          f"{'Om(k=3)':>10} {'k2^?':>6}")
    for (Lq, Lr) in [(2, 2), (3, 2), (2, 3), (4, 2), (3, 3), (4, 3),
                     (4, 4), (5, 3)]:
        hc = K.KagomeTorus(Lq, Lr)
        g = math.gcd(Lq, Lr)
        dim = K.nullity_exact(hc.M_reduced)
        formula = Lq + Lr + g - 1
        assert dim == formula, f"dim formula failed at {Lq},{Lr}: {dim}"
        om2 = K.kernel_count_int(hc.M_reduced, (0, 1))
        om3 = K.kernel_count_int(hc.M_reduced, (0, 1, 2))
        # all-ones translation invariance => the {1,2} count must equal the
        # {0,1} count (value-independence spot check)
        om2b = K.kernel_count_int(hc.M_reduced, (1, 2))
        assert om2 == om2b, "translation invariance failed"
        print(f"{Lq},{Lr:>4} {dim:>4} {formula:>8} {om2:>10} {om3:>10} "
              f"{math.log(max(om2, 2)) / math.log(2):>6.2f}", flush=True)
        rows_csv.append(f"{Lq},{Lr},{hc.F},{dim},{formula},{om2},{om3},"
                        f"{math.log(max(om2, 2)) / math.log(2):.3f},"
                        f"{math.log(max(om3, 3)) / math.log(3):.3f}")
    out = os.path.join(ROOT, "_temp", "kagome_dim_library.csv")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(rows_csv) + "\n")
    print("csv ->", out, flush=True)


# ------------------------------------------------------- 2. patch R sampling

def sample_t_field(W, L, seed, s_lo=0.50):
    """Random R kernel solution of the open-boundary patch (honeycomb-vertex
    t field) with min t ~ s_lo."""
    mat = K.patch_matrix(W, L)
    M = np.array(mat, dtype=float)
    U_, S_, Vh = np.linalg.svd(M, full_matrices=True)
    nrank = np.linalg.matrix_rank(M)
    N = Vh[nrank:].T
    dim = N.shape[1]
    rng = np.random.default_rng(seed)
    c = rng.standard_normal(dim)
    c /= np.linalg.norm(c)

    def sol(eps):
        return 1.0 + eps * (N @ c)

    lo, hi = 0.0, 1.0
    while sol(hi).min() > s_lo and hi < 1e3:
        hi *= 2
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if sol(mid).min() > s_lo:
            lo = mid
        else:
            hi = mid
    t = sol(lo * 0.95)
    t = t / t.mean()
    assert np.allclose(M @ (t * t.mean()), 0, atol=1e-8)
    return t, dim


# ------------------------------------------------------- 3. drawing

def draw():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable

    W = L = 8
    hex_vids, hex_pos, rows, n_vars = K.build_patch(W, L)
    M = np.array(K.patch_matrix(W, L), dtype=float)
    dim_patch = M.shape[1] - np.linalg.matrix_rank(M)

    # vertex coordinates (basis coordinates for adjacency, real space for
    # drawing: real = a*(1,0) + b*(1/2, sqrt(3)/2))
    # + adjacency (basis-coordinate difference in U = bond)
    v2base = {}
    v2real = {}
    for hi in range(len(hex_vids)):
        for k, v in enumerate(hex_vids[hi]):
            a, b = hex_pos[hi][k]
            v2base[v] = (a, b)
            v2real[v] = (a + b * 0.5, b * math.sqrt(3) / 2)
    Uset = set(K.U)
    nbrs = {v: [] for v in range(n_vars)}
    for a in range(n_vars):
        for b in range(a + 1, n_vars):
            d = (v2base[b][0] - v2base[a][0], v2base[b][1] - v2base[a][1])
            if d in Uset:
                nbrs[a].append(b)
                nbrs[b].append(a)
    interior = [v for v in range(n_vars) if len(nbrs[v]) == 3]
    print(f"patch {W}x{L}: n_vars={n_vars} (interior {len(interior)}), "
          f"dim_kernel={dim_patch}", flush=True)

    # ---- BFS embedding of the kagome graph: vertices = bond midpoints,
    # edges = bond pairs at honeycomb vertices, directions fixed (the
    # 30-deg family unit vectors of the reference geometry), lengths = t_hv.
    # Closure constraints (triangles + hexagons) => placing points along a
    # spanning tree of the simply connected patch gives the global embedding.
    mid_ref = {}
    for a in interior:
        for b in nbrs[a]:
            key = (min(a, b), max(a, b))
            if key not in mid_ref:
                mid_ref[key] = ((v2real[a][0] + v2real[b][0]) / 2,
                                (v2real[a][1] + v2real[b][1]) / 2)
    bonds = sorted(mid_ref)
    bid = {bd: i for i, bd in enumerate(bonds)}
    kedges = []          # (hv, i, j, ux, uy, Lref)
    for hv in interior:
        bs = sorted(bid[(min(hv, b), max(hv, b))] for b in nbrs[hv])
        for i, j in ((0, 1), (1, 2), (0, 2)):
            mi, mj = mid_ref[bonds[bs[i]]], mid_ref[bonds[bs[j]]]
            dx, dy = mj[0] - mi[0], mj[1] - mi[1]
            Lref = math.hypot(dx, dy)
            kedges.append((hv, bs[i], bs[j], dx / Lref, dy / Lref, Lref))
    adj = {i: [] for i in range(len(bonds))}
    for hv, i, j, ux, uy, Lref in kedges:
        adj[i].append((j, hv, ux, uy))
        adj[j].append((i, hv, -ux, -uy))

    # reference-geometry normalization: in the equilateral reference the
    # kagome edge length = |m_j - m_i| = sqrt(3)/2 (honeycomb bond length 1),
    # while t=1 is defined as edge length 1 => rescale mid_ref globally,
    # otherwise the all-ones check differs by a linear scaling field
    s0 = kedges[0][5]
    mid_ref = {bd: (x / s0, y / s0) for bd, (x, y) in mid_ref.items()}

    def embed(t):
        """t field -> kagome vertex (bond midpoint) positions. Returns a
        dict bid -> (x,y)."""
        start = adj_start = next(i for i in range(len(bonds)) if adj[i])
        pos = {start: (0.0, 0.0)}
        stack = [start]
        while stack:
            cur = stack.pop()
            for nxt, hv, ux, uy in adj[cur]:
                if nxt in pos:
                    continue
                px, py = pos[cur]
                pos[nxt] = (px + t[hv] * ux, py + t[hv] * uy)
                stack.append(nxt)
        return pos

    # check: embedding of the all-ones solution = reference geometry
    # (differing by a constant vector)
    pos1 = embed(np.ones(n_vars))
    diffs = [(pos1[bid[bd]][0] - mid_ref[bd][0],
              pos1[bid[bd]][1] - mid_ref[bd][1]) for bd in bonds
             if bid[bd] in pos1]
    d0 = diffs[0]
    assert max(max(abs(dx - d0[0]), abs(dy - d0[1])) for dx, dy in diffs) \
        < 1e-9, "embed(all-ones) != reference geometry"

    panels = [("a", None), ("b", 20260927), ("c", 20260928)]
    data = []
    for tag, seed in panels:
        if seed is None:
            t, dim = np.ones(n_vars), dim_patch
        else:
            t, dim = sample_t_field(W, L, seed)
        data.append((tag, t))
    print(f"patch {W}x{L}: n_vars={n_vars}, dim_kernel={dim_patch}", flush=True)

    MM = 1 / 25.4
    fig = plt.figure(figsize=(183 * MM, 64 * MM))
    ax_w = 0.235
    xs = [0.035, 0.035 + (ax_w + 0.035), 0.035 + 2 * (ax_w + 0.035)]
    cmap = plt.get_cmap("viridis")
    norm = Normalize(vmin=0.5, vmax=1.5)

    for (tag, t), x0 in zip(data, xs):
        ax = fig.add_axes([x0, 0.10, ax_w, 0.82])
        ax.set_aspect("equal")
        ax.axis("off")
        pos = embed(t)
        segs, cols = [], []
        for hv, i, j, ux, uy, Lref in kedges:
            if i not in pos or j not in pos:
                continue
            segs.append([pos[i], pos[j]])
            cols.append(cmap(norm(t[hv] / t.mean())) if tag != "a"
                        else "#404040")
        ax.add_collection(LineCollection(segs, colors=cols, linewidths=1.4,
                                         capstyle="round", zorder=3))
        xs_ = [p[0] for p in pos.values()]
        ys_ = [p[1] for p in pos.values()]
        cut = 1.2
        ax.set_xlim(min(xs_) + cut, max(xs_) - cut)
        ax.set_ylim(min(ys_) + cut, max(ys_) - cut)
        ax.text(0.02, 0.985, tag, transform=ax.transAxes, fontsize=9,
                fontweight="bold", va="top", ha="left", zorder=5)

    cax = fig.add_axes([xs[-1] + ax_w + 0.055, 0.10, 0.012, 0.82])
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    cb.set_label("triangle edge length / mean", fontsize=7.5)
    cb.ax.tick_params(labelsize=7)

    for ext in ("png", "pdf"):
        out = os.path.join(ROOT, "z4_kagome",
                           f"KAG_real_metric_random_solution_demo.{ext}")
        fig.savefig(out, dpi=300)
        if ext == "png":
            print("fig ->", out, flush=True)
    plt.close(fig)


if __name__ == "__main__":
    t0 = time.time()
    dim_table()
    print(f"--- dim table done ({time.time()-t0:.1f}s) ---", flush=True)
    draw()
    print(f"--- all done ({time.time()-t0:.1f}s) ---", flush=True)
