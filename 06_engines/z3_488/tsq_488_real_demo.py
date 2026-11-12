# -*- coding: utf-8 -*-
"""
tsq_488_real_demo.py - random-solution demo for the 4.8.8 real metric (R)
plus a three-alphabet dimension comparison.

Question: does the extensive R-space dimension of 4.8.8 allow "disordered
tilings", and which discrete alphabet retains a positive-entropy candidate?

Contents:
  1. Solution-space dimensions of three alphabets (open-boundary patch,
     size series):
     - Z (integer alphabet): exact Fraction rank of rows_split -> the
       dimension is sub-extensive (row splitting rigidifies);
     - R (continuous alphabet): float rank of rows_float -> extensive
       (of the F+2 type);
     - Lambda = Z + (sqrt(2)/2)Z: two-variable rows (m_e, n_e), mod-p rank
       -> rank extensivity restored.
       Derivation: substitute L_e = m_e + n_e*sqrt(2)/2 into
       sum cos(theta) * L = 0; 1 and sqrt(2)/2 are Q-independent, so each
       projection row yields two coupled integer rows:
       sum (a m + b n) = 0 and sum (2b m + a n) = 0
       (a,b = the rational/sqrt(2) cos/sin coefficients of DIR8; unlike the
       Z row split, the m and n blocks are coupled and the axis edges no
       longer decouple from the 45-deg edges, breaking the opposite-side
       equality propagation).
  2. Random R-kernel solutions x 2 seeds -> BFS placement -> three-panel
     figure: (a) equal-edge reference 4.8.8, (b,c) random R solutions
     (edges colored by length, angles preserved).

Outputs:
  TSQ_real_metric_random_solution_demo.png in this directory (300 dpi,
  English labels, no panel titles)
  tsq_488_dim_compare.csv (beside this script)
"""
import os
import sys
import math
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "z3_488"))
import tsq_lib as T  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

P1, P2 = 10**9 + 7, 10**9 + 9


# --------------------------------------------------------------- rank utilities

def rank_modp(rows, ncol, p):
    """Rank of an integer-row matrix via Gaussian elimination mod p (row
    coefficients are small integers, int64-safe)."""
    M = (np.array(rows, dtype=np.int64)) % p
    m = M.shape[0]
    r = 0
    for c in range(ncol):
        if r == m:
            break
        nz = np.nonzero(M[r:, c])[0]
        if len(nz) == 0:
            continue
        pr = r + nz[0]
        if pr != r:
            M[[r, pr]] = M[[pr, r]]
        inv = pow(int(M[r, c]), p - 2, p)
        M[r] = (M[r] * inv) % p
        col = M[:, c].copy()
        col[r] = 0
        nzr = np.nonzero(col)[0]
        if len(nzr):
            M[nzr] = (M[nzr] - np.outer(col[nzr], M[r])) % p
        r += 1
    return r


def frac_rank(rows, ncol):
    """Exact Fraction rank (small systems)."""
    from fractions import Fraction
    mat = [[Fraction(x) for x in r] for r in rows]
    mrow = len(mat)
    r = 0
    for c in range(ncol):
        piv = None
        for rr in range(r, mrow):
            if mat[rr][c] != 0:
                piv = rr
                break
        if piv is None:
            continue
        mat[r], mat[piv] = mat[piv], mat[r]
        pv = mat[r][c]
        mat[r] = [x / pv for x in mat[r]]
        for rr in range(mrow):
            if rr != r and mat[rr][c]:
                f = mat[rr][c]
                mat[rr] = [x - f * y for x, y in zip(mat[rr], mat[r])]
        r += 1
    return r


def lambda_rows(faces, nvars):
    """Lambda = Z + (sqrt(2)/2)Z two-variable rows (m block | n block):
    row1 = rx (+) rx2 (sum a m + sum b n = 0), row2 = 2*rx2 (+) rx
    (sum 2b m + sum a n = 0); same for y. All-zero rows dropped (same rule
    as the tsq_lib row split)."""
    out = []
    for size, sides in faces:
        rx = [0] * nvars; rx2 = [0] * nvars
        ry = [0] * nvars; ry2 = [0] * nvars
        for e, d in sides:
            a, b, c, dd = T.DIR8[d]
            rx[e] += a; rx2[e] += b; ry[e] += c; ry2[e] += dd
        for X, X2 in ((rx, rx2), (ry, ry2)):
            r1 = list(X) + list(X2)
            r2 = [2 * x for x in X2] + list(X)
            for r in (r1, r2):
                if any(r):
                    out.append(r)
    return out


# --------------------------------------------------------------- 1. dimension table

def dim_table():
    rows_csv = ["W,L,F,F8,E,dim_Z,dim_R,dim_Lam,dim_Lam_over_E,dim_R_over_F8"]
    print(f"{'W,L':>8} {'E':>5} {'dim_Z':>7} {'dim_R':>7} {'dim_Lam':>8} "
          f"{'dLam/2E':>8} {'dR/F8':>7}")
    for (W, L) in [(2, 2), (3, 3), (4, 4), (5, 5), (6, 6)]:
        faces, E = T.tsq_patch(W, L)
        rows_i, rows_f = T.tsq_patch_rows(faces, E)
        F8 = sum(1 for s, _ in faces if s == 8)
        # Z: exact rank of the split rows (Fraction for small systems; the
        # min of two mod-p ranks with distinct primes for large ones)
        if E <= 150:
            rz = frac_rank(rows_i, E)
        else:
            rz = min(rank_modp(rows_i, E, P1), rank_modp(rows_i, E, P2))
        dim_Z = E - rz
        # R: float rank
        Mf = np.array(rows_f)
        rank_r = np.linalg.matrix_rank(Mf)
        assert np.allclose(Mf @ np.ones(E), 0, atol=1e-9), "all-ones fails"
        dim_R = E - rank_r
        # Lambda = Z + (sqrt(2)/2)Z: two-variable rows, 2E variables
        rl = lambda_rows(faces, E)
        rl_small = frac_rank(rl, 2 * E) if 2 * E <= 300 else None
        rk = min(rank_modp(rl, 2 * E, P1), rank_modp(rl, 2 * E, P2))
        if rl_small is not None:
            assert rl_small == rk, f"mod-p vs Fraction mismatch {W},{L}"
        dim_L = 2 * E - rk
        # the all-ones solution (m=1, n=0) must satisfy the Lambda rows
        one = np.array([1] * E + [0] * E, dtype=np.int64)
        assert all(sum(r[k] for k in np.nonzero(one)[0]) == 0 for r in rl), \
            "all-ones not a Lambda solution"
        print(f"{W},{L:>5} {E:>5} {dim_Z:>7} {dim_R:>7} {dim_L:>8} "
              f"{dim_L / (2 * E):>8.3f} {dim_R / F8:>7.3f}")
        rows_csv.append(f"{W},{L},{len(faces)},{F8},{E},"
                        f"{dim_Z},{dim_R},{dim_L},{dim_L / (2 * E):.4f},"
                        f"{dim_R / F8:.4f}")
    out = os.path.join(HERE, "tsq_488_dim_compare.csv")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(rows_csv) + "\n")
    print("csv ->", out)


# --------------------------------------------------------------- 2. geometric embedding

def build_geometry(W, L):
    """Replicate the eid order of tsq_patch to build the edge endpoint table
    [(e, vA, vB, d_ab)] together with the bearing truncation points."""
    edges = {}

    def eid(key):
        if key not in edges:
            edges[key] = len(edges)
        return edges[key]

    for i in range(W):
        for j in range(L + 1):
            eid(("ax", 0, i, j))
    for i in range(W + 1):
        for j in range(L):
            eid(("ax", 1, i, j))
    cuts = {}
    for i in range(W + 1):
        for j in range(L + 1):
            for q in range(4):
                dx, dy = T.QUAD_CELL[q]
                if 0 <= i + dx < W and 0 <= j + dy < L:
                    cuts[(i, j, q)] = eid(("cut", i, j, q))

    def node_pt(i, j, k):
        """Truncation-point vid of node (i,j) at bearing k in {0,2,4,6}."""
        if k == 0:
            assert i < W
            return ("H", i, j, 0)
        if k == 2:
            assert j < L
            return ("V", i, j, 0)
        if k == 4:
            assert i > 0
            return ("H", i - 1, j, 1)
        assert j > 0
        return ("V", i, j - 1, 1)

    epi = {}  # edge id -> (vA, vB, d_ab)
    for i in range(W):
        for j in range(L + 1):
            epi[edges[("ax", 0, i, j)]] = (("H", i, j, 0), ("H", i, j, 1), 0)
    for i in range(W + 1):
        for j in range(L):
            epi[edges[("ax", 1, i, j)]] = (("V", i, j, 0), ("V", i, j, 1), 2)
    for (i, j, q), e in cuts.items():
        k = 2 * q
        epi[e] = (node_pt(i, j, k), node_pt(i, j, (k + 2) % 8), (k + 3) % 8)
    return epi


def bfs_embed(epi, lens):
    """Given edge lengths, place vertices along a BFS tree (the patch is
    simply connected, so tree placement is a global embedding)."""
    adj = {}
    for e, (a, b, d) in epi.items():
        adj.setdefault(a, []).append((e, a, b, d))
        adj.setdefault(b, []).append((e, b, a, (d + 4) % 8))
    # real direction vectors
    U = []
    for a, b, c, dd in T.DIR8:
        U.append(((a + b * math.sqrt(2)) / 2, (c + dd * math.sqrt(2)) / 2))
    start = ("H", 0, 0, 0)
    pos = {start: (0.0, 0.0)}
    stack = [start]
    while stack:
        v = stack.pop()
        for e, va, vb, d in adj[v]:
            if vb in pos:
                continue
            ua, ub = pos[va]
            ux, uy = U[d]
            pos[vb] = (ua + lens[e] * ux, ub + lens[e] * uy)
            stack.append(vb)
    assert len(pos) == len({v for a, b, _ in epi.values() for v in (a, b)}), \
        "BFS missed vertices"
    return pos


def face_polygons(faces, epi, pos):
    """CCW vertex sequence of each face."""
    polys = []
    for size, sides in faces:
        pts = []
        for t, (e, d) in enumerate(sides):
            a, b, d_ab = epi[e]
            pa, pb = pos[a], pos[b]
            if d_ab != d:
                pa, pb = pb, pa
            if t == 0:
                pts.append(pa)
            pts.append(pb)
        assert len(pts) == size + 1
        assert max(abs(pts[-1][0] - pts[0][0]),
                   abs(pts[-1][1] - pts[0][1])) < 1e-6, "face not closed"
        polys.append((size, pts))
    return polys


# --------------------------------------------------------------- 3. sampling + figure

def sample_solution(faces, E, seed, s_lo=0.50):
    """All-ones solution plus a random kernel direction; bisect the step so
    that min(s) ~= s_lo."""
    _, rows_f = T.tsq_patch_rows(faces, E)
    M = np.array(rows_f)
    U_, S_, Vh = np.linalg.svd(M, full_matrices=True)
    nrank = np.linalg.matrix_rank(M)
    N = Vh[nrank:].T                      # E x dim_R
    assert N.shape[1] > 0, "no continuous kernel"
    rng = np.random.default_rng(seed)
    c = rng.standard_normal(N.shape[1])
    c /= np.linalg.norm(c)

    def sol(eps):
        return 1.0 + eps * (N @ c)

    lo, hi = 0.0, 1.0
    while (sol(hi).min() > s_lo) and hi < 1e3:
        hi *= 2
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if sol(mid).min() > s_lo:
            lo = mid
        else:
            hi = mid
    s = sol(lo * 0.95)
    s = s / s.mean()
    assert np.allclose(M @ (s * s.mean()), 0, atol=1e-8), "solution invalid"
    return s


def draw():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
    import matplotlib.patches as mpatches

    W = L = 8
    faces, E = T.tsq_patch(W, L)
    epi = build_geometry(W, L)
    F8 = sum(1 for sz, _ in faces if sz == 8)
    print(f"patch {W}x{L}: E={E}, faces={len(faces)} (oct {F8}, sq {len(faces)-F8})")

    samples = [("a", None, None), ("b", 20260927, None), ("c", 20260928, None)]
    panels = []
    for tag, seed, _ in samples:
        if seed is None:
            lens = np.ones(E)
        else:
            lens = sample_solution(faces, E, seed)
        pos = bfs_embed(epi, lens)
        polys = face_polygons(faces, epi, pos)
        panels.append((tag, lens, pos, polys))

    MM = 1 / 25.4
    fig_w, fig_h = 183 * MM, 66 * MM
    fig = plt.figure(figsize=(fig_w, fig_h))
    ax_w = 0.235
    xs = [0.035, 0.035 + (ax_w + 0.035), 0.035 + 2 * (ax_w + 0.035)]
    cmap = plt.get_cmap("viridis")
    norm = Normalize(vmin=0.5, vmax=1.5)

    axes = []
    for (tag, lens, pos, polys), x0 in zip(panels, xs):
        ax = fig.add_axes([x0, 0.10, ax_w, 0.82])
        ax.set_aspect("equal")
        ax.axis("off")
        for size, pts in polys:
            fc = "#c9d4e8" if size == 8 else "#f4d9b8"
            alpha = 0.55 if size == 8 else 0.75
            ax.add_patch(mpatches.Polygon(pts, closed=True, facecolor=fc,
                                          edgecolor="none", alpha=alpha, zorder=1))
        segs, cols = [], []
        for e, (a, b, d) in epi.items():
            segs.append([pos[a], pos[b]])
            if tag == "a":
                cols.append("#404040")
            else:
                cols.append(cmap(norm(lens[e] / lens.mean())))
        ax.add_collection(LineCollection(segs, colors=cols, linewidths=1.5,
                                         capstyle="round", zorder=3))
        xsv = np.array([p[0] for p in pos.values()])
        ysv = np.array([p[1] for p in pos.values()])
        pad = 0.4
        ax.set_xlim(xsv.min() - pad, xsv.max() + pad)
        ax.set_ylim(ysv.min() - pad, ysv.max() + pad)
        ax.text(0.02, 0.985, tag, transform=ax.transAxes, fontsize=9,
                fontweight="bold", va="top", ha="left", zorder=5)
        axes.append(ax)

    cax = fig.add_axes([xs[-1] + ax_w + 0.055, 0.10, 0.012, 0.82])
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    cb.set_label("edge length / mean", fontsize=7.5)
    cb.ax.tick_params(labelsize=7)

    for ext in ("png", "pdf"):
        out = os.path.join(ROOT, "z3_488", f"TSQ_real_metric_random_solution_demo.{ext}")
        fig.savefig(out, dpi=300)
        if ext == "png":
            print("fig ->", out)
    plt.close(fig)


if __name__ == "__main__":
    t0 = time.time()
    dim_table()
    print(f"--- dim table done ({time.time()-t0:.1f}s) ---")
    draw()
    print(f"--- all done ({time.time()-t0:.1f}s) ---")
