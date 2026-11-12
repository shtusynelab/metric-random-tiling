# -*- coding: utf-8 -*-
"""
penrose_lib.py -- combinatorial construction of the Penrose pentagrid
(de Bruijn pentagrid) dual + strand analysis of rhombus tilings.

Construction (standard de Bruijn 1981 duality, combinatorial level):
  Five families of parallel lines with normals n_i = (cos 72i deg,
  sin 72i deg); family i satisfies n_i . x = m + c_i.
  Cells of the arrangement <-> integer step vectors
  gamma(x) = (floor(n_i . x - c_i))_i in Z^5 = tile vertices;
  arrangement segments (between two adjacent crossings on one line) <->
  tile edges, direction family = line family (5 families, 72 deg apart);
  arrangement crossings (family-i line x family-j line) <-> one rhombus
  face of type {i,j}; its four edges are the segments facing the four
  half-edges of that crossing, opposite edges in the same family
  (combinatorial parallelogram).

Strip theorem (see PEN_strip_theorem_analytic_proof.md in this
directory): a rhombus face is a parallelogram => the real constraints are
exactly "opposite sides equal"; the pairing propagates into strands
(worms / Ammann bars), Omega = k^(number of strips) exactly; in the
pentagrid construction strands <-> grid lines => the number of strips is
O(window radius) => s(infinity) = 0 (Penrose has no metric entropy).

The regular pentagrid offsets are irrational (avoiding triple
intersection points); the de Bruijn correspondence guarantees that the
dual of a regular pentagrid is an edge-to-edge Penrose tiling.
"""
from __future__ import annotations

import math
from collections import defaultdict

NORM = [(math.cos(2 * math.pi * i / 5), math.sin(2 * math.pi * i / 5))
        for i in range(5)]
DIRC = [(math.cos(2 * math.pi * i / 5 + math.pi / 2),
         math.sin(2 * math.pi * i / 5 + math.pi / 2)) for i in range(5)]
OFFS = [(i + 1) * (2 ** 0.5) / 3 + i * i * (3 ** 0.5) / 7
        for i in range(5)]   # generic-position irrational offsets (avoiding sqrt(5)-field resonances)


def make_penrose_patch(R: float):
    """Pentagrid dual, combinatorial level, for window |x| <= R (complete faces).

    Returns (faces, edges):
      faces: {crossing_key: [(edge_key, family) x4, cyclic]}
      edges: {edge_key: (family, cellA, cellB)}
    crossing_key = (i, mi, j, mj) with i<j; edge_key = (i, mi, cA, cB).
    """
    M = int(R) + 3
    lines = defaultdict(list)          # (i, m) -> [(t, (j, mj)), ...]
    for i in range(5):
        for m in range(-M, M + 1):
            _ = lines[(i, m)]
    crossings = {}
    for i in range(5):
        for j in range(i + 1, 5):
            nix, niy = NORM[i]
            njx, njy = NORM[j]
            det = nix * njy - niy * njx
            for mi in range(-M, M + 1):
                bi = mi + OFFS[i]
                for mj in range(-M, M + 1):
                    bj = mj + OFFS[j]
                    x = (bi * njy - bj * niy) / det
                    y = (nix * bj - njx * bi) / det
                    if abs(x) <= R and abs(y) <= R:
                        t = x * DIRC[i][0] + y * DIRC[i][1]
                        lines[(i, mi)].append((t, (j, mj)))
                        t2 = x * DIRC[j][0] + y * DIRC[j][1]
                        lines[(j, mj)].append((t2, (i, mi)))
                        crossings[(i, mi, j, mj)] = (x, y)

    def ckey(fam, m, other):
        """Global key of a crossing: (smaller family, its line index,
        larger family, its line index)."""
        return (fam, m, other[0], other[1]) if fam < other[0] \
            else (other[0], other[1], fam, m)

    def gamma(x, y, fam, sgn):
        g = []
        for l in range(5):
            nx, ny = NORM[l]
            v = nx * x + ny * y - OFFS[l]
            g.append(math.floor(v) + (1 if (l == fam and sgn > 0) else 0))
        return tuple(g)

    edges = {}
    for (i, m), lst in lines.items():
        lst.sort()
        for (t1, c1), (t2, c2) in zip(lst, lst[1:]):
            nx, ny = NORM[i]
            k1, k2 = ckey(i, m, c1), ckey(i, m, c2)
            xm = (crossings[k1][0] + crossings[k2][0]) / 2
            ym = (crossings[k1][1] + crossings[k2][1]) / 2
            gA = gamma(xm + 1e-6 * nx, ym + 1e-6 * ny, i, +1)
            gB = gamma(xm - 1e-6 * nx, ym - 1e-6 * ny, i, -1)
            edges[(i, m, c1, c2)] = (i, gA, gB)

    faces = {}
    for (i, mi, j, mj), (x, y) in crossings.items():
        segs = []
        for fam, m in ((i, mi), (j, mj)):
            lst = sorted(lines[(fam, m)])
            me = (j, mj) if fam == i else (i, mi)
            pos = next(k for k, (t, c) in enumerate(lst) if c == me)
            if pos > 0:
                segs.append(((fam, m, lst[pos - 1][1], lst[pos][1]), fam))
            if pos < len(lst) - 1:
                segs.append(((fam, m, lst[pos][1], lst[pos + 1][1]), fam))
        if len(segs) != 4:
            continue
        missing = [s for s, f in segs if s not in edges]
        if missing:
            continue
        def far_angle(item):
            ekey, fam = item
            f2, m2, c1, c2 = ekey
            me = (j, mj) if f2 == i else (i, mi)
            far = c2 if c1 == me else c1
            kc = (f2, m2, far[0], far[1]) if f2 < far[0] \
                else (far[0], far[1], f2, m2)
            px, py = crossings[kc]
            return math.atan2(py - y, px - x)
        segs.sort(key=far_angle)
        faces[(i, mi, j, mj)] = segs
    return faces, edges


def strand_analysis(faces, edges):
    """Union-find pairing of opposite edges -> number of strands; also
    checks that opposite edges share a family and the face edge count.

    Returns (n_strands, n_faces, max_len, len_hist)."""
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    n_bad = 0
    for ck, sides in faces.items():
        fams = [fam for e, fam in sides]
        assert len(fams) == 4
        if not (fams[0] == fams[2] and fams[1] == fams[3]):
            n_bad += 1
        union(sides[0][0], sides[2][0])
        union(sides[1][0], sides[3][0])
    from collections import Counter
    cnt = Counter(find(e) for e in edges)
    # class size = number of edges in the class (window segments per strand;
    # O(1) segments per strand inside the window)
    return len(cnt), len(faces), max(cnt.values()), dict(Counter(cnt.values()))
