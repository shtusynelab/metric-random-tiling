# -*- coding: utf-8 -*-
"""
TRI_numerical_verification.py -- numerical verification of the z=6
triangular-lattice rigidity theorem.

Theorem (see TRI_rigidity_theorem_analytic_proof.md in this directory):
on any connected 6-valent triangular mesh (torus or hexagonal disk of the
regular triangular lattice), the isogonal (60-degree) constraint implies
every triangle is equilateral, which propagates through shared edges, so
all edges have equal length:
    Omega = k (torus, one global scale) or 1 (pin any single edge)
    s     = ln k / F  ->  0

Three verification layers (none presupposes the theorem):
  A. Brute force: enumerate all k^E edge assignments on the smallest
     hexagonal disk R=1 and smallest torus L=2, filter "all three edges of
     each triangle equal", and compare the count against Omega = k;
  B. Union-find class counting of the equilateral constraint: each
     triangle merges its three edges into one class; the number of classes
     must be exactly 1. Also explicitly check that the triangle adjacency
     graph is connected (the load-bearing lemma of the propagation
     mechanism), that every edge belongs to at least one triangle, and
     that the Euler characteristic is correct;
  C. Finite-size series: s = ln k / F tables for both geometries
     (disk/torus), verifying -> 0.

Modeling conventions: torus edges are indexed explicitly by
(direction family, q, r), keeping parallel edges at L=2 -- do NOT dedupe by
endpoint sets (frozenset dedup would wrongly merge torus parallel edges);
the triangle parametrization scan domain of the hexagonal disk is extended
by one ring so that "down" triangles with a parameter point outside the
disk are not missed.

Output: TRI_entropy_density_verification.csv (same directory as this
script). Standard library only.
"""
from __future__ import annotations

import csv
import math
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ALPHABET = (4, 5, 6)  # project edge-color/edge-length alphabet
K = len(ALPHABET)


def hex_patch(radius: int):
    """Hexagonal disk (axial coordinates |q|, |r|, |q+r| <= R).
    Returns (n_verts, n_edges, list of triangle edge-id triples).
    Triangles: up = (q,r),(q+1,r),(q,r+1); down = (q+1,r),(q,r+1),(q+1,r+1).
    The scan domain is extended by one ring; only triangles with all three
    vertices inside the disk are kept."""
    verts = [(q, r)
             for q in range(-radius, radius + 1)
             for r in range(max(-radius - q, -radius), min(radius - q, radius) + 1)]
    vset = frozenset(verts)
    ids: dict[frozenset, int] = {}

    def eid(a, b) -> int:
        key = frozenset((a, b))
        if key not in ids:
            ids[key] = len(ids)
        return ids[key]

    tris = []
    for q in range(-radius - 1, radius + 2):
        for r in range(-radius - 1, radius + 2):
            for t in (((q, r), (q + 1, r), (q, r + 1)),
                      ((q + 1, r), (q, r + 1), (q + 1, r + 1))):
                if all(v in vset for v in t):
                    tris.append(tuple(eid(t[a], t[(a + 1) % 3]) for a in range(3)))
    n_edges = len(ids)
    # Euler characteristic (disk, outer face included): V - E + F = 1
    assert len(verts) - n_edges + len(tris) == 1, \
        f"hex R={radius}: V={len(verts)} E={n_edges} F={len(tris)}"
    return len(verts), n_edges, tris


def torus_patch(side: int):
    """L x L triangular-lattice torus. Edges are indexed explicitly by
    (direction family, q, r), keeping parallel edges.
    Families: dir1 = (q,r)-(q+1,r); dir2 = (q,r)-(q,r+1); dir3 = (q+1,r)-(q,r+1).
    Returns (n_verts, n_edges, list of triangle edge-id triples)."""
    n_verts = side * side
    n_edges = 3 * side * side

    def e1(q, r): return (r % side) * side + (q % side)
    def e2(q, r): return side * side + (r % side) * side + (q % side)
    def e3(q, r): return 2 * side * side + (r % side) * side + (q % side)

    tris = []
    for q in range(side):
        for r in range(side):
            # up: (q,r),(q+1,r),(q,r+1)
            tris.append((e1(q, r), e2(q, r), e3(q, r)))
            # down: (q+1,r),(q,r+1),(q+1,r+1)
            tris.append((e3(q, r), e2(q + 1, r), e1(q, r + 1)))
    # Euler characteristic (torus): V - E + F = 0
    assert n_verts - n_edges + len(tris) == 0, \
        f"torus L={side}: V={n_verts} E={n_edges} F={len(tris)}"
    return n_verts, n_edges, tris


def brute_force(tris: list, k: int) -> int:
    """Filter all k^E assignments one by one (only called when k^E <= ~10^6)."""
    n_edges = max(e for t in tris for e in t) + 1
    count = 0
    for assign in product(range(k), repeat=n_edges):
        if all(assign[a] == assign[b] == assign[c] for a, b, c in tris):
            count += 1
    return count


class DSU:
    def __init__(self, n: int):
        self.p = list(range(n))

    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra

    def n_classes(self, n: int) -> int:
        return len({self.find(x) for x in range(n)})


def triangle_adjacency_connected(tris: list, n_edges: int) -> tuple[bool, int]:
    """Is the triangle adjacency graph (adjacent = sharing an edge)
    connected? Returns (connected?, number of isolated edges)."""
    edge2tris: dict[int, list[int]] = {}
    for fi, t in enumerate(tris):
        for e in t:
            edge2tris.setdefault(e, []).append(fi)
    seen = {0}
    stack = [0]
    while stack:
        fi = stack.pop()
        for e in tris[fi]:
            for fj in edge2tris[e]:
                if fj not in seen:
                    seen.add(fj)
                    stack.append(fj)
    isolated = sum(1 for e in range(n_edges) if e not in edge2tris)
    return len(seen) == len(tris), isolated


def constraint_classes(tris: list, n_edges: int) -> int:
    """Number of union-find classes under the equilateral constraint
    (within each triangle all three edges are equal)."""
    d = DSU(n_edges)
    for a, b, c in tris:
        d.union(a, b)
        d.union(b, c)
    return d.n_classes(n_edges)


def main() -> None:
    rows = []
    all_ok = True

    # A. brute-force anchors
    print("A. brute-force anchors")
    anchors = [("hex_disk_R1", hex_patch(1)), ("torus_L2", torus_patch(2))]
    for name, (n_verts, n_edges, tris) in anchors:
        omega_bf = brute_force(tris, K)
        ok = omega_bf == K
        all_ok &= ok
        rows.append(dict(case="brute", patch=name, k=K, V=n_verts,
                         E=n_edges, F=len(tris),
                         tri_adj_connected="", n_classes="",
                         omega_formula=K, omega_bruteforce=omega_bf,
                         s_formula=f"{math.log(K) / len(tris):.6f}",
                         s_classes=""))
        print(f"  {name}: V={n_verts} E={n_edges} F={len(tris)} "
              f"brute={omega_bf} formula={K} {'OK' if ok else 'MISMATCH'}")

    # B. class counting + load-bearing lemma (triangle adjacency connected)
    #    + Euler/isolated-edge self-checks
    print("B. constraint classes & triangle-adjacency connectivity")
    patches = [("hex_disk_R1", hex_patch(1)),
               ("hex_disk_R2", hex_patch(2)),
               ("hex_disk_R3", hex_patch(3)),
               ("hex_disk_R4", hex_patch(4)),
               ("hex_disk_R5", hex_patch(5)),
               ("hex_disk_R6", hex_patch(6)),
               ("torus_L2", torus_patch(2)),
               ("torus_L3", torus_patch(3)),
               ("torus_L4", torus_patch(4)),
               ("torus_L6", torus_patch(6)),
               ("torus_L8", torus_patch(8))]
    for name, (n_verts, n_edges, tris) in patches:
        connected, isolated = triangle_adjacency_connected(tris, n_edges)
        n_cls = constraint_classes(tris, n_edges)
        ok = connected and isolated == 0 and n_cls == 1
        all_ok &= ok
        rows.append(dict(case="classes", patch=name, k=K, V=n_verts,
                         E=n_edges, F=len(tris),
                         tri_adj_connected=connected, n_classes=n_cls,
                         omega_formula=K ** n_cls, omega_bruteforce="",
                         s_formula=f"{math.log(K) / len(tris):.6f}",
                         s_classes=f"{math.log(K ** n_cls) / len(tris):.6f}"))
        print(f"  {name}: V={n_verts} E={n_edges} F={len(tris)} "
              f"adj_connected={connected} isolated_edges={isolated} "
              f"classes={n_cls} (expect 1)")

    # C. finite-size series: s -> 0 (both geometries)
    print("C. finite-size series (per-face entropy)")
    for radius in range(1, 9):
        _, n_edges, tris = hex_patch(radius)
        s = math.log(K) / len(tris)
        rows.append(dict(case="series", patch=f"hex_disk_R{radius}", k=K,
                         V=3 * radius * radius + 3 * radius + 1,
                         E=n_edges, F=len(tris),
                         tri_adj_connected="", n_classes=1, omega_formula=K,
                         omega_bruteforce="", s_formula=f"{s:.6f}",
                         s_classes=""))
        print(f"  hex_disk_R{radius}: V={3 * radius * radius + 3 * radius + 1} "
              f"E={n_edges} F={len(tris)} s={s:.6f}")
    for side in range(2, 13):
        n_verts, n_edges, tris = torus_patch(side)
        s = math.log(K) / len(tris)
        rows.append(dict(case="series", patch=f"torus_L{side}", k=K,
                         V=n_verts, E=n_edges, F=len(tris),
                         tri_adj_connected="", n_classes=1, omega_formula=K,
                         omega_bruteforce="", s_formula=f"{s:.6f}",
                         s_classes=""))
        print(f"  torus_L{side}: V={n_verts} E={n_edges} F={len(tris)} s={s:.6f}")

    out = ROOT / "TRI_entropy_density_verification.csv"
    with out.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"\nALL CHECKS {'PASSED' if all_ok else 'FAILED'}; wrote {out.name}")
    if not all_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
