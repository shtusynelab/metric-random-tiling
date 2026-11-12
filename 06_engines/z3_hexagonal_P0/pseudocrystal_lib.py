# -*- coding: utf-8 -*-
"""
pseudocrystal_lib.py - unified parameterized library for MOF pseudocrystals
(plan module A1, P0)

Refactors the two-ligand / four-ligand / FFT scripts into one parameterized
library:
  Input: alphabet L subset of {222,223,233,333}, ring count n, boundary
         (periodic/open)
  Output: legal assembly pattern (edge-color field), ligand identity and motif
          statistics, Knuth solution-space estimates, real-space geometry,
          exact structure factor |F(q)|^2

## Model and concept conventions

- Ligand = vertex of the honeycomb graph (3-coordinated); hole = hexagonal
  face; metal bridge = edge.
- Edge-color semantics: edge weight w = sum of the arm numbers of the two
  ligands at the bridge ends, 2+2=4, 2+3=5, 3+3=6. Color index 0/1/2 <->
  weight 4/5/6; color = length = closure datum (three-in-one).
- Ligand identity = multiset of its three arm numbers (222/223/233/333).
- Vertex-level alphabet constraint: among the edges at a vertex, weight 6
  forces arm 3 at that end and weight 4 forces arm 2, while weight 5 admits
  either 2 or 3 at each end (the choice is a degree of freedom). Hence the
  vertex "3-arm" count c3 in [n6, n6+n5] (n6/n5 = number of weight-6/5 edges
  among the vertex's three edges); the alphabet's allowed set
  allowed_c3(L) = {number of 3s in the identity}.
- Library-level constraints are all compiled from vertex constraints:
  (1) equiangular closure L1+L2=L4+L5 and 2L0+L1+L5=2L3+L2+L4 (always
      present);
  (2) adjacent edge pairs within a hole are locally feasible
      ([n6,n6+n5] intersect allowed_c3 non-empty);
  (3) hole perimeter <= 24 + 6*max(allowed_c3) (sum of the 3-arm quotas of
      the hole's six vertices).
  For {222,223}: (2) <=> no 6-6 adjacency within a hole, (3) <=> perimeter
  <= 30 -- one-to-one with the legacy two-ligand library; for the quaternary
  alphabet: (3) is always true (<=36) and the library contains only the
  closure + pair constraints -- matching the legacy four-ligand library.
- All-5 (all six edges weight 5) no-adjacency: enabled only when
  max(allowed_c3) <= 1 (includes {222,223}); a derived theorem of the binary
  system (out-degree counting, evaluation doc 8.5); adding the filter does
  not change the true solution space.

## Approximation statement (scope of the A2 entropy density)

Knuth counting targets the **edge-color field** (same scope as the legacy
scripts): vertex checks only verify [n6,n6+n5] intersect allowed_c3
non-empty, without tracking the global consistency of the arm assignments on
the two sides of weight-5 edges. The approximation biases every alphabet in
the same direction (overestimate), and the weight-5 direction freedom is an
edge-count-scale upper factor, so it does not change the positive/zero
entropy-density verdicts or the magnitude ranking between alphabets. Strict
arm-field counting is left as follow-up work.

## Main interfaces

  build_library(alphabet)          -> TileLibrary
  solve_tessellation(n, lib, ...)  -> solution dict or None (randomized backtracking)
  knuth_estimate(n, lib, ...)      -> Omega_N estimate + alive rate
  extract_geometry(solution, lib)  -> hole centers / metal nodes / bridge edges (endpoints + color)
  motif_stats(solution, lib)       -> 5-5 hole pairs, 6-6 vertices, ligand identities, perimeter distribution
  structure_factor(pts, ...)       -> exact |F(q)|^2 (convex-hull incircle mask)
  crystal_disk_points / amorphous_disk_points -> REF-C / REF-A reference coordinates
"""

import itertools
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np

# ----------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------

LIGAND_ARMS = {'222': (2, 2, 2), '223': (2, 2, 3), '233': (2, 3, 3), '333': (3, 3, 3)}
WEIGHT_OF_COLOR = {0: 4, 1: 5, 2: 6}          # color index -> logical edge length (arm-number sum)
COLOR_OF_WEIGHT = {4: 0, 5: 1, 6: 2}
DIRECTIONS = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]

ALL_SUBSETS = []
for _k in range(1, 5):
    for _combo in itertools.combinations(['222', '223', '233', '333'], _k):
        ALL_SUBSETS.append(tuple(_combo))


def n_holes(n_rings: int) -> int:
    """Number of holes of an n-ring hexagonal disk = 3R(R+1)+1, R=n-1."""
    R = n_rings - 1
    return 3 * R * (R + 1) + 1


# ----------------------------------------------------------------------
# Tile library
# ----------------------------------------------------------------------

def closure_ok(weights) -> bool:
    """Equiangular hexagon closure condition (edge-length sequence)."""
    L = list(weights)
    return (L[1] + L[2] == L[4] + L[5]) and (2 * L[0] + L[1] + L[5] == 2 * L[3] + L[2] + L[4])


def allowed_c3(alphabet) -> frozenset:
    """Set of vertex "3-arm" counts allowed by the alphabet."""
    return frozenset(LIGAND_ARMS[name].count(3) for name in alphabet)


def vertex_feasible(weights3, allowed: frozenset) -> bool:
    """Given the vertex's three edge weights, does a legal arm assignment with
    identity in the alphabet exist."""
    n6 = sum(1 for w in weights3 if w == 6)
    n5 = sum(1 for w in weights3 if w == 5)
    return any(n6 <= c <= n6 + n5 for c in allowed)


@dataclass
class TileLibrary:
    alphabet: tuple
    allowed_c3: frozenset
    colors: np.ndarray      # (K,6) uint8
    weights: np.ndarray     # (K,6) uint8, values in 4/5/6
    all_yellow: np.ndarray  # (K,) bool, all six edges weight 5
    perimeter: np.ndarray   # (K,) int
    n_base: int             # base configurations before rotation dedup

    @property
    def size(self):
        return len(self.colors)


def _make_tile_library(colors_arr, alphabet=(), allowed=frozenset(), n_base=None):
    colors_arr = np.asarray(colors_arr, dtype=np.uint8)
    weights = np.vectorize(WEIGHT_OF_COLOR.get)(colors_arr).astype(np.uint8)
    return TileLibrary(
        alphabet=tuple(alphabet), allowed_c3=frozenset(allowed),
        colors=colors_arr, weights=weights,
        all_yellow=np.all(colors_arr == 1, axis=1),
        perimeter=weights.sum(axis=1).astype(int),
        n_base=len(colors_arr) if n_base is None else n_base,
    )


def edge_weights_allowed(alphabet) -> set:
    """Edge weights that can exist under the alphabet: w = a+b with a, b arms
    present in the alphabet.

    {333} -> {6}; {222} -> {4}; alphabets with both arm 2 and arm 3 -> {4,5,6}.
    """
    arms_present = set()
    for name in alphabet:
        arms_present.update(LIGAND_ARMS[name])
    return {a + b for a in arms_present for b in arms_present}


def build_library(alphabet) -> TileLibrary:
    """Compile an alphabet into a tile library (rotation-deduplicated).

    The perimeter must satisfy both an upper and a lower bound, and the two are
    strictly dual under the 2<->3 arm swap (edge value w -> 10-w): each vertex
    ligand contributes two arms to a hole with contribution sum in
    [4+max(0,c3-1), 4+c3], so the hole perimeter lies in
    [6*(4+max(0,min_c3-1)), 6*(4+max_c3)]. Without the lower bound, the library
    of 3-arm-rich alphabets ({233}/{233,333}) would admit unrealizable tiles
    containing a 4-edge (whose mirror side is correctly excluded by the upper
    bound), and exact counts of mirror alphabets would no longer agree
    bit-for-bit.
    """
    allowed = allowed_c3(alphabet)
    perim_cap = 24 + 6 * max(allowed)
    perim_floor = 6 * (4 + max(0, min(allowed) - 1))
    weights_ok = edge_weights_allowed(alphabet)

    pair_ok = {}
    for a, b in itertools.combinations_with_replacement([4, 5, 6], 2):
        pair_ok[(a, b)] = any(vertex_feasible((a, b, x), allowed) for x in (4, 5, 6))

    base = []
    for w in itertools.product([4, 5, 6], repeat=6):
        if not closure_ok(w) or sum(w) > perim_cap or sum(w) < perim_floor:
            continue
        if any(x not in weights_ok for x in w):
            continue
        if all(pair_ok[tuple(sorted((w[i], w[(i + 1) % 6])))] for i in range(6)):
            base.append(tuple(COLOR_OF_WEIGHT[x] for x in w))

    full, seen = [], set()
    for shape in base:
        curr = list(shape)
        for _ in range(6):
            t = tuple(curr)
            if t not in seen:
                seen.add(t)
                full.append(t)
            last = curr.pop()
            curr.insert(0, last)
    return _make_tile_library(full, alphabet, allowed, n_base=len(base))


def uniform_tile_library(weights6) -> TileLibrary:
    """Hand-built single-tile library (e.g. the REF-A all-5 honeycomb).
    weights6: length-6 weight sequence."""
    cols = np.array([tuple(COLOR_OF_WEIGHT[w] for w in weights6)], dtype=np.uint8)
    return _make_tile_library(cols)


# ----------------------------------------------------------------------
# Coordinates and periodic folding
# ----------------------------------------------------------------------

def hex_coords(radius):
    """Axial coordinates of hole centers of the n-ring disk, sorted by ring
    order (ascending hex distance)."""
    coords = []
    for q in range(-radius, radius + 1):
        for r in range(-radius, radius + 1):
            if abs(q + r) <= radius:
                coords.append((q, r))
    coords.sort(key=lambda p: (abs(p[0]) + abs(p[1]) + abs(p[0] + p[1])))
    return coords


def make_wrap(radius):
    """Periodic (torus) folding; radius=None returns the identity (open
    boundary)."""
    if radius is None:
        return lambda q, r: (q, r)
    R = radius
    centers = [
        (2 * R + 1, -R), (R, R + 1), (-R - 1, 2 * R + 1),
        (-2 * R - 1, R), (-R, -R - 1), (R + 1, -2 * R - 1),
    ]

    def wrap(q, r):
        if abs(q) <= R and abs(r) <= R and abs(q + r) <= R:
            return (q, r)
        for cx, cy in centers:
            wq, wr = q - cx, r - cy
            if abs(wq) <= R and abs(wr) <= R and abs(wq + wr) <= R:
                return (wq, wr)
        return (q, r)

    return wrap


# ----------------------------------------------------------------------
# Arm-number realizability (global consistency of weight-5 edge directions)
# ----------------------------------------------------------------------

def _internal_graph(solution, wrap, libw=None):
    """Extract internal vertices/edges from an edge-color field (vertex =
    three-hole intersection, edge = bridge shared by two holes).

    solution values are tile indices (pass libw) or direct weight sequences.
    Returns (internal_vertex_keys, edges), edges = [(vkeyA, vkeyB, weight)],
    containing only bridges whose two end vertices are internal; dangling
    boundary parts are excluded (boundary effect O(sqrt(N))).
    """
    def tile_w(c):
        return libw[solution[c]] if libw is not None else solution[c]

    vcount = {}
    edge_id = 0
    for (q, r) in solution:
        w = tile_w((q, r))
        for i in range(6):
            vk = vertex_key(q, r, (i - 1) % 6, wrap)
            vcount[vk] = vcount.get(vk, 0) + 1
    internal = {v for v, c in vcount.items() if c == 3}
    edges = []
    seen_e = set()
    for (q, r) in solution:
        w = tile_w((q, r))
        for i in range(6):
            va = vertex_key(q, r, (i - 1) % 6, wrap)
            vb = vertex_key(q, r, i, wrap)
            if va in internal and vb in internal:
                e = (va, vb) if va <= vb else (vb, va)
                if e not in seen_e:
                    seen_e.add(e)
                    edges.append((va, vb, int(w[i])))
    return internal, edges


def vertex_key(q, r, i, wrap):
    h1 = (q, r)
    h2 = wrap(q + DIRECTIONS[i][0], r + DIRECTIONS[i][1])
    j = (i + 1) % 6
    h3 = wrap(q + DIRECTIONS[j][0], r + DIRECTIONS[j][1])
    return tuple(sorted([h1, h2, h3]))


def arms_assignable(edges, max_c3=1):
    """Whether an arm-number realization of the edge-color field exists: each
    weight-5 edge's "3 side" must be accommodated by a vertex, and vertex v
    accepts at most max_c3 - n6(v) of them (weight-6 edges already consume the
    3-arm quota). Kuhn bipartite matching.

    edges: [(vA, vB, weight)]; returns (bool, match_v).
    Note: verifies only the upper bound of the 3-arm quota
    (max_c3 = max(allowed_c3)); the capacity lower bound (alphabets with
    min(allowed_c3) > n6, e.g. {223,233}) is not enforced -- no effect on the
    core alphabets of this study ({222,223} and the quaternary system, whose
    minima are both 0); declared in the report.
    """
    cap = defaultdict(int)
    fives = []
    verts = set()
    for va, vb, w in edges:
        verts.add(va)
        verts.add(vb)
        if w == 6:
            cap[va] -= 1
            cap[vb] -= 1
        elif w == 5:
            fives.append((va, vb))
    for v in verts:
        cap[v] += max_c3
    match_v = {}

    def try_k(e, visited):
        for v in e:
            if cap.get(v, 0) >= 1 and v not in visited:
                visited.add(v)
                if v not in match_v or try_k(match_v[v], visited):
                    match_v[v] = e
                    return True
        return False

    for e in fives:
        if not try_k(e, set()):
            return False, match_v
    return True, match_v


# ----------------------------------------------------------------------
# Assembly solver (randomized backtracking, headless)
# ----------------------------------------------------------------------

def solve_tessellation(n_rings, lib, periodic=False, rng=None,
                       max_backtracks=200000, max_restarts=200):
    """Randomly generate one legal assembly (arm-field scope). On success
    returns (solution, edge_dir).

    solution: {(q,r): tile_idx}; edge_dir: {sorted(vkA,vkB): bool}
    -- direction of the weight-5 edge (True = the first endpoint in sorted
    order is the arm-3 side), i.e. the arm-number realization.

    Built-in constraints: edge matching, all-5 no-adjacency (when
    max(allowed_c3) <= 1), vertex local feasibility, vertex 3-arm count upper
    bound (immediate pruning) and lower bound (checked at completion). The
    weight-5 direction is a native degree of freedom: oriented immediately
    when an edge closes, which guarantees the solution is physically
    realizable.
    """
    rng = rng or random
    R = n_rings - 1
    coords = hex_coords(R)
    wrap = make_wrap(R if periodic else None)
    K = lib.size
    libw = lib.weights
    lib_ay = lib.all_yellow
    enforce_no5x5 = max(lib.allowed_c3) <= 1
    allowed = lib.allowed_c3
    min_c3, max_c3 = min(allowed), max(allowed)

    for _attempt in range(max_restarts):
        solution = {}
        c3 = defaultdict(int)    # vertex 3-arm count
        vdeg = defaultdict(int)  # registered edge count per vertex
        vws = defaultdict(list)  # vertex edge weights (local feasibility check)
        edge_dir = {}
        edge_seen = set()        # registered edges (vertex pairs) -- each edge registered once
        backtracks = 0

        def scan_edges(idx, q, r, undo):
            """Register the hole's 6 edges (mutating state and recording undo
            entries); returns the newly closed weight-5 edges, None = violation."""
            w = libw[idx]
            new5 = []
            for i in range(6):
                vkA = vertex_key(q, r, (i - 1) % 6, wrap)
                vkB = vertex_key(q, r, i, wrap)
                ek = (vkA, vkB) if vkA <= vkB else (vkB, vkA)
                wi = int(w[i])
                if ek in edge_seen:
                    # registered by the neighboring hole; if a 5-edge not yet
                    # oriented, the edge closes now -> orient it
                    if wi == 5 and ek not in edge_dir:
                        new5.append((ek, vkA, vkB))
                    continue
                edge_seen.add(ek)
                undo.append(('seen', ek))
                vdeg[vkA] += 1
                vdeg[vkB] += 1
                vws[vkA].append(wi)
                vws[vkB].append(wi)
                undo.append(('deg', vkA))
                undo.append(('deg', vkB))
                if wi == 6:
                    c3[vkA] += 1
                    c3[vkB] += 1
                    undo.append(('c3', vkA))
                    undo.append(('c3', vkB))
                    if c3[vkA] > max_c3 or c3[vkB] > max_c3:
                        return None
            return new5

        def orient(new5, undo):
            """Roll directions for the new 5-edges (recorded in undo); returns
            False if an upper bound is hit."""
            for ek, vkA, vkB in new5:
                first_is3 = rng.random() < 0.5
                edge_dir[ek] = first_is3
                undo.append(('dir', ek))
                v3 = ek[0] if first_is3 else ek[1]
                c3[v3] += 1
                undo.append(('c3', v3))
                if c3[v3] > max_c3:
                    return False
            return True

        def apply_undo(undo):
            for kind, key in reversed(undo):
                if kind == 'deg':
                    vdeg[key] -= 1
                    vws[key].pop()
                elif kind == 'c3':
                    c3[key] -= 1
                elif kind == 'seen':
                    edge_seen.discard(key)
                else:
                    edge_dir.pop(key, None)

        def vertex_ok(vk):
            if vdeg[vk] == 3:
                if not (min_c3 <= c3[vk] <= max_c3):
                    return False
                if not vertex_feasible(vws[vk], allowed):
                    return False
            return True

        def candidates_at(q, r):
            mask = np.ones(K, dtype=bool)
            for i in range(6):
                nq, nr = wrap(q + DIRECTIONS[i][0], r + DIRECTIONS[i][1])
                nb = solution.get((nq, nr))
                if nb is not None:
                    mask &= libw[:, i] == libw[nb][(i + 3) % 6]
                    if enforce_no5x5 and lib_ay[nb]:
                        mask &= ~lib_ay
            return np.flatnonzero(mask).tolist()

        def solve(remaining):
            nonlocal backtracks
            if not remaining:
                return True
            best, best_cands, best_key = None, None, None
            for c in remaining:
                cands = candidates_at(*c)
                if best is None or len(cands) < best_key:
                    best, best_cands, best_key = c, cands, len(cands)
                    if len(cands) <= 1:
                        break
            if not best_cands:
                backtracks += 1
                return False
            q, r = best
            rest = [c for c in remaining if c != best]
            cand_list = list(best_cands)
            rng.shuffle(cand_list)
            for idx in cand_list:
                solution[(q, r)] = idx
                undo = []
                new5 = scan_edges(idx, q, r, undo)
                ok = new5 is not None
                if ok:
                    ok = orient(new5, undo)
                if ok:
                    touched = {key for kind, key in undo if kind in ('deg', 'c3')}
                    for vk in touched:
                        if not vertex_ok(vk):
                            ok = False
                            break
                if ok and solve(rest):
                    return True
                apply_undo(undo)
                del solution[(q, r)]
                backtracks += 1
                if backtracks > max_backtracks:
                    return False
            return False

        if solve(list(coords)):
            return solution, edge_dir
    return None


# ----------------------------------------------------------------------
# Knuth Monte Carlo solution-space estimation
# ----------------------------------------------------------------------

def knuth_estimate(n_rings, lib, periodic=False, n_samples=2000, seed=None,
                   verify_arms=True):
    """Knuth tree-size estimate (pattern-level scope):
    Omega_N ~ mean(product of branch counts * 1(realizable)).

    Sampling = greedy forward over the edge-color field (branch = number of
    candidate tiles); completed fields additionally get a bipartite matching
    over weight-5 edge directions (arms_assignable); fields without an arm
    realization score 0. The estimator is unbiased for "the number of
    realizable edge-color fields" = the number of physically realizable
    assembly patterns (each pattern counted once, not once per arm
    realization). This scope survives greedy sampling one order of magnitude
    better than native greedy over the arm field. Open boundary is the
    default: torus seam-closure conflicts make greedy survival rates too low
    (the entropy-density limit is boundary-independent).

    Returns dict(n, N_holes, omega_mean, omega_sem, alive_rate,
              ln_omega, ln_omega_sem_lnsp, n_samples, periodic).
    The Omega distribution is heavy-tailed: ln_omega = ln(mean) is the
    reported quantity (as in the legacy scripts); ln_omega_sem_lnsp is the
    log-space sample standard error (surviving samples only, for reference).
    """
    rng = random.Random(seed)
    R = n_rings - 1
    coords = hex_coords(R)
    wrap = make_wrap(R if periodic else None)
    K = lib.size
    libw = lib.weights
    lib_ay = lib.all_yellow
    enforce_no5x5 = max(lib.allowed_c3) <= 1
    allowed = lib.allowed_c3
    min_c3, max_c3 = min(allowed), max(allowed)

    scores = []
    for _s in range(n_samples):
        solution = {}
        c3 = defaultdict(int)
        vdeg = defaultdict(int)
        vws = defaultdict(list)
        edge_seen = set()   # each edge (vertex pair) registered once -- avoids double counting by the two holes
        mult = 1.0
        dead = False
        for (q, r) in coords:
            mask = np.ones(K, dtype=bool)
            for i in range(6):
                nq, nr = wrap(q + DIRECTIONS[i][0], r + DIRECTIONS[i][1])
                nb = solution.get((nq, nr))
                if nb is not None:
                    mask &= libw[:, i] == libw[nb][(i + 3) % 6]
                    if enforce_no5x5 and lib_ay[nb]:
                        mask &= ~lib_ay
            cands = np.flatnonzero(mask)
            if len(cands) == 0:
                dead = True
                break
            mult *= len(cands)
            idx = int(cands[rng.randrange(len(cands))])
            solution[(q, r)] = idx
            w = libw[idx]
            ok = True
            touched = set()
            for i in range(6):
                vkA = vertex_key(q, r, (i - 1) % 6, wrap)
                vkB = vertex_key(q, r, i, wrap)
                ek = (vkA, vkB) if vkA <= vkB else (vkB, vkA)
                if ek in edge_seen:
                    continue
                edge_seen.add(ek)
                wi = int(w[i])
                vdeg[vkA] += 1
                vdeg[vkB] += 1
                vws[vkA].append(wi)
                vws[vkB].append(wi)
                touched.add(vkA)
                touched.add(vkB)
                if wi == 6:
                    c3[vkA] += 1
                    c3[vkB] += 1
                    if c3[vkA] > max_c3 or c3[vkB] > max_c3:
                        ok = False
                        break
            if ok:
                for vk in touched:
                    if vdeg[vk] == 3:
                        if c3[vk] > max_c3 or not vertex_feasible(vws[vk], allowed):
                            ok = False
                            break
            if not ok:
                dead = True
                break
        if not dead and verify_arms:
            _, edges = _internal_graph(solution, wrap, libw)
            ok, _m = arms_assignable(edges, max_c3=max_c3)
            if not ok:
                dead = True
        if not dead:
            scores.append(mult)

    scores = np.array(scores, dtype=float)
    N = n_holes(n_rings)
    if len(scores) == 0:
        return dict(n=n_rings, N_holes=N, omega_mean=0.0, omega_sem=0.0,
                    alive_rate=0.0, ln_omega=-np.inf, ln_omega_sem_lnsp=np.nan,
                    n_samples=n_samples, periodic=periodic)
    omega_mean = float(scores.mean())
    omega_sem = float(scores.std(ddof=1) / math.sqrt(len(scores))) if len(scores) > 1 else float('nan')
    ln_s = np.log(scores)
    return dict(n=n_rings, N_holes=N, omega_mean=omega_mean, omega_sem=omega_sem,
                alive_rate=len(scores) / n_samples,
                ln_omega=math.log(omega_mean),
                ln_omega_sem_lnsp=float(ln_s.std(ddof=1) / math.sqrt(len(ln_s))) if len(ln_s) > 1 else float('nan'),
                n_samples=n_samples, periodic=periodic)


# ----------------------------------------------------------------------
# Geometry extraction (real-space coordinates; same BFS placement as the
# legacy scripts)
# ----------------------------------------------------------------------

def extract_geometry(solution, lib, scale=1.0, periodic=False):
    """Compute real-space geometry from the edge-color field.

    periodic=True folds adjacency on the torus (periodic solutions; the
    geometry of the O(R) seam edges may be misaligned -- acceptable for
    statistics; open solutions are always False).

    Returns dict:
      hole_centers (M,2)          hole centers
      metal_nodes  (V,2)          hexagon geometric vertices (= metal/ligand sites)
      node_keys    [V]            node coordinate keys (x,y)
      edges        [(kA,kB,w,vkA,vkB)]  bridges (deduplicated): geometric endpoint keys,
                                        weight, and the corresponding hole spatial vertex keys
                                        (for edge_dir lookup)
      edge_midpoints (E,2)        bridge midpoints
      edge_weights (E,)           bridge logical edge lengths 4/5/6
    """
    import queue as _q
    R = max(max(abs(q), abs(r)) for q, r in solution)
    wrap = make_wrap(R if periodic else None)
    data = {c: lib.weights[idx] for c, idx in solution.items()}
    pos = {(0, 0): (0.0, 0.0)}
    hole_verts = {}
    hole_vkeys = {}
    bfs = _q.Queue()
    bfs.put((0, 0))
    seen = {(0, 0)}
    while not bfs.empty():
        curr = bfs.get()
        if curr not in data:
            continue
        wx, wy = pos[curr]
        w = data[curr]
        verts = [(wx, wy)]
        vx, vy = float(wx), float(wy)
        for i in range(6):
            vx += float(w[i]) * scale * math.cos(math.radians(i * 60))
            vy += float(w[i]) * scale * math.sin(math.radians(i * 60))
            verts.append((vx, vy))
        hole_verts[curr] = verts
        hole_vkeys[curr] = [vertex_key(curr[0], curr[1], i, wrap) for i in range(6)]
        for i in range(6):
            d = DIRECTIONS[i]
            nb = wrap(curr[0] + d[0], curr[1] + d[1])
            if nb in data and nb not in seen:
                n_w = data[nb]
                k = (i + 3) % 6
                ax = ay = 0.0
                for step in range(k + 1):
                    if step == k:
                        ax_n, ay_n = ax, ay
                    ax += float(n_w[step]) * scale * math.cos(math.radians(step * 60))
                    ay += float(n_w[step]) * scale * math.sin(math.radians(step * 60))
                dest = verts[i + 1]
                # Fix: verts are already absolute coordinates; the legacy
                # version erroneously translated by pos[curr] again
                # ("double-add"), the error accumulating with BFS depth and
                # leaving non-tree shared edges floating for n>=3.
                pos[nb] = (dest[0] - ax_n, dest[1] - ay_n)
                seen.add(nb)
                bfs.put(nb)

    node_key = lambda p: (round(p[0], 4), round(p[1], 4))
    edges = {}
    for c, verts in hole_verts.items():
        vks = hole_vkeys[c]
        for i in range(6):
            k1, k2 = node_key(verts[i]), node_key(verts[i + 1])
            if k1 == k2:
                continue
            if k1 <= k2:
                e = (k1, k2, int(data[c][i]), vks[(i - 1) % 6], vks[i])
            else:
                e = (k2, k1, int(data[c][i]), vks[i], vks[(i - 1) % 6])
            edges[e[:2] + (0,)] = e  # key = geometric endpoints; value carries direction info (later writes overwrite, keeping one consistent copy)
    pts = sorted({(k[0], k[1]) for e in edges for k in (e[:2])})
    metal_nodes = np.array(pts)
    edge_list = list(edges.values())
    mids = np.array([((a[0] + b[0]) / 2, (a[1] + b[1]) / 2) for a, b, _, _, _ in edge_list])
    ews = np.array([w for _, _, w, _, _ in edge_list], dtype=int)
    return dict(
        hole_centers=np.array([pos[c] for c in sorted(hole_verts)]),
        metal_nodes=metal_nodes,
        node_keys=pts,
        edges=edge_list,
        edge_midpoints=mids,
        edge_weights=ews,
    )


# ----------------------------------------------------------------------
# Motif statistics (A3)
# ----------------------------------------------------------------------

def motif_stats(solution, lib, rng=None, arm_mode='auto', edge_dir=None,
                periodic=False):
    """Pattern-level statistics (plan A3).

    - pairs55: number of adjacent all-5 hole pairs (sharing one edge, both
      holes all-5 on all six edges);
    - vertices66: number of degree-3 vertices carrying >= 2 weight-6 edges
      (6-6 adjacency events);
    - ligand_counts: histogram of ligand identities (degree-3 internal
      vertices only). Source of arm directions: edge_dir (the solver's native
      direction field, physically strict) takes precedence over arm_mode:
      'match' (bipartite matching, legacy two-ligand scope) /
      'random' (random assembly, legacy four-ligand scope) /
      'auto' (max(allowed_c3) <= 1 -> match, else random).
    - perimeter_hist / edge_weight_hist.
    """
    rng = rng or random
    geom = extract_geometry(solution, lib, periodic=periodic)

    data = {c: lib.colors[idx] for c, idx in solution.items()}
    yellow = {c for c, col in data.items() if col.min() == 1 and col.max() == 1}
    pairs55 = 0
    for (q, r) in data:
        for d in DIRECTIONS:
            nb = (q + d[0], r + d[1])
            if nb in data and (q, r) < nb and (q, r) in yellow and nb in yellow:
                pairs55 += 1

    incident = defaultdict(list)   # node_key -> [(weight, edge_idx)]
    for ei, (ka, kb, w, vka, vkb) in enumerate(geom['edges']):
        incident[ka].append((w, ei))
        incident[kb].append((w, ei))

    # arm-number direction field
    arm = {}          # edge_idx -> (armAtKA, armAtKB)
    uncertain = set() # 5-edges with undetermined direction (dangling edges; direction belongs to an external ligand)
    for ei, (ka, kb, w, vka, vkb) in enumerate(geom['edges']):
        if w == 4:
            arm[ei] = (2, 2)
        elif w == 6:
            arm[ei] = (3, 3)
        elif edge_dir is not None:
            ek = (vka, vkb) if vka <= vkb else (vkb, vka)
            d = edge_dir.get(ek)
            if d is None:
                uncertain.add(ei)
            else:
                first_is_ka = (ek[0] == vka)
                arm[ei] = ((3, 2) if d else (2, 3)) if first_is_ka \
                    else ((2, 3) if d else (3, 2))
    fives = [ei for ei, e in enumerate(geom['edges']) if e[2] == 5]
    need_arm = [ei for ei in fives if ei not in arm and ei not in uncertain]
    if need_arm:
        if arm_mode == 'auto':
            arm_mode = 'match' if max(lib.allowed_c3) <= 1 else 'random'
        if arm_mode == 'match':
            cap = {k: max(lib.allowed_c3) for k in incident}
            for ka, kb, w, _, _ in geom['edges']:
                if w == 6:
                    cap[ka] -= 1
                    cap[kb] -= 1
            match_v = {}
            unmatched = []
            for ei in need_arm:
                ka, kb = geom['edges'][ei][:2]
                if cap.get(ka, 0) >= 1 and ka not in match_v:
                    match_v[ka] = ei
                elif cap.get(kb, 0) >= 1 and kb not in match_v:
                    match_v[kb] = ei
                else:
                    unmatched.append(ei)

            def dfs_match(ei, visited):
                ka, kb = geom['edges'][ei][:2]
                for v in (ka, kb):
                    if cap.get(v, 0) >= 1 and v not in visited:
                        visited.add(v)
                        if v not in match_v or dfs_match(match_v[v], visited):
                            match_v[v] = ei
                            return True
                return False

            for ei in unmatched:
                dfs_match(ei, set())
            for ei in need_arm:
                ka, kb = geom['edges'][ei][:2]
                if match_v.get(ka) == ei:
                    arm[ei] = (3, 2)
                elif match_v.get(kb) == ei:
                    arm[ei] = (2, 3)
                else:
                    arm[ei] = (2, 3) if rng.random() < 0.5 else (3, 2)
        else:
            for ei in need_arm:
                arm[ei] = (2, 3) if rng.random() < 0.5 else (3, 2)

    lig_count = Counter()
    vertices66 = 0
    n_skipped_vertices = 0
    for v, ws in incident.items():
        if len(ws) != 3:
            continue
        if sum(1 for w, _ in ws if w == 6) >= 2:
            vertices66 += 1
        if any(ei in uncertain for _, ei in ws):
            n_skipped_vertices += 1   # identity not unique (depends on an external ligand); skip
            continue
        arms_at_v = []
        for w, ei in ws:
            ka, kb = geom['edges'][ei][:2]
            a1, a2 = arm[ei]
            arms_at_v.append(a1 if ka == v else a2)
        lig_count[''.join(str(x) for x in sorted(arms_at_v))] += 1

    perim = Counter(int(lib.perimeter[idx]) for idx in solution.values())
    ew = Counter(int(x) for x in geom['edge_weights'])
    return dict(
        n_holes=len(solution),
        pairs55=pairs55,
        n_yellow_holes=len(yellow),
        vertices66=vertices66,
        ligand_counts=dict(lig_count),
        n_boundary_uncertain=n_skipped_vertices,
        perimeter_hist={str(k): v for k, v in sorted(perim.items())},
        edge_weight_hist={str(k): ew.get(k, 0) for k in (4, 5, 6)},
        geometry=geom,
    )


# ----------------------------------------------------------------------
# Structure factor (exact |F(q)|^2; same scope as the legacy FFT
# coordinate-extraction script)
# ----------------------------------------------------------------------

def structure_factor(points, q_max=1.0, q_pixels=401, weights=None,
                     mask_incircle=True):
    """Exact structure factor |F(q)|^2. points (N,2); weights optional
    scattering weights (e.g. edge lengths).

    mask_incircle=True masks by the incircle of the convex hull (x0.99) to
    suppress edge effects.
    Returns (intensity (q_pixels,q_pixels), q_1d).
    """
    from scipy.spatial import ConvexHull
    pts = np.asarray(points, dtype=float)
    w = np.ones(len(pts)) if weights is None else np.asarray(weights, dtype=float)
    pts = pts - (pts.max(axis=0) + pts.min(axis=0)) / 2
    if mask_incircle:
        hull = ConvexHull(pts)
        r_in = np.inf
        for s1, s2 in hull.simplices:
            p1, p2 = pts[s1], pts[s2]
            d = math.hypot(*(p2 - p1))
            if d > 0:
                r_in = min(r_in, abs(p1[0] * p2[1] - p2[0] * p1[1]) / d)
        r_in *= 0.99
        keep = np.linalg.norm(pts, axis=1) <= r_in
        pts, w = pts[keep], w[keep]
    q1d = np.linspace(-q_max, q_max, q_pixels)
    QX, QY = np.meshgrid(q1d, q1d, indexing='xy')
    qx, qy = QX.ravel(), QY.ravel()
    F = np.empty(len(qx))
    B = 4000
    for i in range(0, len(qx), B):
        ph = np.outer(qx[i:i + B], pts[:, 0]) + np.outer(qy[i:i + B], pts[:, 1])
        F[i:i + B] = np.abs((w * np.exp(-1j * ph)).sum(axis=1)) ** 2
    return F.reshape(q_pixels, q_pixels), q1d


# ----------------------------------------------------------------------
# Reference systems (REF-C crystal / REF-A amorphous)
# ----------------------------------------------------------------------

def crystal_disk_points(n_rings, scale=1.0):
    """REF-C: geometry of the all-4 (pure 222) honeycomb disk (same placement
    machinery as assembly solutions)."""
    lib = build_library(('222',))
    sol = {c: 0 for c in hex_coords(n_rings - 1)}
    return extract_geometry(sol, lib, scale=scale)


def amorphous_disk_points(n_rings, jitter=10.0, scale=1.0, seed=None):
    """REF-A: all-5 equilateral honeycomb with random vertex jitter
    (replicates the amorphous graphene generator).

    Hole center spacing = 5*sqrt(3) (equilateral hexagon); vertex jitter drawn
    from the U(0,jitter) disk, as in the legacy scripts.
    """
    rng = random.Random(seed)
    R = n_rings - 1
    d = scale * 5 * math.sqrt(3)
    centers = []
    for q in range(-R - 1, R + 2):
        for r in range(-R - 1, R + 2):
            if abs(q + r) <= R + 1:
                centers.append((d * (q + r / 2.0), d * r * math.sqrt(3) / 2.0))
    node = {}
    for cx, cy in centers:
        for i in range(6):
            a = math.radians(i * 60 + 30)
            p = (round(cx + scale * 5 * math.cos(a), 3),
                 round(cy + scale * 5 * math.sin(a), 3))
            node[p] = 1
    cache = {}
    out = []
    for p in node:
        if jitter > 0:
            if p not in cache:
                a = rng.uniform(0, 2 * math.pi)
                rad = rng.uniform(0, jitter * scale)
                cache[p] = (rad * math.cos(a), rad * math.sin(a))
            dx, dy = cache[p]
            out.append((p[0] + dx, p[1] + dy))
        else:
            out.append(p)
    return np.array(out)



# ----------------------------------------------------------------------
# Transfer-matrix exact counting (arm-field scope, hole-by-hole BFS; P0-A2b)
# ----------------------------------------------------------------------

def _tm_ekey(q, r, i, W):
    """Physical edge key = sorted hole pair (direction-free; row-relative
    coordinates, periodic q)."""
    dq, dr = DIRECTIONS[i]
    a = (q % W, r)
    b = ((q + dq) % W, r + dr)
    return (a, b) if a <= b else (b, a)


def _tm_vkey(q, r, i, W):
    """Physical vertex key = three holes sorted((q%W,r), (neighbor i),
    (neighbor i+1))."""
    j = (i + 1) % 6
    return tuple(sorted([(q % W, r),
                         ((q + DIRECTIONS[i][0]) % W, r + DIRECTIONS[i][1]),
                         ((q + DIRECTIONS[j][0]) % W, r + DIRECTIONS[j][1])]))


def tm_count_strip(W, lib, n_rows=8, verbose=False, max_states=None):
    """Exact solution count Omega of a strip of width W and length n_rows with
    open boundaries (arm-field scope).

    Each complete path = one assembly satisfying all constraints: edge
    matching, equiangular closure (library), weight-5 direction enumeration,
    vertex identity in the alphabet. q is the periodic direction; the row
    direction is open. Returns (omega_total, omega_hist) -- omega_hist is the
    cumulative solution count after each completed row (length n_rows), used
    to check row-by-row entropy convergence.
    """
    allowed_ids = set(tuple(sorted(LIGAND_ARMS[n])) for n in lib.alphabet)
    libw = lib.weights
    K = lib.size
    W_ = W

    def shift_key(k):
        """Re-basing at row end: key coordinates r -= 1. Vertex key = three-hole
        tuple; edge key = (q,r,i)."""
        # vertex key = three-hole tuple; edge key = two-hole tuple -- both are
        # sequences of hole-coordinate tuples
        return tuple(sorted((qq, rr - 1) for qq, rr in k))

    init = (0, (), ())                        # (q_pos, pend_sorted, edges_sorted)
    states = {init: 1}
    states_hist = []
    omega_hist = []
    total_steps = n_rows * W
    for step in range(total_steps):
        if max_states is not None and len(states) > max_states:
            return None, states_hist
        new_states = {}
        for (qp, pend_s, edges_s), cnt in states.items():
            pend = {vk: list(ar) for vk, ar in pend_s}
            edges = {ek: w for ek, w in edges_s}
            first_row = all(rr >= 0 for vk in pend for _, rr in vk) and not edges
            # fixed-edge constraints: 3, 4, 5 (at q=W-1 also 0, usually already
            # enumerated at q=0)
            fixed = {}
            for i in range(6):
                ek = _tm_ekey(qp, 0, i, W_)
                if ek in edges:
                    fixed[i] = edges[ek]
            # enumerate tiles
            for t in range(K):
                w = libw[t]
                if any(int(w[i]) != wi for i, wi in fixed.items()):
                    continue
                # collect the state of the hole's 6 edges: new = not in edges
                # and not a first-row dangling edge (4,5)
                new_edges = []
                skip = False
                for i in range(6):
                    ek = _tm_ekey(qp, 0, i, W_)
                    if ek in edges or i in fixed:
                        continue
                    if i in (4, 5) and first_row and ek not in edges:
                        new_edges.append(i)            # first-row dangling: enumerate this hole's end arm number
                        continue
                    new_edges.append(i)
                # enumerate directions of the w=5 edges among the new edges
                fives = [i for i in new_edges if int(w[i]) == 5]
                for bits in range(1 << len(fives)):
                    added = []       # (vkey, arm) undo records
                    added_e = []
                    dead = False
                    for i in new_edges:
                        ek = _tm_ekey(qp, 0, i, W_)
                        wi = int(w[i])
                        if wi == 4:
                            a1 = a2 = 2
                        elif wi == 6:
                            a1 = a2 = 3
                        else:
                            b = (bits >> fives.index(i)) & 1
                            a1, a2 = (2, 3) if b == 0 else (3, 2)
                        edges[ek] = wi
                        added_e.append(ek)
                        for vk, a in ((_tm_vkey(qp, 0, (i - 1) % 6, W_), a1),
                                      (_tm_vkey(qp, 0, i, W_), a2)):
                            pend.setdefault(vk, []).append(a)
                            added.append((vk, a))
                            if len(pend[vk]) == 3:
                                if tuple(sorted(pend[vk])) not in allowed_ids:
                                    dead = True
                                    break
                        if dead:
                            break
                    if not dead:
                        if qp == W_ - 1:
                            # row end: re-base + drop lagging vertices
                            new_pend = {}
                            for vk, ar in pend.items():
                                sk = shift_key(vk)
                                if all(rr >= -1 for _, rr in sk):
                                    new_pend[sk] = tuple(ar)
                            new_edges_s = tuple(sorted((shift_key(ek), wq)
                                                       for ek, wq in edges.items()
                                                       if min(rr for _, rr in ek) >= -1))
                            ns = (0, tuple(sorted((vk, tuple(ar))
                                                  for vk, ar in new_pend.items())),
                                  new_edges_s)
                            new_states[ns] = new_states.get(ns, 0) + cnt
                        else:
                            ns = (qp + 1,
                                  tuple(sorted((vk, tuple(ar)) for vk, ar in pend.items())),
                                  tuple(sorted(edges.items())))
                            new_states[ns] = new_states.get(ns, 0) + cnt
                    # undo
                    for vk, a in reversed(added):
                        pend[vk].pop()
                        if not pend[vk]:
                            del pend[vk]
                    for ek in added_e:
                        del edges[ek]
        states = new_states
        states_hist.append(len(states))
        if verbose:
            print(f"    step {step+1}/{total_steps}: states={len(states)}", flush=True)
        if not states:
            return 0, states_hist
        if (step + 1) % W_ == 0:
            omega_hist.append(sum(states.values()))
    omega = sum(states.values())
    return omega, omega_hist


def tm_entropy_density(W, lib, n_rows=8, verbose=False):
    """Strip entropy density s(W) = ln Omega / (W*n_rows), with convergence
    diagnostics (per-row cumulative lnOmega/LW)."""
    import math
    omega, hist = tm_count_strip(W, lib, n_rows=n_rows, verbose=verbose)
    out = dict(W=W, n_rows=n_rows, omega=omega,
               s_per_hole=math.log(omega) / (W * n_rows) if omega > 0 else float('-inf'),
               states_final=hist[-1] if hist else 0)
    return out


def brute_count_strip(W, n_rows, lib):
    """Brute-force enumerator: exact count of the same definition
    (open-boundary strip of width W, arm field), for cross-validation."""
    allowed_ids = set(tuple(sorted(LIGAND_ARMS[n])) for n in lib.alphabet)
    libw = lib.weights
    W_ = W
    holes = [(q, r) for r in range(n_rows) for q in range(W_)]
    count = 0

    def rec(idx, edges, pend):
        nonlocal count
        if idx == len(holes):
            count += 1
            return
        q, r = holes[idx]
        fixed = {}
        for i in range(6):
            ek = _tm_ekey(q, r, i, W_)
            if ek in edges:
                fixed[i] = edges[ek]
        for t in range(lib.size):
            w = libw[t]
            if any(int(w[i]) != wi for i, wi in fixed.items()):
                continue
            new_edges = [i for i in range(6)
                         if _tm_ekey(q, r, i, W_) not in edges and i not in fixed]
            fives = [i for i in new_edges if int(w[i]) == 5]
            for bits in range(1 << len(fives)):
                added = []
                added_e = []
                dead = False
                for i in new_edges:
                    ek = _tm_ekey(q, r, i, W_)
                    wi = int(w[i])
                    if wi == 4:
                        a1 = a2 = 2
                    elif wi == 6:
                        a1 = a2 = 3
                    else:
                        b = (bits >> fives.index(i)) & 1
                        a1, a2 = (2, 3) if b == 0 else (3, 2)
                    edges[ek] = wi
                    added_e.append(ek)
                    for vk, a in ((_tm_vkey(q, r, (i - 1) % 6, W_), a1),
                                  (_tm_vkey(q, r, i, W_), a2)):
                        pend.setdefault(vk, []).append(a)
                        added.append((vk, a))
                        if len(pend[vk]) == 3 and tuple(sorted(pend[vk])) not in allowed_ids:
                            dead = True
                            break
                    if dead:
                        break
                if not dead:
                    rec(idx + 1, edges, pend)
                for vk, a in reversed(added):
                    pend[vk].pop()
                    if not pend[vk]:
                        del pend[vk]
                for ek in added_e:
                    del edges[ek]

    rec(0, {}, {})
    return count


# ----------------------------------------------------------------------
# Transfer matrix, packed encoding (state = bytes key; ~60x memory reduction;
# P0-A2b v2)
# ----------------------------------------------------------------------

def tm_count_strip_packed(W, lib, n_rows=10, max_states=None, verbose=False):
    """Memory-optimized version of tm_count_strip: state dict keys = fixed-length
    struct bytes.

    Logic identical to tm_count_strip (edge matching, closure library,
    weight-5 direction enumeration, vertex identity check, open boundary,
    row-end re-basing); only the state representation changes to packed bytes.
    Returns (omega_total, omega_hist).
    """
    import struct
    allowed_ids = set(tuple(sorted(LIGAND_ARMS[n])) for n in lib.alphabet)
    libw = lib.weights
    K = lib.size
    W_ = W
    # fixed-length layout: qp (1B) + npend (1B) + per pending vertex
    #   [narms (1B) + 3 x (arm 1B) + vkey 3 x (q 1B, r 1B)]
    #   + nedge (1B) + per edge [two holes 4 x (q 1B, r 1B) + w 1B]
    VREC = 3 + 6          # 9B = 3B arms + 3 x (q,r)
    EREC = 4 + 1          # 5B
    HDR = 2

    def pack(qp, pend, edges):
        parts = [struct.pack('BB', qp, len(pend))]
        for vk in sorted(pend):
            arms = list(pend[vk]) + [0] * (3 - len(pend[vk]))
            qrs = b''.join(struct.pack('bb', int(qq) + 64, int(rr) + 64) for qq, rr in vk)
            parts.append(struct.pack('BBB', *arms) + qrs)
        ek = sorted(edges)
        parts.append(struct.pack('B', len(ek)))
        for e in ek:
            ka, kb = e
            parts.append(struct.pack('bb', int(ka[0]) + 64, int(ka[1]) + 64)
                         + struct.pack('bb', int(kb[0]) + 64, int(kb[1]) + 64)
                         + struct.pack('B', edges[e]))
        return b''.join(parts)

    def unpack(b):
        qp = b[0]
        npend = b[1]
        off = HDR
        pend = {}
        for _ in range(npend):
            arms = tuple(a for a in b[off:off + 3] if a)
            vk = tuple(sorted((b[off + 3 + 2 * j] - 64,
                               b[off + 4 + 2 * j] - 64) for j in range(3)))
            off += VREC
            pend[vk] = list(arms)
        nedge = b[off]
        off += 1
        edges = {}
        for _ in range(nedge):
            ka = (b[off] - 64, b[off + 1] - 64)
            kb = (b[off + 2] - 64, b[off + 3] - 64)
            edges[(ka, kb)] = b[off + 4]
            off += EREC
        return qp, pend, edges

    init = pack(0, {}, {})
    states = {init: 1}
    omega_hist = []
    total_steps = n_rows * W_

    for step in range(total_steps):
        if max_states is not None and len(states) > max_states:
            return None, omega_hist
        new_states = {}
        for key, cnt in states.items():
            qp, pend_d, edges_d = unpack(key)
            pend = {vk: list(ar) for vk, ar in pend_d.items()}
            edges = dict(edges_d)
            first_row = (step < W_)
            fixed = {}
            for i in range(6):
                ek = _tm_ekey(qp, 0, i, W_)
                if ek in edges:
                    fixed[i] = edges[ek]
            for t in range(K):
                w = libw[t]
                if any(int(w[i]) != wi for i, wi in fixed.items()):
                    continue
                new_edges = []
                for i in range(6):
                    ek = _tm_ekey(qp, 0, i, W_)
                    if ek in edges or i in fixed:
                        continue
                    if i in (4, 5) and first_row:
                        new_edges.append(i)     # first-row dangling: only register this hole's end arm number
                        continue
                    new_edges.append(i)
                fives = [i for i in new_edges if int(w[i]) == 5]
                for bits in range(1 << len(fives)):
                    added = []
                    added_e = []
                    dead = False
                    for i in new_edges:
                        ek = _tm_ekey(qp, 0, i, W_)
                        wi = int(w[i])
                        if wi == 4:
                            a1 = a2 = 2
                        elif wi == 6:
                            a1 = a2 = 3
                        else:
                            bbit = (bits >> fives.index(i)) & 1
                            a1, a2 = (2, 3) if bbit == 0 else (3, 2)
                        if i not in (4, 5) or not first_row:
                            edges[ek] = wi
                        added_e.append((ek, i in (4, 5) and first_row))
                        for vk, a in ((_tm_vkey(qp, 0, (i - 1) % 6, W_), a1),
                                      (_tm_vkey(qp, 0, i, W_), a2)):
                            pend.setdefault(vk, []).append(a)
                            added.append((vk, a))
                            if len(pend[vk]) == 3 and tuple(sorted(pend[vk])) not in allowed_ids:
                                dead = True
                                break
                        if dead:
                            break
                    if not dead:
                        if qp == W_ - 1:
                            np_ = {}
                            for vk, ar in pend.items():
                                sk = tuple(sorted((qq, rr - 1) for qq, rr in vk))
                                if all(rr >= -1 for _, rr in sk):
                                    np_[sk] = tuple(ar)
                            ne = {}
                            for ek2, wq in edges.items():
                                sk = tuple(sorted((qq, rr - 1) for qq, rr in ek2))
                                if all(rr >= -1 for _, rr in sk):
                                    ne[sk] = wq
                            nk = pack(0, {vk: list(ar) for vk, ar in np_.items()}, ne)
                            new_states[nk] = new_states.get(nk, 0) + cnt
                        else:
                            nk = pack(qp + 1,
                                      {vk: list(ar) for vk, ar in pend.items()},
                                      edges)
                            new_states[nk] = new_states.get(nk, 0) + cnt
                    for vk, a in reversed(added):
                        pend[vk].pop()
                        if not pend[vk]:
                            del pend[vk]
                    for ek2, dangling in reversed(added_e):
                        if not dangling:
                            del edges[ek2]
        states = new_states
        if verbose:
            print(f"    step {step+1}/{total_steps}: states={len(states)}", flush=True)
        if not states:
            return 0, omega_hist
        if (step + 1) % W_ == 0:
            omega_hist.append(sum(states.values()))
    return sum(states.values()), omega_hist
