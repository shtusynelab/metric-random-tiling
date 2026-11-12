# -*- coding: utf-8 -*-
"""Numerical cross-check of the affine invariance of edge-value alphabets (2026-09-28).

Question: for the disordered honeycomb tiling with 3 edge-length values, is the
metric entropy the same as for {4,5,6}?
Checks:
  A. Arithmetic-progression class is affine invariant:
     Omega({4,5,6}) = Omega({3,4,5}) = Omega({2,3,4}) = Omega({5,6,7})
     = Omega({8,10,12}) = Omega({3,5,7}) (pointwise equal)
  B. Non-AP counterexamples: Omega({3,4,6}), Omega({4,6,7}) strictly smaller.
Construction: Lx x Ly honeycomb torus (A_{i,j}=(i,j), B_{i,j}=(i+.5,j+.5)),
E=3*Lx*Ly, F=Lx*Ly; face closure uses the same two equations as the library's
closure_ok (eqA: L1+L2=L4+L5; eqB: 2L0+L1+L5=2L3+L2+L4).
Memory: DFS assigns edge by edge, never materializes a state table; peak <10 MB.
"""
import math
import time
from itertools import product

def build_honeycomb_torus(Lx, Ly):
    """Return the edges list and faces (6 edge indices per face, in face-boundary walk order)."""
    def A(i, j): return ('A', i % Lx, j % Ly)
    def B(i, j): return ('B', i % Lx, j % Ly)
    edge_set = []          # (A corner, B corner)
    edge_index = {}
    def add_edge(u, v):
        key = frozenset((u, v))
        if key not in edge_index:
            edge_index[key] = len(edge_set)
            edge_set.append((u, v))
        return edge_index[key]
    for i in range(Lx):
        for j in range(Ly):
            add_edge(A(i, j), B(i, j))
            add_edge(A(i + 1, j), B(i, j))
            add_edge(A(i, j + 1), B(i, j))
    # adjacency table (with geometric angles, for the planar rotation order)
    adj = {v: [] for v in edge_index and set() or set()}
    vertices = set()
    for u, v in edge_set:
        vertices.add(u); vertices.add(v)
    pos = {}
    for i in range(Lx):
        for j in range(Ly):
            pos[('A', i, j)] = (float(i), float(j))
            pos[('B', i, j)] = (i + 0.5, j + 0.5)
    adj = {v: [] for v in vertices}
    for ei, (u, v) in enumerate(edge_set):
        adj[u].append((v, ei)); adj[v].append((u, ei))
    def angle_of(v, w):
        # direction angle of w relative to v; torus displacement is reduced to the
        # nearest integer lattice (unique when |Lx|,|Ly| >= 2)
        pv, pw = pos[v], pos[w]
        dx, dy = pw[0] - pv[0], pw[1] - pv[1]
        if dx > Lx / 2: dx -= Lx
        if dx < -Lx / 2: dx += Lx
        if dy > Ly / 2: dy -= Ly
        if dy < -Ly / 2: dy += Ly
        return math.atan2(dy, dx)
    rot = {}  # v -> [(neighbor, edge index)] sorted CCW by angle
    for v in vertices:
        rot[v] = sorted(adj[v], key=lambda t: angle_of(v, t[0]))
    # face walk: start from a directed edge (u->v), take u's successor in v's CCW order
    faces = []
    visited = set()
    for (u, v0), ei0 in zip(edge_set, range(len(edge_set))):
        for rev in (False, True):
            start = (v0, u) if rev else (u, v0)
            if start in visited:
                continue
            walk = []
            cur = start
            while cur not in visited:
                visited.add(cur)
                a, b = cur
                walk.append((a, b))
                order = rot[b]
                idx = next(k for k, (w, _) in enumerate(order) if w == a)
                nxt = order[(idx + 1) % len(order)]
                cur = (b, nxt[0])
            if len(walk) != 6:
                raise AssertionError(f'face walk length {len(walk)} != 6 at {start}')
            eidx = []
            for a, b in walk:
                key = frozenset((a, b))
                eidx.append(edge_index[key])
            if len(set(eidx)) != 6:
                raise AssertionError('repeated edge in face (torus wrap degeneracy)')
            faces.append(eidx)
    V, E, F = len(vertices), len(edge_set), len(faces)
    assert V - E + F == 0, f'Euler: {V}-{E}+{F} != 0'
    assert F == Lx * Ly, f'F={F} != Lx*Ly={Lx*Ly}'
    assert len(visited) == 2 * E, 'directed edges not fully covered'
    return edge_set, faces

def count_assignments(Lx, Ly, alphabet, report=None):
    """DFS edge-by-edge assignment with per-face early pruning; counts the
    assignments satisfying both face-closure equations."""
    edge_set, faces = build_honeycomb_torus(Lx, Ly)
    E = len(edge_set)
    # edge reordering: lay out face by face so each face's two constraints
    # become decidable as early as possible
    order, seen = [], set()
    for f in faces:
        for e in f:
            if e not in seen:
                seen.add(e); order.append(e)
    assert len(order) == E
    pos_in_order = {e: k for k, e in enumerate(order)}
    # the two equations per face (indices of edges in walk order)
    eqA = [(f[1], f[2], f[4], f[5]) for f in faces]   # L1+L2=L4+L5
    eqB = [(f[0], f[1], f[5], f[3], f[2], f[4]) for f in faces]  # 2L0+L1+L5=2L3+L2+L4
    # edge -> (face, equation) trigger points: position in order of the last edge
    # each equation needs before it becomes decidable
    triggers = {}  # order position -> [(check type, terms as order positions)]
    for fi, (a, b, c, d) in enumerate(eqA):
        terms = tuple(pos_in_order[e] for e in (a, b, c, d))
        triggers.setdefault(max(terms), []).append(('A', terms))
    for fi, (e0, e1, e5, e3, e2, e4) in enumerate(eqB):
        terms = tuple(pos_in_order[e] for e in (e0, e1, e5, e3, e2, e4))
        triggers.setdefault(max(terms), []).append(('B', terms))
    vals = [0] * E
    alpha = list(alphabet)
    count = 0
    def dfs(k):
        nonlocal count
        if k == E:
            count += 1
            return
        for w in alpha:
            vals[k] = w
            ok = True
            for typ, terms in triggers.get(k, ()):
                if typ == 'A':
                    a, b, c, d = terms
                    if vals[a] + vals[b] != vals[c] + vals[d]:
                        ok = False; break
                else:
                    e0, e1, e5, e3, e2, e4 = terms
                    if 2 * vals[e0] + vals[e1] + vals[e5] != 2 * vals[e3] + vals[e2] + vals[e4]:
                        ok = False; break
            if ok:
                dfs(k + 1)
    t0 = time.time()
    dfs(0)
    if report:
        report(f'  ({Lx}x{Ly}) E={E} F={len(faces)} done in {time.time()-t0:.1f}s')
    return count

def brute_count(Lx, Ly, alphabet):
    """Independent anchor by full enumeration of the small (2,2) case."""
    edge_set, faces = build_honeycomb_torus(Lx, Ly)
    E = len(edge_set)
    cnt = 0
    for cand in product(alphabet, repeat=E):
        ok = True
        for f in faces:
            L = [cand[e] for e in f]
            if not ((L[1] + L[2] == L[4] + L[5]) and
                    (2 * L[0] + L[1] + L[5] == 2 * L[3] + L[2] + L[4])):
                ok = False; break
        cnt += ok
    return cnt

if __name__ == '__main__':
    AP_ALPHABETS = [(4, 5, 6), (3, 4, 5), (2, 3, 4), (5, 6, 7), (8, 10, 12), (3, 5, 7)]
    NON_AP = [(3, 4, 6), (4, 6, 7)]
    sizes = [(2, 2), (3, 2), (3, 3)]
    print(f'{"alphabet":>12} | ' + ' | '.join(f'{s[0]}x{s[1]}' for s in sizes) + ' | lnOmega(3x3)/F')
    table = {}
    for alph in AP_ALPHABETS + NON_AP:
        row = []
        for (Lx, Ly) in sizes:
            row.append(count_assignments(Lx, Ly, alph))
        table[alph] = row
        tag = 'AP ' if alph in AP_ALPHABETS else 'non'
        lnF = math.log(row[-1]) / 9.0 if row[-1] > 0 else float('nan')
        print(f'{tag}{str(alph):>10} | ' + ' | '.join(f'{c:>9d}' for c in row) + f' | {lnF:.4f}')
    # anchor: arithmetic-progression classes equal pointwise
    ref = table[(4, 5, 6)]
    for alph in AP_ALPHABETS[1:]:
        assert table[alph] == ref, f'AP invariance FAILED: {alph} {table[alph]} vs {ref}'
    # non-AP: strictly fewer at larger sizes, and lower lnOmega/F (entropy-density direction)
    for alph in NON_AP:
        assert all(a <= b for a, b in zip(table[alph], ref)), f'non-AP exceeds AP: {alph}'
        assert table[alph][1] < ref[1] and table[alph][2] < ref[2], f'non-AP not strictly smaller at (3,2)/(3,3): {alph}'
    # brute-force anchor at (2,2)
    for alph in [(4, 5, 6), (3, 4, 6)]:
        b = brute_count(2, 2, alph)
        assert b == table[alph][0], f'brute mismatch {alph}: {b} vs {table[alph][0]}'
        print(f'brute anchor {alph} (2x2): {b} == DFS PASS')
    print('ALL ASSERTS PASSED')
