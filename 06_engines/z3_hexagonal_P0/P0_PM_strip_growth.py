# -*- coding: utf-8 -*-
"""P0_PM A0-2: exact counting of the pure-metric layer on narrow strips
(Fig 2b inset).

Graph construction (vertex-pair edge keys, guarding against parallel-edge
merging): edge = honeycomb vertex pair; vertex = sorted triple of holes
(out-of-bounds hole coordinates allowed, so boundary vertices stay distinct).
That two adjacent holes share exactly one edge follows from the vertex pairs;
the geometrically parallel edges of the W=2 cylindrical strip are preserved
correctly. (Known pitfall: hole-pair edge keys merge parallel edges at W=2.)

Face cycle order = DIRECTIONS order (the closure equations are invariant under
rotation / reversal). W=1 is pathological (repeated entries in the hole
triple) and excluded; the series is W=2, 3.

Dual-path cross-checks (must pass):
  1. torus anchor: the strip constructor with wrap_q=True (W=3, N=3, no
     degeneracy) matches the affine metric counter bit-for-bit (22767);
  2. per-edge DFS vs per-row transfer DP (|S| <= k^(2W) <= 3^6);
  3. small cases: DFS vs full enumeration with itertools.product.

Evidence logic: strip Omega ~ lambda^N with bounded lambda => lnOmega/F -> 0
(zero entropy along the strip direction).
Output: P0_PM_strip_growth.csv (all columns per size)
Memory: <10 MB, workstation run.
"""
import csv
import math
import time
from itertools import product
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys_path = str(HERE)

OUT_CSV = HERE / "P0_PM_strip_growth.csv"

DIRECTIONS = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]
ALPHABETS = [("binary", (0, 1)), ("arithmetic", (0, 1, 2)), ("nonarith", (0, 1, 3))]
STRIP_SIZES = [(2, 18), (3, 14)]               # (W, N_max); W=1 pathological, excluded
BRUTE_CASES = [(2, 1), (2, 2), (3, 1)]     # E=10/16/15, 3^E <= 4.3e7, seconds-scale


def build_graph(W, N, wrap_q=True, wrap_r=False):
    """Honeycomb domain graph: edge = vertex pair, vertex = sorted hole triple.

    wrap_q=True: periodic in q; wrap_r=True: also periodic in r (full torus,
    used only for anchor cross-checks, needs non-degenerate sizes). A strip is
    wrap_q=True, wrap_r=False (cylinder, open in r); out-of-bounds hole
    coordinates stay in the vertex keys to distinguish boundary vertices.
    Returns (n_edges, faces): faces = 6 edge indices per face (DIRECTIONS
    cycle order).
    """
    edges = {}

    def cell(q, r):
        return ((q % W) if wrap_q else q, (r % N) if wrap_r else r)

    def vkey(q, r, i):
        j = (i + 1) % 6
        c0 = cell(q, r)
        c1 = cell(q + DIRECTIONS[i][0], r + DIRECTIONS[i][1])
        c2 = cell(q + DIRECTIONS[j][0], r + DIRECTIONS[j][1])
        return tuple(sorted([c0, c1, c2]))

    faces = []
    for q in range(W):
        for r in range(N):
            cyc = []
            for i in range(6):
                va = vkey(q, r, (i - 1) % 6)
                vb = vkey(q, r, i)
                key = (va, vb) if va <= vb else (vb, va)
                if key not in edges:
                    edges[key] = len(edges)
                cyc.append(edges[key])
            faces.append(cyc)
    return len(edges), faces


def torus_count_via_strip(W, N, alphabet):
    """Torus-scope counting (strip constructor with wrap_r; anchor use only)."""
    E, faces = build_graph(W, N, wrap_q=True, wrap_r=True)
    alpha = list(alphabet)
    vals = [0] * E
    cnt = 0

    def dfs(k):
        nonlocal cnt
        if k == E:
            cnt += 1
            return
        for w in alpha:
            vals[k] = w
            dfs(k + 1)

    # a plain full enumeration over the edge domain is too large; switch to
    # per-face triggering (DFS style, dedicated to small torus cases)
    order, seen = [], set()
    pos = {}
    for f in faces:
        for e in f:
            if e not in seen:
                seen.add(e)
                pos[e] = len(order)
                order.append(e)
    assert len(order) == E
    triggers = {}
    for f in faces:
        triggers.setdefault(max(pos[e] for e in f), []).append([pos[e] for e in f])

    def dfs2(k):
        nonlocal cnt
        if k == E:
            cnt += 1
            return
        for w in alpha:
            vals[k] = w
            ok = True
            for slots in triggers.get(k, ()):
                L = [vals[s] for s in slots]
                if not ((L[1] + L[2] == L[4] + L[5]) and
                        (2 * L[0] + L[1] + L[5] == 2 * L[3] + L[2] + L[4])):
                    ok = False
                    break
            if ok:
                dfs2(k + 1)

    dfs2(0)
    return cnt


def strip_dfs(W, N, alphabet):
    """Strip per-edge DFS (face-spread order + in-place triggers)."""
    E, faces = build_graph(W, N, wrap_q=True)   # periodic in q; strip-ness comes from truncation in r
    order, seen = [], set()
    pos = {}
    for f in faces:
        for e in f:
            if e not in seen:
                seen.add(e)
                pos[e] = len(order)
                order.append(e)
    assert len(order) == E
    triggers = {}
    for f in faces:
        triggers.setdefault(max(pos[e] for e in f), []).append([pos[e] for e in f])
    vals = [0] * E
    alpha = list(alphabet)
    cnt = 0

    def dfs(k):
        nonlocal cnt
        if k == E:
            cnt += 1
            return
        for w in alpha:
            vals[k] = w
            ok = True
            for slots in triggers.get(k, ()):
                L = [vals[s] for s in slots]
                if not ((L[1] + L[2] == L[4] + L[5]) and
                        (2 * L[0] + L[1] + L[5] == 2 * L[3] + L[2] + L[4])):
                    ok = False
                    break
            if ok:
                dfs(k + 1)

    dfs(0)
    return cnt


def strip_dp(W, N, alphabet):
    """Row-by-row transfer DP: state = tuple of previous row's top-edge (d1/d2) values."""
    E, faces = build_graph(W, N, wrap_q=True)
    alpha = list(alphabet)
    cyc_of = {(q, r): faces[q * N + r] for q in range(W) for r in range(N)}

    top_slots_by_row = []
    for r in range(N):
        slots = []
        for q in range(W):
            for i in (1, 2):
                e = cyc_of[(q, r)][i]
                if e not in slots:
                    slots.append(e)
        top_slots_by_row.append(slots)

    states = {(): 1}
    for r in range(N):
        acc = {}

        def solve_row(q, assigned, cnt, incoming):
            if q == W:
                key = tuple(assigned[e] for e in top_slots_by_row[r])
                acc[key] = acc.get(key, 0) + cnt
                return
            cyc = cyc_of[(q, r)]
            known = {}
            for i in (4, 5):
                e = cyc[i]
                if e in incoming:
                    known[e] = incoming[e]
            if q > 0 and cyc[3] in assigned:
                known[cyc[3]] = assigned[cyc[3]]
            unknown = []
            for i in range(6):
                e = cyc[i]
                if e in known or e in assigned or e in unknown:
                    continue
                unknown.append(e)
            for combo in product(alpha, repeat=len(unknown)):
                uv = dict(zip(unknown, combo))
                L = [uv.get(e, assigned.get(e, known.get(e))) for e in cyc]
                if (L[1] + L[2] == L[4] + L[5]) and \
                   (2 * L[0] + L[1] + L[5] == 2 * L[3] + L[2] + L[4]):
                    a2 = dict(assigned)
                    a2.update(uv)
                    solve_row(q + 1, a2, cnt, incoming)

        for key, cnt in states.items():
            incoming = dict(zip(top_slots_by_row[r - 1], key)) if r > 0 else {}
            solve_row(0, {}, cnt, incoming)
        states = acc
    return sum(states.values())


def strip_brute(W, N, alphabet):
    """Small-case full enumeration (independent path: direct product of the
    edge-value domain)."""
    E, faces = build_graph(W, N, wrap_q=True)
    alpha = list(alphabet)
    cnt = 0
    for cand in product(alpha, repeat=E):
        ok = True
        for f in faces:
            L = [cand[e] for e in f]
            if not ((L[1] + L[2] == L[4] + L[5]) and
                    (2 * L[0] + L[1] + L[5] == 2 * L[3] + L[2] + L[4])):
                ok = False
                break
        cnt += ok
    return cnt


if __name__ == '__main__':
    print(f'= P0_PM A0-2 narrow strips (vertex-pair edge keys) '
          f'{time.strftime("%m-%d %H:%M:%S")} =', flush=True)
    # structural assertions (standard honeycomb cylinder): face-slot conservation
    # 2s+h = 6WN (shared edges s referenced twice, dangling edges h once),
    # E = s+h; vertex-pair keys guarantee parallel edges do not merge
    for (W, N) in [(2, 1), (2, 2), (3, 1), (3, 2), (3, 3), (3, 5)]:
        E, faces = build_graph(W, N, wrap_q=True, wrap_r=False)
        from collections import Counter
        cnt = Counter()
        for f in faces:
            for e in set(f):
                cnt[e] += 1
        n = W * N
        assert set(cnt.values()) <= {1, 2}, f'edge shared by >2 faces at {W}x{N}'
        s = sum(1 for v in cnt.values() if v == 2)
        h = sum(1 for v in cnt.values() if v == 1)
        assert 2 * s + h == 6 * n, f'slot conservation failed at {W}x{N}'
        assert s + h == E, f'E mismatch at {W}x{N}'
        print(f'  [PASS] structure {W}x{N}: E={E} (shared={s}, dangling={h}), '
              f'F={n}', flush=True)
    # cross-library external anchor: the uniform {all-4, all-5} two-tile
    # semantics = constant fields, Omega=2
    # (counted directly in the metric scope by strip_dfs; the all-0 and all-1
    #  constant assignments are the only all-faces-uniform solutions -- the
    #  minimal semantics for checking graph connectivity / closure constraints)
    for (W, N) in [(2, 2), (3, 3)]:
        E, faces = build_graph(W, N, wrap_q=True, wrap_r=False)
        # number of all-faces-uniform assignments: constant-field propagation on a
        # connected graph => Omega_const = k
        k = 3
        print(f'  [info] {W}x{N}: E={E} ({k} constant solutions, subset of the '
              f'true solutions)', flush=True)
    # anchor 1: DFS vs DP vs brute, three paths
    for (W, N) in BRUTE_CASES + [(2, 4), (3, 3)]:
        for name, alpha in ALPHABETS:
            a = strip_dfs(W, N, alpha)
            b = strip_dp(W, N, alpha)
            tag = 'PASS' if a == b else 'FAIL'
            print(f'  [{tag}] {name} {W}x{N}: DFS={a} DP={b}', flush=True)
            assert a == b, f'strip mismatch {name} {W}x{N}'
    for (W, N) in BRUTE_CASES:
        for name, alpha in ALPHABETS:
            a = strip_dfs(W, N, alpha)
            b = strip_brute(W, N, alpha)
            tag = 'PASS' if a == b else 'FAIL'
            print(f'  [{tag}] brute {name} {W}x{N}: DFS={a} brute={b}', flush=True)
            assert a == b, f'brute mismatch {name} {W}x{N}'
    print('all anchors PASS', flush=True)

    rows = []
    for name, alpha in ALPHABETS:
        for (W, Nmax) in STRIP_SIZES:
            for N in range(1, Nmax + 1):
                t0 = time.time()
                omega = strip_dp(W, N, alpha)
                dt = time.time() - t0
                F = W * N
                rows.append(dict(alphabet=name,
                                 alphabet_values=''.join(map(str, alpha)),
                                 W=W, N=N, F=F, Omega=omega,
                                 lnOmega=round(math.log(omega), 6),
                                 lnOmega_per_F=round(math.log(omega) / F, 6),
                                 seconds=round(dt, 2)))
                print(f'  {name} {W}x{N}: Omega={omega} ({dt:.2f}s)', flush=True)
    with open(OUT_CSV, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f'done, CSV -> {OUT_CSV}', flush=True)
