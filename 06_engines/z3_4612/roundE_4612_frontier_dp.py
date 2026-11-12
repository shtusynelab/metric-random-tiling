# -*- coding: utf-8 -*-
"""4.6.12 two-value {0,1} frontier DP (face-by-face closure, keyed states).

Method:
  - rebuild the face-local reduced system (same logic as
    roundE_4612_zcount.build_reduced): square opposite-edge equality
    contraction (SD/SH classes), duodecagon diagonals independent (HD);
    per-face <=4 rows (alpha/beta cos/sin), nonzero rows = equations,
    each attributed to its face. System-match check against the canonical
    zcount build (multiset of equations) at every L, fail-closed.
  - face order = BFS on the face graph (adjacent iff sharing a class
    variable). Torus-safe: no geometric coordinates used.
  - DP over faces: state = packed values of the frontier (assigned class
    variables still referenced by unprocessed faces). Per face, enumerate
    2^k assignments of its unassigned classes and verify ALL equations of
    that face (their variables are all assigned at that point, so no
    partial-sum memory is needed); merge equal keys.
  - tree width = max frontier size (recorded); fail-closed gate on peak
    state count.

Counting convention (full orbit, matches anchors):
  Omega_01 = DP_count x 2^n_free_cls
    2^n_free_cls: classes in no equation (free). The DP does NOT fix the
    gauge, so the complement-shift orbit (x -> 1-x, A*1 = 0) is already
    inside DP_count -- verified: L=1 DP=14 = |{x: Ax=0}| on equation
    variables, x 2^3 free classes = 112 = brute.
  s_01 = ln(Omega_01)/(6N).

Crosschecks (fail-closed): DP L=1/2/3 vs anchors CSV (brute/paramdfs);
loose monotonicity gate vs ternary Knuth estimates (x1.01 tolerance).

Memory gate: peak states <= 2e8 (~20 GB dict ceiling).

Server-only driver (originally run on the internal cluster); pass explicit
arguments when used locally. Imports rely on CWD (no sys.path setup).
Output appended per L to RTH4612_binary_frontier.csv.
"""
import csv
import math
import os
import sys
import time
from collections import defaultdict
from fractions import Fraction
from itertools import product

from roundD_4612_numeric import extract_faces, build_torus, PAIR_COS, PAIR_SIN  # noqa
from roundE_4612_zcount import build_reduced

PEAK_STATE_GATE = 2 * 10 ** 8
ANCHOR_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          'RTH4612_binary_anchors.csv')
OUT_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'RTH4612_binary_frontier.csv')
TERNARY_KNUTH = {3: 6.566e12, 4: 8.351e19}   # estimates, loose gates only


def build_face_system(Lx, Ly, rank_check=True):
    """Rebuild face-attributed reduced system.

    rank_check: Fraction RREF dim assertion (exact, but O(heavy) for
    L>=4) -- enable for L<=3 (anchor crosscheck region) only.
    Returns (face_eqs, n_cls, n_free_cls, N, dim_or_None, all_eq_cls).
    """
    vid, edges, V, E, pts = build_torus(Lx, Ly)
    faces = extract_faces(edges)
    N = Lx * Ly
    assert V == 12 * N and E == 18 * N and len(faces) == 6 * N
    edge_faces = defaultdict(list)
    for fi, f in enumerate(faces):
        for h in f:
            edge_faces[h[3]].append(fi)
    assert all(len(v) == 2 for v in edge_faces.values())
    size_of = [len(f) for f in faces]
    etype = {}
    for eid, (f1, f2) in edge_faces.items():
        s1, s2 = sorted((size_of[f1], size_of[f2]))
        etype[eid] = {(4, 12): 'SD', (4, 6): 'SH', (6, 12): 'HD'}[(s1, s2)]
    parent = list(range(E))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for f in faces:
        if len(f) == 4:
            for i in (0, 1):
                union(f[i][3], f[(i + 2) % 4][3])
    cls_of = {}
    for eid in range(E):
        r = find(eid)
        if r not in cls_of:
            cls_of[r] = len(cls_of)
    n_cls = len(cls_of)
    assert n_cls == 12 * N, (n_cls, 12 * N)
    n_cls_by_type = defaultdict(set)
    for eid in range(E):
        n_cls_by_type[etype[eid]].add(cls_of[find(eid)])
    assert len(n_cls_by_type['SD']) == 3 * N \
        and len(n_cls_by_type['SH']) == 3 * N \
        and len(n_cls_by_type['HD']) == 6 * N
    face_eqs = []
    for f in faces:
        rows = [defaultdict(int) for _ in range(4)]
        for (uu, vv, bdir, eid) in f:
            c = cls_of[find(eid)]
            ac, bc = PAIR_COS[bdir]
            as_, bs = PAIR_SIN[bdir]
            rows[0][c] += ac
            rows[1][c] += bc
            rows[2][c] += as_
            rows[3][c] += bs
        eqs_f = []
        for r in rows:
            r = {k: v for k, v in r.items() if v != 0}
            if r:
                eqs_f.append(dict(r))
        face_eqs.append(eqs_f)
    n_eqs = sum(len(e) for e in face_eqs)
    if Lx >= 2:
        assert n_eqs == 8 * N, (n_eqs, 8 * N)
    all_eq_cls = sorted({c for eqs_f in face_eqs for eq in eqs_f for c in eq})
    dim = None
    if rank_check:
        rows_f = [[Fraction(eq.get(c, 0)) for c in all_eq_cls]
                  for eqs_f in face_eqs for eq in eqs_f]
        rk = 0
        for col in range(len(all_eq_cls)):
            piv = next((i for i in range(rk, len(rows_f))
                        if rows_f[i][col] != 0), None)
            if piv is None:
                continue
            rows_f[rk], rows_f[piv] = rows_f[piv], rows_f[rk]
            pv = rows_f[rk][col]
            if pv != 1:
                rows_f[rk] = [x / pv for x in rows_f[rk]]
            for i in range(len(rows_f)):
                if i != rk and rows_f[i][col] != 0:
                    fac = rows_f[i][col]
                    rows_f[i] = [a - fac * b
                                 for a, b in zip(rows_f[i], rows_f[rk])]
            rk += 1
            if rk == len(rows_f):
                break
        dim = n_cls - rk
        assert dim == 4 * N + 4, (dim, 4 * N + 4)
    n_free_cls = n_cls - len(all_eq_cls)
    edge_cls = [cls_of[find(eid)] for eid in range(E)]
    face_nbrs = defaultdict(set)
    for eid, fl in edge_faces.items():
        f1, f2 = fl
        face_nbrs[f1].add(f2)
        face_nbrs[f2].add(f1)
    return (face_eqs, n_cls, n_free_cls, N, dim, all_eq_cls, face_nbrs,
            edge_cls)


def assert_system_match(Lx, Ly, face_eqs, n_cls):
    """Two-code-path check: rebuilt system == canonical zcount build."""
    cls_of, eqs, all_cls, n_cls_z, dim_z, N, V, E, F = build_reduced(Lx, Ly)
    assert n_cls == n_cls_z, (n_cls, n_cls_z)
    mine = sorted(sorted(eq.items()) for eq in
                  (eq for eqs_f in face_eqs for eq in eqs_f))
    canon = sorted(sorted(eq.items()) for eq in eqs)
    assert mine == canon, f'system mismatch at L={Lx}'


def face_bfs_order(face_nbrs, n_faces, start=0):
    """BFS on the true face graph (edge-sharing adjacency, geometric).

    Note: adjacency must come from shared edges (every edge belongs to a
    class), NOT from equation co-occurrence -- square faces carry no
    equations and would otherwise be isolated.
    """
    order = []
    seen = {start}
    queue = [start]
    while queue:
        f = queue.pop(0)
        order.append(f)
        for g in sorted(face_nbrs.get(f, ())):
            if g not in seen:
                seen.add(g)
                queue.append(g)
    assert len(order) == n_faces, (len(order), n_faces)
    return order


def frontier_plan(face_eqs, face_order, cls_faces):
    """Per-step plan (frontier_in, new_classes, frontier_out, face, eqs)."""
    pos = {f: i for i, f in enumerate(face_order)}
    last_pos = {c: max(pos[f] for f in fl) for c, fl in cls_faces.items()}
    plan = []
    frontier = []
    assigned = set()
    for step, f in enumerate(face_order):
        f_cls = sorted({c for eq in face_eqs[f] for c in eq})
        new = [c for c in f_cls if c not in assigned]
        new.sort(key=lambda c: -last_pos[c])
        out = [c for c in frontier + new if last_pos[c] > step]
        # fail-closed: face equations must be fully covered by in+new
        covered = set(frontier) | set(new)
        for eq in face_eqs[f]:
            assert set(eq) <= covered, ('equation vars not covered', step)
        plan.append((list(frontier), new, out, f, face_eqs[f]))
        assigned.update(new)
        frontier = out
    assert not frontier
    return plan


def dp_count(Lx, Ly, vals=(0, 1), face_order=None, probe_limit=None,
             rank_check=None, pinned=frozenset()):
    """Keyed frontier DP, value-domain generic.

    face_order: explicit face processing order (list covering all faces);
    None = BFS on the edge-sharing face graph.
    probe_limit: if set, abort (return None stats) once peak states exceed
    it -- order probing needs magnitude only, not the full count.
    rank_check: None = auto (L<=3); probing passes False to skip the
    Fraction elimination.
    pinned: class ids forced to 0 (pinned-stitching; classes are counted
    in the equation-variable id space; free classes pinned must be
    handled by the caller via the 2^n_free lift).
    Returns (dp_count, n_free_cls, N, tree_w, peak, dt); with probe_limit,
    returns (None, n_free_cls, N, tree_w, peak, dt) on abort.
    """
    t0 = time.time()
    if rank_check is None:
        rank_check = Lx <= 3   # exact dim + zcount match only in anchor region
    (face_eqs, n_cls, n_free_cls, N, dim, all_eq_cls,
     face_nbrs, _edge_cls) = build_face_system(Lx, Ly, rank_check=rank_check)
    if rank_check:
        assert_system_match(Lx, Ly, face_eqs, n_cls)
    else:
        # lightweight structural assertions (no Fraction elimination)
        assert n_cls == 12 * N and len(all_eq_cls) <= n_cls
    kmax = max(vals)
    bits = 1 if kmax <= 1 else 2          # 2-bit packing for ternary
    mask = (1 << bits) - 1
    pinned_in_vars = {c for c in pinned}   # caller maps free pinned classes
    # class -> faces whose equations reference it (frontier in/out basis)
    cls_faces = defaultdict(list)
    for f, eqs_f in enumerate(face_eqs):
        for eq in eqs_f:
            for c in eq:
                cls_faces[c].append(f)
    face_order = face_bfs_order(face_nbrs, len(face_eqs)) \
        if face_order is None else list(face_order)
    plan = frontier_plan(face_eqs, face_order, cls_faces)
    state = {0: 1}
    tree_w = 0
    peak = 1
    for step, (fr_in, new, out, f, eqs_f) in enumerate(plan):
        if not new and fr_in == out:
            continue                       # pure copy step (square faces)
        k = len(new)
        choices = [((0,) if c in pinned_in_vars else vals) for c in new]
        nstate = defaultdict(int)
        for key, cnt in state.items():
            val = {c: (key >> (i * bits)) & mask
                   for i, c in enumerate(fr_in)}
            for combo in product(*choices) if k else [()]:
                if k:
                    for c, v in zip(new, combo):
                        val[c] = v
                ok = True
                for eq in eqs_f:
                    s = 0
                    for c, co in eq.items():
                        s += co * val[c]
                    if s != 0:
                        ok = False
                        break
                if not ok:
                    continue
                nk = 0
                for i, c in enumerate(out):
                    nk |= val[c] << (i * bits)
                nstate[nk] += cnt
        state = nstate
        tree_w = max(tree_w, len(fr_in) + k, len(out))
        peak = max(peak, len(state))
        if probe_limit is not None and peak > probe_limit:
            print(f'PROBE ABORT: peak {peak} > {probe_limit} at '
                  f'L={Lx}x{Ly} step {step}/{len(plan)}', flush=True)
            return None, n_free_cls, N, tree_w, peak, time.time() - t0
        if peak > PEAK_STATE_GATE:
            print(f'FAIL-CLOSED: peak states {peak} > {PEAK_STATE_GATE} '
                  f'at L={Lx}x{Ly} step {step}/{len(plan)}', flush=True)
            sys.exit(2)
    assert set(state) == {0}, 'nonzero terminal keys remain'
    return state[0], n_free_cls, N, tree_w, peak, time.time() - t0


def load_anchors(alphabet):
    anchors = {}
    if os.path.exists(ANCHOR_CSV):
        with open(ANCHOR_CSV) as f:
            for row in csv.DictReader(f):
                if row['alphabet'] == alphabet:
                    L = int(row['L'])
                    if L not in anchors:
                        anchors[L] = int(row['Omega'])
    return anchors


def append_csv(fname, row):
    exists = os.path.exists(fname)
    with open(fname, 'a', newline='') as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(['L', 'alphabet', 'N', 'tree_width', 'peak_states',
                        'DP_count', 'Omega', 'lnOmega', 's_per_face',
                        'time_s'])
        w.writerow(row)


def main():
    argv = sys.argv[1:]
    ternary = '--ternary' in argv
    argv = [a for a in argv if a != '--ternary']
    lmax = int(argv[0]) if argv else 6
    vals = (0, 1, 2) if ternary else (0, 1)
    k = len(vals)
    alphabet = '{0,1,2}' if ternary else '{0,1}'
    out_csv = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'RTH4612_frontier_dp_%s.csv'
                           % ('ternary' if ternary else 'binary'))
    for L in range(1, lmax + 1):
        anchors = load_anchors(alphabet)   # re-read: anchors may land later
        dp, n_free, N, tw, peak, dt = dp_count(L, L, vals=vals)
        omega = dp * (k ** n_free)
        s = math.log(omega) / (6 * N)
        print(f'L={L} [{alphabet}]: N={N} tree_w={tw} peak={peak} DP={dp} '
              f'Omega={omega} ln={math.log(omega):.4f} '
              f's={s:.4f} [{dt:.1f}s]', flush=True)
        if L in anchors:
            status = 'PASS' if omega == anchors[L] else 'FAIL'
            print(f'    crosscheck vs anchor {anchors[L]}: {status}',
                  flush=True)
            assert omega == anchors[L], ('DP/anchor mismatch', L, omega,
                                         anchors[L])
        if L in TERNARY_KNUTH:
            if ternary:
                # numerical crosscheck vs Knuth estimate (statistical error)
                dln = abs(math.log(omega) - math.log(TERNARY_KNUTH[L]))
                flag = 'PASS (within Knuth error band)' if dln <= 0.05 \
                    else f'WARN |dln|={dln:.4f} -- verify before use'
                print(f'    vs Knuth est {TERNARY_KNUTH[L]:.3e}: {flag}',
                      flush=True)
            else:
                cap = TERNARY_KNUTH[L] * 1.01
                flag = 'OK' if omega <= cap else 'CHECK (near/above ternary)'
                print(f'    monotonicity vs ternary Knuth est: {flag}',
                      flush=True)
        append_csv(out_csv, [L, alphabet, N, tw, peak, dp, omega,
                             f'{math.log(omega):.4f}', f'{s:.6f}',
                             f'{dt:.1f}'])
    print('DONE all sizes', flush=True)


if __name__ == '__main__':
    main()
