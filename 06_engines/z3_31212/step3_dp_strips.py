# -*- coding: utf-8 -*-
"""Step 3: binary width-scaling adjudication (mid/cut separated frontier DP).

Structural basis (Step 1/2):
  the split-row class system decouples into two independent subsystems
  => Omega_01 = Omega_mid * Omega_cut:
    mid: mid-segment edges (3WL variables), 2 equations per dodecagon
        (xrow.b sqrt(3) part + yrow.a rational part),
        dim = L^2+2 (quadratic, extensive) => entropy carrier;
    cut: truncation classes (= triangles, 2WL variables), 2 equations per dodecagon
        (xrow.a rational part + yrow.b sqrt(3) part), dim = 3L-1 (linear)
        => s_cut <= (3L-1) ln2/(3WL) -> 0 (theorem level),
        no contribution to the asymptotic entropy => s_01(inf) = s_mid(inf).
  Equation form (mid, binary): per dodecagon {l0 + l1 = l3 + l4, l1 + l2 = l4 + l5}
  (equal adjacent-pair sums; isomorphic to the cos/sin coefficient RREF; the code
  uses the original coefficients, not hand-derived formulas).

Frontier DP: face graph = dodecagon adjacency graph (via shared keys: mid keys carry
2 faces, cut keys 3 faces); faces closed one by one in BFS order; state = frontier
key-value vector, merged by key; torus wrap-around closes automatically (frontier
must be empty after all faces are processed, asserted). No gauge convention =
full-orbit {0,1} counting (same convention as the Step 2 brute; the complement-shift
orbit is included).

Validation gates:
  A1 structure: rect construction V/E/F/census; mid/cut equation counts = 2WL/2WL.
  A2 decomposition cross-check: square tori L=1,2,3: Omega_mid x Omega_cut =
     32 / 1152 / 20720 (Step 2 anchors).
  A3 L=1 detail: mid = 8 (3 variables, zero equations), cut = 4 (2 classes, zero equations).

Production: W=2 L=2..16; W=3 L=2..12; W=4 L=2..9 (lambda(W) per-column increments
-> s(W) = lambda/(3W)).
Memory: states <= 2^~(3W + const) << the 2e8 gate.
Output: z3_31212/TSD31212_dp_strips.csv + stdout.
"""
import csv
import math
import os
import sys
import time
from collections import defaultdict
from itertools import product

import numpy as np

sys.stdout.reconfigure(encoding='utf-8')
BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.abspath(os.path.join(BASE, 'z4_3464')))
import RTH_numerical_verification as RTH  # noqa: E402

OUT_CSV = os.path.join(BASE, 'z3_31212', 'TSD31212_dp_strips.csv')
_CACHE = {}


def build_31212_rect(W, L):
    """W x L truncated honeycomb torus (rectangular generalization of
    RTH.build_31212, period W*A1 x L*A2)."""
    s, tt = 1.0, 1.0 / math.sqrt(3)
    g = s + 2 * tt
    D0 = math.sqrt(3) * g
    A1, A2 = np.array((D0, 0)), np.array((D0 / 2, D0 * math.sqrt(3) / 2))
    t = RTH.Torus(W * A1, L * A2, 3)
    for i in range(W):
        for j in range(L):
            C = A1 * i + A2 * j
            A = [C + g * np.array(RTH.u(30 + 60 * m)) for m in range(6)]
            dhat = [np.array(RTH.u(150 + 60 * m)) for m in range(6)]
            Pm = [(A[m] + tt * dhat[m], A[(m + 1) % 6] - tt * dhat[m]) for m in range(6)]
            dodec = []
            for m in range(6):
                dodec += [Pm[m][0], Pm[m][1]]
            t.add_face(dodec, 12)
            for m in range(6):
                Ac = A[(m + 1) % 6]
                t.add_face([Ac + tt * np.array(RTH.u(90 + 60 * m)),
                            Ac + tt * np.array(RTH.u(210 + 60 * m)),
                            Ac + tt * np.array(RTH.u(330 + 60 * m))], 3)
    return t.finalize()


def build_subsystems(W, L):
    """mid/cut separated subsystems (keys = mid-segment edges / truncated triangle classes).

    Returns dict(mid_keys, mid_eqs, cut_keys, cut_eqs, n_dodec, N):
      mid_keys/cut_keys: key counts
      mid_eqs/cut_eqs:   [per dodecagon][2 equations]{key: int coef} (x2 scaled to
                         integers, edge direction sign included)
      face_keysets:      the 6 keys of each dodecagon (mid/cut sets)
    """
    hit = _CACHE.get((W, L))
    if hit is not None:
        return hit
    t = build_31212_rect(W, L)
    N = W * L
    assert (t.V, t.E, t.F) == (6 * N, 9 * N, 3 * N), (W, L, t.V, t.E, t.F)
    assert dict(sorted(t.face_census().items())) == {3: 2 * N, 12: N}
    assert t.degs == [3]
    assert all(RTH.rot_norm(f) == (3, 12, 12) for f in t.vertex_figures())
    size_of = [len(f[0]) for f in t.faces]
    edge_faces = defaultdict(list)
    for fi, f in enumerate(t.faces):
        for (eid, sgn) in t.face_edges[fi]:
            edge_faces[eid].append(fi)
    assert all(len(v) == 2 for v in edge_faces.values())
    # triangle collapse (truncation classes)
    parent = list(range(t.E))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for fi in range(t.F):
        if size_of[fi] == 3:
            ids = [eid for (eid, s) in t.face_edges[fi]]
            union(ids[0], ids[1])
            union(ids[1], ids[2])
    mid_of = {}
    cut_of = {}
    for eid in range(t.E):
        fis = edge_faces[eid]
        if all(size_of[fi] == 12 for fi in fis):      # mid-segment edge
            mid_of[eid] = len(mid_of)
        else:                                          # cut edge -> triangle class
            r = find(eid)
            cut_of.setdefault(r, len(cut_of))
    assert len(mid_of) == 3 * N and len(cut_of) == 2 * N, (len(mid_of), len(cut_of), N)
    mid_eqs, cut_eqs = [], []
    mid_keysets, cut_keysets = [], []
    for fi in range(t.F):
        if size_of[fi] != 12:
            continue
        mx = defaultdict(int)
        my = defaultdict(int)
        cx = defaultdict(int)
        cy = defaultdict(int)
        for (eid, sgn) in t.face_edges[fi]:
            if eid in mid_of:
                mx[mid_of[eid]] += 2 * t.xrows[fi][eid].b   # mid X (xrow sqrt(3) part)
                my[mid_of[eid]] += 2 * t.yrows[fi][eid].a   # mid Y (yrow rational part)
            else:
                c = cut_of[find(eid)]
                cx[c] += 2 * t.xrows[fi][eid].a             # cut X (xrow rational part)
                cy[c] += 2 * t.yrows[fi][eid].b             # cut Y (yrow sqrt(3) part)
        mid_eqs.append([dict(v=1, **{}) if False else {k: v for k, v in mx.items() if v},
                        {k: v for k, v in my.items() if v}])
        cut_eqs.append([{k: v for k, v in cx.items() if v},
                        {k: v for k, v in cy.items() if v}])
        mid_keysets.append(sorted(mx))
        cut_keysets.append(sorted(cx))
    assert len(mid_eqs) == N
    assert all(len(e) == 2 for e in mid_eqs) and all(len(e) == 2 for e in cut_eqs)
    hit = dict(mid_keys=len(mid_of), mid_eqs=mid_eqs, cut_keys=len(cut_of),
               cut_eqs=cut_eqs, mid_keysets=mid_keysets, cut_keysets=cut_keysets,
               n_dodec=N, N=N)
    _CACHE[(W, L)] = hit
    return hit


def frontier_dp(n_keys, face_keysets, face_eqs, label='', vals=(0, 1)):
    """Split-row counting frontier DP (full orbit, no gauge; vals parametrizes binary/ternary).

    face_keysets: key list per face (unique key set; zero-valued free keys enumerated too)
    face_eqs:     equation list per face (dict{key: coef}; coefficient sums include
                  direction signs)
    state = frontier key-value tuple (keys in entry order), merged by key.
    Assert: frontier empty after all faces are processed (torus wrap-around closure).
    """
    n_faces = len(face_keysets)
    faces_of = defaultdict(list)
    for fi, ks in enumerate(face_keysets):
        for k in ks:
            faces_of[k].append(fi)
    # face graph BFS order (via shared keys)
    adj = defaultdict(set)
    for ks in face_keysets:
        for a in ks:
            for b in ks:
                if a != b:
                    adj[a].add(b)
    # BFS over faces: start at face 0, expand via shared keys
    order = [0]
    seen = {0}
    queue = [0]
    while queue:
        f = queue.pop(0)
        for k in face_keysets[f]:
            for g in faces_of[k]:
                if g not in seen:
                    seen.add(g)
                    order.append(g)
                    queue.append(g)
    assert len(order) == n_faces, (label, 'face graph not connected', len(order), n_faces)
    # DP (frontier key order = uni_keep, determined by the processed-face set and
    # independent of state values => merge-by-key is legal)
    states = {tuple(): 1}
    frontier = []          # current frontier keys (in entry order)
    processed = [False] * n_faces
    for fi in order:
        processed[fi] = True
        ks = face_keysets[fi]
        eqs = face_eqs[fi]
        # unify frontier key order: old frontier keys still used by unprocessed faces
        # + same-kind keys of this face
        uni_keep = []
        seen_k = set()
        for k in frontier:
            if any(not processed[g] for g in faces_of[k]):
                uni_keep.append(k)
                seen_k.add(k)
        for k in ks:
            if k not in seen_k and any(not processed[g] for g in faces_of[k]):
                uni_keep.append(k)
                seen_k.add(k)
        new_states = defaultdict(int)
        vlist = list(vals)
        vmin, vmax = min(vlist), max(vlist)
        for st, cnt in states.items():
            fvals = dict(zip(frontier, st))
            free = [k for k in ks if k not in fvals]
            # equation preprocessing: fixed contribution + free terms (incremental
            # pruning: interval filter as in paramdfs)
            k_eq = defaultdict(list)            # k -> [(eq_idx, coef)]
            eq_terms = []                       # eq_idx -> [(k, coef)] free terms
            fixed = []                          # eq_idx -> contribution of fixed terms
            for ei, eq in enumerate(eqs):
                fx = 0
                terms = []
                for k, c in eq.items():
                    if k in fvals:
                        fx += c * fvals[k]
                    else:
                        terms.append((k, c))
                        k_eq[k].append((ei, c))
                eq_terms.append(terms)
                fixed.append(fx)
            sum_now = list(fixed)
            nleft = [len(t) for t in eq_terms]
            assign = {}
            nfree = len(free)

            def rec(i):
                if i == nfree:
                    # leaf closure check: when free is empty (all keys already in the
                    # frontier) sum_now was never verified, so check here; redundant but
                    # harmless when free > 0 (already checked when each equation's nleft
                    # hit zero)
                    if all(s == 0 for s in sum_now):
                        new_states[tuple(assign[k] if k in assign else fvals[k]
                                         for k in uni_keep)] += cnt
                    return
                k = free[i]
                touched = k_eq.get(k, ())
                for v in vlist:
                    ok = True
                    for (ei, c) in touched:
                        sum_now[ei] += c * v
                        nleft[ei] -= 1
                    for (ei, c) in touched:
                        if nleft[ei] == 0:
                            if sum_now[ei] != 0:
                                ok = False
                                break
                        else:
                            # the remaining free-term contribution interval must cover 0 - sum_now
                            lo = hi = sum_now[ei]
                            for (kk, cc) in eq_terms[ei]:
                                if kk != k and kk not in assign:
                                    if cc > 0:
                                        lo += cc * vmin
                                        hi += cc * vmax
                                    else:
                                        lo += cc * vmax
                                        hi += cc * vmin
                            if lo > 0 or hi < 0:
                                ok = False
                                break
                    if ok:
                        assign[k] = v
                        rec(i + 1)
                        del assign[k]
                    for (ei, c) in touched:
                        sum_now[ei] -= c * v
                        nleft[ei] += 1
            rec(0)
        frontier = uni_keep
        states = dict(new_states)
        assert len(states) <= len(vals) ** len(frontier) + 1
    assert not frontier, (label, 'frontier not cleared', frontier)
    return sum(states.values())


def main():
    print('=== A1/A2/A3 gates: square-torus decomposition cross-check (Step 2 anchors) ===')
    anchors = {1: 32, 2: 1152, 3: 20720}
    detail = {1: (8, 4)}
    for L in (1, 2, 3):
        sub = build_subsystems(L, L)
        t0 = time.time()
        om = frontier_dp(sub['mid_keys'], sub['mid_keysets'], sub['mid_eqs'],
                         label=f'mid {L}x{L}')
        oc = frontier_dp(sub['cut_keys'], sub['cut_keysets'], sub['cut_eqs'],
                         label=f'cut {L}x{L}')
        prod = om * oc
        print(f'  L={L}: Omega_mid={om} Omega_cut={oc} prod={prod} '
              f'(anchor {anchors[L]}) [{time.time()-t0:.1f}s] '
              f'{"PASS" if prod == anchors[L] else "FAIL"}')
        assert prod == anchors[L], (L, om, oc, anchors[L])
        if L == 1:
            assert (om, oc) == detail[1], (om, oc)

    print('=== production: narrow-torus lambda(W) series ===')
    rows = []
    for W, Lmax in ((2, 16), (3, 12), (4, 9)):
        prev = None
        for L in range(2, Lmax + 1):
            sub = build_subsystems(W, L)
            t0 = time.time()
            om = frontier_dp(sub['mid_keys'], sub['mid_keysets'], sub['mid_eqs'],
                             label=f'mid {W}x{L}')
            oc = frontier_dp(sub['cut_keys'], sub['cut_keysets'], sub['cut_eqs'],
                             label=f'cut {W}x{L}')
            omg = om * oc
            dt = time.time() - t0
            lam_m = math.log(om) - prev[0] if prev else float('nan')
            lam_t = math.log(omg) - prev[1] if prev else float('nan')
            s_m = lam_m / (3 * W) if prev else float('nan')
            s_t = lam_t / (3 * W) if prev else float('nan')
            print(f'  W={W} L={L}: Omega_mid={om} Omega_cut={oc} Omega={omg} '
                  f'ln={math.log(omg):.4f} s_tot={math.log(omg)/(3*W*L):.4f} '
                  f'lambda_mid/col={lam_m:.4f} s_mid={s_m:.4f} s_tot_col={s_t:.4f} [{dt:.1f}s]',
                  flush=True)
            rows.append([W, L, om, oc, omg, f'{math.log(omg):.6f}',
                         f'{math.log(omg)/(3*W*L):.6f}',
                         f'{lam_m:.6f}' if prev else '', f'{s_m:.6f}' if prev else '',
                         f'{dt:.1f}'])
            prev = (math.log(om), math.log(omg))
    with open(OUT_CSV, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['W', 'L', 'Omega_mid', 'Omega_cut', 'Omega', 'lnOmega',
                    's_per_face', 'lambda_mid_per_col', 's_mid_per_col', 'time_s'])
        w.writerows(rows)
    print(f'CSV written {OUT_CSV}')


if __name__ == '__main__':
    main()
