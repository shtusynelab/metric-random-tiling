# -*- coding: utf-8 -*-
"""Enumerate pinned solutions of the 2xL binary torus (L=2,3) to expose
the structure behind n0 = 2^L: which classes stay free, how the rest
are determined. Small-scale (peak ~5 states), DFS with solution lists.
"""
import sys
from collections import defaultdict

import numpy as np

from roundE_4612_frontier_dp import build_face_system, face_bfs_order, \
    frontier_plan
from roundE_4612_pinned_dp import build_quotient, find_circle_candidates, \
    separation_ok
from roundD_4612_numeric import T1


def enum_pinned(L):
    (face_eqs, n_cls, n_free, N, dim, all_eq_cls, face_nbrs,
     edge_cls) = build_face_system(2, L, rank_check=False)
    vid, edges_q, E_q, pts, cls_of_edges_q, raw2q, nbr = build_quotient(2, L)
    cands = find_circle_candidates(nbr, pts, vid, 2 * T1)
    circle = next(c for c in cands if separation_ok(edges_q, c)[0])
    pinned = frozenset(edge_cls[qi] for qi in circle)
    face_order = face_bfs_order(face_nbrs, len(face_eqs))
    cls_faces = defaultdict(list)
    for fi, eqs_f in enumerate(face_eqs):
        for eq in eqs_f:
            for c in eq:
                cls_faces[c].append(fi)
    plan = frontier_plan(face_eqs, face_order, cls_faces)
    # naive DFS over equation variables in plan order, pinned forced 0
    eq_of = defaultdict(list)          # class -> [(fi, eq_idx, eq, coef)]
    for fi, eqs_f in enumerate(face_eqs):
        for ki, eq in enumerate(eqs_f):
            for c, co in eq.items():
                eq_of[c].append((fi, ki, eq, co))
    assigned = {}
    solutions = []
    order = [c for step in plan for c in step[1]]
    assert len(set(order)) == len(order)
    rem = {}
    for fi, eqs_f in enumerate(face_eqs):
        for ki, eq in enumerate(eqs_f):
            rem[(fi, ki)] = set(eq.keys())
    sys.setrecursionlimit(100000)

    def dfs(k, parts):
        # parts[(fi, k)] = partial sum of equation k of face fi
        if k == len(order):
            if all(v == 0 for v in parts.values()):
                solutions.append(dict(assigned))
            return
        c = order[k]
        for v in ((0,) if c in pinned else (0, 1)):
            assigned[c] = v
            touched = eq_of.get(c, [])
            ok = True
            done = []
            for fi, eq_idx, eq, co in touched:
                key = (fi, eq_idx)
                parts[key] = parts.get(key, 0) + co * v
                rem[key].discard(c)
                if not rem[key]:
                    done.append(key)
            for key in done:
                if parts[key] != 0:
                    ok = False
                    break
            if ok:
                dfs(k + 1, parts)
            for fi, eq_idx, eq, co in touched:
                key = (fi, eq_idx)
                parts[key] = parts.get(key, 0) - co * v
                rem[key].add(c)
            del assigned[c]
        return

    dfs(0, {})
    return pinned, order, solutions, all_eq_cls


def main():
    for L in (2, 3):
        pinned, order, sols, all_eq_cls = enum_pinned(L)
        print(f'=== L={L}: pinned={len(pinned)} solutions={len(sols)} '
              f'(expect 2^{L}={2 ** L}) ===', flush=True)
        # per-solution: pattern of NONZERO classes
        for s in sols:
            nz = sorted(c for c, v in s.items() if v != 0)
            print(f'  sol nz={nz}', flush=True)
        # free-class statistics across solutions
        free_stats = defaultdict(int)
        for s in sols:
            for c, v in s.items():
                if v != 0:
                    free_stats[c] += 1
        print(f'  class flip counts: {dict(sorted(free_stats.items()))}',
              flush=True)
        # are solutions pairwise related by single-class flips?
        if len(sols) >= 2:
            base = sols[0]
            for s in sols[1:]:
                diff = sorted(c for c in all_eq_cls if s.get(c, 0) !=
                              base.get(c, 0))
                print(f'  diff vs sol0: {diff}', flush=True)


if __name__ == '__main__':
    main()
