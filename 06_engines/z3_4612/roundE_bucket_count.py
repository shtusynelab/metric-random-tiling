# -*- coding: utf-8 -*-
"""4.6.12 Z-count v3: bucket elimination + honeycomb anchor verification
(whole-file rewrite, clean version).

Algorithm: constraints = integer-coefficient linear homogeneous equations
over class variables; elimination in min-fill greedy order; when
eliminating variable v, enumerate the values of v for each assignment of
the new scope and multiply-sum over the relation tables.
Cost 3^(arity+1) per variable.

The count is rotation invariant (a global 30-degree rotation maps the
sqrt(3)/2 coefficient network to a rational network with a bijection on
solutions) => honeycomb anchors can use true-geometry directions directly.
Verification gates: {0,1} 2x2=64, 3x2=64, 3x3=518; {0,1,2} 3x3=22 767.
Main target: 4.6.12 {0,1,2} L=2..6 => s_Z = ln Omega/(6N) thick/thin read.
Memory: relation tables 3^arity; arity expected <= 10; whole run
< 200 MB locally.
"""
import io
import math
import sys
import time
from collections import defaultdict
from itertools import product
from pathlib import Path

import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.insert(0, str(Path(__file__).resolve().parent))
from roundD_4612_numeric import fund_reduce, PAIR_COS, PAIR_SIN   # noqa
from roundE_4612_zcount import build_reduced as build_4612_reduced  # noqa

VALS = (0, 1, 2)


def bucket_count(n_vars, eqs, vals=VALS):
    """eqs: list of dicts {var: int coef}; returns the number of
        assignments in {0..len(vals)-1}^n satisfying all equations."""
    rels = []
    for eq in eqs:
        scope = tuple(sorted(eq))
        if not scope:
            continue
        table = {}
        for assign in product(vals, repeat=len(scope)):
            if sum(c * x for c, x in zip((eq[u] for u in scope), assign)) == 0:
                table[assign] = 1
        rels.append((scope, table))
    var_rels = defaultdict(list)
    for ri, (scope, tab) in enumerate(rels):
        for v in scope:
            var_rels[v].append(ri)
    alive = list(rels)
    dead = [False] * len(rels)
    eliminated = [False] * n_vars
    for _ in range(n_vars):
        best_v, best_cost = None, None
        for v in range(n_vars):
            if eliminated[v]:
                continue
            scope = set()
            for ri in var_rels[v]:
                if not dead[ri]:
                    scope.update(alive[ri][0])
            scope.discard(v)
            cost = 1
            for _ in scope:
                cost *= len(vals)
            if best_cost is None or cost < best_cost:
                best_v, best_cost = v, cost
        v = best_v
        eliminated[v] = True
        my = [ri for ri in var_rels[v] if not dead[ri]]
        new_scope = set()
        for ri in my:
            new_scope.update(alive[ri][0])
        new_scope.discard(v)
        new_scope = tuple(sorted(new_scope))
        new_tab = defaultdict(int)
        for assign in product(vals, repeat=len(new_scope)):
            env = dict(zip(new_scope, assign))
            total = 0
            for x_v in vals:
                prod = 1
                for ri in my:
                    scope_r, tab_r = alive[ri]
                    key = tuple(x_v if u == v else env[u] for u in scope_r)
                    p = tab_r.get(key, 0)
                    if p == 0:
                        prod = 0
                        break
                    prod *= p
                total += prod
                if total and len(my) == 1:
                    pass
            if total:
                new_tab[assign] = total
        for ri in my:
            dead[ri] = True
        if new_tab:
            alive.append((new_scope, dict(new_tab)))
            dead.append(False)
            ri_new = len(alive) - 1
            for u in new_scope:
                var_rels[u].append(ri_new)
    omega = 1
    for ri, (scope, tab) in enumerate(alive):
        if dead[ri]:
            continue
        assert len(scope) == 0, f'non-empty scope left after elimination: {scope}'
        omega *= sum(tab.values())
    return omega


# ---------------------------------------------------------------- honeycomb verification

def honeycomb_system(Lx, Ly):
    """True geometry of the honeycomb torus: hole center
    c = sqrt(3)*(i+j/2, (sqrt(3)/2)*j); corner = c + unit(30 deg + 60k)
    (circumradius = edge length = 1); edge direction bucket = directed
    bucket on the 30-degree grid."""
    SQ3 = math.sqrt(3.0)
    DIR60 = [(math.cos(math.radians(60 * k)), math.sin(math.radians(60 * k)))
             for k in range(6)]

    def hole(i, j):
        return np.array([SQ3 * (i + j / 2), 1.5 * j])
    T1p = SQ3 * Lx * np.array([1.0, 0.0])
    T2p = SQ3 * Ly * np.array([0.5, SQ3 / 2])

    def canon(p):
        mm, nn, rm = fund_reduce(p, Lx, Ly, T1p, T2p)
        return (round(rm[0], 6), round(rm[1], 6))

    vid, vlist = {}, []

    def vid_of(p):
        k = canon(p)
        if k not in vid:
            vid[k] = len(vlist)
            vlist.append(p)
        return vid[k]

    faces = []
    for i in range(Lx):
        for j in range(Ly):
            c = hole(i, j)
            corners = [c + np.array([math.cos(math.radians(30 + 60 * k)),
                                     math.sin(math.radians(30 + 60 * k))])
                       for k in range(6)]
            keys = [vid_of(p) for p in corners]
            faces.append(keys)
    edge_ids = {}
    for keys in faces:
        for m in range(6):
            u, w = keys[m], keys[(m + 1) % 6]
            key = (min(u, w), max(u, w))
            if key not in edge_ids:
                edge_ids[key] = len(edge_ids)
    E = len(edge_ids)
    eqs = []
    for keys in faces:
        row_c = defaultdict(int)
        row_s = defaultdict(int)
        for m in range(6):
            u, w = keys[m], keys[(m + 1) % 6]
            eid = edge_ids[(min(u, w), max(u, w))]
            dvec = vlist[w] - vlist[u]
            ang = math.degrees(math.atan2(dvec[1], dvec[0])) % 360
            bb = int(round(ang / 30)) % 12
            # count is rotation invariant: rotate everything by 30 deg,
            # odd bucket -> even bucket (rational coefficients, beta=0)
            bb2 = (bb - 1) % 12
            row_c[eid] += PAIR_COS[bb2][0]
            row_s[eid] += PAIR_SIN[bb2][0]
        for r in (row_c, row_s):
            r = {k: v for k, v in r.items() if v}
            if r:
                eqs.append(r)
    return len(vid), E, len(faces), eqs


if __name__ == '__main__':
    print('=== honeycomb anchor verification (rotation invariant: '
          '30-degree-grid directions solve the same system as the project '
          'rational convention) ===')
    for (Lx, Ly), expect in [((2, 2), 64), ((3, 2), 64), ((3, 3), 518)]:
        t0 = time.time()
        V, E, F, eqs = honeycomb_system(Lx, Ly)
        om = bucket_count(E, eqs, vals=(0, 1))
        print(f'honeycomb {Lx}x{Ly} {{0,1}}: V={V} E={E} F={F}  Omega={om}  '
              f'expect {expect}  {"PASS" if om == expect else "FAIL"}  ({time.time()-t0:.1f}s)',
              flush=True)
    t0 = time.time()
    V, E, F, eqs = honeycomb_system(3, 3)
    om = bucket_count(E, eqs, vals=(0, 1, 2))
    print(f'honeycomb 3x3 {{0,1,2}}: Omega={om} expect 22 767  '
          f'{"PASS" if om == 22767 else "FAIL"}  ({time.time()-t0:.1f}s)', flush=True)
    print('=== 4.6.12 Z count ({0,1,2}) ===')
    for L in [2, 3, 4]:
        t0 = time.time()
        cls_of, eqs, all_cls, n_cls, dim, N, V, E, F = build_4612_reduced(L, L)
        cls_idx = {c: i for i, c in enumerate(all_cls)}
        eqs2 = [{cls_idx[c]: co for c, co in eq.items()} for eq in eqs]
        om = bucket_count(n_cls, eqs2)
        sZ = math.log(om) / (6 * N)
        print(f'4.6.12 L={L}: N={N} Omega_Z={om}  ln={math.log(om):.3f}  '
              f's_Z=ln/(6N)={sZ:.4f}  ({time.time()-t0:.1f}s)', flush=True)
