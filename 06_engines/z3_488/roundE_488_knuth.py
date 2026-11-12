# -*- coding: utf-8 -*-
"""Measured Knuth hit rate for the 4.8.8 Z-library (clean version):
strand-constrained system vs the Knuth tool, matching verification.

System: build_488(L) (truncated square lattice, 45-degree mesh, face-equation
coefficients in Q(sqrt(2))).
Torus.xrows/yrows = per-face closure rows (QD coefficients; row sum = 0
self-checked); each row splits into 2 scalar equations (alpha component /
beta component, coefficients scaled by 2 to integers). Variables = edges,
values {0,1,2}.

Reuses rref_parametrize + knuth_run from roundE_4612_knuth (generic engine).
Anchors: 4.8.8 Z three-valued Omega = 3^{4L} exactly (W=2: 6561; W=3: 531441;
TSQ documentation).
A measured hit rate verifies the claim "the strand equality system has a low
hit exponent" (prerequisite test for the feasibility of 4.8.8-Lambda Knuth).

Memory < 50 MB; runs locally.
"""
import io
import math
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'z4_3464'))
sys.path.insert(0, str(HERE.parent / 'z3_4612'))
from RTH_numerical_verification import build_488           # noqa
from roundE_4612_paramdfs import rref_parametrize          # noqa
from roundE_4612_knuth import knuth_run                    # noqa

VALS = (0, 1, 2)


def prepare_488(L):
    t = build_488(L)
    eqs = []
    for rx, ry in zip(t.xrows, t.yrows):
        ra = defaultdict(int)
        rb = defaultdict(int)
        for eid, q in enumerate(rx):
            ra[eid] += int(2 * q.a)
        for eid, q in enumerate(ry):
            rb[eid] += int(2 * q.b)
        for r in (ra, rb):
            r = {k: v for k, v in r.items() if v}
            if r:
                eqs.append(dict(r))
    n_vars = t.E
    free, deps = rref_parametrize(n_vars, eqs)
    fixed = free[0]
    free_eff = [f for f in free if f != fixed]
    pos = {fk: i for i, fk in enumerate(free_eff)}
    nf = len(free_eff)
    dep_list = []
    for p, (Lp, terms) in deps.items():
        tt = [(pos[fk], c) for fk, c in terms if fk in pos]
        if not tt:
            continue
        dep_list.append({'p': p, 'Lp': Lp, 'tt': tt, 'nk': len(tt)})
    p_of_pos = defaultdict(list)
    for di, d in enumerate(dep_list):
        for (pos_i, c) in d['tt']:
            p_of_pos[pos_i].append(di)
    order, seen = [], set()
    for start in range(nf):
        if start in seen:
            continue
        queue = [start]
        seen.add(start)
        while queue:
            k = queue.pop(0)
            order.append(k)
            for di in p_of_pos.get(k, []):
                for (pos_i, c) in dep_list[di]['tt']:
                    if pos_i not in seen:
                        seen.add(pos_i)
                        queue.append(pos_i)
    assert len(order) == nf
    return nf, dep_list, p_of_pos, order, t


if __name__ == '__main__':
    rng = random.Random(20261001)
    anchors = {2: 3 ** 8, 3: 3 ** 12}
    for L in [2, 3]:
        t0 = time.time()
        nf, dep_list, p_of_pos, order, t = prepare_488(L)
        n_samp = 20000
        mean, se, hit = knuth_run(nf, dep_list, p_of_pos, order, n_samp, rng)
        om_gauge = mean
        anchor = anchors[L]
        print(f'4.8.8 Z L={L}: nf={nf}  Knuth Omega(gauge)~{om_gauge:.4g}  '
              f'anchor 3^{4*L}={anchor}  ratio={om_gauge/anchor:.3f}  '
              f'hit={hit:.3f}  ({time.time()-t0:.1f}s)', flush=True)
        print(f'  reading: high hit => strand equalities are Knuth-friendly '
              f'(revised claim); low hit => confirms the exponential decay of '
              f'the strand hit rate', flush=True)
