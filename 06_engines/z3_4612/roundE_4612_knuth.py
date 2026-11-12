# -*- coding: utf-8 -*-
"""4.6.12 Z-count Knuth tree-sampling rough estimate (thick/thin pre-check).

Estimator (classical Knuth): assign {0,1,2} variable by variable in
dependency-clustered order; at each step, for each value, tentatively apply
the incremental partial sums and test feasibility (closed equations must
take values in {0, Lp, 2Lp} / open equations pruned by partial-sum
bounds); undo immediately if infeasible. At step end, f = number of
feasible branches: 0 => the path is dead (X=0); otherwise commit one
feasible branch at random, undo the rest, weight x 3/f.
Omega(gauge) = E[X] (unbiased).
Gauge: x_{free[0]} pinned to 0 (removes the 1/3 shift orbit)
=> Omega_total = 3 * E[X].

Validation: L=1 crosschecked against brute force 2349 (gauge 783);
L=2 crosschecked against parametric DFS 9 907 839.
Memory: <50 MB; speed: per sample O(nf x dep) integer ops,
seconds-to-minutes for thousands of samples.
"""
import math
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from roundE_4612_zcount import build_reduced          # noqa
from roundE_4612_paramdfs import rref_parametrize     # noqa

VALS = (0, 1, 2)


def prepare(L):
    cls_of, eqs, all_cls, n_cls, dim, N, V, E, F = build_reduced(L, L)
    free, deps = rref_parametrize(n_cls, eqs)
    fixed = free[0]                          # gauge-fixed to 0
    free_eff = [f for f in free if f != fixed]
    pos = {fk: i for i, fk in enumerate(free_eff)}
    nf = len(free_eff)
    dep_list = []
    for p, (Lp, terms) in deps.items():
        tt = [(pos[fk], c) for fk, c in terms if fk in pos]
        if not tt:
            continue                          # value identically 0
        dep_list.append({'p': p, 'Lp': Lp, 'tt': tt, 'nk': len(tt)})
    p_of_pos = defaultdict(list)
    for di, d in enumerate(dep_list):
        for (pos_i, c) in d['tt']:
            p_of_pos[pos_i].append(di)
    # BFS cluster order
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
    return nf, dep_list, p_of_pos, order, N


def knuth_run(nf, dep_list, p_of_pos, order, n_samples, rng):
    assign = [None] * nf
    hits = 0
    total = 0.0
    total2 = 0.0
    for s in range(n_samples):
        part = defaultdict(int)               # (di) -> partial sum
        nleft = [d['nk'] for d in dep_list]
        w = 1.0
        alive = True
        for k in order:
            feas = []                         # (v, [(di, coef)]) branches kept applied
            for v in VALS:
                touched = []
                ok = True
                for di in p_of_pos.get(k, []):
                    d = dep_list[di]
                    coef = dict(d['tt'])[k]
                    part[di] += coef * v
                    nleft[di] -= 1
                    touched.append((di, coef))
                for (di, coef) in touched:
                    d = dep_list[di]
                    if nleft[di] == 0:
                        if part[di] not in (0, d['Lp'], 2 * d['Lp']):
                            ok = False
                            break
                    else:
                        lo = hi = part[di]
                        for (pos_i, c) in d['tt']:
                            if assign[pos_i] is None:
                                if c > 0:
                                    hi += 2 * c
                                else:
                                    lo += 2 * c
                        tg = (0, d['Lp'], 2 * d['Lp'])
                        if hi < tg[0] or lo > tg[2]:
                            ok = False
                            break
                for (di, coef) in touched:    # undo after trial (branches tried independently)
                    part[di] -= coef * v
                    nleft[di] += 1
                if ok:
                    feas.append((v, touched))
            if not feas:
                alive = False
                break
            v, kept = feas[rng.randrange(len(feas))]
            for (di, coef) in kept:           # re-apply the selected branch
                part[di] += coef * v
                nleft[di] -= 1
            assign[k] = v
            w *= len(feas)      # weight = product of feasible-branch counts (on the pruned tree E[prod f] = Omega)
        if alive and all(assign[d['tt'][0][0]] is not None for d in dep_list):
            # leaf: spot-check that all equations are closed
            ok = all(part[di] in (0, dep_list[di]['Lp'], 2 * dep_list[di]['Lp'])
                     for di in range(len(dep_list)))
            if ok:
                hits += 1
                total += w
                total2 += w * w
        for k in order:                        # reset
            assign[k] = None
    mean = total / n_samples
    se = math.sqrt(max(total2 / n_samples - mean * mean, 0.0) / n_samples)
    return mean, se, hits / n_samples


def _worker(args):
    L, n_samp, seed = args
    rng = random.Random(seed)
    nf, dep_list, p_of_pos, order, N = prepare(L)
    mean, se, hit = knuth_run(nf, dep_list, p_of_pos, order, n_samp, rng)
    return mean, se, hit, n_samp


if __name__ == '__main__':
    import argparse
    from multiprocessing import Pool
    ap = argparse.ArgumentParser()
    ap.add_argument('--L', type=int, required=True)
    ap.add_argument('--samples', type=int, default=20000)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--chunks', type=int, default=10)
    a = ap.parse_args()
    per = a.samples // a.chunks
    t0 = time.time()
    jobs = [(a.L, per, 20261001 + 7919 * c) for c in range(a.chunks)]
    tot = t2 = n_tot = 0
    hits = 0
    with Pool(a.workers) as pool:
        for mean, se, hit, n in pool.imap_unordered(_worker, jobs):
            tot += mean * n
            t2 += (se * se + mean * mean) * n   # second-moment pooling
            n_tot += n
            hits += hit * n
    mean = tot / n_tot
    se = math.sqrt(max(t2 / n_tot - mean * mean, 0.0) / n_tot)
    om_est = mean * 3
    sZ = math.log(om_est) / (6 * a.L * a.L)
    dln_se = se / mean if mean > 0 else float('inf')
    if om_est > 0:
        sZ = math.log(om_est) / (6 * a.L * a.L)
        print(f'L={a.L}: samples={n_tot}  Omega(gauge)~{om_est:.4g}  Omega_total~{om_est*3:.4g}  '
              f's_Z={sZ:.4f} +/- {dln_se / (6 * a.L * a.L):.4f}  '
              f'dln={dln_se:.4f}  hit={hits/n_tot:.3f}  ({time.time()-t0:.1f}s)', flush=True)
    else:
        print(f'L={a.L}: Omega estimated 0 (hits {hits}/{n_tot} -- '
              f'insufficient samples or bound too loose); increase samples', flush=True)
    import os
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), f'RTH4612_knuth_L{a.L}.csv'), 'a') as f:
        f.write(f'{a.L},{n_tot},{om_est:.6g},{dln_se:.4f},{hits/n_tot:.4f},{time.time()-t0:.0f}\n')
