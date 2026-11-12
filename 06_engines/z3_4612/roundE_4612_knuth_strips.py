# -*- coding: utf-8 -*-
"""4.6.12 binary strip Knuth sampler: W x L tori beyond DP reach.

Calibrated on W=2xL=4 (DP anchor Omega=2569312). Hit rates measured
(W=2xL=4: 1.0%, decay ~6x per column) put W=3xL<=8 and W=4xL<=4 within
reach -- these are the lambda(W) points the frontier DP cannot deliver
(peak states hit the 2e8 gate at W=3, L>=6).

Counting convention: Omega_est = E[X] x 2 x 2^{n_free} (gauge shift +
free classes). Single-seed stderr ~ sigma_ln/sqrt(hits).
"""
import math
import sys
import time
from collections import defaultdict
import random
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from roundE_4612_binary_anchors import rref_parametrize
from roundE_4612_zcount import build_reduced


def prepare_binary(Lx, Ly):
    cls_of, eqs, all_cls, n_cls, dim, N, V, E, F = build_reduced(Lx, Ly)
    free, deps = rref_parametrize(n_cls, eqs)
    fixed = free[0]
    free_eff = [f for f in free if f != fixed]
    pos = {fk: i for i, fk in enumerate(free_eff)}
    nf = len(free_eff)
    dep_list = []
    for p, (Lp, terms) in deps.items():
        tt = [(pos[fk], c) for fk, c in terms if fk in pos]
        if not tt:
            continue
        dep_list.append({"p": p, "Lp": Lp, "tt": tt, "nk": len(tt)})
    p_of_pos = defaultdict(list)
    for di, d in enumerate(dep_list):
        for (pos_i, c) in d["tt"]:
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
                for (pos_i, c) in dep_list[di]["tt"]:
                    if pos_i not in seen:
                        seen.add(pos_i)
                        queue.append(pos_i)
    assert len(order) == nf
    n_free = n_cls - len({c for eq in eqs for c in eq})
    return nf, dep_list, p_of_pos, order, N, n_free


def knuth_binary(nf, dep_list, p_of_pos, order, n_samples, seed):
    rng = random.Random(seed)
    total_X = 0.0
    hits = 0
    for _ in range(n_samples):
        part = [0] * len(dep_list)
        nleft = [d["nk"] for d in dep_list]
        X = 1.0
        alive = True
        for k in order:
            feas = []
            for v in (0, 1):
                ok = True
                touched = []
                for di in p_of_pos.get(k, []):
                    coef = dict(dep_list[di]["tt"])[k]
                    part[di] += coef * v
                    nleft[di] -= 1
                    touched.append((di, coef))
                for di, coef in touched:
                    Lp = dep_list[di]["Lp"]
                    if nleft[di] == 0:
                        S = part[di]
                        if S != 0 and S != Lp:
                            ok = False
                            break
                    else:
                        lo = hi = part[di]
                        for (pos_i, c) in dep_list[di]["tt"]:
                            lo += min(0, c)
                            hi += max(0, c)
                        if hi < 0 or lo > Lp:
                            ok = False
                            break
                for di, coef in touched:
                    part[di] -= coef * v
                    nleft[di] += 1
                if ok:
                    feas.append(v)
            f = len(feas)
            if f == 0:
                alive = False
                X = 0.0
                break
            v = rng.choice(feas)
            X *= 2.0 / f
            for di in p_of_pos.get(k, []):
                coef = dict(dep_list[di]["tt"])[k]
                part[di] += coef * v
                nleft[di] -= 1
        if alive:
            total_X += X
            hits += 1
    return total_X / n_samples, hits


def main():
    jobs = [(2, 4, 200000, 2569312),
            (3, 4, 400000, None),
            (3, 6, 800000, None),
            (4, 3, 400000, None),
            (4, 5, 1500000, None)]
    for (W, L, nsamp, anchor) in jobs:
        nf, dep_list, p_of_pos, order, N, n_free = prepare_binary(W, L)
        t0 = time.time()
        mean, hits = knuth_binary(nf, dep_list, p_of_pos, order, nsamp,
                                  20260777)
        dt = time.time() - t0
        tag = ''
        if mean > 0:
            om = mean * 2 * (2 ** n_free)
            s = math.log(om) / (6 * N)
            tag = (f'Omega_est={om:.6g} s={s:.4f}')
            if anchor:
                dln = abs(math.log(om) - math.log(anchor))
                tag += f' vs DP ln={math.log(anchor):.4f} |dln|={dln:.4f}'
        else:
            tag = 'ZERO HITS'
        print(f'W={W} L={L}: nf={nf} hit={hits}/{nsamp}={hits/nsamp:.5f} '
              f'{tag} [{dt:.0f}s]', flush=True)


if __name__ == '__main__':
    main()
