# -*- coding: utf-8 -*-
"""Step 4: ternary {0,1,2} Knuth strip estimates (W >= 3 width points).

Work split (as in 4.6.12): W=2 full series + W=3 L=2,4 already covered by exact DP
(15 calibration points); remaining W >= 3 width points via Knuth estimation; the
transpose pair (W=3 L=4 <-> W=4 L=3) serves as an independent check.

Ported from roundE_4612_knuth_strips2.py (weight convention: walk = uniform choice
among feasible branches => X = prod f; the estimator must be calibrated first,
production only with residual < 2%). 3.12.12 adaptation:
  - joint class system = mid (0..3WL-1) + cut (offset 3WL), equations = 4 per dodecagon;
  - closure generalization: S == 0 (mod Lp) and S in {0, Lp, ..., kmax*Lp} (kmax=2);
  - interval-check generalization: remaining-term contributions [kmin*c, kmax*c]
    (the 4.6.12 version hardcodes kmax=1);
  - Omega_est = E[X] x k x k^{n_free} (gauge factor x k: the mod-k translation group
    acts freely, rationale: row sums vanish identically (all-ones reference solution)
    => coefficient sums == 0 mod k; adjudicated against the L=1 brute force (243)
    and DP anchors).
Output: z3_31212/TSD31212_knuth_ternary.csv + stdout.
"""
import csv
import math
import os
import sys
import time
from collections import defaultdict
import random

sys.stdout.reconfigure(encoding='utf-8')
BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib31212 import rref_parametrize  # noqa: E402
from step3_dp_strips import build_subsystems  # noqa: E402

OUT_CSV = os.path.join(BASE, 'z3_31212', 'TSD31212_knuth_ternary.csv')
_CACHE = {}


def prepare_joint(W, L):
    """Joint class-system RREF parametrization + dependency-connected order
    (isomorphic to the 4.6.12 prepare_binary)."""
    hit = _CACHE.get((W, L))
    if hit is not None:
        return hit
    sub = build_subsystems(W, L)
    n_mid = sub['mid_keys']
    n_cls = n_mid + sub['cut_keys']
    eqs = []
    for fi in range(sub['n_dodec']):
        for eq in sub['mid_eqs'][fi]:
            if eq:
                eqs.append(dict(eq))
        for eq in sub['cut_eqs'][fi]:
            if eq:
                eqs.append({k + n_mid: v for k, v in eq.items()})
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
    coef_of = [{pos_i: c for (pos_i, c) in d["tt"]} for d in dep_list]
    n_free = n_cls - len({c for eq in eqs for c in eq})
    hit = (nf, dep_list, p_of_pos, order, coef_of, n_free, n_cls)
    _CACHE[(W, L)] = hit
    return hit


def knuth_ternary(nf, dep_list, p_of_pos, order, coef_of, n_samples, seed,
                  vals=(0, 1, 2)):
    """Knuth tree sampling: walk free variables one by one, uniform choice among
    feasible branches, X *= f."""
    rng = random.Random(seed)
    kmax = max(vals)
    kmin = min(vals)
    total_X = 0.0
    hits = 0
    for _ in range(n_samples):
        part = [0] * len(dep_list)
        nleft = [d["nk"] for d in dep_list]
        X = 1.0
        alive = True
        for k in order:
            feas = []
            touched_by_k = p_of_pos.get(k, [])
            for v in vals:
                ok = True
                touched = []
                for di in touched_by_k:
                    coef = coef_of[di][k]
                    part[di] += coef * v
                    nleft[di] -= 1
                    touched.append((di, coef))
                for di, coef in touched:
                    Lp = dep_list[di]["Lp"]
                    if nleft[di] == 0:
                        S = part[di]
                        if S % Lp != 0 or not (0 <= S <= kmax * Lp):
                            ok = False
                            break
                    else:
                        lo = hi = part[di]
                        for (pos_i, c) in dep_list[di]["tt"]:
                            lo += min(c * kmin, c * kmax)
                            hi += max(c * kmin, c * kmax)
                        if hi < 0 or lo > kmax * Lp:
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
            X *= f
            for di in touched_by_k:
                coef = coef_of[di][k]
                part[di] += coef * v
                nleft[di] -= 1
        if alive:
            total_X += X
            hits += 1
    return total_X / n_samples, hits


def main():
    k = 3
    # calibration/production job table (anchors = exact DP values, joint = mid x cut)
    dp_anchors = {
        (2, 3): 1017 * 67, (2, 5): 16041 * 379, (2, 7): 294585 * 2707,
        (2, 9): 6000777 * 21739, (3, 2): 1017 * 67, (3, 4): 30321 * 235,
        (1, 1): 243,
    }
    jobs = [
        # calibration points (exact DP values available; production gate residual < 2%)
        (2, 3, 200000), (2, 5, 200000), (2, 7, 200000), (2, 9, 300000),
        (3, 2, 200000), (3, 4, 300000),
        # production points (remaining W >= 3 widths)
        (3, 5, 400000), (3, 7, 600000), (3, 8, 800000), (3, 10, 1000000),
        (4, 3, 400000), (4, 5, 800000), (4, 7, 1000000),
        (5, 2, 500000), (5, 3, 800000),
    ]
    rows = []
    for (W, L, nsamp) in jobs:
        nf, dep_list, p_of_pos, order, coef_of, n_free, n_cls = prepare_joint(W, L)
        t0 = time.time()
        mean, hits = knuth_ternary(nf, dep_list, p_of_pos, order, coef_of,
                                   nsamp, 20260777)
        dt = time.time() - t0
        tag = ''
        if mean > 0:
            om = mean * k * (k ** n_free)
            s_mid_face = math.log(om) / (3 * W * L) if om > 0 else float('nan')
            tag = f'Omega_est={om:.6g} s_face={s_mid_face:.4f}'
            if (W, L) in dp_anchors:
                anchor = dp_anchors[(W, L)]
                dln = math.log(om) - math.log(anchor)
                tag += f' vs DP {anchor:.6g} dln={dln:+.4f} ({100*dln/math.log(anchor):+.1f}%)'
                tag += ' [CALIB PASS]' if abs(dln) / math.log(anchor) < 0.02 else ' [CALIB FAIL]'
        else:
            tag = 'ZERO HITS'
        print(f'W={W} L={L}: nf={nf} n_free={n_free} hit={hits}/{nsamp}'
              f'={hits/nsamp:.2e} {tag} [{dt:.0f}s]', flush=True)
        rows.append([W, L, nsamp, hits, nf, n_free, f'{mean:.6g}',
                     f'{mean * k * (k ** n_free):.6g}' if mean > 0 else 'ZERO',
                     f'{dt:.0f}'])
    with open(OUT_CSV, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['W', 'L', 'n_samples', 'hits', 'nf', 'n_free', 'E_X',
                    'Omega_est', 'time_s'])
        w.writerows(rows)
    print(f'CSV written {OUT_CSV}')


if __name__ == '__main__':
    main()
