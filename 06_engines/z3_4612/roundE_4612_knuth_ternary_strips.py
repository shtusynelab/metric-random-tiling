# -*- coding: utf-8 -*-
"""4.6.12 TERNARY strip Knuth (audit-registered discriminator #1):
does lambda_tern(W) grow ~ 6sW (area law, ternary window survives) or
stay flat (perimeter-type, whole network joins the 3.12.12 class)?
Walk = uniform among feasible values, weight X *= f (corrected convention,
same fix as the binary sampler); closure S in {0, Lp, 2Lp};
Omega = E[X] x 3 x 3^{n_free} (gauge + free classes).
Calibrate: square 2x2 exact 29,723,517.
"""
import math, random, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from collections import defaultdict
from roundE_4612_binary_anchors import rref_parametrize
from roundE_4612_zcount import build_reduced

def prepare(Lx, Ly):
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
    n_free = n_cls - len({c for eq in eqs for c in eq})
    return nf, dep_list, p_of_pos, order, N, n_free

def knuth_ternary(nf, dep_list, p_of_pos, order, n_samples, seed):
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
            for v in (0, 1, 2):
                ok = True
                for di in p_of_pos.get(k, []):
                    coef = dict(dep_list[di]["tt"])[k]
                    if nleft[di] - 1 == 0:
                        S = part[di] + coef * v
                        Lp = dep_list[di]["Lp"]
                        if S % Lp != 0 or not (0 <= S <= 2 * Lp):
                            ok = False
                            break
                if ok:
                    feas.append(v)
            f = len(feas)
            if f == 0:
                alive = False
                X = 0.0
                break
            v = rng.choice(feas)
            X *= f
            for di in p_of_pos.get(k, []):
                coef = dict(dep_list[di]["tt"])[k]
                part[di] += coef * v
                nleft[di] -= 1
        if alive:
            total_X += X
            hits += 1
    return total_X / n_samples, hits

jobs = [(2, 2, 100000, 29723517),   # calibration (exact)
        (2, 4, 200000, None), (2, 6, 300000, None), (2, 8, 400000, None),
        (3, 2, 200000, None), (3, 3, 300000, 6.566e12), (3, 4, 400000, None),
        (3, 5, 500000, None)]
for (W, L, nsamp, anchor) in jobs:
    nf, dep_list, p_of_pos, order, N, n_free = prepare(W, L)
    t0 = time.time()
    mean, hits = knuth_ternary(nf, dep_list, p_of_pos, order, nsamp, 20260777)
    dt = time.time() - t0
    tag = ""
    if mean > 0:
        om = mean * 3 * (3 ** n_free)
        s = math.log(om) / (6 * N)
        lam = math.log(om) / L
        tag = f"Omega={om:.5g} s={s:.4f} lam=ln/L={lam:.3f}"
        if anchor:
            if isinstance(anchor, float):
                dln = abs(math.log(om) - math.log(anchor))
                tag += f" vs anchor {anchor:.3e} |dln|={dln:.4f}"
            else:
                tag += f" vs exact {anchor} ratio={om/anchor:.4f}"
    else:
        tag = "ZERO HITS"
    print(f"W={W} L={L}: nf={nf} hit={hits}/{nsamp}={hits/nsamp:.5f} "
          f"{tag} [{dt:.0f}s]", flush=True)
print("TERNARY STRIPS DONE", flush=True)
