# -*- coding: utf-8 -*-
"""P0-A2b strip Knuth estimator: reliability verification + estimates for
extreme cells.

Estimator: the same enumeration tree as tm_count_strip exact counting
(tile x 5-edge direction combinations as branches; row-end rebasing and
lagging vertices handled by the same rules); single random root-to-leaf
path, weight = product of branch counts (leaves dying on a mid-way
violation get weight 0). E[weight] = exact Omega (unbiased); no
step-by-full-state expansion is needed -- that is why it is feasible and
cheap for very large state spaces.

Workflow:
  1. reliability verification: for (alphabet, W) cells that already have
     exact Omega in the main CSV, 3 seeds x N samples; compare Omega_hat
     vs Omega (relative error, z-score, alive rate);
  2. for the 14 states_exceeded extreme cells, produce Omega_hat and s
     with error bars;
  3. merge locally (Knuth rows marked source=knuth) and extrapolate in
     1/W -> extrapolation CSV.
"""
import sys, math, random, argparse
from pathlib import Path
from collections import Counter
import multiprocessing as mp
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pseudocrystal_lib as pl

L_ROWS = 10
allowed_ids_ = [None]  # injected at runtime (avoids deep parameter passing)
SEEDS = (1, 2, 3)


def _enum_opts(qp, pend, edges, libw, K, W_):
    first_row = all(rr >= 0 for vk in pend for _, rr in vk) and not edges
    fixed = {}
    for i in range(6):
        ek = pl._tm_ekey(qp, 0, i, W_)
        if ek in edges:
            fixed[i] = edges[ek]
    opts = []
    for t in range(K):
        w = libw[t]
        if any(int(w[i]) != wi for i, wi in fixed.items()):
            continue
        new_edges = []
        for i in range(6):
            ek = pl._tm_ekey(qp, 0, i, W_)
            if ek in edges or i in fixed:
                continue
            new_edges.append(i)
        fives = [i for i in new_edges if int(w[i]) == 5]
        for bits in range(1 << len(fives)):
            opts.append((t, bits, new_edges, fives))
    return opts, first_row


def _try_apply(qp, pend, edges, opt, libw, W_, undo):
    """Trial-apply; return False on violation (caller reverts via undo)."""
    t, bits, new_edges, fives = opt
    w = libw[t]
    for i in new_edges:
        ek = pl._tm_ekey(qp, 0, i, W_)
        wi = int(w[i])
        if wi == 4:
            a1 = a2 = 2
        elif wi == 6:
            a1 = a2 = 3
        else:
            b = (bits >> fives.index(i)) & 1
            a1, a2 = (2, 3) if b == 0 else (3, 2)
        edges[ek] = wi
        undo.append(('e', ek))
        for vk, arm in ((pl._tm_vkey(qp, 0, (i - 1) % 6, W_), a1),
                        (pl._tm_vkey(qp, 0, i, W_), a2)):
            pend.setdefault(vk, []).append(arm)
            undo.append(('v', vk))
            if len(pend[vk]) == 3 and tuple(sorted(pend[vk])) not in allowed_ids_[0]:
                return False
    return True


def _undo_apply(pend, edges, undo):
    for kind, item in reversed(undo):
        if kind == 'e':
            del edges[item]
        else:
            pend[item].pop()
            if not pend[item]:
                del pend[item]


def _advance(qp, pend, edges, W_):
    """Row-end rebasing (non-mutating): return a new view (qp2, pend2, edges2)."""
    if qp != W_ - 1:
        return qp + 1, pend, edges
    new_pend = {}
    for vk, ar in pend.items():
        sk = tuple(sorted((qq, rr - 1) for qq, rr in vk))
        if all(rr >= -1 for _, rr in sk):
            new_pend[sk] = ar
    new_edges = {tuple(sorted((qq, rr - 1) for qq, rr in ek)): wq
                 for ek, wq in edges.items()
                 if min(rr for _, rr in ek) >= -1}
    return 0, new_pend, new_edges


def knuth_strip_one(W, libw, K, allowed_ids, rng, la2=True):
    """Pruning (Rosenbluth-growth) Knuth with second-level lookahead.

    Each step only follows branches that do not die immediately and (when
    la2) whose children still have a live continuation; the weight is
    multiplied by the number of surviving branches.
    E[prod s_i * 1(completion)] = Omega unbiased: every complete solution
    path necessarily passes all survival filters.
    """
    W_ = W
    qp = 0
    pend = {}
    edges = {}
    mult = 1.0
    for _step in range(L_ROWS * W_):
        opts, _fr = _enum_opts(qp, pend, edges, libw, K, W_)
        if not opts:
            return 0.0
        survivors = []
        for opt in opts:
            undo = []
            if _try_apply(qp, pend, edges, opt, libw, W_, undo):
                survivors.append(opt)
            _undo_apply(pend, edges, undo)
        if not survivors:
            return 0.0
        if la2:
            viable = []
            for opt in survivors:
                undo = []
                _try_apply(qp, pend, edges, opt, libw, W_, undo)
                qp2, p2, e2 = _advance(qp, pend, edges, W_)
                o2, _ = _enum_opts(qp2, p2, e2, libw, K, W_)
                ok2 = False
                for opt2 in o2:
                    undo2 = []
                    if _try_apply(qp2, p2, e2, opt2, libw, W_, undo2):
                        ok2 = True
                    _undo_apply(p2, e2, undo2)
                    if ok2:
                        break
                _undo_apply(pend, edges, undo)
                if ok2:
                    viable.append(opt)
            if not viable:
                return 0.0
            survivors = viable
        mult *= len(survivors)
        opt = survivors[rng.randrange(len(survivors))]
        if not _try_apply(qp, pend, edges, opt, libw, W_, []):
            return 0.0
        qp, pend, edges = _advance(qp, pend, edges, W_)
    return mult


def knuth_strip(args):
    alpha, W, n_samples, seed = args
    lib = pl.build_library(alpha)
    libw = lib.weights
    K = lib.size
    allowed_ids_[0] = set(tuple(sorted(pl.LIGAND_ARMS[n])) for n in lib.alphabet)
    rng = random.Random(seed)
    ws = np.empty(n_samples)
    for s in range(n_samples):
        ws[s] = knuth_strip_one(W, libw, K, None, rng)
    alive = float((ws > 0).mean())
    mean = float(ws.mean())
    sem = float(ws.std(ddof=1) / math.sqrt(n_samples)) if n_samples > 1 else float('nan')
    return (alpha, W, mean, sem, alive)


def estimate_cell(alpha, W, n_samples):
    """Serial over 3 seeds (parallel across cells); return pooled mean and
    between-seed standard error."""
    res = [knuth_strip((alpha, W, n_samples, sd)) for sd in SEEDS]
    means = np.array([r[2] for r in res])
    alive = float(np.mean([r[4] for r in res]))
    pool = float(means.mean())
    sem_between = float(means.std(ddof=1) / math.sqrt(len(means))) if len(means) > 1 else float('nan')
    s = math.log(pool) / (W * L_ROWS) if pool > 0 else float('-inf')
    s_err = sem_between / (pool * W * L_ROWS) if pool > 0 and sem_between > 0 else float('nan')
    return dict(omega_hat=pool, omega_sem=sem_between, alive_rate=round(alive, 4),
                s=round(s, 4) if pool > 0 else None,
                s_err=round(s_err, 4) if pool > 0 else None,
                per_seed_means=[round(float(x), 1) for x in means])


def run_val(job):
    alpha, W, omega_exact, n_samples = job
    r = estimate_cell(alpha, W, n_samples)
    rel = abs(r['omega_hat'] - omega_exact) / omega_exact if r['omega_hat'] > 0 else float('inf')
    z = (r['omega_hat'] - omega_exact) / r['omega_sem'] if r['omega_sem'] else float('nan')
    return dict(alphabet='+'.join(alpha), W=W, omega_exact=omega_exact,
                rel_err=round(rel, 4), z=round(z, 3) if z == z else None,
                alive_rate=r['alive_rate'],
                per_seed_means=r['per_seed_means'])


def run_ext(job):
    alpha, W, n_samples = job
    return estimate_cell(alpha, W, n_samples)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-val', type=int, default=8000)
    ap.add_argument('--n-ext', type=int, default=25000)
    ap.add_argument('--procs', type=int, default=8)
    a = ap.parse_args()

    df = pd.read_csv(HERE / 'P0_A2b_TM_main_table.csv')
    ok = df[df['status'] == 'ok']
    ex = df[df['status'] == 'states_exceeded']
    print(f'main table: {len(ok)} ok / {len(ex)} exceeded')

    # ---- 1. verification ----
    val_whitelist = {('222+223+233', 3), ('223+233+333', 3),
                     ('222+223+233+333', 3)}
    val_jobs = [(tuple(r['alphabet'].split('+')), int(r['W']), float(r['omega']),
                 a.n_val) for _, r in ok.iterrows()
                if (r['alphabet'], int(r['W'])) in val_whitelist]
    print(f'verification: {len(val_jobs)} exact cells x {len(SEEDS)} seeds '
          f'x {a.n_val} samples ...', flush=True)
    with mp.Pool(a.procs) as pool:
        val_res = pool.map(run_val, val_jobs)
    vd = pd.DataFrame(val_res)
    vd.to_csv(HERE / 'P0_A2b_Knuth_verification.csv', index=False, encoding='utf-8-sig')
    print('\n=== verification (Omega_hat vs exact Omega) ===')
    for _, r in vd.iterrows():
        print(f"{r['alphabet']:22s} W={r['W']} rel_err={r['rel_err']:.4f} "
              f"z={r['z']} alive={r['alive_rate']}")
    good = vd[vd['alive_rate'] > 0]
    print(f"\nsummary: alive cells {len(good)}/{len(vd)}; "
          f"rel_err median {good['rel_err'].median():.4f}, "
          f"p90 {good['rel_err'].quantile(0.9):.4f}, max {good['rel_err'].max():.4f}; "
          f"|z|<2 coverage {100*((good['z'].abs() < 2)).mean():.0f}%")
    dead = vd[vd['alive_rate'] == 0]
    if len(dead):
        print('alive rate 0 (solutions too sparse, estimator not applicable):',
              [(r['alphabet'], r['W']) for _, r in dead.iterrows()])

    # ---- 2. extreme cells ----
    starve_skip = {('222+223', 5), ('222+233', 5), ('223+333', 5),
                   ('233+333', 5), ('222+223+333', 5), ('222+233+333', 5)}
    ext_jobs = [(tuple(r['alphabet'].split('+')), int(r['W']), a.n_ext)
                for _, r in ex.iterrows()
                if (r['alphabet'], int(r['W'])) not in starve_skip]
    print(f"\nextreme cells: {len(ext_jobs)} x {len(SEEDS)} seeds "
          f"x {a.n_ext} samples ...", flush=True)
    with mp.Pool(a.procs) as pool:
        ext_res = pool.map(run_ext, ext_jobs)
    rows = []
    for (alpha, W, _), r in zip(ext_jobs, ext_res):
        rows.append(dict(alphabet='+'.join(alpha), W=W, status='knuth_est',
                         omega=r['omega_hat'], omega_sem=r['omega_sem'],
                         L=L_ROWS, s=r['s'], s_err=r['s_err'],
                         alive_rate=r['alive_rate']))
    ed = pd.DataFrame(rows)
    ed.to_csv(HERE / 'P0_A2b_Knuth_estimates.csv', index=False, encoding='utf-8-sig')
    print('\n=== Knuth estimates for extreme cells ===')
    for _, r in ed.iterrows():
        print(f"{r['alphabet']:22s} W={r['W']} s={r['s']:.4f}+/-{r['s_err']:.4f} "
              f"Omega_hat={r['omega']:.3e} alive={r['alive_rate']}")

    # ---- 3. merge + 1/W extrapolation ----
    merged = []
    for _, r in ok.iterrows():
        d = dict(r)
        d['source'] = 'exact'
        merged.append(d)
    for _, r in ed.iterrows():
        merged.append(dict(alphabet=r['alphabet'], W=r['W'], status='knuth_est',
                           omega=r['omega'], L=L_ROWS, s=r['s'], source='knuth'))
    md = pd.DataFrame(merged)
    print('\n=== s(W) extrapolation (linear in 1/W) ===')
    summ = []
    for alpha, g in md.groupby('alphabet'):
        g = g.sort_values('W')
        if len(g) >= 2 and (g['s'] > -np.inf).all():
            x = 1.0 / g['W'].values
            y = g['s'].values
            b, s0 = np.polyfit(x, y, 1)
            src = '+'.join(sorted(set(g['source'])))
            summ.append(dict(alphabet=alpha, s_inf=round(s0, 4),
                             slope=round(b, 4), kpts=len(g), source=src))
            print(f"{alpha:22s} s(inf)~{s0:+.4f} (slope {b:+.3f}, "
                  f"W={list(g['W'].values)}, {src})")
    pd.DataFrame(summ).to_csv(HERE / 'P0_A2b_TM_extrapolation.csv',
                              index=False, encoding='utf-8-sig')
    print(f"\nextrapolation table written to P0_A2b_TM_extrapolation.csv "
          f"({len(summ)} alphabets)")


if __name__ == '__main__':
    main()
