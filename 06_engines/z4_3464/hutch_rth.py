# -*- coding: utf-8 -*-
"""
Server-only driver (originally run on the internal cluster); pass explicit
arguments when used locally.

hutch_rth.py - Hutchinson stochastic trace for the 3.4.6.4 torus count
Tr(T^L).
Purpose: the L=11 safety point (tests whether the upper branch keeps rising
vs a crossover). The W=11 result enters Fig 1c in the stochastic-trace
convention (+/-1 s.d.); precedent = the 4.8.8 W=5 point.

Estimator: Tr(T^L) ~ (1/m) sum_i v_i^T T^L v_i with Rademacher +/-1 probes,
batched X = n x b. T is a nonnegative integer count matrix; bias and variance
are calibrated for free against the known exact values at W=9/W=10.

== Registered predictions (made before any W=11 computation) ==
  lnOmega(11) global 10-point quadratic fit
    (lnOmega = 0.7332L + 0.1859L^2 + 2.9122):                     33.47
  lnOmega(11) second-difference extrapolation
    (increments 4.024/4.498 -> ~4.97):                            33.99
  Perimeter-law reference line (increments frozen ~2.3):          31.4
  Early form Omega ~ 2^{3.3L} (fit at L<=5, excluded by L>=6):    25.2
  Acceptance line: lnOmega(11) >= 33.0 judged asymptotic-side;
  below 33.0 = crossover alert, stop and re-examine.

== Calibration protocol (must pass before production) ==
  W=9 (exact 44,571,806,560) and W=10 (exact 4,002,790,772,868), m=2e5:
  produce a convergence table (running mean vs exact, z scores should be
  ~N(0,1)) + single-probe distribution diagnostics
  (std/skew/kurtosis/max|h|). If z deviates systematically or std fails to
  shrink as 1/sqrt(m) => stop.

Usage (run in the directory containing rth_torus_trace_sparse.py):
  calibration: python3 hutch_rth.py --W 10 --L 10 --m 200000 --exact 4002790772868 \
             --probe-workers 40 --tag calW10
  production:  python3 hutch_rth.py --W 11 --L 11 --m 200000 \
             --build-workers 6 --probe-workers 24 --tag prodW11 --done DONE_prodW11
"""
import os, sys, json, time, math
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import rth_torus_trace_sparse as rts   # noqa: E402

# ---- fail-closed gate: the parallel build path rts._build_batch has no gate
# in the original; add one ----
rts.MAX_INTER = 26_000_000


def _gated_build_batch(batch):
    """rts._build_batch plus an intermediate-state gate after each face
    (zero semantic change)."""
    import numpy as _np
    from itertools import product
    tm = rts._G["tm"]
    k = rts._G["k"]
    alpha0 = rts._G["alpha0"]
    fr0 = rts._G["fr0"]
    done0 = rts._G["done0"]
    A = range(alpha0, alpha0 + k)
    id2mid = rts._G["id2mid"]
    Minv = rts._G["Minv"]

    def canon(fr):
        def key(e):
            uv = Minv @ _np.array(id2mid[e], float)
            return (int(round(uv[0])) % tm.t.W, round(float(uv[1]), 4))
        return sorted(fr, key=key)

    def to_canon(states, fr):
        o = canon(fr)
        out = {}
        for st, cnt in states.items():
            d = dict(st)
            kk = tuple(d[e] for e in o)
            out[kk] = out.get(kk, 0) + cnt
        return out

    def process_column(states, j, done):
        for fi in [f for f in tm.order if tm.col[f] == j]:
            _, eqs, _ = tm.face_eqs[fi]
            segs = tm.t.face_edges[fi]
            eids = [eid for eid, _ in segs]
            coef = {eid: eqs[i] for i, (eid, _) in enumerate(segs)}
            new_states = {}
            for st, cnt in states.items():
                cur = dict(st)
                old_e = [e for e in eids if e in cur]
                new_e = [e for e in eids if e not in cur]
                acc0 = [0, 0, 0, 0]
                for e in old_e:
                    m = coef[e]
                    v = cur[e]
                    for i in range(4):
                        acc0[i] += m[i] * v
                for combo in product(A, repeat=len(new_e)):
                    acc = list(acc0)
                    for e, v in zip(new_e, combo):
                        m = coef[e]
                        for i in range(4):
                            acc[i] += m[i] * v
                    if any(acc):
                        continue
                    upd = dict(cur)
                    for e, v in zip(new_e, combo):
                        upd[e] = v
                    nxt = tuple(sorted(upd.items()))
                    new_states[nxt] = new_states.get(nxt, 0) + cnt
            if len(new_states) > rts.MAX_INTER:
                raise MemoryError(
                    f"in-face intermediate states exceed gate {rts.MAX_INTER}")
            states = new_states
            for e in eids:
                done[e] = done.get(e, 0) + 1
            merged = {}
            for st, cc in states.items():
                key = tuple(sorted((e, v) for e, v in st
                                   if done.get(e, 0) < len(tm.edge_faces[e])))
                merged[key] = merged.get(key, 0) + cc
            states = merged
        return states

    part = {}
    for s_in in batch:
        out = process_column({tuple(zip(fr0, s_in)): 1}, 1, dict(done0))
        outs = {}
        for st, cnt in to_canon(out, tm.frontier_after(1)).items():
            outs[st] = outs.get(st, 0) + cnt
        part[s_in] = outs
    return part


rts._build_batch = _gated_build_batch


# ---- Hutchinson batched probes ----
def _init_probe(T, L):
    rts._G["T"] = T
    rts._G["Lh"] = L


def _hutch_chunk(args):
    """h_i = v_i^T T^L v_i for a chunk of probes; fork inherits _G['T']."""
    seed0, nprobes, batch = args
    T = rts._G["T"]
    L = rts._G["Lh"]
    rng = np.random.default_rng(seed0)
    n = T.shape[0]
    hs = np.empty(nprobes, dtype=np.float64)
    done = 0
    while done < nprobes:
        b = min(batch, nprobes - done)
        V = (rng.integers(0, 2, size=(b, n)).astype(np.float64)
             * 2.0 - 1.0).T                     # n x b
        X = V.copy()
        for _ in range(L):
            X = T @ X
        hs[done:done + b] = np.sum(V * X, axis=0)
        done += b
    return hs


def hutch_trace(T, L, m, base_seed, batch, probe_workers, tag, ck_path, log):
    """Chunked parallel probes with checkpointing; returns the h array
    (np.float64)."""
    import multiprocessing as mp
    hs_all = None
    if os.path.exists(ck_path):
        try:
            hs_all = np.load(ck_path)["hs"]
            log(f"  hutch resume: {len(hs_all)} probes already present")
        except Exception:
            hs_all = None
    todo_m = m - (0 if hs_all is None else len(hs_all))
    if todo_m > 0:
        chunk = max(500, min(2500, todo_m // 8 + 1))
        tasks = []
        ci = 0
        left = todo_m
        while left > 0:
            c = min(chunk, left)
            tasks.append((base_seed * 104729 + ci * 7919, c, batch))
            left -= c
            ci += 1
        pieces = [] if hs_all is None else [hs_all]
        t0 = time.time()
        with mp.Pool(min(probe_workers, len(tasks)),
                     initializer=_init_probe, initargs=(T, L)) as pool:
            for k, h in enumerate(pool.imap_unordered(_hutch_chunk, tasks)):
                pieces.append(h)
                np.savez_compressed(ck_path, hs=np.concatenate(pieces))
                acc = np.concatenate(pieces)
                mu = float(acc.mean())
                sd = float(acc.std(ddof=1))
                log(f"  [{tag}] probes {len(acc)}/{m}  mean={mu:.6e}  "
                    f"rel.sd={(sd / abs(mu)) if mu else float('nan'):.3e}  "
                    f"stderr/mean={(sd / math.sqrt(len(acc)) / abs(mu)) if mu else float('nan'):.3e}  "
                    f"({time.time() - t0:.0f}s)")
        hs = np.concatenate(pieces)
    else:
        hs = hs_all
    return hs


def diagnose(hs, m_target, exact, tag, log):
    """Convergence table + distribution diagnostics."""
    mu = float(hs.mean())
    sd = float(hs.std(ddof=1))
    n = len(hs)
    m2 = float(((hs - mu) ** 2).mean())
    m4 = float(((hs - mu) ** 4).mean())
    skew = float(((hs - mu) ** 3).mean() / max(m2 ** 1.5, 1e-300))
    kurt = float(m4 / max(m2 ** 2, 1e-300))
    log(f"  [{tag}] diagnostics: n={n} mean={mu:.10e} std={sd:.4e} "
        f"stderr={sd / math.sqrt(n):.4e} skew={skew:.2f} kurt={kurt:.1f} "
        f"max|h|={float(np.abs(hs).max()):.4e}")
    if exact:
        err = mu - exact
        z = err / (sd / math.sqrt(n))
        log(f"  [{tag}] vs exact {exact}: rel.dev={(err / exact):+.3e}  "
            f"z={z:+.2f}")
    log(f"  [{tag}] convergence table (running mean vs exact, z scores):")
    for mm in sorted({100, 300, 1000, 3000, 10000, 30000, 100000,
                      200000, len(hs)}):
        if mm > len(hs):
            continue
        a = hs[:mm]
        muj = float(a.mean())
        sdj = float(a.std(ddof=1))
        line = f"    m={mm:>7d}  lnOmega_hat={math.log(abs(muj)):.4f}"
        if exact:
            line += (f"  dln={math.log(abs(muj) / exact):+.4f}"
                     f"  z={(muj - exact) / (sdj / math.sqrt(mm)):+.2f}")
        log(line)
    est = mu
    sdm = sd / math.sqrt(n)
    lo, hi = est - sdm, est + sdm
    ln = math.log(est)
    ln_err = max(abs(math.log(hi) - ln), abs(ln - math.log(max(lo, 1e-300))))
    return est, sdm, ln, ln_err


REGISTERED = {
    "lnOmega11_global_fit": 33.47,
    "lnOmega11_second_diff": 33.99,
    "lnOmega11_perimeter_frozen": 31.4,
    "lnOmega11_early_perimeter_excluded": 25.2,
    "accept_line": 33.0,
}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--W", type=int, required=True)
    ap.add_argument("--L", type=int, default=0)
    ap.add_argument("--m", type=int, default=200000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seed", type=int, default=20261001)
    ap.add_argument("--build-workers", type=int, default=1)
    ap.add_argument("--probe-workers", type=int, default=8)
    ap.add_argument("--max-inter", type=int, default=26_000_000,
                    help="fail-closed gate on in-face intermediate states "
                         "(default 26e6; W=11 needs larger)")
    ap.add_argument("--exact", type=float, default=0.0)
    ap.add_argument("--tag", default="hutch")
    ap.add_argument("--done", default="")
    args = ap.parse_args()
    L = args.L or args.W
    rts.MAX_INTER = args.max_inter

    def log(msg):
        print(msg, flush=True)

    log(f"= hutch_rth {args.tag} W={args.W} L={L} m={args.m} "
        f"{time.strftime('%F %T')} =")
    t0 = time.time()
    rows, cols, vals, n, _ = rts.build_T_triplets(
        args.W, log=log, workers=args.build_workers)
    import scipy.sparse as sp
    T = sp.csr_matrix((np.array(vals, dtype=np.float64),
                       (np.array(rows), np.array(cols))), shape=(n, n))
    nnz = T.nnz
    log(f"  |S|={n}  nnz={nnz}  avg_deg={nnz / n:.2f}  "
        f"build {time.time() - t0:.0f}s")
    if args.W >= 2:
        want = 2 ** (2 * args.W + 1)
        log(f"  state-law check |S|==2^(2W+1)={want}: "
            f"{'PASS' if n == want else 'DEVIATE (recorded as is)'}")

    ck = os.path.join(HERE, f"hutch_{args.tag}_probes.npz")
    hs = hutch_trace(T, L, args.m, args.seed, args.batch,
                     args.probe_workers, args.tag, ck, log)
    est, sdm, ln, ln_err = diagnose(hs, args.m,
                                    args.exact if args.exact else 0.0,
                                    args.tag, log)
    res = {"W": args.W, "L": L, "m": int(len(hs)), "n_states": int(n),
           "nnz": int(nnz), "est": est, "sd": float(hs.std(ddof=1)),
           "stderr": sdm, "lnOmega": ln, "lnOmega_err": ln_err,
           "exact": args.exact, "registered": REGISTERED}
    if args.exact:
        res["z_vs_exact"] = (est - args.exact) / sdm
    with open(os.path.join(HERE, f"hutch_{args.tag}_result.json"),
              "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    log(f"  ==> [{args.tag}] Tr(T^{L}) ~ {est:.6e} +/- {sdm:.2e}  "
        f"lnOmega = {ln:.4f} +/- {ln_err:.4f}  "
        f"(total {time.time() - t0:.0f}s)")
    if args.W == 11:
        log("  == registered-prediction comparison (lnOmega(11)) ==")
        for k, v in REGISTERED.items():
            log(f"     {k}: {v}")
        verdict = ("ASYMPTOTIC-SIDE (>=33.0, trend confirmed)" if ln >= 33.0
                   else "CROSSOVER ALERT (<33.0) - stop and re-examine")
        log(f"  == verdict: {verdict} ==")
    if args.done:
        with open(os.path.join(HERE, args.done), "w") as f:
            f.write(json.dumps(res) + "\n")
        log(f"  DONE marker written: {args.done}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
