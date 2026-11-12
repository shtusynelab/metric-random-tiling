# -*- coding: utf-8 -*-
"""
Server-only driver (originally run on the internal cluster); pass explicit
arguments when used locally.

rth_torus_trace_sparse.py - sparse/blocked version of
Omega_tor(L x L) = Tr(T^L).
Used for the L=8 morphology decision: |S| = 2^{2L+1} = 131072, so a dense
matrix of 137 GB is infeasible; sparse CSR + blocked unit-vector trace
(memory <= ~3 GB).

Algorithm: Tr(T^L) = sum_s (T^L)[s,s]; take unit-vector blocks X = I[:, B]
(n x chunk), apply X <- T X a total of L times, and accumulate X[j, j] for
j in B. scipy CSR @ dense block.
Verification: L=5,6,7 must reproduce 86956 / 1,114,700 / 23,325,388 exactly
(cross-checked against the dense version) before L=8 is counted.
The T construction is the same mathematics as rth_torus_trace.build_T
(sparse-triplet version).
"""
import os, sys, math, time
import numpy as np
from collections import Counter
from itertools import product

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE
for p in (HERE, os.path.join(ROOT, "z4_3464")):
    if p not in sys.path:
        sys.path.insert(0, p)
from rth_cyl_tm import CylinderRTH, CylStripTM  # noqa: E402
import scipy.sparse as sp  # noqa: E402

MAX_INTER = 12_000_000    # in-face intermediate-state gate; raised after
                         # W=10 legitimately tripped the earlier 2e6 gate
                         # (single-seed-column completions ~3^{1.4W},
                         # millions)



_G = {}

def _init_worker_trace(T):
    _G["T"] = T


def _init_worker(tm, k, alpha0, fr0, done0, id2mid, Minv):
    _G.update(tm=tm, k=k, alpha0=alpha0, fr0=fr0, done0=done0,
              id2mid=id2mid, Minv=Minv)


def _build_batch_tagged(tagged):
    bi, batch = tagged
    return bi, _build_batch(batch)


def _build_batch(batch):
    """Bulk single-column transfer for one batch of states (fork inherits
    _G['tm'] etc.; zero serialization)."""
    import numpy as _np
    tm, k, alpha0, fr0, done0 = (_G["tm"], _G["k"], _G["alpha0"],
                                 _G["fr0"], _G["done0"])
    A = range(alpha0, alpha0 + k)
    id2mid = _G["id2mid"]
    Minv = _G["Minv"]

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


def build_T_triplets(W, k=2, alpha0=0, log=None, workers=1):
    """Bulk single-column transfer operator of the width-W cylinder
    (sparse triplets + state table)."""
    t0 = time.time()
    c = CylinderRTH(W, 3)
    tm = CylStripTM(c)
    A = range(alpha0, alpha0 + k)
    id2mid = {v: kk for kk, v in c.edge_ids.items()}
    Minv = c._Minv

    def canon(fr):
        def key(e):
            uv = Minv @ np.array(id2mid[e], float)
            return (int(round(uv[0])) % W, round(float(uv[1]), 4))
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
                    if len(new_states) > MAX_INTER:
                        raise MemoryError(
                            f"in-face intermediate states exceed gate "
                            f"{MAX_INTER}")
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

    done0 = {}
    states = to_canon(process_column({(): 1}, 0, done0), tm.frontier_after(0))
    n = len(states)
    ordered = sorted(states)
    fr0 = canon(tm.frontier_after(0))
    if workers > 1 and n >= 4096:
        import multiprocessing as mp
        import pickle
        if n > 1_000_000:          # large W: single-seed-column completion
            workers = min(workers, 12)  # counts in the tens of millions; cap
        ck_dir = os.path.join(HERE, f"ck_build_W{W}")
        os.makedirs(ck_dir, exist_ok=True)
        all_batches = [(bi, ordered[i:i + 8192])
                       for bi, i in enumerate(range(0, n, 8192))]
        todo = [(bi, bs) for bi, bs in all_batches
                if not os.path.exists(os.path.join(ck_dir, f"b{bi:05d}.pkl"))]
        T = {}
        for bi, bs in all_batches:
            fck = os.path.join(ck_dir, f"b{bi:05d}.pkl")
            if os.path.exists(fck):
                with open(fck, "rb") as f:
                    for s_in, outs in pickle.load(f).items():
                        T[s_in] = Counter(outs)
        t0c = time.time()
        if todo:
            with mp.Pool(workers, initializer=_init_worker,
                         initargs=(tm, k, alpha0, fr0, done0, id2mid,
                                   Minv)) as pool:
                for bi, part in pool.imap_unordered(
                        _build_batch_tagged, todo):
                    for s_in, outs in part.items():
                        T[s_in] = Counter(outs)
                    with open(os.path.join(ck_dir, f"b{bi:05d}.pkl"),
                              "wb") as f:
                        pickle.dump(part, f, protocol=4)
                    if log:
                        print(f"    T-build batch {bi+1}/{len(all_batches)} "
                              f"|T|={len(T)} ({time.time()-t0c:.0f}s)",
                              flush=True)
        if log:
            print(f"    T-build done x{workers} (including "
                  f"{len(all_batches)-len(todo)} checkpointed batches, "
                  f"{time.time()-t0:.0f}s)", flush=True)
    else:
        T = {}
        for i, s_in in enumerate(ordered):
            out = process_column({tuple(zip(fr0, s_in)): 1}, 1, dict(done0))
            T[s_in] = Counter()
            for st, cnt in to_canon(out, tm.frontier_after(1)).items():
                T[s_in][st] += cnt
    idx = {s: i for i, s in enumerate(sorted(T))}
    rows, cols, vals = [], [], []
    for s, row in T.items():
        for s2, cnt in row.items():
            rows.append(idx[s])
            cols.append(idx[s2])
            vals.append(cnt)
    return rows, cols, vals, n, idx


def _trace_worker_tagged(tagged):
    pi, s0, e, L, c = tagged
    return pi, _trace_worker((s0, e, L, c))


def _trace_worker(args):
    """Partial trace sum for one block range (fork inherits _G['T'])."""
    start, end, L, chunk = args
    T = _G["T"]
    X = np.zeros((end - start, min(chunk, end - start))) if False else None
    m_all = end - start
    total = 0.0
    for c0 in range(start, end, chunk):
        m = min(chunk, end - c0)
        idx = np.arange(c0, c0 + m)
        X = np.zeros((T.shape[0], m))
        X[idx - 0, np.arange(m)] = 1.0
        for _ in range(L):
            X = T @ X
        total += float(np.sum(X[idx, np.arange(m)]))
    return total


def trace_power(rows, cols, vals, n, L, chunk=2048, log=None, workers=1, W=None, skip_states=0, fine=0, fine_ranges=()):
    """Tr(T^L) sparse blocked (multiprocessing-capable; memory =
    workers x n x chunk x 8 B)."""
    T = sp.csr_matrix((np.array(vals, dtype=np.float64),
                       (np.array(rows), np.array(cols))), shape=(n, n))
    nnz = T.nnz
    if workers > 1 and n >= 8192:
        import multiprocessing as mp
        import pickle
        ck = os.path.join(HERE, f"ck_trace_W{W}_L{L}.pkl")
        parts = {}
        if os.path.exists(ck):
            try:
                with open(ck, "rb") as f:
                    loaded = pickle.load(f)
                if isinstance(loaded, dict):
                    parts = loaded
                    print(f"    trace resume: {len(parts)} segments already "
                          f"present", flush=True)
            except Exception:
                pass
        if fine and fine_ranges:
            step = fine
            tasks = []
            j = 100
            for (a, b) in fine_ranges:
                for s0 in range(a, b, step):
                    if j not in parts:
                        tasks.append((j, s0, min(s0 + step, b), L, chunk))
                    j += 1
        else:
            per = (n + workers - 1) // workers
            tasks = [(pi, s0, min(s0 + per, n), L, chunk)
                     for pi, s0 in enumerate(range(0, n, per))
                     if pi not in parts]
        if tasks:
            t0 = time.time()
            nt = min(len(tasks), 6)   # 6 x 32 GiB transient = 192 GB (safety
                                      # margin; 12 workers at once OOMs)
            with mp.Pool(nt, initializer=_init_worker_trace,
                         initargs=(T,)) as pool:
                for pi, val in pool.imap_unordered(_trace_worker_tagged,
                                                   tasks):
                    parts[pi] = val
                    with open(ck, "wb") as f:
                        pickle.dump(parts, f)
                    if log:
                        print(f"    trace segment {pi+1}/{len(tasks)} done "
                              f"({time.time()-t0:.0f}s)", flush=True)
        total = float(sum(parts.values()))

        r = round(total)
        assert abs(total - r) < 1e-6 * max(1.0, abs(total)), f"non-integer {total}"
        return int(r), nnz
    total = 0.0
    t0 = time.time()
    for start in range(0, n, chunk):
        idx = np.arange(start, min(start + chunk, n))
        m = len(idx)
        X = np.zeros((n, m))
        X[idx, np.arange(m)] = 1.0
        for _ in range(L):
            X = T @ X
        total += float(np.sum(X[idx, np.arange(m)]))
        if log:
            print(f"    trace block {start}/{n}: running total {total:.6g} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    r = round(total)
    assert abs(total - r) < 1e-6 * max(1.0, abs(total)), f"non-integer {total}"
    return int(r), nnz


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--Ws", default="5,6,7,8")
    ap.add_argument("--Ls", default="")
    ap.add_argument("--fine", type=int, default=0)
    ap.add_argument("--fine_ranges", default="")     # default = same as W
    args = ap.parse_args()
    Ws = [int(x) for x in args.Ws.split(",")]
    Ls = [int(x) for x in args.Ls.split(",")] if args.Ls else Ws
    known = {2: 128, 3: 1036, 4: 8800, 5: 86956, 6: 1114700, 7: 23325388}
    import multiprocessing as mp
    workers = min(64, mp.cpu_count() - 4)
    print(f"= sparse Tr(T^L) workers={workers} {time.strftime('%F %T')} =",
          flush=True)
    for W, L in zip(Ws, Ls):
        t0 = time.time()
        print(f"W={W} (torus {L}x{L}, k=2):", flush=True)
        rows, cols, vals, n, _ = build_T_triplets(W, log=True, workers=workers)
        print(f"  |S|={n} nnz={len(vals)} build {time.time()-t0:.0f}s",
              flush=True)
        # memory model: steady state workers x n x chunk x 8; old+new coexist
        # during rebind => transient x2, plus ~8 GB in the parent; budget
        # 260 GB
        chunk = 4096 if n <= 524288 else 1024   # W=10: X = n x 1024 x 8 =
                                                # 16 GiB/worker (coexists with
                                                # parallel tasks)
        tw = max(1, min(8, int(260e9 / (2 * 8 * n * chunk))))
        fine_ranges = tuple(
            tuple(int(x) for x in seg.split(":"))
            for seg in args.fine_ranges.split(";") if seg)
        om, _nnz = trace_power(rows, cols, vals, n, L, chunk=chunk,
                               log=(W >= 7), workers=tw, W=W,
                               fine=args.fine, fine_ranges=fine_ranges)
        ln = math.log(om)
        s_face = ln / (6 * L * L)
        tag = ""
        if L in known:
            tag = "PASS" if om == known[L] else f"FAIL (known {known[L]})"
        print(f"  ==> Omega_tor({L}x{L},k=2) = {om}  lnOmega={ln:.3f}  "
              f"s/face={s_face:.4f}  s*L={s_face*L:.4f}  {tag} "
              f"({time.time()-t0:.0f}s)", flush=True)
        if L in known and om != known[L]:
            return 1
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
