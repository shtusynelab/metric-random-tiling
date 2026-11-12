# -*- coding: utf-8 -*-
"""T_builder_v2.py - clean rebuild of the row transfer matrix for
4.8.8-Lambda.

Semantics are taken strictly from the authoritative dense version build_T.py
(verified W=2 = 6641); vectorized implementation, LIB-parameterized (exp/int
control libraries), component-domain checks throughout (no code-table
collisions), hard budget gates (refuse before materializing).

Semantics (one-to-one with build_T.py):
  state = (up segment, hv); up = 2W cells with one LIB value each (alpha,
  beta interleaved), hv = W cells;
  transition (up,hv) -> (nr,hn):
    c1 = up[2i].n - nr[2(i+1)%W].n     c2 = up[2i].m - nr[2(i+1)%W].m
    c3 = up[2(i+1)%W+1].n - nr[2i+1].n c4 = up[2(i+1)%W+1].m - nr[2i+1].m
    parity: (c1+c3)%2==0 (per cell)
    hst(i) = ((c1+c3)/2, c2+c4)  -> hn = hv + hst must be in LIB cellwise
    vst(i) = ((c1-c3)/2, c2-c4); a closed loop requires sum(vst)=0; weight
    w = number of legal V-walk starting points
  Omega(W) = trace(T^W).
Indexing convention matches build_T.py / ref_T_W2.py (itertools.product
order: cell 0 is the highest digit), so W=2 can be compared elementwise
against ref_T_W2.npy.

Anchor ladder: W=2 exp 6641 / int 6561=3^8; W=3 exp 563553 / int
531441=3^12; W=4 exp 52864817 / int 43046721=3^16; W=5 int
3486784401=3^20 (known analytically).

Usage:
  python T_builder_v2.py --W 2 --lib exp --job selftest   # compare vs ref_T_W2.npy
  python T_builder_v2.py --W 3 --lib exp --job trace      # exact block trace
  python T_builder_v2.py --W 5 --lib int --job hutch --probes 2000 --seed 1
  python T_builder_v2.py --W 5 --lib exp --job info       # report scale only
"""
import argparse
import math
import time

import numpy as np
import scipy.sparse as sp

LIBS = {
    "exp": [(1, 0), (0, 1), (1, 1)],
    "int": [(1, 0), (2, 0), (3, 0)],
}


class FlatLUT:
    """Flat lookup table (m,n) -> library index / -1; idx = (m-lo_m)*nn +
    (n-lo_n) as int32."""

    def __init__(self, lib, W):
        ms = [p[0] for p in lib]
        ns = [p[1] for p in lib]
        self.lo_m = min(ms) - 2 * W - 2
        self.lo_n = min(ns) - 2 * W - 2
        nn = max(ns) + 2 * W + 3 - self.lo_n + 1
        nm = max(ms) + 2 * W + 3 - self.lo_m + 1
        grid = np.full((nm, nn), -1, dtype=np.int8)
        for k, (m, n) in enumerate(lib):
            grid[m - self.lo_m, n - self.lo_n] = k
        self.flat = grid.reshape(-1)
        self.nn = nn

    def idx(self, M, N):
        """Same-shape int arrays M,N -> flat index int32 (guaranteed not to
        overflow)."""
        return ((M.astype(np.int32) - self.lo_m) * self.nn +
                (N.astype(np.int32) - self.lo_n))

    def member(self, M, N):
        return self.flat[self.idx(M, N)] >= 0

    def libidx(self, M, N):
        return self.flat[self.idx(M, N)]


def seg_digits(nseg, ncell):
    """id -> per-cell value indices (cell 0 = highest digit, same order as
    itertools.product). Returns (nseg,ncell) int8."""
    ids = np.arange(nseg, dtype=np.int64)
    out = np.empty((nseg, ncell), dtype=np.int8)
    for c in range(ncell):
        out[:, c] = ((ids // 3 ** (ncell - 1 - c)) % 3).astype(np.int8)
    return out


def build_pairs(W, lib, u_chunk=64, verbose_every=200):
    """Segment-pair precomputation: returns the live pairs
    (up_ids, nr_ids, w, H1, H2). H1/H2 = the m/n components of hst,
    (npairs,W) int8. Per-chunk peak < ~400 MB (W <= 5)."""
    lut = FlatLUT(lib, W)
    nseg = 3 ** (2 * W)
    digs = seg_digits(nseg, 2 * W)
    segvals = np.array(lib, dtype=np.int8)[digs]     # (nseg,2W,2)
    SM, SN = segvals[..., 0], segvals[..., 1]
    i = np.arange(W)
    i1 = (i + 1) % W
    c_a, c_b = 2 * i, 2 * i + 1
    c1b, c3b = 2 * i1, 2 * i1 + 1
    up_l, nr_l, w_l, H1_l, H2_l = [], [], [], [], []
    t0 = time.time()
    for u0 in range(0, nseg, u_chunk):
        u1_ = min(nseg, u0 + u_chunk)
        SMu, SNu = SM[u0:u1_], SN[u0:u1_]
        C1 = SNu[:, c_a][:, None, :] - SN[:, c1b][None, :, :]   # (cu,nseg,W)
        C2 = SMu[:, c_a][:, None, :] - SM[:, c1b][None, :, :]
        C3 = SNu[:, c3b][:, None, :] - SN[:, c_b][None, :, :]
        C4 = SMu[:, c3b][:, None, :] - SM[:, c_b][None, :, :]
        par = ((C1 + C3) & 1) == 0
        ok = par.all(axis=2)
        V1 = (C1 - C3) >> 1
        V1[~par] = 0
        V2 = C2 - C4
        ok &= (V1.sum(axis=2) == 0) & (V2.sum(axis=2) == 0)
        if not ok.any():
            continue
        cum1 = np.cumsum(V1, axis=2, dtype=np.int16)
        cum2 = np.cumsum(V2, axis=2, dtype=np.int16)
        cnt = np.zeros(ok.shape, dtype=np.int8)
        for (sm, sn) in lib:
            pm = (sm + cum1).astype(np.int8)
            pn = (sn + cum2).astype(np.int8)
            inlib = lut.member(pm, pn)               # (cu,nseg,W) bool
            cnt += (inlib.all(axis=2) & ok).astype(np.int8)
        uu, vv = np.nonzero(cnt)
        if uu.size:
            H1 = ((C1 + C3) >> 1).astype(np.int8)
            H2 = (C2 + C4).astype(np.int8)
            up_l.append(uu + u0)
            nr_l.append(vv)
            w_l.append(cnt[uu, vv])
            H1_l.append(H1[uu, vv])
            H2_l.append(H2[uu, vv])
        if verbose_every and (u0 // u_chunk) % verbose_every == 0:
            done = sum(a.shape[0] for a in up_l)
            print(f"  pairs u {u1_}/{nseg} live pairs so far {done} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    up_ids = np.concatenate(up_l)
    nr_ids = np.concatenate(nr_l)
    w = np.concatenate(w_l)
    H1 = np.concatenate(H1_l, axis=0)
    H2 = np.concatenate(H2_l, axis=0)
    return up_ids, nr_ids, w, H1, H2


def assemble_csr(W, lib, up_ids, nr_ids, w, H1, H2, pair_chunk=50_000,
                 verbose_every=10, nnz_cap=700_000_000):
    """Expand live pairs x hv into a CSR matrix (n = 3^{3W}).
    Budget gate: incremental - accumulate the actual nnz per pair chunk and
    fail-closed once the cap is exceeded (the naive upper bound
    live_pairs x nhv is far too pessimistic; measured survival rate ~5%).
    Indices int32 (n < 2^31)."""
    lut = FlatLUT(lib, W)
    nseg = 3 ** (2 * W)
    nhv = 3 ** W
    n = nseg * nhv
    hvd = seg_digits(nhv, W)
    hvvals = np.array(lib, dtype=np.int8)[hvd]
    HM, HN = hvvals[..., 0], hvvals[..., 1]
    wexp = (3 ** np.arange(W - 1, -1, -1)).astype(np.int64)
    P = up_ids.shape[0]
    rr_l, cc_l, vv_l = [], [], []
    nnz_acc = 0
    t0 = time.time()
    for p0 in range(0, P, pair_chunk):
        p1_ = min(P, p0 + pair_chunk)
        hn_m = H1[p0:p1_, None, :] + HM[None, :, :]     # (pc,nhv,W) int16
        hn_n = H2[p0:p1_, None, :] + HN[None, :, :]
        li = lut.libidx(hn_m, hn_n)                     # (pc,nhv,W) int8
        valid = (li >= 0).all(axis=2)
        if not valid.any():
            continue
        pos_hn = np.where(li >= 0, li, 0).astype(np.int64)
        pos_hn = (pos_hn * wexp[None, None, :]).sum(axis=2)
        pp, hh = np.nonzero(valid)
        nnz_acc += pp.shape[0]
        assert nnz_acc <= nnz_cap, \
            f"accumulated nnz {nnz_acc} exceeds cap {nnz_cap}, fail-closed"
        rr_l.append((pos_hn[pp, hh] * nseg + nr_ids[p0 + pp]).astype(np.int32))
        cc_l.append((hh.astype(np.int64) * nseg + up_ids[p0 + pp]).astype(np.int32))
        vv_l.append(w[p0 + pp].astype(np.float64))
        if verbose_every and (p0 // pair_chunk) % verbose_every == 0:
            print(f"  csr pairs {p1_}/{P} nnz so far {nnz_acc} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    rr = np.concatenate(rr_l)
    cc = np.concatenate(cc_l)
    vval = np.concatenate(vv_l)
    T = sp.csr_matrix((vval, (rr, cc)), shape=(n, n))
    T.sum_duplicates()
    return T


def exact_trace_power(T, k, chunk=1024, verbose_every=20):
    """Exact trace(T^k): X = I[:, block], X <- T X applied k times, then sum
    the offset diagonal of X: sum_j X[j+c0, j] = sum_{i in block} (T^k)_{ii}.
    WARNING: np.trace(X) must NOT be used (that would be X[p,p] =
    (T^k)[p,p+c0], a misaligned diagonal that varies with the block size).
    float64 carries exact integers throughout (< 2^53), no rounding; assert
    near-integer per block."""
    n = T.shape[0]
    tot = 0.0
    t0 = time.time()
    for c0 in range(0, n, chunk):
        c1_ = min(n, c0 + chunk)
        b = c1_ - c0
        X = np.zeros((n, b))
        X[np.arange(c0, c1_), np.arange(b)] = 1.0
        for _ in range(k):
            X = T @ X
            m = np.abs(X).max()
            assert m < 2.0 ** 52, f"intermediate value {m} exceeds the float64 integer range"
        tot += float(X[np.arange(c0, c1_), np.arange(b)].sum())
        if verbose_every and (c0 // chunk) % verbose_every == 0:
            print(f"  trace block {c1_}/{n} cumulative {tot:.0f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    ri = round(tot)
    assert abs(tot - ri) < 1e-3, f"trace not integral: {tot}"
    return ri


def hutchinson(T, k, probes, seed, verbose_every=200):
    """Hutchinson +/-1 estimator for tr(T^k); asserts the matvec intermediates
    stay in the integer range."""
    rng = np.random.default_rng(seed)
    n = T.shape[0]
    est = np.empty(probes)
    for p in range(probes):
        z = rng.choice(np.array([-1.0, 1.0]), size=n)
        y = z
        for _ in range(k):
            y = T @ y
            m = np.abs(y).max()
            if m >= 2.0 ** 52:
                raise AssertionError(f"matvec intermediate {m} exceeds 2^52")
        est[p] = float(z @ y)
        if verbose_every and (p + 1) % verbose_every == 0:
            a = est[:p + 1]
            print(f"  probe {p+1}: {a.mean():.10g} +/- "
                  f"{a.std(ddof=1)/math.sqrt(p+1):.3g}", flush=True)
    return est


def power_iter(T, iters=200, seed=12345):
    """Power iteration for lambda_max (only a directional indicator for the
    nearly degenerate spectrum; known to have misread W=2/4)."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(T.shape[0])
    lam = 0.0
    for t in range(iters):
        x = T @ x
        nx = np.abs(x).max()
        if nx == 0:
            return 0.0
        lam = nx
        x = x / nx
    return lam


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--W", type=int, required=True)
    ap.add_argument("--lib", choices=list(LIBS), default="exp")
    ap.add_argument("--job",
                    choices=["selftest", "info", "trace", "hutch",
                             "savecsr", "hutchfile"],
                    required=True)
    ap.add_argument("--probes", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--u-chunk", type=int, default=64)
    ap.add_argument("--file", default="T.npz")
    args = ap.parse_args()
    W, lib = args.W, LIBS[args.lib]
    t0 = time.time()
    print(f"[{args.lib} W={W} {args.job}] segment-pair precomputation...",
          flush=True)
    up_ids, nr_ids, w, H1, H2 = build_pairs(W, lib, u_chunk=args.u_chunk)
    P = up_ids.shape[0]
    print(f"live segment pairs = {P} (of {3**(4*W)}, share {P/3**(4*W):.3e}); "
          f"mean w = {w.mean():.4f}, w histogram {np.bincount(w).tolist()} "
          f"({time.time()-t0:.0f}s)", flush=True)
    nseg, nhv, n = 3 ** (2 * W), 3 ** W, 3 ** (3 * W)
    print(f"n = {n} = {nseg}x{nhv}; nnz upper bound = {P*nhv}; "
          f"CSR upper-bound memory = {P*nhv*12/1e9:.2f} GB", flush=True)
    if args.job == "info":
        return
    T = assemble_csr(W, lib, up_ids, nr_ids, w, H1, H2)
    print(f"T: dim={n} nnz={T.nnz} mean out-degree {T.nnz/n:.3f} "
          f"({time.time()-t0:.0f}s total)", flush=True)
    if args.job == "selftest":
        assert W == 2, "selftest only supports W=2"
        Td = T.toarray()
        tr = float(np.trace(Td @ Td))
        print(f"trace(T^2) = {tr:.0f} (exp anchor 6641 / int anchor 6561)",
              flush=True)
        if args.lib == "exp":
            ref = np.load("ref_T_W2.npy")
            same = np.array_equal(Td, ref)
            md = np.abs(Td - ref).max()
            print(f"elementwise equal to ref_T_W2.npy: {same} (max|diff|={md})",
                  flush=True)
        return
    if args.job == "trace":
        tr = exact_trace_power(T, W)
        anchor = ({"exp": {2: 6641, 3: 563553, 4: 52864817}.get(W),
                   "int": 3 ** (4 * W)}[args.lib])
        ok = "OK" if (anchor is None or tr == anchor) else f"MISMATCH (anchor {anchor})"
        s = math.log(tr) / (2 * W * W)
        g = tr / 3 ** (4 * W)
        print(f"=== exact trace(T^{W}) = {tr} {ok}", flush=True)
        print(f"    Omega={tr} s={s:.6f} g={g:.6f}", flush=True)
        if W == 4:
            tA = time.time()
            A = (T @ T).tocsr()
            tr_b = float(A.multiply(A.T).sum())
            ri = round(tr_b)
            assert abs(tr_b - ri) < 1e-3, f"sum(T^2 o T^2^T) not integral: {tr_b}"
            print(f"[cross-check] sum(T^2 o T^2^T) = {ri} (block method {tr}, "
                  f"{'match' if ri == tr else 'MISMATCH'}) "
                  f"({time.time()-tA:.0f}s, T^2 nnz={A.nnz})", flush=True)
    if args.job == "savecsr":
        sp.save_npz(args.file, T, compressed=False)
        print(f"saved {args.file} (nnz={T.nnz})", flush=True)
        return
    if args.job == "hutchfile":
        T = sp.load_npz(args.file)
        print(f"loaded {args.file}: dim={T.shape[0]} nnz={T.nnz}", flush=True)
        lam = power_iter(T, iters=300)
        lnW = math.log(lam) if lam > 0 else float("nan")
        print(f"lambda_max (power iteration, 300 iters) = {lam:.6f} "
              f"lnLambda/W = {lnW/W:.6f}", flush=True)
        est = hutchinson(T, W, args.probes, args.seed)
        m, se = est.mean(), est.std(ddof=1) / math.sqrt(len(est))
        print(f"=== trace(T^{W}) = {m:.10g} +/- {se:.3g} "
              f"(seed {args.seed}, {args.probes} probes)", flush=True)
        ri = round(m)
        print(f"    integer snap = {ri} (deviation {abs(m-ri):.4g}, "
              f"{abs(m-ri)/max(abs(m),1)*100:.4f}%)", flush=True)
        s = math.log(m) / (2 * W * W)
        g = m / 3 ** (4 * W)
        print(f"    Omega~={m:.6g} s={s:.6f} g={g:.6f}", flush=True)
        np.savetxt(f"hutch_W{W}_{args.lib}_s{args.seed}.csv", est,
                   header="probe_estimate", comments="")
        return
    if args.job == "hutch":
        est = hutchinson(T, W, args.probes, args.seed)
        m, se = est.mean(), est.std(ddof=1) / math.sqrt(len(est))
        print(f"=== trace(T^{W}) = {m:.10g} +/- {se:.3g} "
              f"(seed {args.seed}, {args.probes} probes)", flush=True)
        ri = round(m)
        print(f"    integer snap = {ri} (deviation {abs(m-ri):.4g}, "
              f"{abs(m-ri)/max(abs(m),1)*100:.4f}%)", flush=True)
        s = math.log(m) / (2 * W * W)
        g = m / 3 ** (4 * W)
        print(f"    Omega~={m:.6g} s={s:.6f} g={g:.6f}", flush=True)


if __name__ == "__main__":
    main()
