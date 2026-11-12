# -*- coding: utf-8 -*-
"""trace_hutch.py - vectorized construction of the 4.8.8-Lambda row transfer
matrix + Hutchinson stochastic trace.

Omega(W) = trace(T^W) (W=2 independently verified by build_T.py = 6641).
Vectorization: a segment = independent (alpha, beta) per cell (2W cells);
SM/SN (nseg x 2W) int8; per u-block broadcast computation of C1..C4 ->
parity/loop/V-walk filtering -> H-library expansion -> COO blocks -> CSR.
Usage: python trace_hutch.py --W 4 --probes 600 [--check]
  --check: W=2 exact (trace = 6641) + W=3 stochastic (vs 563553) priors.
"""
import sys, math, time, argparse
import numpy as np
import scipy.sparse as sp

LIB = np.array([[1, 0], [0, 1], [1, 1]], dtype=np.int64)   # (m,n) x 3
LUT = np.full(4, -1, dtype=np.int64)                        # code = 2m+n -> LIB index
LUT[1], LUT[2], LUT[3] = 0, 1, 2
def lutp(W):
    L = np.full(6 * W + 8, -1, dtype=np.int64)
    L[1 + 3 * W], L[2 + 3 * W], L[3 + 3 * W] = 0, 1, 2
    return L


def seg_arrays(W):
    """Segment enumeration: 2W cells with 3 values each -> nseg = 3^{2W};
    SM/SN (nseg, 2W) int8."""
    nseg = 3 ** (2 * W)
    ids = np.arange(nseg, dtype=np.int64)
    SM = np.zeros((nseg, 2 * W), dtype=np.int8)
    SN = np.zeros((nseg, 2 * W), dtype=np.int8)
    for c in range(2 * W):
        d = (ids // 3 ** (2 * W - 1 - c)) % 3
        SM[:, c] = LIB[d, 0]
        SN[:, c] = LIB[d, 1]
    return nseg, SM, SN


def hv_arrays(W):
    nhv = 3 ** W
    ids = np.arange(nhv, dtype=np.int64)
    HM = np.zeros((nhv, W), dtype=np.int8)
    HN = np.zeros((nhv, W), dtype=np.int8)
    for c in range(W):
        d = (ids // 3 ** (W - 1 - c)) % 3
        HM[:, c] = LIB[d, 0]
        HN[:, c] = LIB[d, 1]
    libidx = LUT[2 * HM + HN]           # (nhv, W) in {0,1,2}
    pos = np.zeros(nhv, dtype=np.int64)
    for c in range(W):
        pos += libidx[:, c] * 3 ** (W - 1 - c)
    return nhv, HM, HN, libidx, pos


def assemble_T(W, u_chunk=128, verbose=True):
    """Vectorized construction of the sparse T (dim = 3^{3W}). Returns a csr
    matrix."""
    nseg, SM, SN = seg_arrays(W)
    nhv, HM, HN, hvlib, hvpos = hv_arrays(W)
    LUTP = lutp(W)
    n = nseg * nhv
    i1 = np.roll(np.arange(W), -1)          # (i+1) % W
    blocks = []
    t0 = time.time()
    nnz_tot = 0
    for u0 in range(0, nseg, u_chunk):
        u1_ = min(nseg, u0 + u_chunk)
        SMu = SM[u0:u1_]                     # (cu, 2W)
        SNu = SN[u0:u1_]
        cu = u1_ - u0
        # C1..C4: (cu, nseg, W) int16
        C1 = SNu[:, 2*np.arange(W)][:, None, :] - SN[:, 2*i1][None, :, :]
        C2 = SMu[:, 2*np.arange(W)][:, None, :] - SM[:, 2*i1][None, :, :]
        C3 = SNu[:, 2*i1+1][:, None, :] - SN[:, 2*np.arange(W)+1][None, :, :]
        C4 = SMu[:, 2*i1+1][:, None, :] - SM[:, 2*np.arange(W)+1][None, :, :]
        par = ((C1 + C3) % 2 == 0)           # (cu, nseg, W)
        ok = par.all(axis=2)                 # all parity checks pass
        # V steps (vst) and the closed loop
        V1 = (C1 - C3) // 2
        V2 = C2 - C4
        sV1 = V1.sum(axis=2); sV2 = V2.sum(axis=2)
        ok &= (sV1 == 0) & (sV2 == 0)
        if not ok.any():
            continue
        # V walk: partial sums P_k (3 starting points x all k intermediate
        # states in LIB)
        cum1 = np.cumsum(V1, axis=2)         # (cu, nseg, W)
        cum2 = np.cumsum(V2, axis=2)
        cnt = np.zeros((cu, nseg), dtype=np.int64)
        for si, start in enumerate(LIB):
            posm = start[0] + cum1           # start + cumulative (last point = start, closed loop)
            posn = start[1] + cum2
            inlib = ((posm >= 0) & (posm <= 1) & (posn >= 0) & (posn <= 1) &
                 ~((posm == 0) & (posn == 0)))   # library points: component domain + not (0,0)
            cnt += inlib.all(axis=2) * (par.all(axis=2) & (sV1 == 0) & (sV2 == 0))
        cnt = cnt * ok   # double guard
        # H steps: hst(i) = ((c1+c3)/2, c2+c4)
        H1 = (C1 + C3) // 2
        H2 = C2 + C4
        # expand (v, hv): hn = HVvals + hst must be in the library -> COO
        # per-v sub-blocks avoid huge tensors: nv full dim 6561 x 81 hv x W -
        # (nv,81,W,2) int16 ~4MB, fine
        # target H values (per i): hn = HV + hst -> (cu, nv, nhv, W) broadcast
        hn_m = H1[:, :, None, :] + HM[None, None, :, :]   # (cu, nv, nhv, W)
        hn_n = H2[:, :, None, :] + HN[None, None, :, :]
        code = 2 * hn_m + hn_n
        vrange = ((hn_m >= 0) & (hn_m <= 1) & (hn_n >= 0) & (hn_n <= 1) &
                  ~((hn_m == 0) & (hn_n == 0)))             # (cu, nv, nhv, W) library membership
        valid = vrange.all(axis=3)                          # (cu, nv, nhv)
        # target H library index
        tgt_libidx = (2*hn_m + hn_n - 1).astype(np.int8)    # after the range check code in {1,2,3} -> index 0..2
        w_exp = (3 ** (W - 1 - np.arange(W))).astype(np.int32)
        pos_hn = (tgt_libidx.astype(np.int32) *
                  w_exp[None, None, None, :]).sum(axis=3, dtype=np.int32)
        # source state hv position = hvpos (independent of u)
        # COO: for valid (u,v,hv): src = hvpos[hv]*nseg + u_id; dst = pos_hn*nseg + v_id
        uu, vv, hh = np.nonzero(valid)
        u_ids = uu + u0
        w_seg = cnt[uu, vv]                                # weight (= number of V-walk starting points)
        src = hvpos[hh] * nseg + u_ids
        dst = pos_hn[uu, vv, hh] * nseg + vv
        blk = sp.coo_matrix((w_seg.astype(np.float64), (dst, src)),
                            shape=(n, n)).tocsr()
        blocks.append(blk)
        nnz_tot += blk.nnz
        if verbose:
            print(f"  u-block {u0}/{nseg}: nnz so far {nnz_tot} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    T = blocks[0]
    for b in blocks[1:]:
        T = T + b
    T = T.tocsr()
    T.sum_duplicates()
    return T, nnz_tot


def hutchinson_trace(T, k, probes, seed):
    """Hutchinson +/-1 estimator for tr(T^k)."""
    rng = np.random.default_rng(seed)
    n = T.shape[0]
    est = np.zeros(probes)
    for p in range(probes):
        z = rng.choice(np.array([-1.0, 1.0]), size=n)
        y = z
        for _ in range(k):
            y = T @ y
        est[p] = z @ y
    return est


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--W", type=int, required=True)
    ap.add_argument("--probes", type=int, default=600)
    ap.add_argument("--u-chunk", type=int, default=64)
    ap.add_argument("--seed", type=int, default=777)
    args = ap.parse_args()
    W = args.W
    t0 = time.time()
    T, nnz = assemble_T(W, u_chunk=args.u_chunk)
    print(f"T assembled: nnz={T.nnz}  {time.time()-t0:.0f}s", flush=True)
    if W == 2:
        # exact check: dense trace(T^2)
        Td = T.toarray()
        tr = float(np.trace(Td @ Td))
        print(f"W=2 exact trace(T^2) = {tr:.0f} (anchor 6641)"
              f"{' OK' if abs(tr-6641) < 0.5 else ' MISMATCH'}", flush=True)
        return
    est = hutchinson_trace(T, W, args.probes, args.seed)
    m, se = est.mean(), est.std(ddof=1) / math.sqrt(len(est))
    om = m
    s = math.log(om) / (2 * W)
    g = om / 3 ** (4 * W)
    print(f"=== W={W}: trace(T^{W}) = {om:.5g} +/- {se:.3g}", flush=True)
    print(f"    Omega ~= {om:.4g}; g(W) = Omega/3^{4*W} = {g:.3f}; "
          f"s(W) = {s:.4f}", flush=True)


if __name__ == "__main__":
    main()
