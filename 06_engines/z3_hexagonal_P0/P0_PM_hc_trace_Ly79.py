# -*- coding: utf-8 -*-
"""P0_PM trace method, Ly=7/9 family: honeycomb {0,1}/{0,1,2} x7/x9 torus
counts Tr(T^Lx), Lx=2..20.

Scale: |S| = k^(2Ly): binary Ly7 2^14=16384 / Ly9 2^18=262144 (light);
arith Ly7 3^14=4,782,969 (light) / Ly9 3^18=387,420,489 (heavy; dict-based
approach infeasible).
Method:
  Build T = prefix-chunked parallel enumeration of all k^(2Ly) cut states
  (enumerate in code order, no parent sort), per-state column_transfer,
  int64 encoding; dead-target edges dropped (same semantics as the dict
  trace); scipy CSR.
  Count = Hutchinson randomized trace: v Rademacher; a single propagation
  chain w=T^k v gives all estimates v.w for k=2..20; mean +/- stderr.
  Exact integer trace is infeasible at 3^18 states; the randomized trace is
  the standard surrogate (tsq W=5 precedent).
Validation (all must pass):
  1) smoke: binary Ly=5 parallel pipeline vs archived exact CSV (pipeline
     calibration, runs locally);
  2) binary Ly=7/9: non-resonant points vs closed form Omega=2^(Lx+Ly+1)
     (end-to-end validation);
  3) small DFS anchors: 2xLy serial (skip on 1800s timeout), 3x7 chunked
     (skip on 7200s timeout).
Memory gate: check /proc/meminfo MemAvailable before heavy steps, exit if
insufficient (thresholds in constants).
Usage: python3 P0_PM_hc_trace_Ly79.py [smoke]
Output: P0_PM_torus_trace_shape_classes_Ly7Ly9.csv + DONE marker.
"""
import csv
import math
import sys
import time
from itertools import product
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import scipy.sparse as sp

import os

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from P0_PM_hc_trace import TorusCtx                     # noqa: E402
import P0_PM_torus_growth as tg                         # noqa: E402

OUT_CSV = HERE / "P0_PM_torus_trace_shape_classes_Ly7Ly9.csv"
DONE = HERE / "DONE"
LX_MIN, LX_MAX = 2, 20
N_PROBE = {"binary": 800, "arithmetic": 300}
BATCH = {"binary": 64, "arithmetic": 4}
NW_BUILD = 64 if os.name != "nt" else 30   # Windows handle limit 63
MEM_NEED_BUILD = 55 << 30        # Ly9 arith build+merge peak estimate ~55GB
MEM_NEED_PROBE = 45 << 30        # CSR + batched-vector peak estimate ~41GB

JOBS = [("Ly7", 7, "binary", (0, 1)),
        ("Ly7", 7, "arithmetic", (0, 1, 2)),
        ("Ly9", 9, "binary", (0, 1)),
        ("Ly9", 9, "arithmetic", (0, 1, 2))]


def mem_avail():
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return 1 << 62


def encode(s, k):
    code = 0
    for i, d in enumerate(s):
        code += d * k ** i
    return code


_G = {}


def _winit(Lx, Ly, alpha, m):
    _G["ctx"] = TorusCtx(3, Ly, alpha)
    _G["alpha"] = tuple(alpha)
    _G["k"] = len(alpha)
    _G["n"] = 2 * Ly
    _G["m"] = m


def _wchunk(pnum):
    """Enumerate all suffixes of prefix pnum (numeric order = code order);
    the prefix occupies the high digits m..n-1 and suffixes count up on a
    mixed-radix counter => src codes ascend within a chunk and code ranges
    of different chunks are disjoint."""
    k, n, m = _G["k"], _G["n"], _G["m"]
    alpha = _G["alpha"]
    pref = [(pnum // k ** j) % k for j in range(m)]
    pref_code = sum(d * k ** (n - m + j) for j, d in enumerate(pref))
    nsuf = k ** (n - m)
    src = []
    dst = []
    wt = []
    suf = [0] * (n - m)
    for num in range(nsuf):
        t = num
        for j in range(n - m):
            suf[j] = t % k
            t //= k
        code = pref_code + num * k ** m
        s = tuple(pref) + tuple(suf)
        o = _G["ctx"].column_transfer({s: 1})
        for ot, wt_ in o.items():
            src.append(code)
            dst.append(sum(d * k ** i for i, d in enumerate(ot)))
            wt.append(wt_)
    return (np.array(src, dtype=np.int64), np.array(dst, dtype=np.int64),
            np.array(wt, dtype=np.int64))


def build_parallel(Ly, alpha, tag):
    k = len(alpha)
    n = 2 * Ly
    m = 1
    while k ** m < 512 and m < n:
        m += 1
    ntask = k ** m
    t0 = time.time()
    print(f"  [{tag}] build: |S|<={k ** n}, {ntask} chunks x {k ** (n - m)}",
          flush=True)
    chunks = []
    with Pool(NW_BUILD, initializer=_winit,
              initargs=(3, Ly, alpha, m)) as pool:
        for i, (s, d, w) in enumerate(pool.imap_unordered(
                _wchunk, range(ntask), chunksize=1)):
            chunks.append((s, d, w))
            if (i + 1) % 100 == 0:
                print(f"    [build] {i + 1}/{ntask} chunks "
                      f"({time.time() - t0:.0f}s, "
                      f"avail {mem_avail() >> 30}GB)", flush=True)
    src = np.concatenate([c[0] for c in chunks])
    dst = np.concatenate([c[1] for c in chunks])
    wt = np.concatenate([c[2] for c in chunks])
    del chunks
    live = np.unique(src)
    n = len(live)
    src_i = np.searchsorted(live, src).astype(np.int64)
    idx = np.searchsorted(live, dst)
    valid = (idx < n) & (live[np.minimum(idx, n - 1)] == dst)
    print(f"  [{tag}] states {n}, edges {int(valid.sum())}/{len(dst)} "
          f"({time.time() - t0:.0f}s, avail {mem_avail() >> 30}GB)",
          flush=True)
    src_i = src_i[valid]
    dst_i = idx[valid].astype(np.int32)
    ww = wt[valid].astype(np.float64)
    order = np.argsort(src_i, kind="stable")
    src_i, dst_i, ww = src_i[order], dst_i[order], ww[order]
    counts = np.bincount(src_i, minlength=n)
    indptr = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    csr = sp.csr_matrix((ww, dst_i, indptr), shape=(n, n))
    print(f"  [{tag}] CSR nnz={csr.nnz} ({time.time() - t0:.0f}s, "
          f"avail {mem_avail() >> 30}GB)", flush=True)
    return csr, live, n, time.time() - t0


def hutchinson(csr, n, n_max, nprobe, seed, tag, batch):
    rng = np.random.default_rng(seed)
    est = np.zeros((nprobe, n_max + 1))
    t0 = time.time()
    done = 0
    while done < nprobe:
        nb = min(batch, nprobe - done)
        v = rng.choice(np.array([-1.0, 1.0]), size=(n, nb))
        w = v.copy()
        for k in range(1, n_max + 1):
            w = csr @ w
            if k >= 2:
                est[done:done + nb, k] = np.sum(v * w, axis=0)
        done += nb
        print(f"    [hutch] {tag} {done}/{nprobe} probes "
              f"({time.time() - t0:.0f}s)", flush=True)
    mean = est.mean(axis=0)
    stderr = est.std(axis=0, ddof=1) / math.sqrt(nprobe)
    return mean, stderr


def closed_form_binary(Lx, Ly, alpha):
    return (Lx + Ly + 1) * math.log(len(alpha))


def dfs_anchor(Lx, Ly, alpha, kind):
    """Independent DFS anchor; returns None on timeout/exception (non-fatal)."""
    try:
        ctx = tg.Ctx(Lx, Ly, alpha)
        t0 = time.time()
        if kind == "serial":
            om = tg.count_serial(ctx)
        else:
            om, _nch, timed_out = tg.count_chunked(
                ctx, n_workers=32, deadline=7200)
            if timed_out or om is None:
                print(f"  [DFS] {Lx}x{Ly} timeout -> skip", flush=True)
                return None
        print(f"  [DFS] {Lx}x{Ly}: Omega={om} ({time.time() - t0:.0f}s)",
              flush=True)
        return om
    except Exception as exc:                               # noqa: BLE001
        print(f"  [DFS] {Lx}x{Ly} skipped ({exc})", flush=True)
        return None


def run_job(tag, Ly, name, alpha, rows, fields, smoke):
    print(f"== {name} {tag} ==", flush=True)
    csr, live, n, t_build = build_parallel(Ly, alpha, tag)
    nprobe = 200 if smoke else N_PROBE[name]
    mean, stderr = hutchinson(csr, n, LX_MAX, nprobe,
                              seed=100000 * Ly + len(name), tag=tag,
                              batch=BATCH[name])
    del csr
    # ---- validation ----
    if name == "binary":
        npass = nfail = 0
        for Lx in range(LX_MIN, LX_MAX + 1):
            if Lx % Ly == 0:
                continue
            exact = closed_form_binary(Lx, Ly, alpha)
            err = abs(mean[Lx] - exact)
            tol = max(5 * stderr[Lx], 2e-3)
            ok = err <= tol
            npass, nfail = npass + ok, nfail + (not ok)
            print(f"  [val] {name} {Lx}x{Ly}: est {mean[Lx]:.4f} vs closed "
                  f"{exact:.4f} (err {err:.4f} tol {tol:.4f}) "
                  f"{'PASS' if ok else 'FAIL'}", flush=True)
        print(f"  [val] {name} {tag}: {npass} PASS / {nfail} FAIL",
              flush=True)
    if name == "arithmetic":
        om_dfs = dfs_anchor(2, Ly, alpha, "serial")
        if om_dfs is not None:
            err = abs(mean[2] - math.log(om_dfs))
            print(f"  [anchor] {name} 2x{Ly}: hutch {mean[2]:.4f} vs DFS "
                  f"{math.log(om_dfs):.4f} (err {err:.4f})", flush=True)
        if Ly == 7:
            om_dfs = dfs_anchor(3, Ly, alpha, "chunked")
            if om_dfs is not None:
                err = abs(mean[3] - math.log(om_dfs))
                print(f"  [anchor] {name} 3x{Ly}: hutch {mean[3]:.4f} vs DFS "
                      f"{math.log(om_dfs):.4f} (err {err:.4f})", flush=True)
    # ---- CSV rows ----
    av = "".join(str(v) for v in alpha)
    for Lx in range(LX_MIN, LX_MAX + 1):
        if mean[Lx] <= 0:
            continue
        rows.append(dict(
            alphabet=name, alphabet_values=av, Lx=Lx, Ly=Ly, F=Lx * Ly,
            E=None, method=f"hutchinson(n={nprobe})",
            Omega=int(round(mean[Lx])),
            lnOmega=round(math.log(mean[Lx]), 6),
            lnOmega_stderr=round(stderr[Lx] / mean[Lx], 6),
            lnOmega_per_F=round(math.log(mean[Lx]) / (Lx * Ly), 6),
            n_probes=nprobe, n_states=n, seconds=round(t_build, 1),
            status="validation" if name == "binary" else "hutchinson"))
    with open(OUT_CSV, "w", newline="", encoding="utf-8-sig") as fo:
        w = csv.DictWriter(fo, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"  CSV updated ({len(rows)} rows)", flush=True)


def main():
    smoke = "smoke" in sys.argv
    print(f"= honeycomb trace Ly=7/9 {time.strftime('%F %T')} =",
          flush=True)
    fields = ["alphabet", "alphabet_values", "Lx", "Ly", "F", "E",
              "method", "Omega", "lnOmega", "lnOmega_stderr",
              "lnOmega_per_F", "n_probes", "n_states", "seconds", "status"]
    rows = []
    if smoke:
        # pipeline calibration: binary Ly=5 parallel pipeline vs exact CSV
        # (16041/185523/...)
        run_job("smoke", 5, "binary", (0, 1), rows, fields, smoke=True)
    else:
        for tag, Ly, name, alpha in JOBS:
            run_job(tag, Ly, name, alpha, rows, fields, smoke=False)
    DONE.write_text("done", encoding="utf-8")
    print(f"all done -> {OUT_CSV} + DONE", flush=True)


if __name__ == "__main__":
    main()
