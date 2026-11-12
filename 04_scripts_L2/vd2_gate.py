# -*- coding: utf-8 -*-
"""Server-only driver (originally run on the internal cluster); pass explicit
arguments when used locally.

vd2_gate.py -- mirror gate and equation-of-state analysis for the
re-sampled tension ensembles (vd_torus_run_v2.py outputs).

Waits for all expected npz files, then for each |h| pools the chains and
reports:
  * m(h) = <sum l>/F per face and the mirror identity m(h) + m(-h) = 12;
  * the value marginals P_{+h}(v) against the mirrored P_{-h}(4-v),
    with per-configuration two-sample errors;
  * the h = 0 anchor from the archived uniform ensemble (torus13_a123.npz);
  * the entropy cost by trapezoid integration of m(h), anchored at h = 0.

usage: python3 vd2_gate.py <ensdir> [expect_chains]
writes <ensdir>/gate_summary.csv and prints everything.
ASCII-only source.
"""
import csv
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths_common as PC  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent / "external"))

H_LIST = (-0.30, -0.15, 0.15, 0.30)          # mirror pairs (-h, +h)
F = 169                                       # faces per 13x13 torus


def wait_for(ensdir, nchain, timeout=8 * 3600):
    t0 = time.time()
    while True:
        files = sorted(ensdir.glob("vd2_h*_c*.npz"))
        if len(files) >= 4 * nchain:
            time.sleep(20)                     # let writers finish
            return files
        if time.time() - t0 > timeout:
            print("TIMEOUT waiting for files", flush=True)
            return files
        time.sleep(60)


def pool(files, h, nchain):
    cfgs = []
    for k in range(nchain):
        cand = [f for f in files
                if f.name.startswith(f"vd2_h{h:+.2f}_c{k}")]
        if not cand:
            raise SystemExit(f"missing chain {k} for h={h}")
        d = np.load(cand[0])
        cfgs.append(d["cfgs"].astype(np.int64))
    return np.concatenate(cfgs, axis=0)


def main():
    ensdir = Path(sys.argv[1])
    nchain = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    files = wait_for(ensdir, nchain)
    print(f"found {len(files)} chain files", flush=True)

    E = None
    data = {}
    for h in H_LIST:
        X = pool(files, h, nchain)
        E = X.shape[1]
        data[h] = X
        print(f"h={h:+.2f}: {X.shape[0]} configs pooled", flush=True)

    # h = 0 anchor from the archived uniform ensemble
    anchor = PC.DATA / "raw" / "torus13_a123.npz"
    if not anchor.exists():
        anchor = ensdir / "torus13_a123.npz"
    d0 = np.load(anchor)
    X0 = d0["cfgs"].astype(np.int64)

    def m_of(X):
        return X.sum(axis=1) / F

    rows = []
    for h in (-0.30, -0.15, 0.0, 0.15, 0.30):
        X = X0 if h == 0.0 else data[h]
        mm = m_of(X)
        se = mm.std(ddof=1) / math.sqrt(len(mm))
        mean_l = X.mean()
        # marginals per configuration
        pv = {v: (X == v).mean(axis=1) for v in (1, 2, 3)}
        rows.append(dict(h=h, n=len(mm), m_mean=mm.mean(), m_se=se,
                         mean_l=mean_l,
                         P1=pv[1].mean(), P1_se=pv[1].std(ddof=1) / math.sqrt(len(mm)),
                         P2=pv[2].mean(), P2_se=pv[2].std(ddof=1) / math.sqrt(len(mm)),
                         P3=pv[3].mean(), P3_se=pv[3].std(ddof=1) / math.sqrt(len(mm))))
    print("\n=== m(h) equation of state ===")
    for r in rows:
        print(f"  h={r['h']:+.2f}: m = {r['m_mean']:.4f} +- {r['m_se']:.4f}"
              f"   P(1)={r['P1']:.4f} P(2)={r['P2']:.4f} P(3)={r['P3']:.4f}",
              flush=True)

    print("\n=== mirror identity ===")
    for h in (0.15, 0.30):
        ap, am = data[h], data[-h]
        mp, mm = m_of(ap), 4.0 - m_of(am)         # mirror of -h maps l -> 4-l
        d = mp.mean() - mm.mean()
        se = math.sqrt(mp.var(ddof=1) / len(mp) + mm.var(ddof=1) / len(mm))
        print(f"  |h|={h:.2f}: per-edge m+ + m- = {(m_of(ap).mean() + m_of(am).mean()):.4f}"
              f" (target 4)   difference {d:+.5f} +- {se:.5f} ({abs(d)/se:.2f} sigma)", flush=True)
        for v in (1, 2, 3):
            pa = (ap == v).mean(axis=1)
            pb = (am == (4 - v)).mean(axis=1)
            dd = pa.mean() - pb.mean()
            s = math.sqrt(pa.var(ddof=1) / len(pa) + pb.var(ddof=1) / len(pb))
            print(f"     P_P({v})={pa.mean():.4f} vs mirrored P_M({4-v})={pb.mean():.4f}"
                  f"  diff={dd:+.4f} ({abs(dd)/s:.2f} sigma)", flush=True)

    print("\n=== entropy cost (trapezoid, h=0 anchor) ===")
    hs = np.array([r["h"] for r in rows])
    ms = np.array([r["m_mean"] for r in rows])
    o = np.argsort(hs)
    hs, ms = hs[o], ms[o]
    cum = np.concatenate([[0.0], np.cumsum((ms[1:] + ms[:-1]) / 2 * np.diff(hs))])
    for h, m, c in zip(hs, ms, cum):
        print(f"  h={h:+.2f}: Delta_s/F = {h*m - c:+.4f}", flush=True)

    out = ensdir / "gate_summary.csv"
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
