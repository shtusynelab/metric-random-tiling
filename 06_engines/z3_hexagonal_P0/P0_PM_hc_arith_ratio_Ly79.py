# -*- coding: utf-8 -*-
"""P0_PM exact DFS anchors: arithmetic 2xLy / 3xLy, Ly=7/9.

Purpose: five-point ratio protocol of the arith panel of fig2 --
lambda(Ly) = ln(Omega(3xLy)/Omega(2xLy)), consistent with the same protocol
at Ly=2/3/5 (numbers already in CSV). The 3x9 count may take hours
(chunked); run in the background.
Output: P0_PM_hc_arith_ratio.csv (row by row) + DONE marker.
"""
import csv
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import P0_PM_torus_growth as tg                         # noqa: E402

OUT_CSV = HERE / "P0_PM_hc_arith_ratio.csv"
DONE = HERE / "DONE"
ALPHA = (0, 1, 2)
NW = 48
JOBS = [(2, 7, 3600), (3, 7, 14400), (2, 9, 14400), (3, 9, 259200)]


def main():
    print(f"= arith ratio anchors {time.strftime('%F %T')} =\n"
          f"workers={NW}", flush=True)
    rows = []
    fields = ["Lx", "Ly", "Omega", "lnOmega", "lambda_ratio", "seconds",
              "status"]
    for Lx, Ly, deadline in JOBS:
        t0 = time.time()
        try:
            ctx = tg.Ctx(Lx, Ly, ALPHA)
            om, _nch, timed_out = tg.count_chunked(
                ctx, n_workers=NW, deadline=deadline,
                chunk_timeout=(1800.0 if (Lx, Ly) == (3, 9) else 30.0))
            if timed_out or om is None:
                print(f"[TIMEOUT] {Lx}x{Ly} (deadline {deadline}s)",
                      flush=True)
                rows.append(dict(Lx=Lx, Ly=Ly, Omega="", lnOmega="",
                                 lambda_ratio="", seconds=round(
                                     time.time() - t0, 1),
                                 status="timeout"))
                continue
        except Exception as exc:                           # noqa: BLE001
            print(f"[ERROR] {Lx}x{Ly}: {exc}", flush=True)
            rows.append(dict(Lx=Lx, Ly=Ly, Omega="", lnOmega="",
                             lambda_ratio="", seconds=round(
                                 time.time() - t0, 1),
                             status=f"error:{exc}"))
            continue
        lnom = __import__("math").log(om)
        # ratio protocol lambda: pair with the Lx=2 row of the same Ly
        lam = ""
        prev = [r for r in rows if r["Ly"] == Ly and r["Lx"] == 2
                and r["Omega"] != ""]
        if prev:
            lam = round(lnom - prev[0]["lnOmega"], 6)
        rows.append(dict(Lx=Lx, Ly=Ly, Omega=om, lnOmega=round(lnom, 6),
                         lambda_ratio=lam, seconds=round(time.time() - t0, 1),
                         status="ok"))
        print(f"[OK] {Lx}x{Ly}: Omega={om} lnOmega={lnom:.4f} "
              f"lambda_ratio={lam} ({time.time() - t0:.0f}s)", flush=True)
        with open(OUT_CSV, "w", newline="", encoding="utf-8-sig") as fo:
            w = csv.DictWriter(fo, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
    DONE.write_text("done", encoding="utf-8")
    print(f"all done -> {OUT_CSV} + DONE", flush=True)


if __name__ == "__main__":
    main()
