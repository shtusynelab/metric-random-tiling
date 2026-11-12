# -*- coding: utf-8 -*-
"""
TSQ_Lambda_DP_series.py - exact counting W-series runner for the 4.8.8
Lambda library.

For each (W, lib) calls tsq_lambda_lib.dp_count_fast (vectorized frontier DP
with future-functional linear merging; exact Python-int counting) and appends
one result row per line to the CSV.
The control library {1,2,3} has a built-in anchor assertion Omega = 3^{4W}
(methodology anchor shared with TSQ_entropy_density_verification.csv).

Usage (server):
  python3 TSQ_Lambda_DP_series.py --Ws 3 4 5 6 --libs exp int \
      --out TSQ_Lambda_DP.csv [--max-states 150000000] [-v]
Memory scale (measured/estimated, for reporting):
  W=3 ~0.5-1 GB; W=4 measured 7-8 GB (pre-optimization) -> expected 1-2 GB
  (post-optimization); W=5 est 5-30 GB; W=6 est 20-150 GB (fail-closed:
  stop as soon as max_states is exceeded).
  Run serially; running multiple W in parallel is forbidden.
"""
import argparse, csv, math, os, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from tsq_lambda_lib import dp_count_row, dp_count_row_batched, M_EXP, M_INT

LIBS = {"exp": M_EXP, "int": M_INT}
FIELDS = ["W", "lib", "omega", "s_per_face", "peak_states", "t_s", "note"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--Ws", type=int, nargs="+", required=True)
    ap.add_argument("--libs", nargs="+", choices=["exp", "int"], default=["exp", "int"])
    ap.add_argument("--out", default=os.path.join(HERE, "TSQ_Lambda_DP.csv"))
    ap.add_argument("--max-states", type=int, default=150_000_000)
    ap.add_argument("--workers", type=int, default=1,
                    help="number of chunk-level parallel workers (Linux fork; 1 = serial)")
    ap.add_argument("--batched", action="store_true",
                    help="row-batched driver (dp_count_row_batched; W=5-scale memory control)")
    ap.add_argument("--batch-size", type=int, default=1_000_000)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    newfile = not os.path.exists(args.out)
    with open(args.out, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if newfile:
            w.writeheader()
        for W in args.Ws:
            for tag in args.libs:
                M = LIBS[tag]
                # Preflight (fail-closed): face-0 output ~ k^{3W} x b,
                # b ~ 38 (calibrated on W=4: 20017611/531441; W=3 same order
                # of magnitude). Refuse directly above the gate to avoid an
                # OOM endangering co-running jobs.
                b_est = 40
                est_face0 = len(M) ** (3 * W) * b_est
                if est_face0 > args.max_states and not args.batched:
                    # in batched mode the face-0 transient is bounded by the
                    # batch (~batch x 38); the preflight only blocks direct runs
                    w.writerow({"W": W, "lib": tag, "omega": "", "s_per_face": "",
                                "peak_states": "", "t_s": 0.0,
                                "note": ("PREFLIGHT-REFUSED: face0 states ~%d > "
                                         "cap %d (k^3W=%d x b~%d, W=4 calibrated)"
                                         % (est_face0, args.max_states,
                                            len(M) ** (3 * W), b_est))})
                    f.flush()
                    print(f"W={W} {tag}: PREFLIGHT-REFUSED (est face0 "
                          f"{est_face0:.3g} > {args.max_states})", flush=True)
                    continue
                t0 = time.time()
                try:
                    driver = dp_count_row_batched if args.batched else dp_count_row
                    om, pk, st = driver(W, M=M, max_states=args.max_states,
                                        verbose=args.verbose,
                                        workers=args.workers,
                                        batch_size=args.batch_size)
                except RuntimeError as ex:
                    w.writerow({"W": W, "lib": tag, "omega": "", "s_per_face": "",
                                "peak_states": "", "t_s": round(time.time() - t0, 1),
                                "note": f"ABORTED: {ex}"})
                    f.flush()
                    print(f"W={W} {tag}: ABORTED {ex}", flush=True)
                    continue
                dt = time.time() - t0
                pk = max((row[4] for row in st), default=0)
                s = math.log(om) / (2 * W * W) if om > 0 else ""
                note = ""
                if tag == "int":
                    anchor = 3 ** (4 * W)
                    note = "anchor3^%d=%s" % (4 * W, "OK" if om == anchor else "FAIL")
                    print(f"W={W} {tag}: Omega={om} anchor={'OK' if om == anchor else 'FAIL'}",
                          flush=True)
                    assert om == anchor, f"control-library anchor failed at W={W}"
                w.writerow({"W": W, "lib": tag, "omega": om,
                            "s_per_face": f"{s:.6f}" if s != "" else "",
                            "peak_states": pk, "t_s": round(dt, 1), "note": note})
                f.flush()
                print(f"W={W} {tag}: Omega={om} s={s if s == '' else round(s, 6)} "
                      f"peak={pk} t={dt:.1f}s", flush=True)


if __name__ == "__main__":
    main()
