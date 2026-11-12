# -*- coding: utf-8 -*-
"""P0-A2b reinforced Knuth long run, v2.

Convention: N_SAMPLES=2e5 per seed gives row-level +/-0.012 with 4 seeds
(the older 10k-sample configuration gives +/-0.054).

v2 conventions:
  - progress lines only reference fields of the run_one return value;
  - main() never references variables from outside its own loop scope;
  - KN_LONG_SMOKE=1 environment variable switches to smoke mode
    (200 samples x 1 seed x 1 alphabet) to validate the completion path
    (result collection + CSV + printing) before the real run.
"""
import sys, os, time, math
import argparse
from pathlib import Path
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import P0_A2b_Knuth_reliability_verification as kn

SMOKE = os.environ.get("KN_LONG_SMOKE", "") == "1"
N_SAMPLES = 200 if SMOKE else 200_000
N_SEEDS = 1 if SMOKE else 4
N_WORKERS = 2 if SMOKE else 32

CALIB = [(("222", "233"), 3), (("222", "233"), 4),
         (("222", "223", "333"), 3), (("222", "223", "333"), 4),
         (("222", "233", "333"), 3), (("222", "233", "333"), 4)]
LOAD = [(("223", "233"), 4), (("223", "233"), 5),
        (("222", "223", "233"), 4), (("222", "223", "233"), 5),
        (("223", "233", "333"), 4), (("223", "233", "333"), 5),
        (("222", "223", "233", "333"), 4),
        (("222", "223", "233", "333"), 5)]
TASKS = (CALIB[:1] if SMOKE else CALIB + LOAD)


def run_one(args):
    alpha, W, seed = args
    t0 = time.time()
    alpha_s, W_, mean, sem, alive = kn.knuth_strip(
        (alpha, W, N_SAMPLES, seed))
    return dict(alphabet='+'.join(alpha), W=W, seed=seed,
                s=(f"{math.log(mean) / (W * kn.L_ROWS):.6f}"
                   if mean > 0 else "nan"),
                s_sem=(f"{sem / (mean * W * kn.L_ROWS):.6f}"
                       if mean > 0 and sem > 0 else "nan"),
                alive=str(alive), omega_mean=f"{mean:.4e}",
                seconds=f"{time.time()-t0:.0f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=str(HERE / 'P0_A2b_Knuth_long.csv'),
                    help='output CSV path (default: P0_A2b_Knuth_long.csv here)')
    args = ap.parse_args()
    CSV_OUT = args.out
    done = set()
    results = []
    if os.path.exists(CSV_OUT) and not SMOKE:
        results = [dict(r) for _, r in pd.read_csv(CSV_OUT).iterrows()]
        done = {(r['alphabet'], int(r['W']), int(r['seed']))
                for r in results}
    csv_out = CSV_OUT + ".smoke" if SMOKE else CSV_OUT
    todo = [(a, W, s) for a, W in TASKS for s in range(1, N_SEEDS + 1)
            if ('+'.join(a), W, s) not in done]
    print(f'Knuth-long v2 smoke={SMOKE}: {len(todo)} tasks '
          f'({N_SAMPLES:,} samples per seed, {N_WORKERS} workers) '
          f'started {time.strftime("%m-%d %H:%M:%S")}', flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        futs = {pool.submit(run_one, t): t for t in todo}
        for i, fut in enumerate(as_completed(futs)):
            res = fut.result()
            results.append(res)
            pd.DataFrame(results).to_csv(csv_out, index=False,
                                         encoding='utf-8-sig')
            print(f"[{i+1}/{len(todo)}] {res['alphabet']:16s} W={res['W']} "
                  f"seed={res['seed']} s={res['s']}+/-{res['s_sem']}"
                  f" ({res['seconds']}s) @{time.strftime('%H:%M:%S')}",
                  flush=True)
    print(f'Knuth-long v2 done in {(time.time()-t0)/60:.1f} min', flush=True)


if __name__ == '__main__':
    main()
