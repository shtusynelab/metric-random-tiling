# -*- coding: utf-8 -*-
"""7x3 arithmetic independent cross-check: prefix-chunked DFS (a separate
code path from the trace method). Reference value = the 7x3 output of the
trace method P0_PM_hc_trace.py. Memory < 10 MB per worker, 6 workers.
"""
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from P0_PM_torus_growth import Ctx, count_chunked  # noqa: E402

if __name__ == "__main__":
    t0 = time.time()
    ctx = Ctx(7, 3, (0, 1, 2))
    omega, n_chunks, timed_out = count_chunked(ctx, 6, 7200.0, k_min=16)
    tag = "DNF" if timed_out else f"Omega={omega}"
    print(f"[DFS 7x3 arithmetic] {tag} chunks={n_chunks} "
          f"({time.time() - t0:.0f}s)", flush=True)
