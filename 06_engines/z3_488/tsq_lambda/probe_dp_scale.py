# -*- coding: utf-8 -*-
"""W=3,4 DP scale probe: state peaks / runtimes, to extrapolate the server
job budget."""
import sys, time, math
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tsq_lambda_lib import dp_count, M_EXP, M_INT

for W in (3, 4):
    for tag, M in (("exp", M_EXP), ("int", M_INT)):
        t0 = time.time()
        om, pk = dp_count(W, M=M)
        dt = time.time() - t0
        F = 2 * W * W
        s = math.log(om) / F
        anchor = ""
        if M is M_INT:
            ok = (om == 3 ** (4 * W))
            anchor = f"  anchor 3^{4*W}={'OK' if ok else 'FAIL!'}"
            assert ok
        print(f"W={W} {tag}: Omega={om} s(W)={s:.6f} peak={pk} t={dt:.1f}s{anchor}",
              flush=True)
