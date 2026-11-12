# -*- coding: utf-8 -*-
"""
SNH_numerical_verification.py -- independent rigidity check for the
(3,3,3,3,6) snub hexagonal tiling.
Verifies dim_R = dim_Q = 1 for k = 1..5, same conventions as
z4_3464/RTH_numerical_verification.py part 7.
snub_lib.py in this directory is the canonical archived copy (a duplicate
in z4_3464/ is kept for joint regression).
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
# snub_lib's Q(sqrt(d)) arithmetic (field_rank_qd/QD etc.) comes from
# RTH_numerical_verification -- the archived copy still depends on that module
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_ROOT, "z4_3464"))
from snub_lib import dims_snub


def main():
    n_ok = 0
    for k in range(1, 6):
        dR, dQ = dims_snub(k)
        ok = (dR == 1 and dQ == 1)
        n_ok += ok
        print(f"snub hexagonal k={k}: dim_R={dR} dim_Q={dQ} (expect 1,1) "
              f"{'PASS' if ok else 'FAIL'}")
    print(f"{n_ok}/5 PASS")
    return 0 if n_ok == 5 else 1


if __name__ == "__main__":
    sys.exit(main())
