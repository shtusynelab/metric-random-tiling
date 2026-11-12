# -*- coding: utf-8 -*-
"""W=4 exact trace(T^4): T^2 = T@T (monitor nnz), tr(T^4) = sum(T^2 o T^2^T).

Server-only driver (originally run on the internal cluster); pass explicit
arguments when used locally.
"""
import sys, math, time
import argparse
from pathlib import Path
HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument("--workdir", default=str(HERE),
                help="directory containing trace_hutch.py (default: script dir)")
args = ap.parse_args()
sys.path.insert(0, str(args.workdir))
import numpy as np
import importlib.util
spec = importlib.util.spec_from_file_location(
    "th", str(Path(args.workdir) / "trace_hutch.py"))
th = importlib.util.module_from_spec(spec)
sys.argv = ["x", "--W", "4"]
spec.loader.exec_module(th)
W = 4
t0 = time.time()
T, _ = th.assemble_T(W, u_chunk=64, verbose=True)
print(f"T: dim={T.shape[0]} nnz={T.nnz} built in {time.time()-t0:.0f}s", flush=True)
t1 = time.time()
T2 = (T @ T).tocsr()
print(f"T^2 nnz = {T2.nnz} ({time.time()-t1:.0f}s)", flush=True)
t2 = time.time()
Tt = T.T.tocsr()
tr = float(T2.multiply(Tt).sum())
print(f"=== exact trace(T^4) = {tr:.1f} ({time.time()-t2:.0f}s)", flush=True)
om = tr
g4 = om / 3 ** 16
s4 = math.log(om) / 32
print(f"Omega(4) = {om:.0f} = 3^16 x {g4:.4f} (predicted g(4) ~ 1.11)")
print(f"s(4) = {s4:.4f} (chain: 1.1001 / 0.7357 / ...)")
lnO = math.log(om)
print(f"linear-law check: lnOmega/W = {lnO/W:.4f} "
      f"(W=2: 4.4006; W=3: 4.4137; predicted ~4.42)")
