# -*- coding: utf-8 -*-
"""Server-only driver (originally run on the internal cluster); pass explicit
arguments when used locally.

Current route: states_exceeded tasks are handled by short-strip exact
counting (L=3..6; P0_A2b_shortL.py on the server) + 1/L extrapolation, no
long-strip reruns.

Logic:
  1. status polling: main run (row count / process / last log line),
     short-strip sweep (row count / process / log tail), free memory;
  2. main run finished and the sweep process gone, but the sweep incomplete
     (log lacks the end marker) -> restart the sweep on the server
     (resumable), report "[scan restart triggered]";
  3. main run and sweep both finished -> run P0_A2b_collect.py on the
     server (merge exact main-table values + short-strip 1/L extrapolation
     -> 1/W extrapolation table) and scp the CSVs back to the local
     directory, report "[collection done]";
  4. otherwise report status only.
"""
import argparse
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--host', required=True,
                    help='SSH host (user@host)')
    ap.add_argument('--local-dir', default=str(HERE),
                    help='local directory the CSVs are copied back into')
    ap.add_argument('--remote-dir', required=True,
                    help='working directory on the server')
    args = ap.parse_args()
    rd = args.remote_dir.rstrip('/')
    ssh = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
           "-o", "StrictHostKeyChecking=accept-new", args.host]

    def sh(cmd, timeout=120):
        r = subprocess.run(ssh + [cmd], capture_output=True, text=True,
                           timeout=timeout)
        return r.stdout.strip()

    report = []
    n_main_rows = sh(f"wc -l < {rd}/P0_A2b_TM_entropy_density.csv")
    n_main_proc = sh("pgrep -fc P0_A2b_server || true")
    main_log = sh(f"tail -1 {rd}/A2b_run.log")
    n_sw_rows = sh(f"wc -l < {rd}/P0_A2b_TM_short_strip_exact_counting.csv "
                   "2>/dev/null || echo 0")
    n_sw_proc = sh("pgrep -fc P0_A2b_shortL || true")
    sw_log = sh(f"tail -1 {rd}/A2b_shortL.log 2>/dev/null || true")
    mem = sh("free -g | awk 'NR==2{print $7}'")
    main_done = (n_main_proc.strip() == "0")
    sw_done = ("scan finished" in sw_log) and (n_sw_proc.strip() == "0")
    report.append(f"main run: {n_main_rows} lines (incl. header), "
                  f"process {n_main_proc}, last line: {main_log}")
    report.append(f"short-strip sweep: {n_sw_rows} lines (incl. header), "
                  f"process {n_sw_proc}, last line: {sw_log}")
    report.append(f"memory available: {mem} GB")

    if main_done and (not sw_done):
        sh(f"cd {rd} && setsid python3 P0_A2b_shortL.py "
           "> A2b_shortL.log 2>&1 < /dev/null & echo go")
        report.append("[scan restart triggered] (resumable)")

    if main_done and sw_done:
        out = sh(f"cd {rd} && python3 P0_A2b_collect.py", timeout=300)
        report.append("=== collection output ===")
        report.append(out)
        for f in ("P0_A2b_TM_entropy_density.csv",
                  "P0_A2b_TM_short_strip_exact_counting.csv",
                  "P0_A2b_TM_extrapolation.csv"):
            subprocess.run(["scp", "-o", "BatchMode=yes",
                            f"{args.host}:{rd}/{f}",
                            args.local_dir.replace("\\", "/") + "/" + f],
                           capture_output=True, text=True, timeout=180)
        report.append("[collection done] CSVs copied back to the local directory")

    print("\n".join(report))


if __name__ == "__main__":
    main()
