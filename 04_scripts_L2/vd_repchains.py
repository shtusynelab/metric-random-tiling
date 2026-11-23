# -*- coding: utf-8 -*-
"""vd_repchains.py -- independent-chain replication of the Fig. 4b diffraction
statistics (v6.2 addition; production-same calibre via vd_ladder_ext).

Three new sampling runs (same driver/protocol as the production ensembles,
new seeds -- see Table R6):

  a123r   : {1,2,3}, 13x13, seed 20264100, eq 1.5e6, thin 4e4, ncfg 20
  b12r    : {1,2},   13x13, seed 20264200, eq 1.5e6, thin 4e4, ncfg 20
  b12r21  : {1,2},   21x21, seed 20264300, eq 3.0e6, thin 4e4, ncfg 20

Pipeline (anchored; the run aborts the dependent stage if any anchor misses
its target by >15%):
  A1  original torus13_a123.npz -> f_coh ~ 9.0e-4, halo/core ~ 16.7   (gates R1)
  A2  original torus13_b12.npz  -> f_coh ~ 1.8e-4                     (gates R2)
  A3  original tor21_b12.npz (R = 7 patch) -> f_coh ~ 1.0e-2          (gates R3)
  R1  a123r: f_coh at shifts c = 0..6 ({1,2,3}..{7,8,9}) by the same exact
      translation relabeling as production; endpoint ratio f_coh(c=6)/
      f_coh(c=0) with 10k paired-resample 95% quantiles (S6.2 procedure).
  R2  b12r: f_coh (13x13, second seed) -> within-parent spread.
  R3  b12r21: f_coh (21x21, second seed, R = 7 / 169-face patch).

Stages run selectively: default all; pass stage names (A1 R1 R2 A3 R3 ...) to
run a subset, e.g. the 13x13 block before the 21x21 chain lands:
  python vd_repchains.py A1 A2 R1 R2 [--raw DIR]
Outputs: <DATA>/repchain_summary.csv, <DATA>/repchain_gates.json
"""
import sys, time, json
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import paths_common as PC                     # noqa: E402

sys.path.insert(0, str(HERE / "external"))
import vd_ladder_ext as LX                    # noqa: E402  (build_reference/run_domain)
import vd_suite3 as S                         # noqa: E402  (F_pts)
from vd_patch import embed_config             # noqa: E402

argv = list(sys.argv[1:])
RAW_NEW = LX.RAW
if "--raw" in argv:
    RAW_NEW = Path(argv[argv.index("--raw") + 1]).expanduser()
    argv = [a for a in argv if a != "--raw" and a != RAW_NEW.as_posix()
            and a != str(RAW_NEW)]
STAGES = argv if argv else ["A1", "A2", "A3", "R1", "R2", "R3"]
T0 = time.time()
ROWS = []
GATES = {}
RATIO = None


def log(msg):
    print(f"[{time.time()-T0:6.0f}s] {msg}", flush=True)


def bond_mean(ref, pts):
    return float(np.mean([np.hypot(*(pts[a] - pts[b])) for a, b in ref["bonds"]]))


def spot_F(ref, cfgs, shift):
    Fs = []
    for x in cfgs:
        pts, _cv = embed_config(ref["G"], ref["T"], np.asarray(x) + shift)
        Fs.append(S.F_pts(pts / bond_mean(ref, pts), ref["pix"], ref["cc"],
                          ref["dqq"]))
    return np.array(Fs)


def paired_ratio(ref, cfgs, c0, c1, ndraw=10000, seed=20261011):
    """Endpoint coherent-intensity ratio + paired bootstrap (S6.2 procedure)."""
    F0 = spot_F(ref, cfgs, c0)
    F1 = spot_F(ref, cfgs, c1)
    point = float((np.abs(F1.mean(axis=0)) ** 2).mean()
                  / (np.abs(F0.mean(axis=0)) ** 2).mean())
    rng = np.random.default_rng(seed)
    rats = np.empty(ndraw)
    for b in range(ndraw):
        idx = rng.integers(0, len(cfgs), len(cfgs))
        rats[b] = ((np.abs(F1[idx].mean(axis=0)) ** 2).mean()
                   / (np.abs(F0[idx].mean(axis=0)) ** 2).mean())
    lo, hi = (float(v) for v in np.percentile(rats, [2.5, 97.5]))
    return dict(point=point, lo=lo, hi=hi)


def meta(d, key):
    return np.asarray(d[key]).reshape(-1)[0]


def run_A1(ref13):
    d = np.load(LX.RAW / "torus13_a123.npz")
    chk = LX.run_domain(ref13, d["cfgs"].astype(np.int64), (1, 2, 3), tag="A1")
    GATES["A1_fcoh"] = chk["f_coh"]
    log(f"A1 a123 recheck f_coh={chk['f_coh']:.4g} (target ~9.0e-4) "
        f"halo/core={chk['halo_core']:.2f} (target ~16.7)")
    ROWS.append(dict(row="A1", tag="a123", f_coh=chk["f_coh"],
                     halo_core=chk["halo_core"]))


def run_A2(ref13):
    d = np.load(LX.RAW / "torus13_b12.npz")
    chk = LX.run_domain(ref13, d["cfgs"].astype(np.int64), (1, 2), tag="A2")
    GATES["A2_fcoh"] = chk["f_coh"]
    log(f"A2 b12 recheck f_coh={chk['f_coh']:.4g} (target ~1.8e-4)")
    ROWS.append(dict(row="A2", tag="b12", f_coh=chk["f_coh"],
                     halo_core=chk["halo_core"]))


def run_A3(ref21):
    d = np.load(LX.RAW / "tor21_b12.npz")
    chk = LX.run_domain(ref21, d["cfgs"].astype(np.int64), (1, 2), boot=False,
                        tag="A3")
    GATES["A3_fcoh"] = chk["f_coh"]
    log(f"A3 tor21_b12 R=7 recheck f_coh={chk['f_coh']:.4g} (target ~1.0e-2)")
    ROWS.append(dict(row="A3", tag="b12_R7", f_coh=chk["f_coh"],
                     halo_core=chk["halo_core"]))


def gate(name, target):
    return abs(GATES[name] - target) / target < 0.15


def run_R1(ref13):
    global RATIO
    if not gate("A1_fcoh", 9.013e-4):
        log("R1 ABORTED: A1 anchor gate failed")
        return False
    d = np.load(RAW_NEW / "torus13_a123r.npz")
    cfgs = d["cfgs"].astype(np.int64)
    log(f"a123r seed={int(meta(d, 'seed'))} swap={float(meta(d, 'swap')):.3f} "
        f"cfgs={cfgs.shape}")
    for c in range(7):
        A = tuple(v + c for v in (1, 2, 3))
        tag = "a123r" if c == 0 else f"a{1+c}{2+c}{3+c}r"
        r = LX.run_domain(ref13, cfgs + c, A, boot=(c in (0, 6)), tag=tag)
        ROWS.append(dict(row=f"R1_c{c}", tag=tag, f_coh=r["f_coh"],
                         f_inc=r["f_inc"], halo_core=r["halo_core"], r=r["r"]))
        log(f"R1 shift +{c} ({A}) f_coh={r['f_coh']:.4g} "
            f"halo/core={r['halo_core']:.2f} r={r['r']:.3f}")
    ratio = paired_ratio(ref13, cfgs, 0, 6)
    RATIO = ratio
    ROWS.append(dict(row="R1_ratio", tag="a789r/a123r", f_coh=ratio["point"],
                     ci_lo=ratio["lo"], ci_hi=ratio["hi"]))
    log(f"R1 endpoint ratio = {ratio['point']:.1f} (95% paired quantiles "
        f"{ratio['lo']:.0f}-{ratio['hi']:.0f}; production 640.45, 257-1260)")
    return True


def run_R2(ref13):
    if not gate("A2_fcoh", 1.8e-4):
        log("R2 ABORTED: A2 anchor gate failed")
        return False
    d = np.load(RAW_NEW / "torus13_b12r.npz")
    cfgs = d["cfgs"].astype(np.int64)
    r = LX.run_domain(ref13, cfgs, (1, 2), boot=True, tag="b12r")
    ROWS.append(dict(row="R2", tag="b12r", f_coh=r["f_coh"],
                     halo_core=r["halo_core"], fcoh_lo=r.get("fcoh_lo"),
                     fcoh_hi=r.get("fcoh_hi")))
    log(f"R2 b12r (13x13, seed {int(meta(d, 'seed'))}) "
        f"f_coh={r['f_coh']:.4g} (production 1.8e-4)")
    return True


def run_R3(ref21):
    if not gate("A3_fcoh", 1.0e-2):
        log("R3 ABORTED: A3 anchor gate failed")
        return False
    d = np.load(RAW_NEW / "tor21_b12r21.npz")
    cfgs = d["cfgs"].astype(np.int64)
    r = LX.run_domain(ref21, cfgs, (1, 2), boot=True, tag="b12r21_R7")
    ROWS.append(dict(row="R3", tag="b12r21_R7", f_coh=r["f_coh"],
                     halo_core=r["halo_core"], fcoh_lo=r.get("fcoh_lo"),
                     fcoh_hi=r.get("fcoh_hi")))
    log(f"R3 b12r21 (21x21, seed {int(meta(d, 'seed'))}) "
        f"f_coh={r['f_coh']:.4g} (production 1.0e-2)")
    return True


def save():
    pd.DataFrame(ROWS).to_csv(PC.DATA / "repchain_summary.csv", index=False,
                              encoding="utf-8-sig")
    (PC.DATA / "repchain_gates.json").write_text(json.dumps(
        {"gates": GATES, "ratio": RATIO, "stages": STAGES}, indent=2))
    log("saved repchain_summary.csv / repchain_gates.json")


def main():
    ref13 = LX.build_reference(13, 7) if ("A1" in STAGES or "A2" in STAGES
                                          or "R1" in STAGES or "R2" in STAGES) else None
    ref21 = LX.build_reference(21, 7) if ("A3" in STAGES or "R3" in STAGES) else None
    if "A1" in STAGES:
        run_A1(ref13)
    if "A2" in STAGES:
        run_A2(ref13)
    if "A3" in STAGES:
        run_A3(ref21)
    if "R1" in STAGES:
        run_R1(ref13)
    if "R2" in STAGES:
        run_R2(ref13)
    if "R3" in STAGES:
        run_R3(ref21)
    save()


if __name__ == "__main__":
    main()
