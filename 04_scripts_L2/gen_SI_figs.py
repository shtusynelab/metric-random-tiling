# -*- coding: utf-8 -*-
"""Generate the resonance-sector plot (Fig. S4)."""
import csv
import math
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import paths_common as PC  # noqa: E402

ROOT = PC.HC
FIGS = PC.FIGURES
sys.path.insert(0, str(ROOT))

from l1_style import plt, SINGLE_COL, DOUBLE_COL, savefig  # noqa: E402
from matplotlib.collections import PolyCollection          # noqa: E402
from matplotlib.colors import Normalize                    # noqa: E402
from matplotlib.cm import ScalarMappable                   # noqa: E402

EDGE_GRAY = '#3a3a3a'
CMAP = plt.get_cmap('viridis')


def _load_long_chain():
    """TSDHC_long_trace.csv -> {(alphabet, Ly): ([Lx...], [lnOmega...])}.

    Parsing conventions as in fig2cd_symmetric_forms.py: multiprocess
    append, no header, BOM-polluted rows - skip rows whose row[2] is not
    all digits and strip any BOM from row[0];
    columns: row[0]=alphabet, row[2]=Ly, row[3]=Lx, row[5]=lnOmega."""
    out = {}
    with open(ROOT / "TSDHC_long_trace.csv", encoding="utf-8-sig") as f:
        for row in csv.reader(f):
            if len(row) < 7 or not row[2].isdigit():
                continue
            key = (row[0].lstrip("\ufeff"), int(row[2]))
            out.setdefault(key, ([], []))
            out[key][0].append(int(row[3]))
            out[key][1].append(float(row[5]))
    return out


def fig_S4():
    df3 = pd.read_csv(ROOT / PC.fname("P0_PM_torus_trace_shape_classes_Ly2Ly3.csv"))
    df5 = pd.read_csv(ROOT / PC.fname("P0_PM_torus_trace_shape_classes_Ly5.csv"))
    spec = {"binary": (2, "{1,2} (k = 2)"), "nonarith": (3, "{1,2,4} (k = 3)")}
    cls_col = {"binary": "#2166ac", "nonarith": "#e6811e"}
    w_mk = {2: "o", 3: "s", 5: "^"}
    fig = plt.figure(figsize=(SINGLE_COL, 2.9))
    ax_a = fig.add_axes([0.115, 0.145, 0.36, 0.78])
    ax_b = fig.add_axes([0.585, 0.145, 0.36, 0.78])
    nm = "binary"
    k = spec[nm][0]
    d = df3[(df3.alphabet == nm) & (df3.Ly == 3)].sort_values("Lx")
    lx = d.Lx.values.astype(float)
    y = d.lnOmega.values.astype(float)
    res = (lx % 3) == 0
    xr = np.array([1.5, 20.5])
    ax_a.plot(xr, np.log(2) * xr + 4 * np.log(2), color="#9a9a9a", lw=0.9,
              ls=(0, (3, 1.6)), zorder=1)
    ax_a.plot(lx[~res], y[~res], "s", ms=3.2, mfc="#2166ac", mec="#2166ac",
              mew=0.0, zorder=3, label="non-resonant (exact)")
    ax_a.plot(lx[res], y[res], "o", ms=3.6, mfc="white", mec="#c0392b",
              mew=0.8, zorder=4, label="resonant $L_x\\equiv 0$ (mod 3)")
    ax_a.set_xlabel("$L_x$")
    ax_a.set_ylabel("$\\ln\\Omega$")
    ax_a.set_xticks([2, 6, 10, 14, 18])
    ax_a.legend(loc="upper left", borderaxespad=0.2, handlelength=1.0)
    ax_a.annotate("closed form $2^{L_x+4}$", xy=(0.97, 0.06),
                  xycoords="axes fraction", ha="right", fontsize=5.5,
                  color="#777777")
    for nm, (k, lab) in spec.items():
        for ly in (2, 3, 5):
            dd = df3 if ly in (2, 3) else df5
            d = dd[(dd.alphabet == nm) & (dd.Ly == ly)].sort_values("Lx")
            lx = d.Lx.values.astype(float)
            y = d.lnOmega.values.astype(float)
            m = (lx % ly) == 0
            if not m.any():
                continue
            excess = y[m] - (lx[m] + ly + 1) * np.log(k)
            ax_b.plot(lx[m], np.maximum(excess, 1e-3), w_mk[ly], ms=3.0,
                      mfc=cls_col[nm], mec=cls_col[nm], mew=0.0,
                      label=f"{lab}, $L_y={ly}$")
        # long-chain exact counts (TSDHC_long_trace, Lx<=150): resonant-sector
        # excess extends into the asymptotic regime, supporting the reading
        # that the winding sectors carry the entropy
        long = _load_long_chain()
        for ly in (2, 3, 5):
            if (nm, ly) not in long:
                continue
            lxs = np.array(long[(nm, ly)][0], dtype=float)
            yl = np.array(long[(nm, ly)][1], dtype=float)
            keep = (lxs % ly == 0) & (lxs > 20)   # Lx<=20 already covered by the shape-class CSVs
            if not keep.any():
                continue
            exc = yl[keep] - (lxs[keep] + ly + 1) * np.log(spec[nm][0])
            ax_b.plot(lxs[keep], np.maximum(exc, 1e-3), w_mk[ly], ms=2.0,
                      mfc=cls_col[nm], mec=cls_col[nm], mew=0.0, alpha=0.55,
                      zorder=2)
    ax_b.set_xscale("log")
    ax_b.set_yscale("log")
    from matplotlib.ticker import ScalarFormatter, NullFormatter
    ax_b.xaxis.set_major_formatter(ScalarFormatter())
    ax_b.xaxis.set_minor_formatter(NullFormatter())
    ax_b.set_xticks([2, 5, 10, 20, 50, 150])
    ax_b.set_xlabel("$L_x$ (resonant sizes)")
    ax_b.set_ylabel("$\\Delta=\\ln\\Omega-\\ln\\Omega_{\\rm closed\\;form}$")
    ax_b.legend(loc="upper left", borderaxespad=0.2, handlelength=1.0,
                labelspacing=0.3)
    for ax, s in ((ax_a, "a"), (ax_b, "b")):
        ax.text(-0.14, 1.04, s, transform=ax.transAxes, fontsize=8,
                fontweight="bold", va="bottom", ha="right")
    savefig(fig, str(FIGS / "S4_resonance_sectors"))
    plt.close(fig)
    print("[OK] S4_resonance_sectors")


# ---------------------------------------------------------------- Fig. S4

if __name__ == "__main__":
    fig_S4()
