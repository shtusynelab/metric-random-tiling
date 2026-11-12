# -*- coding: utf-8 -*-
"""SI figure: entropy split -- ideal alloy-mixing bound vs measured entropy.

Convention (erratum fix, SI v3.0c): the honeycomb torus carries E = 3F edges
at F hexagonal faces (SI Sect. S1.2).  The ideal alloy-mixing reference for E
independent edges over k equiprobable species is

    S_mix/F = (E/F) ln k = 3 ln k                    [grey line / grey bars]

which for k = 2, 3 gives 2.08 and 3.30 per face.  (An earlier version of this
script used E/F = 1.5 -- half the true edge count -- halving the reference and
the closure cost; the measured entropies were never affected.)

    measured entropy density  S_obs/F = ln Omega/F                    [blue]
    closure cost = S_mix/F - S_obs/F   (entropy tax of the metric constraint)

(a) {1,2,3}: S_obs/F decreases with F towards the manuscript asymptote
    s_infinity = 0.544; the ideal-mixing line 3.30 sits far above, and the gap
    (closure cost) grows with the counting domain.
(b) F = 15: three-alphabet comparison.  Each alphabet is drawn against its own
    ideal-mixing line (2.08 -> 3.30 with k); {1,2,3} and {1,2,4} share k = 3
    and the same bound, yet measure 0.81 vs 0.66 => the difference is
    geometric/arithmetic, not compositional.

Data: si_entropy_split.csv (exact enumeration; ln Omega = H_comp + H_cond
asserted at every point).  Labels all English; output PDF + PNG (600 dpi),
written to the canonical figures directory as S11_entropy_split.*.
"""
import csv
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths_common as PC  # noqa: E402

DATA = PC.DATA / "si_entropy_split.csv"
OUT = PC.FIGURES

C_MIX = "#8c8c8c"
C_OBS = "#1f77b4"
S_REF = 0.544

DROP = {("12", 2, 2), ("123", 2, 2), ("124", 2, 2), ("124", 2, 3)}
ALPHA_LABEL = {"12": r"$\{1,2\}$", "123": r"$\{1,2,3\}$",
               "124": r"$\{1,2,4\}$"}


def load():
    rows = list(csv.DictReader(open(DATA, encoding="utf-8")))
    out = []
    for r in rows:
        for k in ("Lx", "Ly", "F", "E", "omega", "ln_Omega", "H_comp",
                  "H_cond", "H_comp_per_F", "H_cond_per_F"):
            r[k] = float(r[k])
        if (r["alpha"], int(r["Lx"]), int(r["Ly"])) in DROP:
            continue
        out.append(r)
    # honeycomb torus edge count: E = 3F exactly (SI Sect. S1.2)
    ef = {r["E"] / r["F"] for r in rows}
    assert ef == {3.0}, f"E/F must be 3 on the honeycomb torus, got {ef}"
    return out


def ideal_mix(alpha):
    """Ideal alloy-mixing entropy per face, (E/F) ln k with E/F = 3."""
    return 3.0 * math.log(len(set(alpha)))


def main():
    rows = load()
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8.2,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "mathtext.fontset": "dejavusans",
    })
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0),
                             gridspec_kw={"width_ratios": [1.0, 0.9]})
    axA, axB = axes

    # ---------------- (a) {1,2,3}: measured vs ideal mixing bound ----------
    sub = sorted([r for r in rows if r["alpha"] == "123"], key=lambda r: r["F"])
    F = np.array([r["F"] for r in sub])
    obs = np.array([r["ln_Omega"] / r["F"] for r in sub])
    m123 = ideal_mix("123")

    axA.axhline(m123, color=C_MIX, lw=1.3)
    axA.text(9.0, m123 + 0.07,
             r"ideal mixing  $(E/F)\ln 3 = 3.30$",
             ha="center", va="bottom", fontsize=7.0, color="0.3")
    axA.plot(F, obs, "o-", color=C_OBS, ms=4.2, lw=1.4,
             label=r"measured  $\ln\Omega/F$")
    axA.axhline(S_REF, color="0.55", lw=1.0, ls="--")
    axA.text(F.max() + 0.9, S_REF + 0.03, r"$s_\infty = 0.544$",
             ha="right", va="bottom", fontsize=7.0, color="0.35")
    xa = 14.2
    axA.annotate("", xy=(xa, m123 - 0.06), xytext=(xa, obs[-1] + 0.05),
                 arrowprops=dict(arrowstyle="<->", color="0.45", lw=0.9,
                                 shrinkA=1, shrinkB=1))
    axA.text(xa - 0.35, (m123 + obs[-1]) / 2,
             "closure\ncost", ha="right", va="center", fontsize=6.9,
             color="0.35", linespacing=1.15)
    axA.set_xlabel(r"number of faces  $F$")
    axA.set_ylabel(r"entropy per face  [$k_B$]")
    axA.set_xlim(F.min() - 0.9, F.max() + 1.0)
    axA.set_ylim(0, 3.75)
    axA.set_yticks([0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5])
    axA.legend(frameon=False, fontsize=7.0, loc="lower right",
               bbox_to_anchor=(1.0, 0.02),
               handlelength=1.5, borderpad=0.2, labelspacing=0.3)
    axA.text(0.035, 0.06, r"honeycomb  $\{1,2,3\}$",
             transform=axA.transAxes, ha="left", va="bottom", fontsize=7.6)

    # ---------------- (b) F = 15: three alphabets ----------------
    tags = ["12", "123", "124"]
    sel = {t: [r for r in rows if r["alpha"] == t and r["F"] == 15][0]
           for t in tags}
    x = np.arange(len(tags))
    w = 0.42
    mixv = [ideal_mix(t) for t in tags]
    obsv = [sel[t]["ln_Omega"] / sel[t]["F"] for t in tags]

    axB.bar(x - w / 2, mixv, width=w, color=C_MIX, edgecolor="black",
            linewidth=0.4, label=r"ideal mixing  $(E/F)\ln k$")
    axB.bar(x + w / 2, obsv, width=w, color=C_OBS, edgecolor="black",
            linewidth=0.4, label=r"measured  $\ln\Omega/F$")
    for xi, (m, o) in enumerate(zip(mixv, obsv)):
        axB.text(xi - w / 2, m + 0.05, f"{m:.2f}", ha="center", va="bottom",
                 fontsize=6.8, color="0.25")
        axB.text(xi + w / 2, o + 0.05, f"{o:.2f}", ha="center", va="bottom",
                 fontsize=6.8, color="0.25")
    axB.set_xticks(x)
    axB.set_xticklabels([ALPHA_LABEL[t] for t in tags], fontsize=8.0)
    axB.set_ylabel(r"entropy per face  [$k_B$]")
    axB.set_ylim(0, 3.95)
    axB.set_xlim(-0.62, 2.62)
    axB.legend(frameon=False, fontsize=7.0, loc="upper left",
               handlelength=1.5, borderpad=0.2, labelspacing=0.3)
    axB.text(0.965, 0.60, r"$F=15$", transform=axB.transAxes,
             ha="right", va="top", fontsize=7.6)

    for ax in axes:
        ax.tick_params(labelsize=7.2)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    fig.tight_layout(pad=0.6)
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        p = OUT / f"S11_entropy_split.{ext}"
        fig.savefig(p, dpi=600 if ext == "png" else None, bbox_inches="tight")
        print("wrote", p)

    # ---- programmatic self-check ----
    print(f"\n[a] {{1,2,3}}: ideal mixing = {m123:.3f} per face")
    for r in sub:
        o = r["ln_Omega"] / r["F"]
        print(f"   F={int(r['F']):2d}  measured={o:.4f}  "
              f"cost={m123 - o:.4f}")
    print("\n[b] F=15:")
    for t in tags:
        print(f"   {{{t}}}: ideal={ideal_mix(t):.3f}  "
              f"measured={sel[t]['ln_Omega']/15:.3f}")


if __name__ == "__main__":
    main()
