# -*- coding: utf-8 -*-
"""Shared paper-figure style -- Nature Materials conventions.

Spec: double column 183 mm / single column 89 mm; Arial; minimum body font
size 5 pt (in-panel labels 5.5-8 pt); line widths 0.5-1.5 pt; lowercase bold
panel labels (a, b, c); vector PDF (editable text) + 300 dpi PNG; RGB.
Colour scheme (colour-vision safe, consistent across figures):
  crystal REF-C green #1b7837 | pseudocrystal PC red #c0392b | quaternary
  PC4 purple #6a3d9a | amorphous REF-A blue #2166ac | neutral grey #7f7f7f
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

MM = 1 / 25.4
DOUBLE_COL = 183 * MM          # 7.205 in
SINGLE_COL = 89 * MM           # 3.504 in
ONEFIVE_COL = 120 * MM         # 1.5-column width

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 7,
    'axes.titlesize': 7.5,
    'axes.labelsize': 7,
    'axes.linewidth': 0.6,
    'xtick.labelsize': 6,
    'ytick.labelsize': 6,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'xtick.major.size': 2.5,
    'ytick.major.size': 2.5,
    'legend.fontsize': 5.5,
    'legend.frameon': False,
    'legend.handlelength': 1.2,
    'axes.unicode_minus': False,
    'svg.fonttype': 'none',
    'pdf.fonttype': 42,          # editable text (journal requirement)
    'ps.fonttype': 42,
    'figure.dpi': 300,
    'savefig.dpi': 300,
})

# unified colour scheme across figures
C_REF_C = '#1b7837'
C_PC = '#c0392b'
C_PC4 = '#6a3d9a'
C_REF_A = '#2166ac'
C_NEUTRAL = '#7f7f7f'
COLORS = {'REF-C': C_REF_C, 'PC': C_PC, 'PC4': C_PC4, 'REF-A': C_REF_A}


def panel_label(ax, s, dx=-0.08, dy=1.02):
    """Nature-Materials-style panel label: lowercase bold, placed just
    outside the upper-left of the axes."""
    ax.text(dx, dy, s, transform=ax.transAxes, fontsize=8,
            fontweight='bold', va='bottom', ha='right')


def savefig(fig, out_base):
    fig.savefig(out_base + '.png', dpi=300, bbox_inches='tight')
    fig.savefig(out_base + '.pdf', bbox_inches='tight')
    print(f"saved: {out_base}.png / .pdf")
