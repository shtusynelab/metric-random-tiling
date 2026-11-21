"""SI Fig. S8 - constructive lower-bound certificates (four panels).

Panel a: honeycomb flower-pair double completion (phi+ / phi-), source
flip_certificate.png; panels b-d: the three thinner carriers (3.4.6.4,
3.12.12 mid, 4.6.12), source cert_*.png. Sources are real-metric renderings
(edge length = edge value; per-face closure asserted at generation time,
scripts flip_certificate_fig.py / a4_cert_figs_all.py). This script only
composes the four panels; it redraws nothing.

Figure policy: English-only labels, no panel titles (letters a-d outside the
top-left corner of each axes), png + pdf at dpi 300.
"""

import io
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt   # noqa
import numpy as np                # noqa
from PIL import Image             # noqa
from matplotlib.lines import Line2D  # noqa

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import paths_common as PC  # noqa: E402

FIGS = PC.FIGURES
SRC = FIGS / 'assets_certificates'

PANELS = [
    ('a', 'flip_certificate.png', 'honeycomb {1,2,3}'),
    ('b', 'cert_3464.png', '3.4.6.4 {1,2,3}'),
    ('c', 'cert_31212.png', '3.12.12 {1,2,3}'),
    ('d', 'cert_4612.png', '4.6.12 {1,2,3}'),
]


def load(p):
    im = np.asarray(Image.open(p).convert('RGB'))
    # trim the white margin so the tiling and its labels fill the panel
    mask = np.any(im < 250, axis=2)
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    r0, r1 = np.where(rows)[0][[0, -1]]
    c0, c1 = np.where(cols)[0][[0, -1]]
    pad = 6
    r0 = max(0, r0 - pad)
    r1 = min(im.shape[0] - 1, r1 + pad)
    c0 = max(0, c0 - pad)
    c1 = min(im.shape[1] - 1, c1 + pad)
    return im[r0:r1 + 1, c0:c1 + 1]


def main():
    imgs = {k: (load(SRC / f), cap) for k, f, cap in PANELS}
    # row height ratios follow the source aspect ratios (w/h)
    r1 = (imgs['a'][0].shape[1] / imgs['a'][0].shape[0]
          + imgs['b'][0].shape[1] / imgs['b'][0].shape[0]) / 2
    r2 = (imgs['c'][0].shape[1] / imgs['c'][0].shape[0]
          + imgs['d'][0].shape[1] / imgs['d'][0].shape[0]) / 2
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.15),
                             gridspec_kw={'height_ratios': [1 / r1, 1 / r2]})
    for ax, (k, _, cap) in zip(axes.flat, PANELS):
        im, _ = imgs[k]
        ax.imshow(im, interpolation='none')
        ax.set_axis_off()
        ax.text(0.0, 1.03, k, transform=ax.transAxes, fontsize=8,
                fontweight='bold', va='bottom', ha='left')
    fig.subplots_adjust(left=0.015, right=0.985, bottom=0.10, top=0.952,
                        wspace=0.04, hspace=0.16)
    col = {1: '#1f77b4', 2: '#8c8c8c', 3: '#d62728'}
    handles = [Line2D([0], [0], color=col[x], lw=3) for x in (1, 2, 3)]
    fig.legend(handles, ['edge length 1', 'edge length 2', 'edge length 3'],
               loc='lower center', bbox_to_anchor=(0.5, 0.01), ncol=3,
               fontsize=7.5, frameon=False)
    out = FIGS / 'S7_lower_bound_certificates'
    fig.savefig(out.with_suffix('.png'), dpi=300)
    fig.savefig(str(out) + '.pdf')
    print('written:', out.with_suffix('.png'),
          (out.with_suffix('.png')).stat().st_size, 'bytes')


if __name__ == '__main__':
    main()
