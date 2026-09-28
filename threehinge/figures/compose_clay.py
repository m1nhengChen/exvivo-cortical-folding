#!/usr/bin/env python3
"""Compose the two Cycles clay passes into panel f's single image.

blender_net.py renders the cortex opaque (shell) and the hinge network alone (net), both on a
transparent film. Here the shell is pushed toward the page so it becomes a pale relief map of the
folding rather than a solid object, and the network is laid over it at full strength. Doing the
two as separate passes is what buys the look: an in-render translucent cortex has to be crossed a
dozen times by every ray and turns to fog, whereas a lightened opaque pass keeps every gyrus and
sulcus at full contrast while still sitting behind the fibres.

Output keeps the alpha, so the panel sits on the figure's own white."""
# --- project paths and import roots: see src/threehinge/_env.py ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
import _env

import numpy as np
from PIL import Image

OUT = (_env.ROOT + "/surf/three_hinge/results")
LAMBDA = 0.44            # shell -> page; the folding must stay legible but never rival the fibres
MARGIN = 12              # px of transparent padding kept around the silhouette


def rgba(path):
    return np.asarray(Image.open(path).convert("RGBA"), dtype=np.float32) / 255.0


def main():
    sh, nt = rgba(f"{OUT}/_net_clay_shell.png"), rgba(f"{OUT}/_net_clay_net.png")
    sh[..., :3] = sh[..., :3] * (1 - LAMBDA) + LAMBDA

    a = nt[..., 3:4]
    out = np.empty_like(sh)
    out[..., 3:4] = a + sh[..., 3:4] * (1 - a)
    # premultiplied "over"; guard the fully transparent border against 0/0
    num = nt[..., :3] * a + sh[..., :3] * sh[..., 3:4] * (1 - a)
    out[..., :3] = np.where(out[..., 3:4] > 1e-6, num / np.maximum(out[..., 3:4], 1e-6), 1.0)

    keep = np.argwhere(out[..., 3] > 0.004)
    (y0, x0), (y1, x1) = keep.min(0), keep.max(0) + 1
    y0, x0 = max(0, y0 - MARGIN), max(0, x0 - MARGIN)
    y1, x1 = min(out.shape[0], y1 + MARGIN), min(out.shape[1], x1 + MARGIN)
    out = out[y0:y1, x0:x1]

    im = Image.fromarray((np.clip(out, 0, 1) * 255).round().astype(np.uint8), "RGBA")
    im.save(f"{OUT}/_net_clay.png")
    print("wrote _net_clay.png", im.size, "aspect %.3f" % (im.size[0] / im.size[1]))


if __name__ == "__main__":
    main()
