"""
Generate the max-RATTLE (polytope-width) deliverables for the report's new
section — the analogue of the departure-from-centre figures, but for the full
peak-to-peak polytope width.

Width vs departure (they behave oppositely):
  * departure-from-centre is MINIMISED at the centred assembly (~1.55 mm) and
    GROWS for offset assemblies (worst ~2.98 mm) — the frictionless metric;
  * polytope WIDTH is MAXIMISED at the centred assembly (~3.09 mm) and SHRINKS
    for offset assemblies — the realistic (stiction-walk) envelope ceiling.

Outputs (docs/):
  plots/rattle_dashboard.{png,pdf}   width distribution + CDF
  plots/two_bounds.{png,pdf}         the two metrics on the 6 mm budget axis
  strips/strip_nominal_rattle.png    nominal full-width key-frame strip
  animations/rattle_width_nominal.gif  nominal full-diameter sweep

    uv run python vv_rattle_figures.py
"""
from __future__ import annotations

import base64
import os

import numpy as np

import vv_viz
from vv_viz import N

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
DOCS = os.path.join(HERE, "docs")


def _load_or_make_mc():
    up = os.path.join(DATA, "u_mc_5k.npy")
    dp = os.path.join(DATA, "rattle_mc_5k.npy")
    if not (os.path.exists(up) and os.path.exists(dp)):
        import vv_mc_generator
        vv_mc_generator.main()
    return np.load(up), np.load(dp)


def widths_cached(u_mc: np.ndarray, nd: int = 72) -> np.ndarray:
    """Polytope width per assembly, cached to data/width_mc_5k.npy."""
    wp = os.path.join(DATA, "width_mc_5k.npy")
    if os.path.exists(wp):
        w = np.load(wp)
        if len(w) == len(u_mc):
            return w
    print(f"Computing polytope widths for {len(u_mc)} assemblies (nd={nd}) …")
    w = vv_viz.mc_widths(u_mc, nd=nd)
    np.save(wp, w)
    return w


def main():
    plots = os.path.join(DOCS, "plots")
    strips = os.path.join(DOCS, "strips")
    anim = os.path.join(DOCS, "animations")
    for d in (plots, strips, anim):
        os.makedirs(d, exist_ok=True)

    u_mc, dep_mm = _load_or_make_mc()
    widths = widths_cached(u_mc)

    nom_w = vv_viz.max_rattle_mm(np.zeros(N), nd=72)
    print(f"nominal width (ceiling) = {nom_w:.3f} mm   "
          f"MC width: median {np.median(widths):.3f}, P5 {np.percentile(widths,5):.3f}, "
          f"min {widths.min():.3f} mm")
    print(f"departure: nominal {vv_viz.max_departure_mm(np.zeros(N),nd=72):.3f}, "
          f"worst {dep_mm.max():.3f} mm")

    print("rattle dashboard …")
    vv_viz.figure_rattle_dashboard(widths, out_base=os.path.join(plots, "rattle_dashboard"))
    print("two-bounds figure …")
    vv_viz.figure_two_bounds(dep_mm, widths, out_base=os.path.join(plots, "two_bounds"))

    print("nominal full-width key-frame strip …")
    vv_viz.make_keyframe_strip(np.zeros(N), os.path.join(strips, "strip_nominal_rattle.png"),
                               mode="translation")

    print("nominal full-diameter rattle GIF …")
    out, w = vv_viz.make_gif(np.zeros(N), os.path.join(anim, "rattle_width_nominal.gif"))
    print(f"  → {out}  (width {w:.2f} mm)")

    print("done.")


if __name__ == "__main__":
    main()
