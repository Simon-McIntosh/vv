"""
Canonical Monte-Carlo generator for the ITER VV lateral-rattle (n=1) study.

WHAT THIS PRODUCES
------------------
The committed, reproducible dataset behind every Monte-Carlo number in the
report (``docs/vv-research-findings.html``).  Run once; the arrays it writes
to ``data/`` are loaded by ``vv_viz.py`` to build the figures.

RATTLE METRIC  (peak-to-peak range)
-----------------------------------
"Rattle" is the **peak-to-peak range** of the VV-centre lateral displacement:
the full diameter of the reachable polytope (forward + backward reach along
the worst lateral direction), in millimetres.  Physically this is the
**n = 1 first-wall displacement envelope** — a rigid horizontal shift of the
vacuum vessel makes the plasma-wall gap vary as cos(toroidal angle), i.e. an
n = 1 perturbation.  When the plasma is limited on the inner wall during
start-up, the gap asymmetry localises the heat load; the peak-to-peak range
bounds how much the wall position can move between shots.  (The pure-rotation
DOF about the vertical axis is n = 0 — it does not perturb the plasma-wall gap
— and is reported separately in vv_viz, not folded into this number.)

The range is computed by ``vv_viz.max_rattle_mm`` (scipy HiGHS LP), the single
source of LP truth shared with the figure code.

REPRODUCIBILITY
---------------
Fixed seed (PCG64) -> bit-for-bit reproducible.  Assembly offsets are drawn
iid Uniform(-1.5, +1.5) mm at each of the 9 supports.

    uv run python vv_mc_generator.py
"""
from __future__ import annotations

import os
import numpy as np

from vv_viz import max_rattle_mm, N, DELTA_M

N_SAMPLES = 5000
SEED = 20260527          # fixed -> reproducible
N_DIR = 72               # direction sweep for the range metric (converged)
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def generate(n_samples: int = N_SAMPLES, seed: int = SEED, nd: int = N_DIR):
    """Sample assembly offsets and compute the peak-to-peak rattle range (mm)."""
    rng = np.random.default_rng(seed)
    u = rng.uniform(-DELTA_M, DELTA_M, size=(n_samples, N))
    rattles = np.array([max_rattle_mm(u[k], nd=nd) for k in range(n_samples)])
    return rattles, u


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    rattles, u = generate()
    np.save(os.path.join(DATA_DIR, "rattle_mc_5k.npy"), rattles)
    np.save(os.path.join(DATA_DIR, "u_mc_5k.npy"), u)

    nominal = max_rattle_mm(np.zeros(N), nd=N_DIR)
    print(f"N={N_SAMPLES}  seed={SEED}  nd={N_DIR}  metric=peak-to-peak range (mm)")
    for p in (50, 75, 90, 95, 99):
        print(f"  P{p:<2} = {np.percentile(rattles, p):.4f} mm")
    print(f"  mode ~= {_mode(rattles):.4f} mm")
    print(f"  max  = {rattles.max():.4f} mm")
    print(f"  nominal (u=0) = {nominal:.4f} mm   (hard theoretical ceiling)")
    print(f"  -> wrote {DATA_DIR}/rattle_mc_5k.npy, u_mc_5k.npy")


def _mode(x, bins=60):
    h, e = np.histogram(x, bins=bins)
    i = int(h.argmax())
    return 0.5 * (e[i] + e[i + 1])


if __name__ == "__main__":
    main()
