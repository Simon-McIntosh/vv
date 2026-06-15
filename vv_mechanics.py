"""
Reproducible mechanics ledger for the ITER VV lateral-displacement study.

This module computes — from first principles, with one committed source of
truth — the force, friction, energy and thermal numbers that decide *which*
lateral-displacement bound the vessel actually occupies:

  * the frictionless **self-centring** picture (departure-from-centre metric),
    which is only a LOWER bound; and
  * the realistic **stiction-walk** picture (polytope-width metric), in which
    the vessel holds wherever it is pushed and ratchets within the slot.

All geometry/material constants follow the ITER VV Load Specification + mass
collection table (ITER_D_6TLUDY): support ring R_s ~ 10 m, supported mass
M ~ 9000 t, 15 deg inward hinge inclination, +/-1.5 mm toroidal slot.

    uv run python vv_mechanics.py
"""
from __future__ import annotations

import numpy as np

from vv_viz import A, ANGLES, DELTA_M, N, R_M, lp_rattle, max_departure_mm, max_rattle_mm

# ── constants (ITER_D_6TLUDY) ───────────────────────────────────────────────
G          = 9.81        # m/s^2
M_TONNES   = 9000.0      # supported mass (VV + in-vessel), t
R_S        = R_M         # support-ring radius, m (10.0; shared with vv_viz)
INCL_DEG   = 15.0        # hinge inclination from vertical, deg
Z_COG      = 5.0         # estimated CoG height above support attachment, m
MU         = 0.10        # effective friction coefficient at the sliding interface
SLOT_MM    = DELTA_M * 1e3   # half-slot, mm (1.5)

# Reviewer-anchored figures (ITER VVLS): axial DW force per VVGS ~ 10 MN;
# toroidal sliding (breakaway) force per VVGS ~ 1 MN.
ALPHA_SS   = 16e-6       # 316L(N) coefficient of thermal expansion, 1/K
DT_BAKE    = 78.0        # warm-up room (22 C) -> 100 C, K  (full bake ~200 C -> dT~178)


# ── gravitational pendulum (lateral centring) ──────────────────────────────
def weight_N() -> float:
    return M_TONNES * 1e3 * G


def h_conv_m() -> float:
    """Height of the support-axis convergence point above the support ring."""
    return R_S / np.tan(np.radians(INCL_DEG))


def L_eff_m() -> float:
    return h_conv_m() - Z_COG


def K_N_per_m() -> float:
    return weight_N() / L_eff_m()


def Tn_s() -> float:
    return 2 * np.pi * np.sqrt(L_eff_m() / G)


def F_restore_at_stop_N() -> float:
    """Gravitational lateral restoring force at the +/-1.5 mm stop."""
    return K_N_per_m() * DELTA_M


# ── support load & friction ─────────────────────────────────────────────────
def support_vertical_N() -> float:
    return weight_N() / N


def support_axial_N() -> float:
    """Axial force in the 15 deg-inclined strut from dead weight."""
    return support_vertical_N() / np.cos(np.radians(INCL_DEG))


def stiction_per_support_N(mu: float = MU) -> float:
    """Toroidal breakaway force at one support = mu * axial clamping load."""
    return mu * support_axial_N()


def aggregate_breakaway_N(mu: float = MU, nd: int = 360):
    """Lateral force to translate the whole vessel quasi-statically, overcoming
    stiction at all 9 supports simultaneously (virtual work).

    For a unit lateral displacement s in direction theta, support i slides by
    |d(delta_i)/ds| = |sin(theta - phi_i)|, so the work-conjugate breakaway
    force is  F(theta) = sum_i f_i * |sin(theta - phi_i)|, f_i = mu * axial.
    Returns (mean, min, max) over direction.
    """
    f = stiction_per_support_N(mu)
    th = np.linspace(0, 2 * np.pi, nd, endpoint=False)
    F = np.array([np.sum(f * np.abs(np.sin(t - ANGLES))) for t in th])
    return float(F.mean()), float(F.min()), float(F.max())


# ── disruption energy / walk budget ─────────────────────────────────────────
def friction_work_full_slide_J(mu: float = MU) -> float:
    """Work done against aggregate breakaway over one full +/-1.5 mm slide."""
    Fmean, _, _ = aggregate_breakaway_N(mu)
    return Fmean * (2 * DELTA_M)        # full 3 mm diameter traverse


def free_mass_KE_J(impulse_Ns: float = 1e5) -> tuple[float, float]:
    """Old-report free-mass estimate: v = J/M, KE = 1/2 M v^2.
    Returns (v_mm_per_s, KE_J)."""
    M = M_TONNES * 1e3
    v = impulse_Ns / M
    return v * 1e3, 0.5 * M * v ** 2


def walk_per_event_mm(KE_J: float, mu: float = MU) -> float:
    """Distance a given kinetic energy can push the vessel against breakaway."""
    Fmean, _, _ = aggregate_breakaway_N(mu)
    return KE_J / Fmean * 1e3


# ── thermal ──────────────────────────────────────────────────────────────────
def radial_breathing_mm(dT: float = DT_BAKE) -> float:
    """Free radial growth at the support radius (accommodated by dowel rotation)."""
    return R_S * ALPHA_SS * dT * 1e3


def offcentre_toroidal_demand_mm(d_mm: float, dT: float = DT_BAKE) -> float:
    """Toroidal slot demand induced by isotropic expansion of a vessel whose
    centroid is offset by |d|. Because delta_i^th = alpha*dT*(d . et_i) =
    A[i,:] . (alpha*dT*dx, alpha*dT*dy, 0), it is exactly a representable rigid
    shift of magnitude alpha*dT*|d| -> the worst-support toroidal demand."""
    return ALPHA_SS * dT * d_mm


def force_to_lock_micron_N(demand_mm: float, k_tor_N_per_m: float = 1e9) -> float:
    """Elastic force to absorb the toroidal mismatch if a slot is stiction-locked,
    for an assumed support toroidal stiffness (default 1 GN/m)."""
    return demand_mm * 1e-3 * k_tor_N_per_m


# ── kinematic bounds (distribution-free) ─────────────────────────────────────
def nominal_envelope_mm() -> float:
    """Max one-sided departure from centre for a centred (u=0) assembly."""
    return max_departure_mm(np.zeros(N), nd=144)


def nominal_width_mm() -> float:
    """Polytope diameter (peak-to-peak rattle) for a centred assembly."""
    return max_rattle_mm(np.zeros(N), nd=180)


# ── ledger ────────────────────────────────────────────────────────────────────
def main() -> None:
    W = weight_N()
    Fmean, Fmin, Fmax = aggregate_breakaway_N()
    fss = stiction_per_support_N()
    Fr = F_restore_at_stop_N()
    v_mm, KE = free_mass_KE_J()
    Wfric = friction_work_full_slide_J()

    print("=" * 72)
    print("ITER VV lateral mechanics ledger  (ITER_D_6TLUDY constants)")
    print("=" * 72)
    print(f"M = {M_TONNES:.0f} t   W = {W/1e6:.1f} MN   R_s = {R_S:.0f} m   "
          f"incl = {INCL_DEG:.0f} deg   slot = +/-{SLOT_MM:.1f} mm   mu = {MU:.2f}")
    print("-" * 72)
    print("GRAVITATIONAL CENTRING (lateral pendulum)")
    print(f"  h_conv             = {h_conv_m():8.2f} m")
    print(f"  L_eff              = {L_eff_m():8.2f} m   (h - z_CoG, z_CoG={Z_COG:.0f} m est.)")
    print(f"  K = W/L_eff        = {K_N_per_m()/1e6:8.3f} kN/mm   (was ~3.0 at R_s=8m,M=8kt)")
    print(f"  T_n                = {Tn_s():8.2f} s")
    print(f"  F_restore @ stop   = {Fr/1e3:8.2f} kN   (K * 1.5 mm)")
    print("-" * 72)
    print("SUPPORT LOAD & FRICTION")
    print(f"  vertical/support   = {support_vertical_N()/1e6:8.2f} MN")
    print(f"  axial/support(/cos)= {support_axial_N()/1e6:8.2f} MN   (reviewer: ~10 MN)")
    print(f"  stiction/support   = {fss/1e6:8.2f} MN   (mu*axial; reviewer: ~1 MN)")
    print(f"  aggregate breakaway= {Fmean/1e6:8.2f} MN   (mean over dir; "
          f"range {Fmin/1e6:.2f}-{Fmax/1e6:.2f})")
    print("-" * 72)
    print("STICTION vs CENTRING  (does gravity recentre the vessel?)")
    print(f"  aggregate breakaway / F_restore = {Fmean/Fr:8.0f} x   (~1000x => NO)")
    print(f"  per-support stiction / F_restore = {fss/Fr:8.0f} x")
    print("-" * 72)
    print("DISRUPTION / WALK BUDGET")
    print(f"  old-report impulse J=1e5 N.s -> v={v_mm:.1f} mm/s, KE={KE:.0f} J")
    print(f"  friction work over full 3 mm slide = {Wfric/1e3:8.2f} kJ")
    print(f"  => old KE moves vessel only ~{walk_per_event_mm(KE):.3f} mm "
          f"(friction work >> KE: old impulse is far below VVLS loads)")
    print(f"  VVLS bounding loads (MN-tens of MN) exceed {Fmean/1e6:.1f} MN breakaway "
          f"-> vessel SLIDES and walks")
    print("-" * 72)
    print("THERMAL CYCLING (22 -> 100 C, dT=%.0f K)" % DT_BAKE)
    print(f"  free radial breathing       = {radial_breathing_mm():8.2f} mm  (dowel rotation, free)")
    for d in (1.5, 3.0):
        dem = offcentre_toroidal_demand_mm(d)
        fl = force_to_lock_micron_N(dem)
        print(f"  off-centre d={d:.1f}mm: toroidal demand = {dem*1e3:6.2f} um "
              f"-> lock force ~{fl/1e3:6.1f} kN  (<< {fss/1e6:.1f} MN stiction)")
    print(f"  => thermal expansion CANNOT break stiction; no coordinated recentring")
    print("-" * 72)
    print("DISTRIBUTION-FREE KINEMATIC BOUNDS  (independent of the gap prior)")
    print(f"  nominal centred envelope (departure) = {nominal_envelope_mm():.3f} mm")
    print(f"  nominal polytope WIDTH (rattle)      = {nominal_width_mm():.3f} mm")
    print("=" * 72)


if __name__ == "__main__":
    main()
