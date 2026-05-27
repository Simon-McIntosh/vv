"""
ITER Vacuum Vessel - Correct LP-Based Rattle MC Analysis

Implements the CORRECT bounding analysis using scipy.optimize.linprog (HiGHS),
avoiding the pseudoinverse/bounding_box approach which over-estimates bounds.

Physical setup:
- R = 8.0 m gravity support radius
- 9 supports evenly spaced from top: φᵢ = π/2 - 2πi/9
- TOROIDAL_TOLERANCE = 3 mm total (±1.5 mm per side)

Rigid body DOFs: q = [dx (m), dy (m), dθ (rad)]
Constraint matrix A (9×3): τᵢ = A[i,:] @ q  (toroidal displacement at support i)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.optimize import linprog

os.makedirs("plots", exist_ok=True)

# Physical parameters
R = 8.0                      # gravity support radius (m)
N_SUPPORTS = 9
TOROIDAL_TOLERANCE = 3e-3    # 3 mm total
HALF_TOL = TOROIDAL_TOLERANCE / 2.0  # 1.5 mm per side


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def build_geometry():
    """
    Build support geometry and constraint matrix.

    Returns
    -------
    A : ndarray (9, 3)
        Constraint matrix: τᵢ = A[i,:] @ q
    A_pinv : ndarray (3, 9)
        Moore-Penrose pseudoinverse of A
    positions : ndarray (9, 2)
        (x, y) of each support in metres
    angles : ndarray (9,)
        Angular position of each support (rad), starting from top
    """
    angles = np.array([np.pi / 2 - 2 * np.pi * i / N_SUPPORTS for i in range(N_SUPPORTS)])
    positions = np.column_stack([R * np.cos(angles), R * np.sin(angles)])

    A = np.zeros((N_SUPPORTS, 3))
    A[:, 0] = -np.sin(angles)    # toroidal contribution from dx
    A[:, 1] =  np.cos(angles)    # toroidal contribution from dy
    A[:, 2] =  R                 # toroidal contribution from dθ (= R for all i)

    A_pinv = np.linalg.pinv(A)
    return A, A_pinv, positions, angles


# ---------------------------------------------------------------------------
# LP bounds
# ---------------------------------------------------------------------------

def compute_lp_bounds(A, half_tol, num_directions=72):
    """
    Compute the reachable VV centre envelope using LP (HiGHS).

    For each direction θ in [0, 2π): maximise [cos θ, sin θ, 0] @ q
    subject to -half_tol ≤ A @ q ≤ half_tol.

    Also computes axis-aligned extremes dx_max, dy_max, dθ_max.

    Parameters
    ----------
    A : ndarray (9, 3)
    half_tol : float
        Per-side tolerance in metres.
    num_directions : int
        Number of radial directions for the envelope polygon.

    Returns
    -------
    envelope : ndarray (num_directions, 2)
        Extreme (dx, dy) for each direction (metres).
    dx_max : float
        Maximum |dx| (metres).
    dy_max : float
        Maximum |dy| (metres).
    dtheta_max : float
        Maximum |dθ| (rad).
    """
    bounds_q = [(None, None)] * 3
    b_ub_upper = np.full(N_SUPPORTS, half_tol)   # A @ q ≤ half_tol
    b_ub_lower = np.full(N_SUPPORTS, half_tol)   # -A @ q ≤ half_tol  → A_ineq = [-A; A]
    A_ub = np.vstack([-A, A])
    b_ub = np.concatenate([b_ub_lower, b_ub_upper])

    def _lp_max(c_obj):
        """Maximise c_obj @ q (minimise -c_obj @ q)."""
        res = linprog(-c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds_q, method='highs')
        assert res.status == 0, f"LP infeasible/unbounded: status={res.status}, message={res.message}"
        return -res.fun, res.x

    # Axis-aligned extremes
    _, x_dx = _lp_max(np.array([1.0, 0.0, 0.0]))
    dx_max = x_dx[0]
    _, x_dy = _lp_max(np.array([0.0, 1.0, 0.0]))
    dy_max = x_dy[1]
    _, x_dt = _lp_max(np.array([0.0, 0.0, 1.0]))
    dtheta_max = x_dt[2]

    # Envelope over directions
    thetas = np.linspace(0, 2 * np.pi, num_directions, endpoint=False)
    envelope = np.zeros((num_directions, 2))
    for k, theta in enumerate(thetas):
        c = np.array([np.cos(theta), np.sin(theta), 0.0])
        val, x = _lp_max(c)
        envelope[k] = x[:2]

    return envelope, dx_max, dy_max, dtheta_max


# ---------------------------------------------------------------------------
# MC sampling
# ---------------------------------------------------------------------------

def sample_assembly_states(A, half_tol, N_samples=100_000):
    """
    Sample VV assembly positions uniformly from the feasible polytope.

    Uses rejection sampling with an oversampled bounding box.

    Parameters
    ----------
    A : ndarray (9, 3)
    half_tol : float
    N_samples : int

    Returns
    -------
    q_samples : ndarray (N_samples, 3)
        Each row is [dx (m), dy (m), dθ (rad)].
    """
    np.random.seed(42)

    # Conservative bounding box (slightly larger than LP bounds)
    # LP-derived bounds (from verified physics)
    dx_bnd    = 1.6e-3    # ±1.6 mm
    dy_bnd    = 1.6e-3    # ±1.6 mm
    dtheta_bnd = 0.20e-3  # ±0.20 mrad

    box = np.array([dx_bnd, dy_bnd, dtheta_bnd])

    collected = []
    total_collected = 0
    oversample = 10

    while total_collected < N_samples:
        n_try = oversample * N_samples
        q_try = (2 * np.random.rand(n_try, 3) - 1) * box[None, :]
        tau = q_try @ A.T                        # (n_try, 9)
        feasible = np.all(np.abs(tau) <= half_tol, axis=1)
        q_ok = q_try[feasible]
        collected.append(q_ok)
        total_collected += len(q_ok)

    q_all = np.vstack(collected)
    return q_all[:N_samples]


# ---------------------------------------------------------------------------
# Rattle computation
# ---------------------------------------------------------------------------

def compute_rattle_from_assembly(q0_batch, A, half_tol, num_directions=36):
    """
    Compute max additional LP displacement in each direction from each assembly state.

    For assembly state q0 and direction ψ (unit vector in XY), the maximum
    additional displacement in direction ψ while remaining feasible is:

        rattle(ψ, q0) = M(ψ) − ψᵀ q0[:2]

    where M(ψ) = LP_bound(ψ) = max_{|A@q|≤half_tol} ψᵀ q[:2].

    This is exact (not an approximation) and follows directly from:
        max_{q1 feasible} ψᵀ(q1 − q0)[:2]
        = max_{q1 feasible} ψᵀ q1[:2] − ψᵀ q0[:2]
        = M(ψ) − ψᵀ q0[:2]

    INVARIANT: rattle(ψ, q0) + rattle(−ψ, q0) = 2·M(ψ) for all feasible q0,
    because M(−ψ) = M(ψ) (feasible polytope is symmetric about q=0) and
    ψᵀ q0[:2] cancels.

    Parameters
    ----------
    q0_batch : ndarray (N, 3)
    A : ndarray (9, 3)
    half_tol : float
    num_directions : int

    Returns
    -------
    rattle : ndarray (N, num_directions)
        rattle[n, d] = max additional displacement from state n in direction d (metres)
    """
    thetas = np.linspace(0, 2 * np.pi, num_directions, endpoint=False)
    psi = np.column_stack([np.cos(thetas), np.sin(thetas)])   # (D, 2)

    # LP bounds M(ψ) for each direction (computed once)
    A_ub = np.vstack([-A, A])
    b_ub = np.full(2 * N_SUPPORTS, half_tol)
    M = np.zeros(num_directions)
    for k in range(num_directions):
        c = np.array([psi[k, 0], psi[k, 1], 0.0])
        res = linprog(-c, A_ub=A_ub, b_ub=b_ub, bounds=[(None, None)] * 3, method='highs')
        assert res.status == 0, f"LP infeasible: status={res.status}"
        M[k] = -res.fun

    # rattle[n, d] = M[d] - psi[d] @ q0[n, :2]
    proj = q0_batch[:, :2] @ psi.T   # (N, D): proj[n, d] = psi[d] · q0[n, :2]
    rattle = M[None, :] - proj        # (N, D), always ≥ 0 for feasible q0

    return rattle


def _verify_rattle_invariant(A, half_tol, num_directions=36):
    """
    Assert that rattle_fwd(θ) + rattle_bwd(θ) = 2·LP_bound(θ) for a small test batch.

    This invariant holds analytically because the feasible polytope is symmetric
    about q=0, so LP_bound(−ψ) = LP_bound(ψ), and the projection term cancels.
    """
    np.random.seed(0)
    q_test = sample_assembly_states(A, half_tol, N_samples=50)

    rattle = compute_rattle_from_assembly(q_test, A, half_tol, num_directions)

    # Forward + backward: direction d and d + D/2 are opposites
    D2 = num_directions // 2
    fwd = rattle[:, :D2]
    bwd = rattle[:, D2:]
    total = fwd + bwd   # (N, D//2) — must equal 2·M[d] for d in 0..D/2-1

    # Recompute LP bounds for first half of directions
    thetas = np.linspace(0, 2 * np.pi, num_directions, endpoint=False)[:D2]
    A_ub = np.vstack([-A, A])
    b_ub = np.full(2 * N_SUPPORTS, half_tol)
    expected = np.zeros(D2)
    for k, theta in enumerate(thetas):
        c = np.array([np.cos(theta), np.sin(theta), 0.0])
        res = linprog(-c, A_ub=A_ub, b_ub=b_ub, bounds=[(None, None)] * 3, method='highs')
        assert res.status == 0
        expected[k] = 2 * (-res.fun)

    err = np.abs(total - expected[None, :]).max()
    assert err < 1e-9, f"Rattle invariant FAILED: max error = {err:.2e}"
    print(f"  [OK] Rattle invariant verified: max error = {err:.2e}")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_feasible_envelope(envelope_xy, A, half_tol, assembly_samples):
    """
    Three-panel plot: LP envelope + MC scatter, dx histogram, dy histogram.

    Parameters
    ----------
    envelope_xy : ndarray (D, 2)  in metres
    A : ndarray (9, 3)
    half_tol : float
    assembly_samples : ndarray (N, 3)
    """
    dx_mm = assembly_samples[:, 0] * 1e3
    dy_mm = assembly_samples[:, 1] * 1e3
    env_mm = envelope_xy * 1e3

    fig = plt.figure(figsize=(14, 5))
    gs = GridSpec(1, 3, figure=fig, wspace=0.35)

    # Panel 1: envelope + scatter
    ax1 = fig.add_subplot(gs[0])
    env_closed = np.vstack([env_mm, env_mm[0]])
    ax1.plot(env_closed[:, 0], env_closed[:, 1], 'b-', lw=2, label='LP envelope')
    ax1.scatter(dx_mm[::50], dy_mm[::50], s=1, alpha=0.3, color='grey', label='MC samples (1/50)')
    ax1.set_xlabel('dx (mm)')
    ax1.set_ylabel('dy (mm)')
    ax1.set_title('VV Centre Reachable Envelope')
    ax1.set_aspect('equal')
    ax1.axhline(0, color='k', lw=0.5)
    ax1.axvline(0, color='k', lw=0.5)
    ax1.legend(fontsize=8)

    # Panel 2: dx histogram
    ax2 = fig.add_subplot(gs[1])
    ax2.hist(dx_mm, bins=80, density=True, color='steelblue', edgecolor='none', alpha=0.8)
    ax2.set_xlabel('dx (mm)')
    ax2.set_ylabel('Density')
    ax2.set_title('Assembly X offset distribution')

    # Panel 3: dy histogram
    ax3 = fig.add_subplot(gs[2])
    ax3.hist(dy_mm, bins=80, density=True, color='coral', edgecolor='none', alpha=0.8)
    ax3.set_xlabel('dy (mm)')
    ax3.set_ylabel('Density')
    ax3.set_title('Assembly Y offset distribution')

    fig.suptitle('ITER VV Feasible Assembly Positions  (LP + MC)', fontsize=12, fontweight='bold')
    fig.savefig('plots/feasible_envelope.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  Saved plots/feasible_envelope.png")


def plot_rattle_analysis(assembly_samples_q, rattle_fwd, rattle_bwd, lp_dx_max, lp_dy_max):
    """
    Four-panel rattle analysis plot.

    Parameters
    ----------
    assembly_samples_q : ndarray (N, 3)
    rattle_fwd : ndarray (N,)  max additional +X displacement (m)
    rattle_bwd : ndarray (N,)  max additional -X displacement (m)
    lp_dx_max : float
    lp_dy_max : float
    """
    dx_mm  = assembly_samples_q[:, 0] * 1e3
    dy_mm  = assembly_samples_q[:, 1] * 1e3
    rfwd_mm = rattle_fwd * 1e3
    rbwd_mm = rattle_bwd * 1e3
    total_mm = (rfwd_mm + rbwd_mm)
    rmax_mm  = np.maximum(rfwd_mm, rbwd_mm)

    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 2, figure=fig, wspace=0.35, hspace=0.45)

    # Panel 1: scatter coloured by rattle magnitude
    ax1 = fig.add_subplot(gs[0, 0])
    sc = ax1.scatter(dx_mm, dy_mm, c=rmax_mm, s=1, cmap='viridis')
    fig.colorbar(sc, ax=ax1, label='Max rattle (mm)')
    ax1.set_xlabel('dx₀ (mm)')
    ax1.set_ylabel('dy₀ (mm)')
    ax1.set_title('Assembly state coloured by rattle')
    ax1.set_aspect('equal')

    # Panel 2: PDF of rattle_fwd
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(rfwd_mm, bins=80, density=True, color='steelblue', edgecolor='none', alpha=0.8)
    ax2.axvline(lp_dx_max * 1e3, color='r', lw=1.5, ls='--', label=f'LP max = {lp_dx_max*1e3:.3f} mm')
    ax2.set_xlabel('Rattle fwd X (mm)')
    ax2.set_ylabel('Density')
    ax2.set_title('PDF of max forward X rattle from assembly state')
    ax2.legend(fontsize=8)

    # Panel 3: PDF of total rattle (should be delta at 2×lp_dx_max)
    ax3 = fig.add_subplot(gs[1, 0])
    total_range = total_mm.max() - total_mm.min()
    if total_range < 1e-6:
        # Constant — plot a vertical spike
        ax3.axvline(total_mm[0], color='coral', lw=3, label='delta function')
    else:
        ax3.hist(total_mm, bins=80, density=True, color='coral', edgecolor='none', alpha=0.8)
    ax3.axvline(2 * lp_dx_max * 1e3, color='r', lw=1.5, ls='--',
                label=f'2×LP = {2*lp_dx_max*1e3:.3f} mm')
    ax3.set_xlabel('Total rattle X (fwd + bwd) (mm)')
    ax3.set_ylabel('Density')
    ax3.set_title('Total rattle range (CONSTANT = 2×LP bound)')
    ax3.legend(fontsize=8)

    # Panel 4: text summary
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')
    lines = [
        f"LP bounds (exact):",
        f"  dx_max = ±{lp_dx_max*1e3:.3f} mm",
        f"  dy_max = ±{lp_dy_max*1e3:.3f} mm",
        f"",
        f"Total rattle range (constant):",
        f"  X = {2*lp_dx_max*1e3:.3f} mm",
        f"  Y = {2*lp_dy_max*1e3:.3f} mm",
        f"",
        f"Rattle fwd X:",
        f"  mean = {rfwd_mm.mean():.3f} mm",
        f"  std  = {rfwd_mm.std():.3f} mm",
        f"  95th = {np.percentile(rfwd_mm, 95):.3f} mm",
    ]
    ax4.text(0.05, 0.95, '\n'.join(lines), transform=ax4.transAxes,
             va='top', ha='left', fontsize=10, family='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    fig.suptitle('ITER VV Rattle Analysis', fontsize=13, fontweight='bold')
    fig.savefig('plots/rattle_analysis.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  Saved plots/rattle_analysis.png")


def plot_summary_for_assembly_team(lp_bounds, assembly_stats, rattle_stats):
    """
    Clean 2-panel summary plot for the assembly team.

    Parameters
    ----------
    lp_bounds : dict with keys dx_max, dy_max, dtheta_max, envelope (metres)
    assembly_stats : dict with keys dx_mean, dx_std, dx_p95, dy_mean, dy_std, dy_p95
    rattle_stats : dict with keys p95, p99 (metres)
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel 1: feasible envelope
    ax = axes[0]
    env_mm = lp_bounds['envelope'] * 1e3
    env_closed = np.vstack([env_mm, env_mm[0]])
    ax.fill(env_closed[:, 0], env_closed[:, 1], alpha=0.2, color='blue')
    ax.plot(env_closed[:, 0], env_closed[:, 1], 'b-', lw=2)
    ax.axhline(0, color='k', lw=0.5)
    ax.axvline(0, color='k', lw=0.5)
    ax.set_xlabel('VV centre dx (mm)')
    ax.set_ylabel('VV centre dy (mm)')
    ax.set_title('Maximum VV Centre Travel\n(LP feasible envelope)')
    ax.set_aspect('equal')
    ax.annotate(f'±{lp_bounds["dx_max"]*1e3:.3f} mm', xy=(lp_bounds['dx_max']*1e3, 0),
                xytext=(lp_bounds['dx_max']*1e3 * 0.5, lp_bounds['dy_max']*1e3 * 0.7),
                fontsize=9, color='blue',
                arrowprops=dict(arrowstyle='->', color='blue'))

    # Panel 2: table
    ax2 = axes[1]
    ax2.axis('off')

    dx_m = lp_bounds['dx_max'] * 1e3
    dy_m = lp_bounds['dy_max'] * 1e3
    dt_m = lp_bounds['dtheta_max'] * 1e3   # mrad

    rows = [
        ['Parameter', 'Value', 'Unit'],
        ['Max VV centre X', f'±{dx_m:.3f}', 'mm'],
        ['Max VV centre Y', f'±{dy_m:.3f}', 'mm'],
        ['Max VV rotation', f'±{dt_m:.4f}', 'mrad'],
        ['Total X rattle range', f'{2*dx_m:.3f}', 'mm'],
        ['Total Y rattle range', f'{2*dy_m:.3f}', 'mm'],
        ['Assembly X 95th %ile', f'{assembly_stats["dx_p95"]:.3f}', 'mm'],
        ['Assembly Y 95th %ile', f'{assembly_stats["dy_p95"]:.3f}', 'mm'],
        ['Rattle 95th %ile', f'{rattle_stats["p95"]*1e3:.3f}', 'mm'],
        ['Rattle 99th %ile', f'{rattle_stats["p99"]*1e3:.3f}', 'mm'],
    ]
    tbl = ax2.table(
        cellText=rows[1:],
        colLabels=rows[0],
        loc='center',
        cellLoc='center',
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.2, 1.6)
    ax2.set_title('Key Numbers for Assembly Team', fontsize=12, fontweight='bold', pad=20)

    fig.suptitle('ITER VV Assembly Tolerance Summary', fontsize=14, fontweight='bold')
    fig.savefig('plots/assembly_team_summary.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  Saved plots/assembly_team_summary.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Run the full VV rattle MC analysis and produce summary plots."""
    print("=" * 60)
    print("  ITER VV RATTLE MC — CORRECT LP-BASED ANALYSIS")
    print("=" * 60)

    # 1. Geometry
    print("\n[1] Building geometry...")
    A, A_pinv, positions, angles = build_geometry()
    print(f"  A shape: {A.shape},  rank: {np.linalg.matrix_rank(A)}")

    # 2. LP bounds
    print("\n[2] Computing LP bounds (72 directions)...")
    envelope, dx_max, dy_max, dtheta_max = compute_lp_bounds(A, HALF_TOL, num_directions=72)
    print(f"  dx_max    = ±{dx_max*1e3:.3f} mm")
    print(f"  dy_max    = ±{dy_max*1e3:.3f} mm")
    print(f"  dtheta_max = ±{dtheta_max*1e3:.4f} mrad")

    # Verify dtheta matches analytic value 1.5mm/8m
    analytic_dtheta = HALF_TOL / R
    assert abs(dtheta_max - analytic_dtheta) < 1e-9, \
        f"dtheta_max mismatch: {dtheta_max:.6e} vs {analytic_dtheta:.6e}"
    print(f"  [OK] dθ_max matches analytic {analytic_dtheta*1e3:.4f} mrad = 1.5mm/8m")

    # 3. Sample assembly states
    print("\n[3] Sampling 100k assembly states (rejection MC)...")
    q_samples = sample_assembly_states(A, HALF_TOL, N_samples=100_000)
    print(f"  Sampled: {q_samples.shape[0]} feasible states")

    # Verify constraints satisfied
    tau_check = q_samples @ A.T
    assert np.all(np.abs(tau_check) <= HALF_TOL + 1e-12), "Constraint violation in MC samples!"
    print("  [OK] All samples satisfy toroidal constraints")

    # 4. Rattle invariant check
    print("\n[4] Verifying rattle invariant (small batch)...")
    _verify_rattle_invariant(A, HALF_TOL, num_directions=36)

    # 5. Full rattle computation
    print("\n[5] Computing rattle for all 100k assembly states (36 directions)...")
    rattle_all = compute_rattle_from_assembly(q_samples, A, HALF_TOL, num_directions=36)
    # rattle_all shape: (100k, 36)
    # Direction 0 → +X, direction 18 → -X (opposite)
    rattle_fwd_x = rattle_all[:, 0]
    rattle_bwd_x = rattle_all[:, 18]
    rattle_fwd_y = rattle_all[:, 9]   # direction π/2 → +Y
    rattle_bwd_y = rattle_all[:, 27]  # direction 3π/2 → -Y

    max_rattle_any = rattle_all.max(axis=1)

    # 6. Plots
    print("\n[6] Generating plots...")
    plot_feasible_envelope(envelope, A, HALF_TOL, q_samples)
    plot_rattle_analysis(q_samples, rattle_fwd_x, rattle_bwd_x, dx_max, dy_max)

    assembly_stats = {
        'dx_mean': q_samples[:, 0].mean() * 1e3,
        'dx_std':  q_samples[:, 0].std()  * 1e3,
        'dx_p95':  np.percentile(np.abs(q_samples[:, 0]), 95) * 1e3,
        'dy_mean': q_samples[:, 1].mean() * 1e3,
        'dy_std':  q_samples[:, 1].std()  * 1e3,
        'dy_p95':  np.percentile(np.abs(q_samples[:, 1]), 95) * 1e3,
    }
    rattle_stats = {
        'p95': np.percentile(max_rattle_any, 95),
        'p99': np.percentile(max_rattle_any, 99),
    }
    lp_info = {
        'dx_max':    dx_max,
        'dy_max':    dy_max,
        'dtheta_max': dtheta_max,
        'envelope':  envelope,
    }
    plot_summary_for_assembly_team(lp_info, assembly_stats, rattle_stats)

    # 7. Summary
    print()
    print("=" * 60)
    print("  === BOUNDING RESULTS FOR ASSEMBLY TEAM ===")
    print()
    print("  LP Feasible Bounds (EXACT):")
    print(f"    Max VV centre displacement:  ±{dx_max*1e3:.3f} mm (X),  ±{dy_max*1e3:.3f} mm (Y)")
    print(f"    Max VV rotation:             ±{dtheta_max*1e3:.4f} mrad")
    print()
    print("  Total Rattle Range (CONSTANT, all assembly positions):")
    print(f"    X-direction: {2*dx_max*1e3:.3f} mm total range")
    print(f"    Y-direction: {2*dy_max*1e3:.3f} mm total range")
    print()
    print("  Assembly Position Distribution (100k MC samples):")
    print(f"    X offset: mean={assembly_stats['dx_mean']:.4f}, "
          f"std={assembly_stats['dx_std']:.4f}, |95th|={assembly_stats['dx_p95']:.4f} mm")
    print(f"    Y offset: mean={assembly_stats['dy_mean']:.4f}, "
          f"std={assembly_stats['dy_std']:.4f}, |95th|={assembly_stats['dy_p95']:.4f} mm")
    print()
    print("  Rattle from assembly (max in any direction):")
    print(f"    95th percentile: {rattle_stats['p95']*1e3:.4f} mm")
    print(f"    99th percentile: {rattle_stats['p99']*1e3:.4f} mm")
    print("=" * 60)


if __name__ == "__main__":
    main()
