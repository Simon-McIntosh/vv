"""
ITER VV Rattle Monte Carlo Analysis — Correct Physics

The physical model:
- 9 gravity supports at R=8m, evenly spaced
- Each support has a 3mm toroidal gap (±1.5mm)
- Assembly state: each gap sampled uniformly and independently
- Rattle: max additional rigid body motion within remaining gaps
- Uses LP (scipy HiGHS) — NOT pseudoinverse — for correct constraint handling
"""

import numpy as np
from scipy.optimize import linprog
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import os

os.makedirs("plots", exist_ok=True)
np.random.seed(42)

VESSEL_RADIUS = 8.0      # metres
NUM_SUPPORTS = 9
HALF_TOL = 1.5e-3        # ±1.5 mm in metres

def build_geometry():
    """Build support geometry and constraint matrix."""
    angles = np.pi/2 - np.linspace(0, 2*np.pi, NUM_SUPPORTS, endpoint=False)
    positions = np.column_stack([VESSEL_RADIUS*np.cos(angles), VESSEL_RADIUS*np.sin(angles)])
    toroidal_dirs = np.column_stack([-np.sin(angles), np.cos(angles)])
    
    # A[i,:] = [t_x, t_y, -y_i*t_x + x_i*t_y] where last col = R (rotation coeff)
    A = np.zeros((NUM_SUPPORTS, 3))
    for i in range(NUM_SUPPORTS):
        t = toroidal_dirs[i]
        x, y = positions[i]
        A[i, 0] = t[0]
        A[i, 1] = t[1]
        A[i, 2] = -y * t[0] + x * t[1]   # = R for all supports
    
    return A, np.linalg.pinv(A), positions, angles, toroidal_dirs

def max_rattle_in_direction(u, A, half_tol, theta):
    """
    Solve rattle LP: max (cos θ, sin θ) · (δdx, δdy)
    subject to: A @ δq ≤ (half_tol - u)   [remaining forward gap]
               -A @ δq ≤ (half_tol + u)   [remaining backward gap]
    
    Returns: max additional displacement in direction theta (metres).
             Returns 0.0 if LP infeasible.
    """
    g_plus  = half_tol - u
    g_minus = half_tol + u
    A_ub = np.vstack([A, -A])
    b_ub = np.concatenate([g_plus, g_minus])
    c = [-np.cos(theta), -np.sin(theta), 0.0]
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=[(None,None)]*3, method='highs')
    return -res.fun if res.status == 0 else 0.0

def compute_rattle_envelope(u, A, half_tol, num_directions=72):
    """
    Compute the full rattle envelope for one assembly state.
    Returns array of (δdx, δdy) extreme points in num_directions.
    """
    thetas = np.linspace(0, 2*np.pi, num_directions, endpoint=False)
    g_plus  = half_tol - u
    g_minus = half_tol + u
    A_ub = np.vstack([A, -A])
    b_ub = np.concatenate([g_plus, g_minus])
    
    points = []
    for theta in thetas:
        c = [-np.cos(theta), -np.sin(theta), 0.0]
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=[(None,None)]*3, method='highs')
        if res.status == 0:
            points.append(res.x[:2])
        else:
            points.append([0.0, 0.0])
    return np.array(points)

def max_rattle_magnitude(u, A, half_tol, num_directions=36):
    """Max rattle magnitude in any direction for one assembly state."""
    envelope = compute_rattle_envelope(u, A, half_tol, num_directions)
    return np.max(np.linalg.norm(envelope, axis=1))

def assembly_position(u, A):
    """Best-fit rigid body position for assembly gap state u."""
    return np.linalg.lstsq(A, u, rcond=None)[0]

# ─────────────────────────────────────────────────────────────────────────────
# DEMONSTRATION PLOTS (understanding the physics)
# ─────────────────────────────────────────────────────────────────────────────

def plot_support_state(ax, A, positions, angles, u, title, half_tol):
    """Draw the VV with supports showing gap state."""
    theta = np.linspace(0, 2*np.pi, 200)
    ax.plot(VESSEL_RADIUS*np.cos(theta), VESSEL_RADIUS*np.sin(theta),
            'b-', lw=2, label='VV outline')
    ax.plot(0, 0, 'ko', ms=6)
    
    g_plus  = half_tol - u
    g_minus = half_tol + u
    
    for i, (pos, ang, ui) in enumerate(zip(positions, angles, u)):
        ax.plot([0, pos[0]], [0, pos[1]], 'gray', lw=0.8, alpha=0.4)
        
        # Toroidal direction at this support
        tor = np.array([-np.sin(ang), np.cos(ang)])
        
        # Scale up for visibility (×2000 to convert m → visible units at 8m scale)
        scale = 2000

        # Full gap extent
        p_pos = pos + tor * half_tol * scale
        p_neg = pos - tor * half_tol * scale
        ax.plot([p_neg[0], p_pos[0]], [p_neg[1], p_pos[1]], 'lightgray', lw=4, solid_capstyle='round')
        
        # Consumed gap (where pin currently is)
        p_pin = pos + tor * ui * scale
        ax.plot([pos[0], p_pin[0]], [pos[1], p_pin[1]],
                'red' if ui > 0 else 'royalblue', lw=4, solid_capstyle='round',
                alpha=0.8)
        
        # Remaining forward gap
        if g_plus[i] > 1e-9:
            ax.plot([p_pin[0], p_pos[0]], [p_pin[1], p_pos[1]], 'lime', lw=4, solid_capstyle='round', alpha=0.8)
        # Remaining backward gap
        if g_minus[i] > 1e-9:
            ax.plot([p_neg[0], p_pin[0]], [p_neg[1], p_pin[1]], 'cyan', lw=4, solid_capstyle='round', alpha=0.8)
        
        ax.annotate(f'S{i+1}\nu={ui*1000:+.1f}', pos*1.15,
                    fontsize=7, ha='center', va='center')
    
    ax.set_xlim(-12, 12); ax.set_ylim(-12, 12)
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.2)

def plot_demonstration():
    """
    Three-panel plot demonstrating the physics.
    Panel A: Nominal state (all u=0) → full rattle
    Panel B: Alternating state (max blocking) → zero rattle  
    Panel C: Random state → partial rattle
    """
    A, A_pinv, positions, angles, toroidal_dirs = build_geometry()
    half_tol = HALF_TOL
    
    # Three cases
    u_nominal    = np.zeros(NUM_SUPPORTS)
    u_alternating = np.array([+half_tol if i%2==0 else -half_tol for i in range(NUM_SUPPORTS)])
    rng = np.random.default_rng(7)
    u_random     = rng.uniform(-half_tol, half_tol, NUM_SUPPORTS)
    
    cases = [
        (u_nominal,     "A: Nominal assembly (u=0)\nAll gaps symmetric → maximum rattle"),
        (u_alternating, "B: Alternating max blocking\nZero rattle (VV locked)"),
        (u_random,      "C: Random assembly state\nPartial rattle"),
    ]
    
    fig = plt.figure(figsize=(18, 14))
    
    for col, (u, title) in enumerate(cases):
        # Top: support state visualisation
        ax_top = fig.add_subplot(3, 3, col+1)
        plot_support_state(ax_top, A, positions, angles, u, title, half_tol)
        
        # Compute rattle envelope
        envelope = compute_rattle_envelope(u, A, half_tol, num_directions=72)
        max_mag  = np.max(np.linalg.norm(envelope, axis=1))
        
        # Also compute via pseudoinverse (WRONG) for comparison
        c_x = A_pinv[:2,:].T @ np.array([1.0, 0.0])
        g_plus = half_tol - u
        g_minus = half_tol + u
        pseudo_rattle = np.sum(np.where(c_x > 0, c_x*g_plus, -c_x*g_minus)) * 1000
        
        # Middle: rattle envelope (in mm)
        ax_mid = fig.add_subplot(3, 3, col+4)
        env_mm = envelope * 1000
        env_closed = np.vstack([env_mm, env_mm[0]])
        ax_mid.fill(env_closed[:,0], env_closed[:,1], alpha=0.25, color='steelblue')
        ax_mid.plot(env_closed[:,0], env_closed[:,1], 'steelblue', lw=2, label='Rattle envelope (LP)')
        ax_mid.plot(0, 0, 'ko', ms=6, zorder=5, label='Assembly position')
        ax_mid.set_xlabel('δdx (mm)'); ax_mid.set_ylabel('δdy (mm)')
        ax_mid.set_aspect('equal'); ax_mid.grid(True, alpha=0.3)
        lim = max(max_mag*1000*1.3, 0.1)
        ax_mid.set_xlim(-lim, lim); ax_mid.set_ylim(-lim, lim)
        ax_mid.set_title(f'Rattle envelope\nMax |rattle| = {max_mag*1000:.4f} mm\n'
                         f'(pseudoinverse would give {pseudo_rattle:.4f} mm — WRONG for case B)',
                         fontsize=9)
        ax_mid.legend(fontsize=8)
        
        # Bottom: bar chart of per-direction rattle
        ax_bot = fig.add_subplot(3, 3, col+7)
        thetas = np.linspace(0, 2*np.pi, 72, endpoint=False)
        radii  = np.linalg.norm(envelope, axis=1) * 1000
        ax_bot.bar(np.degrees(thetas), radii, width=5, color='steelblue', alpha=0.7)
        ax_bot.set_xlabel('Direction (°)'); ax_bot.set_ylabel('Rattle reach (mm)')
        ax_bot.set_title('Rattle vs direction', fontsize=9)
        ax_bot.set_xlim(0, 360)
        ax_bot.set_xticks([0, 90, 180, 270, 360])
    
    # Legend for gap colours
    leg_items = [
        mpatches.Patch(color='lightgray', label='Total gap (±1.5 mm)'),
        mpatches.Patch(color='red',       label='Pin +forward (consumed)'),
        mpatches.Patch(color='royalblue', label='Pin -backward (consumed)'),
        mpatches.Patch(color='lime',      label='Remaining forward gap'),
        mpatches.Patch(color='cyan',      label='Remaining backward gap'),
    ]
    fig.legend(handles=leg_items, loc='lower center', ncol=5, fontsize=9,
               bbox_to_anchor=(0.5, 0.01))
    
    plt.suptitle('ITER VV Rattle Physics — Three Assembly Scenarios\n'
                 'Rattle = max additional rigid-body motion within remaining gaps (correct LP)',
                 fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    plt.savefig('plots/rattle_physics_demonstration.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: plots/rattle_physics_demonstration.png")
    
    # Print key verification numbers
    for u, label in [(u_nominal, 'Nominal'), (u_alternating, 'Alternating'), (u_random, 'Random')]:
        env = compute_rattle_envelope(u, A, half_tol, 36)
        max_r = np.max(np.linalg.norm(env, axis=1))
        c_x = A_pinv[:2,:].T @ np.array([1.0, 0.0])
        g_plus = half_tol - u
        g_minus = half_tol + u
        pseudo = np.sum(np.where(c_x > 0, c_x*g_plus, -c_x*g_minus))
        print(f"  {label}: LP rattle = {max_r*1000:.4f} mm, pseudoinverse (WRONG) = {pseudo*1000:.4f} mm")

def run_mc_simulation(N_samples=2000):
    """
    MC simulation: sample N assembly states, compute max rattle for each.
    Uses correct LP formulation.
    Returns: u_samples (N,9), assembly_positions (N,3), max_rattles (N,)
    """
    A, A_pinv, positions, angles, toroidal_dirs = build_geometry()
    half_tol = HALF_TOL
    
    print(f"Running MC with {N_samples} samples (LP per sample)...")
    u_samples = np.random.uniform(-half_tol, half_tol, (N_samples, NUM_SUPPORTS))
    assembly_pos = np.array([np.linalg.lstsq(A, u, rcond=None)[0] for u in u_samples])
    
    max_rattles = []
    for i, u in enumerate(u_samples):
        if i % 200 == 0:
            print(f"  Sample {i}/{N_samples}...")
        mr = max_rattle_magnitude(u, A, half_tol, num_directions=24)
        max_rattles.append(mr)
    
    return A, u_samples, assembly_pos, np.array(max_rattles)

def plot_mc_results(A, u_samples, assembly_pos, max_rattles):
    """Plot MC statistics: distribution of rattle vs assembly position."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    dx0_mm = assembly_pos[:, 0] * 1000
    dy0_mm = assembly_pos[:, 1] * 1000
    mr_mm  = max_rattles * 1000
    
    # 1. Assembly position scatter coloured by max rattle
    ax = axes[0, 0]
    sc = ax.scatter(dx0_mm, dy0_mm, c=mr_mm, cmap='plasma', s=8, alpha=0.6)
    plt.colorbar(sc, ax=ax, label='Max rattle (mm)')
    ax.set_xlabel('Assembly dx₀ (mm)'); ax.set_ylabel('Assembly dy₀ (mm)')
    ax.set_title('Assembly position distribution\n(coloured by max rattle)')
    ax.set_aspect('equal'); ax.grid(True, alpha=0.3)
    ax.axhline(0, color='k', lw=0.5); ax.axvline(0, color='k', lw=0.5)
    
    # 2. Max rattle PDF
    ax = axes[0, 1]
    ax.hist(mr_mm, bins=50, density=True, color='steelblue', edgecolor='k', alpha=0.7)
    ax.axvline(np.percentile(mr_mm, 95), color='orange', ls='--', lw=2,
               label=f'P95 = {np.percentile(mr_mm,95):.3f} mm')
    ax.axvline(np.percentile(mr_mm, 99), color='red', ls='--', lw=2,
               label=f'P99 = {np.percentile(mr_mm,99):.3f} mm')
    ax.axvline(np.max(mr_mm), color='darkred', ls='-', lw=1.5,
               label=f'Max = {np.max(mr_mm):.3f} mm')
    ax.set_xlabel('Max rattle (mm)'); ax.set_ylabel('PDF')
    ax.set_title('Max rattle magnitude distribution\n(all assembly states)')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    
    # 3. Rattle vs assembly offset magnitude
    ax = axes[0, 2]
    offset_mm = np.sqrt(dx0_mm**2 + dy0_mm**2)
    ax.scatter(offset_mm, mr_mm, s=4, alpha=0.3, color='steelblue')
    ax.set_xlabel('Assembly offset magnitude (mm)'); ax.set_ylabel('Max rattle (mm)')
    ax.set_title('Rattle vs assembly offset\n(larger offset → smaller rattle)')
    ax.grid(True, alpha=0.3)
    
    # 4. X rattle in +X and -X directions
    ax = axes[1, 0]
    A2, A_pinv2, positions, angles, toroidal_dirs = build_geometry()
    half_tol = HALF_TOL
    rattle_px = np.array([max_rattle_in_direction(u, A2, half_tol, 0) for u in u_samples]) * 1000
    rattle_nx = np.array([max_rattle_in_direction(u, A2, half_tol, np.pi) for u in u_samples]) * 1000
    ax.hist(rattle_px, bins=40, density=True, alpha=0.6, color='blue', label='+X rattle')
    ax.hist(rattle_nx, bins=40, density=True, alpha=0.6, color='red', label='-X rattle')
    ax.set_xlabel('Rattle distance (mm)'); ax.set_ylabel('PDF')
    ax.set_title('X-direction rattle distribution\n(from random assembly states)')
    ax.legend(); ax.grid(True, alpha=0.3)
    
    # 5. Total X rattle (should NOT be constant — unlike pseudoinverse claim)
    ax = axes[1, 1]
    total_x = rattle_px + rattle_nx
    ax.hist(total_x, bins=40, density=True, color='purple', edgecolor='k', alpha=0.7)
    ax.set_xlabel('Total X rattle range (mm)'); ax.set_ylabel('PDF')
    ax.set_title(f'Total X rattle range distribution\n'
                 f'(mean={np.mean(total_x):.3f}, max={np.max(total_x):.3f} mm)\n'
                 f'NOT constant — pseudoinverse claim was wrong')
    ax.grid(True, alpha=0.3)
    
    # 6. Statistics table
    ax = axes[1, 2]
    ax.axis('off')
    stats_text = f"""MC RATTLE STATISTICS
{'─'*40}
N samples:  {len(max_rattles):,}

MAX RATTLE (any direction):
  Mean:     {np.mean(mr_mm):.4f} mm
  Std:      {np.std(mr_mm):.4f} mm
  P50:      {np.percentile(mr_mm,50):.4f} mm
  P95:      {np.percentile(mr_mm,95):.4f} mm
  P99:      {np.percentile(mr_mm,99):.4f} mm
  Maximum:  {np.max(mr_mm):.4f} mm

X RATTLE (+X direction):
  Mean:     {np.mean(rattle_px):.4f} mm
  P95:      {np.percentile(rattle_px,95):.4f} mm
  Max:      {np.max(rattle_px):.4f} mm

ASSEMBLY POSITION:
  X std:    {np.std(dx0_mm):.4f} mm
  Y std:    {np.std(dy0_mm):.4f} mm
  |r| P95:  {np.percentile(np.sqrt(dx0_mm**2+dy0_mm**2),95):.4f} mm

KEY INSIGHT:
  Alternating assembly → 0 rattle
  Uniform same-side → max rattle
  MC captures the full distribution"""
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=9,
            va='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', fc='lightyellow', alpha=0.9))
    
    plt.suptitle('ITER VV Rattle MC Analysis — Correct LP Physics\n'
                 '(Each sample: random independent gap state → LP for max rattle)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig('plots/rattle_mc_results.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: plots/rattle_mc_results.png")
    return {
        'n': len(max_rattles),
        'mean_mm': float(np.mean(mr_mm)),
        'p95_mm': float(np.percentile(mr_mm, 95)),
        'p99_mm': float(np.percentile(mr_mm, 99)),
        'max_mm': float(np.max(mr_mm)),
        'rattle_px_p95': float(np.percentile(rattle_px, 95)),
    }

def main():
    print("="*60)
    print("ITER VV Rattle MC — Correct LP Physics")
    print("="*60)
    
    # Step 1: Verify key cases
    A, A_pinv, positions, angles, toroidal_dirs = build_geometry()
    print("\nVERIFICATION of key cases:")
    u_alternating = np.array([HALF_TOL if i%2==0 else -HALF_TOL for i in range(NUM_SUPPORTS)])
    rattle_alt = max_rattle_magnitude(u_alternating, A, HALF_TOL, 36)
    print(f"  Alternating case max rattle: {rattle_alt*1000:.6f} mm  (expect ~0)")
    
    u_nominal = np.zeros(NUM_SUPPORTS)
    rattle_nom = max_rattle_magnitude(u_nominal, A, HALF_TOL, 36)
    print(f"  Nominal case max rattle:     {rattle_nom*1000:.4f} mm  (expect ~1.547)")
    
    # Step 2: Demonstration plots
    print("\nGenerating physics demonstration plots...")
    plot_demonstration()
    
    # Step 3: MC
    A2, u_samples, assembly_pos, max_rattles = run_mc_simulation(N_samples=2000)
    
    # Step 4: MC results plots
    print("\nGenerating MC results plots...")
    stats = plot_mc_results(A2, u_samples, assembly_pos, max_rattles)
    
    print("\n" + "="*60)
    print("RESULTS FOR ASSEMBLY TEAM")
    print("="*60)
    print(f"  Max rattle (P95): {stats['p95_mm']:.3f} mm")
    print(f"  Max rattle (P99): {stats['p99_mm']:.3f} mm")
    print(f"  Max rattle (worst case): {stats['max_mm']:.3f} mm")
    print(f"  +X rattle P95: {stats['rattle_px_p95']:.3f} mm")
    print(f"\nPlots saved to plots/")

if __name__ == "__main__":
    main()
