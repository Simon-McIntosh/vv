"""
ITER Vacuum Vessel - Bounding Box and Kinematic Analysis

This analysis addresses:
1. What is the total range of motion (bounding box) in X and Y?
2. Is there kinematic blocking from the support geometry?
3. What is the "shake envelope" - how much can we move the VV?

Key distinction from previous analysis:
- Previous: displacement magnitude from nominal (one random sample)
- This: extreme positions achievable (optimization over all valid support configs)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os

os.makedirs("plots", exist_ok=True)

# Physical parameters
VESSEL_RADIUS = 8.0  # meters
NUM_SUPPORTS = 9
TOROIDAL_TOLERANCE = 3e-3  # 3mm total, so ±1.5mm

def get_support_geometry(radius, num_supports):
    """Get support positions and directions."""
    angles = np.linspace(0, 2*np.pi, num_supports, endpoint=False)
    angles = np.pi/2 - angles  # Start from top
    
    positions = np.array([[radius * np.cos(a), radius * np.sin(a)] for a in angles])
    # Toroidal direction: perpendicular to radial, counter-clockwise
    toroidal_dirs = np.array([[-np.sin(a), np.cos(a)] for a in angles])
    # Radial direction: outward from center
    radial_dirs = np.array([[np.cos(a), np.sin(a)] for a in angles])
    
    return positions, angles, toroidal_dirs, radial_dirs

def rigid_body_from_toroidal_displacements(toroidal_disps, positions, angles, toroidal_dirs):
    """
    Compute rigid body motion (dx, dy, dtheta) from toroidal displacements.
    Uses least squares since system is over-determined (9 equations, 3 unknowns).
    """
    N = len(positions)
    A = np.zeros((N, 3))
    for i in range(N):
        t = toroidal_dirs[i]
        x, y = positions[i]
        A[i, 0] = t[0]
        A[i, 1] = t[1]
        A[i, 2] = -y * t[0] + x * t[1]
    
    result, residuals, rank, s = np.linalg.lstsq(A, toroidal_disps, rcond=None)
    return result[0], result[1], result[2], residuals

def find_extreme_position(direction, positions, angles, toroidal_dirs, half_tol):
    """
    Find the extreme VV center position in a given direction.
    
    Since the rigid body motion is LINEAR in toroidal displacements,
    the extremes are achieved by setting each support to its limit
    in the direction that maximizes the objective.
    
    direction: 'x+', 'x-', 'y+', 'y-' or angle in radians
    """
    N = len(positions)
    
    if direction == 'x+':
        target = np.array([1, 0])
    elif direction == 'x-':
        target = np.array([-1, 0])
    elif direction == 'y+':
        target = np.array([0, 1])
    elif direction == 'y-':
        target = np.array([0, -1])
    else:
        target = np.array([np.cos(direction), np.sin(direction)])
    
    # Build constraint matrix: [dx, dy, dtheta] = pinv(A) @ toroidal_disps
    A = np.zeros((N, 3))
    for i in range(N):
        t = toroidal_dirs[i]
        x, y = positions[i]
        A[i, 0] = t[0]
        A[i, 1] = t[1]
        A[i, 2] = -y * t[0] + x * t[1]
    
    # Pseudo-inverse maps toroidal displacements to rigid body motion
    A_pinv = np.linalg.pinv(A)
    
    # We want to maximize target · [dx, dy] = target · A_pinv[:2, :] @ toroidal_disps
    # The gradient of this w.r.t. toroidal_disps is: A_pinv[:2, :].T @ target
    gradient = A_pinv[:2, :].T @ target
    
    # For linear objective with box constraints, optimal is at the boundary
    # Set each support to +half_tol if gradient > 0, else -half_tol
    optimal_toroidal = np.where(gradient > 0, half_tol, -half_tol)
    
    dx, dy, dtheta, residuals = rigid_body_from_toroidal_displacements(
        optimal_toroidal, positions, angles, toroidal_dirs
    )
    
    return dx, dy, dtheta, optimal_toroidal, residuals

def analyze_kinematic_blocking(positions, angles, toroidal_dirs):
    """
    Analyze the kinematic constraints of the support system.
    
    With 9 supports each allowing 1 DOF (toroidal), and a rigid body
    having 3 DOFs (dx, dy, dtheta), the system is over-constrained.
    
    This analyzes:
    1. The constraint matrix rank
    2. Which motions are blocked vs allowed
    3. The degree of redundancy
    """
    N = len(positions)
    
    # Build the constraint matrix A where: toroidal_disp = A @ [dx, dy, dtheta]
    A = np.zeros((N, 3))
    for i in range(N):
        t = toroidal_dirs[i]
        x, y = positions[i]
        A[i, 0] = t[0]
        A[i, 1] = t[1]
        A[i, 2] = -y * t[0] + x * t[1]
    
    # SVD analysis
    U, S, Vt = np.linalg.svd(A)
    
    # Rank of constraint matrix
    rank = np.sum(S > 1e-10)
    
    # Null space of A^T - directions in toroidal space that don't affect rigid body motion
    null_dim = N - rank
    
    print("\n" + "="*60)
    print("KINEMATIC CONSTRAINT ANALYSIS")
    print("="*60)
    print(f"\nNumber of supports: {N}")
    print(f"Rigid body DOFs: 3 (dx, dy, dθ)")
    print(f"Constraint equations: {N}")
    print(f"Constraint matrix rank: {rank}")
    print(f"Over-constraint (redundancy): {N - 3} equations")
    print(f"Null space dimension: {null_dim}")
    
    print(f"\nSingular values: {S}")
    
    # The columns of Vt are the principal directions in rigid-body space
    print("\nPrincipal constraint directions (in [dx, dy, dθ] space):")
    for i in range(3):
        print(f"  Mode {i+1}: [{Vt[i,0]:.4f}, {Vt[i,1]:.4f}, {Vt[i,2]:.4f}] "
              f"(strength: {S[i]:.4f})")
    
    # Check if any rigid body motion is blocked
    if rank == 3:
        print("\n✓ All 3 rigid body DOFs are constrained by toroidal limits")
        print("  (No pure translation or rotation is unconstrained)")
    else:
        print(f"\n⚠ Only {rank} of 3 rigid body DOFs are constrained!")
    
    return A, S, Vt, rank

def compute_bounding_box(positions, angles, toroidal_dirs, half_tol):
    """Compute the bounding box of achievable VV center positions."""
    
    print("\n" + "="*60)
    print("BOUNDING BOX ANALYSIS")
    print("="*60)
    
    # Find extremes in X and Y
    dx_max, dy_at_xmax, _, config_xmax, _ = find_extreme_position('x+', positions, angles, toroidal_dirs, half_tol)
    dx_min, dy_at_xmin, _, config_xmin, _ = find_extreme_position('x-', positions, angles, toroidal_dirs, half_tol)
    dx_at_ymax, dy_max, _, config_ymax, _ = find_extreme_position('y+', positions, angles, toroidal_dirs, half_tol)
    dx_at_ymin, dy_min, _, config_ymin, _ = find_extreme_position('y-', positions, angles, toroidal_dirs, half_tol)
    
    x_range = dx_max - dx_min
    y_range = dy_max - dy_min
    
    print(f"\nExtreme positions (mm):")
    print(f"  X: [{dx_min*1000:.4f}, {dx_max*1000:.4f}] → Range: {x_range*1000:.4f} mm")
    print(f"  Y: [{dy_min*1000:.4f}, {dy_max*1000:.4f}] → Range: {y_range*1000:.4f} mm")
    
    print(f"\nBounding box dimensions:")
    print(f"  Width (X):  {x_range*1000:.4f} mm")
    print(f"  Height (Y): {y_range*1000:.4f} mm")
    
    # Check asymmetry
    x_center = (dx_max + dx_min) / 2
    y_center = (dy_max + dy_min) / 2
    print(f"\nBounding box center offset from nominal:")
    print(f"  X offset: {x_center*1000:.6f} mm")
    print(f"  Y offset: {y_center*1000:.6f} mm")
    
    if abs(x_center) < 1e-9 and abs(y_center) < 1e-9:
        print("  → Symmetric about origin (as expected for symmetric supports)")
    
    return {
        'x_min': dx_min, 'x_max': dx_max,
        'y_min': dy_min, 'y_max': dy_max,
        'x_range': x_range, 'y_range': y_range,
        'configs': {
            'x_max': config_xmax, 'x_min': config_xmin,
            'y_max': config_ymax, 'y_min': config_ymin
        }
    }

def compute_directional_extremes(positions, angles, toroidal_dirs, half_tol, num_directions=72):
    """Compute extreme positions in many directions to map the reachable envelope."""
    
    directions = np.linspace(0, 2*np.pi, num_directions, endpoint=False)
    extremes = []
    
    for theta in directions:
        dx, dy, dtheta, _, _ = find_extreme_position(theta, positions, angles, toroidal_dirs, half_tol)
        extremes.append([dx, dy])
    
    return np.array(extremes), directions

def monte_carlo_envelope(positions, angles, toroidal_dirs, half_tol, num_samples=100000):
    """
    Monte Carlo sampling to show distribution within the envelope.
    Also computes the "shake range" - if starting from any point,
    how far can we move?
    """
    N = len(positions)
    
    results = {
        'dx': np.zeros(num_samples),
        'dy': np.zeros(num_samples),
        'dtheta': np.zeros(num_samples)
    }
    
    for i in range(num_samples):
        toroidal_disps = np.random.uniform(-half_tol, half_tol, N)
        dx, dy, dtheta, _ = rigid_body_from_toroidal_displacements(
            toroidal_disps, positions, angles, toroidal_dirs
        )
        results['dx'][i] = dx
        results['dy'][i] = dy
        results['dtheta'][i] = dtheta
    
    return results

def plot_bounding_box_analysis(bbox, envelope, mc_results, positions, toroidal_dirs, half_tol):
    """Create comprehensive bounding box visualization."""
    
    fig = plt.figure(figsize=(18, 12))
    
    # 1. Reachable envelope with Monte Carlo samples
    ax1 = fig.add_subplot(2, 3, 1)
    
    # Plot Monte Carlo samples
    ax1.scatter(mc_results['dx']*1000, mc_results['dy']*1000, 
                alpha=0.05, s=1, c='blue', rasterized=True)
    
    # Plot envelope boundary
    envelope_closed = np.vstack([envelope, envelope[0]])
    ax1.plot(envelope_closed[:, 0]*1000, envelope_closed[:, 1]*1000, 
             'r-', linewidth=2, label='Reachable Envelope')
    
    # Plot bounding box
    x_min, x_max = bbox['x_min']*1000, bbox['x_max']*1000
    y_min, y_max = bbox['y_min']*1000, bbox['y_max']*1000
    rect = plt.Rectangle((x_min, y_min), x_max-x_min, y_max-y_min,
                          fill=False, edgecolor='green', linewidth=2, 
                          linestyle='--', label='Bounding Box')
    ax1.add_patch(rect)
    
    ax1.axhline(0, color='gray', linewidth=0.5)
    ax1.axvline(0, color='gray', linewidth=0.5)
    ax1.set_xlabel('X Displacement (mm)')
    ax1.set_ylabel('Y Displacement (mm)')
    ax1.set_title('Reachable Position Envelope\n(Monte Carlo + Boundary)')
    ax1.legend(loc='upper right')
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    
    # 2. X displacement PDF with full range
    ax2 = fig.add_subplot(2, 3, 2)
    dx_mm = mc_results['dx'] * 1000
    ax2.hist(dx_mm, bins=100, density=True, alpha=0.7, color='steelblue', edgecolor='black')
    
    # Mark extremes
    ax2.axvline(x_min, color='red', linestyle='--', linewidth=2, label=f'Min: {x_min:.3f}')
    ax2.axvline(x_max, color='red', linestyle='--', linewidth=2, label=f'Max: {x_max:.3f}')
    ax2.axvline(0, color='black', linestyle='-', alpha=0.5)
    
    ax2.set_xlabel('X Displacement (mm)')
    ax2.set_ylabel('Probability Density')
    ax2.set_title(f'X-Direction PDF\nTotal Range: {bbox["x_range"]*1000:.3f} mm')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Y displacement PDF with full range
    ax3 = fig.add_subplot(2, 3, 3)
    dy_mm = mc_results['dy'] * 1000
    ax3.hist(dy_mm, bins=100, density=True, alpha=0.7, color='steelblue', edgecolor='black')
    
    ax3.axvline(y_min, color='red', linestyle='--', linewidth=2, label=f'Min: {y_min:.3f}')
    ax3.axvline(y_max, color='red', linestyle='--', linewidth=2, label=f'Max: {y_max:.3f}')
    ax3.axvline(0, color='black', linestyle='-', alpha=0.5)
    
    ax3.set_xlabel('Y Displacement (mm)')
    ax3.set_ylabel('Probability Density')
    ax3.set_title(f'Y-Direction PDF\nTotal Range: {bbox["y_range"]*1000:.3f} mm')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Geometry with extreme configs
    ax4 = fig.add_subplot(2, 3, 4)
    
    # Draw vessel
    theta = np.linspace(0, 2*np.pi, 100)
    ax4.plot(VESSEL_RADIUS * np.cos(theta), VESSEL_RADIUS * np.sin(theta), 
             'b-', linewidth=2)
    ax4.plot(0, 0, 'ko', markersize=8)
    
    # Draw supports with tolerance bands
    for i, (pos, tdir) in enumerate(zip(positions, toroidal_dirs)):
        ax4.plot([0, pos[0]], [0, pos[1]], 'gray', linewidth=1, alpha=0.5)
        # Tolerance band
        p1 = pos + tdir * half_tol
        p2 = pos - tdir * half_tol
        ax4.plot([p1[0], p2[0]], [p1[1], p2[1]], 'r-', linewidth=3)
        ax4.plot(pos[0], pos[1], 'rs', markersize=8)
    
    ax4.set_xlim(-12, 12)
    ax4.set_ylim(-12, 12)
    ax4.set_aspect('equal')
    ax4.set_xlabel('X (m)')
    ax4.set_ylabel('Y (m)')
    ax4.set_title('Support Geometry with Toroidal Tolerance Bands\n(Red bars = ±1.5mm toroidal play)')
    ax4.grid(True, alpha=0.3)
    
    # 5. Rotation angle
    ax5 = fig.add_subplot(2, 3, 5)
    dtheta_mrad = mc_results['dtheta'] * 1000
    ax5.hist(dtheta_mrad, bins=100, density=True, alpha=0.7, color='steelblue', edgecolor='black')
    
    theta_min, theta_max = dtheta_mrad.min(), dtheta_mrad.max()
    ax5.axvline(theta_min, color='red', linestyle='--', linewidth=2)
    ax5.axvline(theta_max, color='red', linestyle='--', linewidth=2)
    
    ax5.set_xlabel('Rotation Angle (mrad)')
    ax5.set_ylabel('Probability Density')
    ax5.set_title(f'Rotation PDF\nRange: [{theta_min:.4f}, {theta_max:.4f}] mrad')
    ax5.grid(True, alpha=0.3)
    
    # 6. Summary statistics
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')
    
    summary_text = f"""
    BOUNDING BOX ANALYSIS SUMMARY
    ════════════════════════════════════════════
    
    Input Parameters:
    ─────────────────
    • Vessel Radius: {VESSEL_RADIUS} m
    • Number of Supports: {NUM_SUPPORTS}
    • Toroidal Tolerance: ±{half_tol*1000:.1f} mm per support
    • Radial: Unconstrained (thermal expansion)
    
    BOUNDING BOX (Total Shake Range):
    ─────────────────────────────────
    X-Direction:
      • Min: {bbox['x_min']*1000:+.4f} mm
      • Max: {bbox['x_max']*1000:+.4f} mm
      • Total Range: {bbox['x_range']*1000:.4f} mm
    
    Y-Direction:
      • Min: {bbox['y_min']*1000:+.4f} mm
      • Max: {bbox['y_max']*1000:+.4f} mm
      • Total Range: {bbox['y_range']*1000:.4f} mm
    
    Rotation:
      • Range: [{dtheta_mrad.min():.4f}, {dtheta_mrad.max():.4f}] mrad
      • Total: {dtheta_mrad.max() - dtheta_mrad.min():.4f} mrad
    
    Monte Carlo Statistics (100k samples):
    ──────────────────────────────────────
    X: mean={np.mean(dx_mm):.4f}mm, std={np.std(dx_mm):.4f}mm
    Y: mean={np.mean(dy_mm):.4f}mm, std={np.std(dy_mm):.4f}mm
    
    INTERPRETATION:
    ───────────────
    Once installed, the VV can "shake" by up to:
    • {bbox['x_range']*1000:.2f}mm in X (total range)
    • {bbox['y_range']*1000:.2f}mm in Y (total range)
    
    This is NOT symmetric displacement from center,
    but the total possible excursion range.
    """
    
    ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, fontsize=9,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    
    plt.suptitle('ITER Vacuum Vessel - Bounding Box Analysis\n(Total Range of Motion from Support Tolerances)', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('plots/bounding_box_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\nSaved: plots/bounding_box_analysis.png")

def plot_kinematic_analysis(A, S, Vt, positions, toroidal_dirs):
    """Visualize kinematic constraint analysis."""
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 1. Constraint matrix heatmap
    ax = axes[0]
    im = ax.imshow(A, aspect='auto', cmap='RdBu', vmin=-1, vmax=1)
    ax.set_xlabel('Rigid Body DOF')
    ax.set_ylabel('Support Index')
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(['dx', 'dy', 'dθ'])
    ax.set_yticks(range(9))
    ax.set_yticklabels([f'S{i+1}' for i in range(9)])
    ax.set_title('Constraint Matrix A\n(toroidal_disp = A × [dx,dy,dθ])')
    plt.colorbar(im, ax=ax)
    
    # 2. Singular values
    ax = axes[1]
    ax.bar(range(3), S, color='steelblue', edgecolor='black')
    ax.set_xlabel('Mode Index')
    ax.set_ylabel('Singular Value')
    ax.set_title('Singular Values of Constraint Matrix\n(All > 0 means all DOFs constrained)')
    ax.set_xticks([0, 1, 2])
    ax.grid(True, alpha=0.3, axis='y')
    
    # 3. Principal constraint directions
    ax = axes[2]
    colors = ['red', 'green', 'blue']
    labels = ['Mode 1 (strongest)', 'Mode 2', 'Mode 3 (weakest)']
    
    # Plot as vectors in dx-dy plane (ignoring dtheta for viz)
    for i in range(3):
        v = Vt[i, :2]  # Just dx, dy components
        v = v / np.linalg.norm(v) * S[i] / S[0]  # Scale by relative strength
        ax.arrow(0, 0, v[0], v[1], head_width=0.05, head_length=0.02, 
                 fc=colors[i], ec=colors[i], linewidth=2, label=labels[i])
    
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_aspect('equal')
    ax.axhline(0, color='gray', linewidth=0.5)
    ax.axvline(0, color='gray', linewidth=0.5)
    ax.set_xlabel('dx component')
    ax.set_ylabel('dy component')
    ax.set_title('Principal Constraint Directions\n(in translation plane)')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('Kinematic Constraint Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('plots/kinematic_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: plots/kinematic_analysis.png")

def plot_shake_envelope_detail(envelope, directions, bbox):
    """Detailed plot of the shake envelope."""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 1. Envelope with direction indicators
    ax = axes[0]
    
    # Plot envelope
    envelope_mm = envelope * 1000
    envelope_closed = np.vstack([envelope_mm, envelope_mm[0]])
    ax.fill(envelope_closed[:, 0], envelope_closed[:, 1], alpha=0.3, color='blue')
    ax.plot(envelope_closed[:, 0], envelope_closed[:, 1], 'b-', linewidth=2)
    
    # Plot axes to extremes
    ax.plot([bbox['x_min']*1000, bbox['x_max']*1000], [0, 0], 'r-', linewidth=2, label='X range')
    ax.plot([0, 0], [bbox['y_min']*1000, bbox['y_max']*1000], 'g-', linewidth=2, label='Y range')
    
    # Mark extreme points
    ax.plot(bbox['x_max']*1000, 0, 'r>', markersize=10)
    ax.plot(bbox['x_min']*1000, 0, 'r<', markersize=10)
    ax.plot(0, bbox['y_max']*1000, 'g^', markersize=10)
    ax.plot(0, bbox['y_min']*1000, 'gv', markersize=10)
    ax.plot(0, 0, 'ko', markersize=8, label='Nominal position')
    
    ax.set_xlabel('X Displacement (mm)')
    ax.set_ylabel('Y Displacement (mm)')
    ax.set_title('Shake Envelope\n(All achievable center positions)')
    ax.legend(loc='upper right')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # 2. Radial reach vs direction
    ax = axes[1]
    
    radial_reach = np.sqrt(envelope[:, 0]**2 + envelope[:, 1]**2) * 1000
    ax.plot(np.degrees(directions), radial_reach, 'b-', linewidth=2)
    ax.fill_between(np.degrees(directions), 0, radial_reach, alpha=0.3)
    
    ax.set_xlabel('Direction (degrees from +X)')
    ax.set_ylabel('Maximum Reach (mm)')
    ax.set_title('Directional Reach\n(How far VV can move in each direction)')
    ax.set_xlim(0, 360)
    ax.set_xticks(np.arange(0, 361, 45))
    ax.grid(True, alpha=0.3)
    
    # Annotate min/max reach
    max_reach = radial_reach.max()
    min_reach = radial_reach.min()
    max_dir = directions[np.argmax(radial_reach)]
    min_dir = directions[np.argmin(radial_reach)]
    
    ax.axhline(max_reach, color='red', linestyle='--', alpha=0.7, label=f'Max: {max_reach:.3f}mm @ {np.degrees(max_dir):.0f}°')
    ax.axhline(min_reach, color='green', linestyle='--', alpha=0.7, label=f'Min: {min_reach:.3f}mm @ {np.degrees(min_dir):.0f}°')
    ax.legend(loc='upper right')
    
    plt.suptitle('ITER Vacuum Vessel - Shake Envelope Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('plots/shake_envelope.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: plots/shake_envelope.png")

def main():
    print("="*60)
    print("ITER Vacuum Vessel - Bounding Box & Kinematic Analysis")
    print("="*60)
    
    # Get geometry
    positions, angles, toroidal_dirs, radial_dirs = get_support_geometry(VESSEL_RADIUS, NUM_SUPPORTS)
    half_tol = TOROIDAL_TOLERANCE / 2
    
    # Kinematic analysis
    A, S, Vt, rank = analyze_kinematic_blocking(positions, angles, toroidal_dirs)
    
    # Bounding box analysis
    bbox = compute_bounding_box(positions, angles, toroidal_dirs, half_tol)
    
    # Compute detailed envelope
    print("\nComputing reachable envelope (72 directions)...")
    envelope, directions = compute_directional_extremes(positions, angles, toroidal_dirs, half_tol)
    
    # Monte Carlo for distribution
    print("Running Monte Carlo for distribution (100k samples)...")
    mc_results = monte_carlo_envelope(positions, angles, toroidal_dirs, half_tol, 100000)
    
    # Generate plots
    print("\nGenerating plots...")
    plot_kinematic_analysis(A, S, Vt, positions, toroidal_dirs)
    plot_bounding_box_analysis(bbox, envelope, mc_results, positions, toroidal_dirs, half_tol)
    plot_shake_envelope_detail(envelope, directions, bbox)
    
    # Final summary
    print("\n" + "="*60)
    print("FINAL SUMMARY - SHAKE RANGE")
    print("="*60)
    print(f"""
Once installed, the ITER vacuum vessel can potentially move:

  X-Direction: {bbox['x_range']*1000:.3f} mm total range
               (from {bbox['x_min']*1000:+.3f} to {bbox['x_max']*1000:+.3f} mm)
  
  Y-Direction: {bbox['y_range']*1000:.3f} mm total range
               (from {bbox['y_min']*1000:+.3f} to {bbox['y_max']*1000:+.3f} mm)

KINEMATIC BLOCKING:
  The 9-support geometry with toroidal-only DOFs provides
  {"COMPLETE" if rank == 3 else "PARTIAL"} kinematic constraint on rigid body motion.
  Rank of constraint matrix: {rank}/3
  
  This means all translational and rotational DOFs are constrained
  by the toroidal tolerance limits - there is no "free" direction.
  
  However, the constraints are NOT identical in all directions.
  The envelope is roughly circular but not perfectly so.
""")
    
    print("Plots saved to plots/ directory:")
    print("  - kinematic_analysis.png")
    print("  - bounding_box_analysis.png")
    print("  - shake_envelope.png")
    print("="*60)

if __name__ == "__main__":
    main()
