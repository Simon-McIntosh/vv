"""
ITER Vacuum Vessel Lateral Movement Monte Carlo Simulation

This simulation assesses the lateral movement potential of the ITER vacuum vessel
based on the tolerances in its 9 gravity supports:
- Unconstrained radial movement (for thermal expansion)
- 3mm allowable movement in toroidal direction (assembly tolerance)

The vacuum vessel is treated as a rigid body.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import minimize
import os

# Create output directory for plots
os.makedirs("plots", exist_ok=True)

# Physical parameters
VESSEL_RADIUS = 8.0  # meters (approximate major radius for support locations)
NUM_SUPPORTS = 9  # 9 gravity supports evenly distributed
TOROIDAL_TOLERANCE = 3e-3  # 3mm in meters

# Monte Carlo parameters
NUM_SIMULATIONS = 100000

def get_support_positions(radius, num_supports):
    """
    Get the nominal positions of gravity supports.
    Supports are evenly distributed around the circle.
    """
    angles = np.linspace(0, 2*np.pi, num_supports, endpoint=False)
    # Start from top (90 degrees) and go clockwise to match typical diagrams
    angles = np.pi/2 - angles
    positions = np.array([[radius * np.cos(a), radius * np.sin(a)] for a in angles])
    return positions, angles

def compute_vessel_displacement_rigid_body(support_toroidal_displacements, support_positions, support_angles):
    """
    For a rigid vacuum vessel, compute the center displacement given
    toroidal displacements at each support.
    
    The vessel can translate (dx, dy) and rotate (dtheta) as a rigid body.
    We find the best-fit rigid body motion that matches the toroidal displacements.
    
    Parameters:
    - support_toroidal_displacements: array of toroidal displacements at each support (m)
    - support_positions: (N, 2) array of support positions
    - support_angles: angles of supports from center
    
    Returns:
    - dx, dy: center displacement
    - dtheta: rotation angle
    """
    N = len(support_positions)
    
    # For each support, the toroidal direction is perpendicular to radial
    # Toroidal unit vectors (perpendicular to radial, counter-clockwise positive)
    toroidal_dirs = np.array([[-np.sin(a), np.cos(a)] for a in support_angles])
    
    # For a rigid body motion (dx, dy, dtheta), the displacement at support i is:
    # d_i = [dx, dy] + dtheta * [-y_i, x_i]  (rotation about center)
    # The toroidal component is: d_i · t_i
    
    # Set up least squares: find (dx, dy, dtheta) that best matches toroidal displacements
    # toroidal_disp[i] = dx*t_ix + dy*t_iy + dtheta*(-y_i*t_ix + x_i*t_iy)
    
    A = np.zeros((N, 3))
    for i in range(N):
        t = toroidal_dirs[i]
        x, y = support_positions[i]
        A[i, 0] = t[0]  # dx coefficient
        A[i, 1] = t[1]  # dy coefficient
        A[i, 2] = -y * t[0] + x * t[1]  # dtheta coefficient
    
    b = support_toroidal_displacements
    
    # Solve least squares
    result, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
    dx, dy, dtheta = result
    
    return dx, dy, dtheta

def run_monte_carlo_simulation(num_simulations, vessel_radius, num_supports, toroidal_tolerance):
    """
    Run Monte Carlo simulation for vacuum vessel lateral movement.
    
    Each support has a random toroidal displacement uniformly distributed
    within the tolerance band.
    """
    print(f"Running {num_simulations:,} Monte Carlo simulations...")
    
    # Get support geometry
    support_positions, support_angles = get_support_positions(vessel_radius, num_supports)
    
    # Storage for results
    center_dx = np.zeros(num_simulations)
    center_dy = np.zeros(num_simulations)
    center_dtheta = np.zeros(num_simulations)
    
    # Half tolerance (±1.5mm if 3mm total)
    half_tol = toroidal_tolerance / 2
    
    for i in range(num_simulations):
        # Random toroidal displacements at each support (uniform distribution)
        toroidal_disps = np.random.uniform(-half_tol, half_tol, num_supports)
        
        # Compute rigid body motion
        dx, dy, dtheta = compute_vessel_displacement_rigid_body(
            toroidal_disps, support_positions, support_angles
        )
        
        center_dx[i] = dx
        center_dy[i] = dy
        center_dtheta[i] = dtheta
    
    # Compute radial displacement magnitude
    center_displacement = np.sqrt(center_dx**2 + center_dy**2)
    
    print(f"Simulation complete.")
    
    return {
        'dx': center_dx,
        'dy': center_dy,
        'dtheta': center_dtheta,
        'displacement': center_displacement,
        'support_positions': support_positions,
        'support_angles': support_angles
    }

def plot_geometry(support_positions, vessel_radius):
    """Plot the vacuum vessel and support geometry."""
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Draw vacuum vessel circle
    theta = np.linspace(0, 2*np.pi, 100)
    ax.plot(vessel_radius * np.cos(theta), vessel_radius * np.sin(theta), 
            'b-', linewidth=2, label='Vacuum Vessel')
    
    # Draw center
    ax.plot(0, 0, 'ko', markersize=8, label='Center')
    
    # Draw supports
    for i, (x, y) in enumerate(support_positions):
        # Line from center to support
        ax.plot([0, x], [0, y], 'r-', linewidth=1.5)
        # Support marker
        ax.plot(x, y, 'rs', markersize=12)
        ax.annotate(f'S{i+1}', (x*1.1, y*1.1), fontsize=10, ha='center', va='center')
    
    ax.set_xlim(-12, 12)
    ax.set_ylim(-12, 12)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_title('ITER Vacuum Vessel - Gravity Support Geometry\n(9 Supports, 8m Radius)', fontsize=14)
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig('plots/geometry.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: plots/geometry.png")

def plot_displacement_pdf(results, toroidal_tolerance):
    """Plot PDF of center displacement magnitude."""
    displacement_mm = results['displacement'] * 1000  # Convert to mm
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Histogram with density
    counts, bins, patches = ax.hist(displacement_mm, bins=100, density=True, 
                                     alpha=0.7, color='steelblue', edgecolor='black')
    
    # Fit and plot kernel density estimate
    kde = stats.gaussian_kde(displacement_mm)
    x_range = np.linspace(0, displacement_mm.max() * 1.1, 200)
    ax.plot(x_range, kde(x_range), 'r-', linewidth=2, label='KDE Fit')
    
    # Statistics
    mean_disp = np.mean(displacement_mm)
    std_disp = np.std(displacement_mm)
    p95 = np.percentile(displacement_mm, 95)
    p99 = np.percentile(displacement_mm, 99)
    max_disp = np.max(displacement_mm)
    
    # Add vertical lines for statistics
    ax.axvline(mean_disp, color='green', linestyle='--', linewidth=2, label=f'Mean: {mean_disp:.3f} mm')
    ax.axvline(p95, color='orange', linestyle='--', linewidth=2, label=f'95th %ile: {p95:.3f} mm')
    ax.axvline(p99, color='red', linestyle='--', linewidth=2, label=f'99th %ile: {p99:.3f} mm')
    
    ax.set_xlabel('Center Displacement Magnitude (mm)', fontsize=12)
    ax.set_ylabel('Probability Density', fontsize=12)
    ax.set_title(f'ITER Vacuum Vessel - Lateral Movement PDF\n(Toroidal Tolerance: {toroidal_tolerance*1000:.1f} mm)', fontsize=14)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Add text box with statistics
    textstr = f'Statistics:\n' \
              f'Mean: {mean_disp:.4f} mm\n' \
              f'Std Dev: {std_disp:.4f} mm\n' \
              f'95th %ile: {p95:.4f} mm\n' \
              f'99th %ile: {p99:.4f} mm\n' \
              f'Maximum: {max_disp:.4f} mm'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.98, 0.55, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right', bbox=props)
    
    plt.tight_layout()
    plt.savefig('plots/displacement_pdf.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: plots/displacement_pdf.png")
    
    return {'mean': mean_disp, 'std': std_disp, 'p95': p95, 'p99': p99, 'max': max_disp}

def plot_xy_displacement_pdf(results):
    """Plot 2D PDF of center displacement (dx, dy)."""
    dx_mm = results['dx'] * 1000
    dy_mm = results['dy'] * 1000
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 2D histogram / heatmap
    ax = axes[0]
    h = ax.hist2d(dx_mm, dy_mm, bins=50, cmap='hot', density=True)
    plt.colorbar(h[3], ax=ax, label='Probability Density')
    ax.set_xlabel('X Displacement (mm)', fontsize=11)
    ax.set_ylabel('Y Displacement (mm)', fontsize=11)
    ax.set_title('2D Displacement Distribution', fontsize=12)
    ax.set_aspect('equal')
    ax.axhline(0, color='white', linewidth=0.5, alpha=0.5)
    ax.axvline(0, color='white', linewidth=0.5, alpha=0.5)
    
    # X displacement PDF
    ax = axes[1]
    ax.hist(dx_mm, bins=80, density=True, alpha=0.7, color='steelblue', edgecolor='black')
    kde_x = stats.gaussian_kde(dx_mm)
    x_range = np.linspace(dx_mm.min(), dx_mm.max(), 200)
    ax.plot(x_range, kde_x(x_range), 'r-', linewidth=2)
    ax.axvline(0, color='black', linestyle='--', alpha=0.5)
    ax.set_xlabel('X Displacement (mm)', fontsize=11)
    ax.set_ylabel('Probability Density', fontsize=11)
    ax.set_title(f'X Displacement PDF\nMean: {np.mean(dx_mm):.4f} mm, Std: {np.std(dx_mm):.4f} mm', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Y displacement PDF
    ax = axes[2]
    ax.hist(dy_mm, bins=80, density=True, alpha=0.7, color='steelblue', edgecolor='black')
    kde_y = stats.gaussian_kde(dy_mm)
    y_range = np.linspace(dy_mm.min(), dy_mm.max(), 200)
    ax.plot(y_range, kde_y(y_range), 'r-', linewidth=2)
    ax.axvline(0, color='black', linestyle='--', alpha=0.5)
    ax.set_xlabel('Y Displacement (mm)', fontsize=11)
    ax.set_ylabel('Probability Density', fontsize=11)
    ax.set_title(f'Y Displacement PDF\nMean: {np.mean(dy_mm):.4f} mm, Std: {np.std(dy_mm):.4f} mm', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('ITER Vacuum Vessel - Directional Displacement Analysis', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig('plots/xy_displacement_pdf.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: plots/xy_displacement_pdf.png")

def plot_rotation_pdf(results):
    """Plot PDF of rotation angle."""
    dtheta_mrad = results['dtheta'] * 1000  # Convert to milliradians
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.hist(dtheta_mrad, bins=100, density=True, alpha=0.7, color='steelblue', edgecolor='black')
    
    kde = stats.gaussian_kde(dtheta_mrad)
    x_range = np.linspace(dtheta_mrad.min(), dtheta_mrad.max(), 200)
    ax.plot(x_range, kde(x_range), 'r-', linewidth=2, label='KDE Fit')
    
    mean_rot = np.mean(dtheta_mrad)
    std_rot = np.std(dtheta_mrad)
    
    ax.axvline(mean_rot, color='green', linestyle='--', linewidth=2, label=f'Mean: {mean_rot:.4f} mrad')
    ax.axvline(0, color='black', linestyle='-', alpha=0.3)
    
    ax.set_xlabel('Rotation Angle (mrad)', fontsize=12)
    ax.set_ylabel('Probability Density', fontsize=12)
    ax.set_title('ITER Vacuum Vessel - Rotation Angle PDF', fontsize=14)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    textstr = f'Statistics:\n' \
              f'Mean: {mean_rot:.5f} mrad\n' \
              f'Std Dev: {std_rot:.5f} mrad\n' \
              f'Max: {np.max(np.abs(dtheta_mrad)):.5f} mrad'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.98, 0.95, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right', bbox=props)
    
    plt.tight_layout()
    plt.savefig('plots/rotation_pdf.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: plots/rotation_pdf.png")

def plot_polar_displacement(results):
    """Plot displacement in polar coordinates."""
    displacement_mm = results['displacement'] * 1000
    dx_mm = results['dx'] * 1000
    dy_mm = results['dy'] * 1000
    
    # Calculate angles of displacement
    angles = np.arctan2(dy_mm, dx_mm)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Polar histogram
    ax = plt.subplot(121, projection='polar')
    ax.hist(angles, bins=36, density=True, alpha=0.7, color='steelblue')
    ax.set_title('Angular Distribution of\nDisplacement Direction', fontsize=12, pad=20)
    
    # Scatter plot in Cartesian
    ax = axes[1]
    scatter = ax.scatter(dx_mm, dy_mm, c=displacement_mm, cmap='viridis', 
                         alpha=0.1, s=1, rasterized=True)
    plt.colorbar(scatter, ax=ax, label='Displacement Magnitude (mm)')
    
    # Add circles for reference
    for r in [0.2, 0.4, 0.6, 0.8, 1.0]:
        circle = plt.Circle((0, 0), r, fill=False, color='gray', linestyle='--', alpha=0.5)
        ax.add_patch(circle)
        ax.annotate(f'{r:.1f}mm', (r/np.sqrt(2), r/np.sqrt(2)), fontsize=8, alpha=0.7)
    
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_aspect('equal')
    ax.set_xlabel('X Displacement (mm)', fontsize=11)
    ax.set_ylabel('Y Displacement (mm)', fontsize=11)
    ax.set_title('Displacement Scatter Plot', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.axvline(0, color='black', linewidth=0.5)
    
    plt.suptitle('ITER Vacuum Vessel - Displacement Distribution Analysis', fontsize=14)
    plt.tight_layout()
    plt.savefig('plots/polar_displacement.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: plots/polar_displacement.png")

def plot_summary(results, stats_dict, toroidal_tolerance, vessel_radius, num_supports):
    """Create a summary plot with all key information."""
    fig = plt.figure(figsize=(16, 12))
    
    # Geometry
    ax1 = fig.add_subplot(2, 3, 1)
    support_positions = results['support_positions']
    theta = np.linspace(0, 2*np.pi, 100)
    ax1.plot(vessel_radius * np.cos(theta), vessel_radius * np.sin(theta), 'b-', linewidth=2)
    ax1.plot(0, 0, 'ko', markersize=6)
    for i, (x, y) in enumerate(support_positions):
        ax1.plot([0, x], [0, y], 'r-', linewidth=1.5)
        ax1.plot(x, y, 'rs', markersize=8)
    ax1.set_xlim(-12, 12)
    ax1.set_ylim(-12, 12)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Support Geometry', fontsize=11)
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    
    # Displacement PDF
    ax2 = fig.add_subplot(2, 3, 2)
    displacement_mm = results['displacement'] * 1000
    ax2.hist(displacement_mm, bins=80, density=True, alpha=0.7, color='steelblue', edgecolor='black')
    kde = stats.gaussian_kde(displacement_mm)
    x_range = np.linspace(0, displacement_mm.max() * 1.1, 200)
    ax2.plot(x_range, kde(x_range), 'r-', linewidth=2)
    ax2.axvline(stats_dict['p95'], color='orange', linestyle='--', linewidth=2, label=f"95%: {stats_dict['p95']:.3f}mm")
    ax2.set_xlabel('Displacement (mm)')
    ax2.set_ylabel('PDF')
    ax2.set_title('Displacement Magnitude PDF', fontsize=11)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 2D scatter
    ax3 = fig.add_subplot(2, 3, 3)
    dx_mm = results['dx'] * 1000
    dy_mm = results['dy'] * 1000
    ax3.hist2d(dx_mm, dy_mm, bins=50, cmap='hot', density=True)
    ax3.set_xlabel('X Displacement (mm)')
    ax3.set_ylabel('Y Displacement (mm)')
    ax3.set_title('2D Displacement Heatmap', fontsize=11)
    ax3.set_aspect('equal')
    
    # CDF
    ax4 = fig.add_subplot(2, 3, 4)
    sorted_disp = np.sort(displacement_mm)
    cdf = np.arange(1, len(sorted_disp) + 1) / len(sorted_disp)
    ax4.plot(sorted_disp, cdf * 100, 'b-', linewidth=2)
    ax4.axhline(95, color='orange', linestyle='--', alpha=0.7, label='95%')
    ax4.axhline(99, color='red', linestyle='--', alpha=0.7, label='99%')
    ax4.axvline(stats_dict['p95'], color='orange', linestyle='--', alpha=0.7)
    ax4.axvline(stats_dict['p99'], color='red', linestyle='--', alpha=0.7)
    ax4.set_xlabel('Displacement (mm)')
    ax4.set_ylabel('Cumulative Probability (%)')
    ax4.set_title('Cumulative Distribution Function', fontsize=11)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(0, None)
    ax4.set_ylim(0, 100)
    
    # X-Y marginal PDFs
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.hist(dx_mm, bins=60, density=True, alpha=0.6, color='blue', label='X', edgecolor='black')
    ax5.hist(dy_mm, bins=60, density=True, alpha=0.6, color='red', label='Y', edgecolor='black')
    ax5.set_xlabel('Displacement (mm)')
    ax5.set_ylabel('PDF')
    ax5.set_title('X and Y Marginal PDFs', fontsize=11)
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Statistics text
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')
    
    stats_text = f"""
    ITER Vacuum Vessel Lateral Movement
    Monte Carlo Simulation Results
    ═══════════════════════════════════════
    
    Input Parameters:
    ─────────────────
    • Vessel Radius: {vessel_radius} m
    • Number of Supports: {num_supports}
    • Toroidal Tolerance: {toroidal_tolerance*1000:.1f} mm
    • Radial Constraint: Unconstrained
    • Number of Simulations: {NUM_SIMULATIONS:,}
    
    Results (Displacement Magnitude):
    ──────────────────────────────────
    • Mean: {stats_dict['mean']:.4f} mm
    • Standard Deviation: {stats_dict['std']:.4f} mm
    • 95th Percentile: {stats_dict['p95']:.4f} mm
    • 99th Percentile: {stats_dict['p99']:.4f} mm
    • Maximum: {stats_dict['max']:.4f} mm
    
    X-Direction:
    • Mean: {np.mean(dx_mm):.4f} mm
    • Std Dev: {np.std(dx_mm):.4f} mm
    
    Y-Direction:
    • Mean: {np.mean(dy_mm):.4f} mm
    • Std Dev: {np.std(dy_mm):.4f} mm
    """
    ax6.text(0.1, 0.95, stats_text, transform=ax6.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.suptitle('ITER Vacuum Vessel - Lateral Movement Assessment Summary', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('plots/summary.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: plots/summary.png")

def main():
    print("=" * 60)
    print("ITER Vacuum Vessel Lateral Movement Monte Carlo Simulation")
    print("=" * 60)
    print(f"\nParameters:")
    print(f"  - Vessel Radius: {VESSEL_RADIUS} m")
    print(f"  - Number of Supports: {NUM_SUPPORTS}")
    print(f"  - Toroidal Tolerance: {TOROIDAL_TOLERANCE*1000} mm")
    print(f"  - Radial Constraint: Unconstrained")
    print(f"  - Number of Simulations: {NUM_SIMULATIONS:,}")
    print()
    
    # Get support positions for geometry plot
    support_positions, support_angles = get_support_positions(VESSEL_RADIUS, NUM_SUPPORTS)
    
    # Plot geometry
    print("Generating geometry plot...")
    plot_geometry(support_positions, VESSEL_RADIUS)
    
    # Run Monte Carlo simulation
    results = run_monte_carlo_simulation(
        NUM_SIMULATIONS, VESSEL_RADIUS, NUM_SUPPORTS, TOROIDAL_TOLERANCE
    )
    
    # Generate plots
    print("\nGenerating PDF plots...")
    stats_dict = plot_displacement_pdf(results, TOROIDAL_TOLERANCE)
    plot_xy_displacement_pdf(results)
    plot_rotation_pdf(results)
    plot_polar_displacement(results)
    plot_summary(results, stats_dict, TOROIDAL_TOLERANCE, VESSEL_RADIUS, NUM_SUPPORTS)
    
    # Print summary statistics
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"\nCenter Displacement Statistics:")
    print(f"  Mean:           {stats_dict['mean']:.4f} mm")
    print(f"  Std Dev:        {stats_dict['std']:.4f} mm")
    print(f"  95th Percentile: {stats_dict['p95']:.4f} mm")
    print(f"  99th Percentile: {stats_dict['p99']:.4f} mm")
    print(f"  Maximum:        {stats_dict['max']:.4f} mm")
    
    print(f"\nX-Direction Displacement:")
    print(f"  Mean: {np.mean(results['dx'])*1000:.4f} mm")
    print(f"  Std:  {np.std(results['dx'])*1000:.4f} mm")
    
    print(f"\nY-Direction Displacement:")
    print(f"  Mean: {np.mean(results['dy'])*1000:.4f} mm")
    print(f"  Std:  {np.std(results['dy'])*1000:.4f} mm")
    
    print(f"\nRotation Angle:")
    print(f"  Mean: {np.mean(results['dtheta'])*1000:.5f} mrad")
    print(f"  Std:  {np.std(results['dtheta'])*1000:.5f} mrad")
    
    print("\n" + "=" * 60)
    print("All plots saved to 'plots/' directory")
    print("=" * 60)

if __name__ == "__main__":
    main()
