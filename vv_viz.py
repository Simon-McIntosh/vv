"""
VV lateral rattle — clean scriptable visualisation.

Physical mechanism
------------------
Each of the 9 gravity supports is a four-bar linkage that allows large radial
displacements of the VV (thermal expansion, assembly movement).  At the top of
the four-bar a pin is mounted with its axis aligned in the toroidal direction.
The VV bracket slides on this toroidal pin and is allowed ±1.5 mm of travel
about the nominal (assembly) position.  This ±1.5 mm compliance is the source
of the lateral "rattle" studied here.

Constraint model: the toroidal slide at support i when the VV moves by
[Δx, Δy, Δθ] is   δᵢ = −sin(φᵢ)·Δx + cos(φᵢ)·Δy + R·Δθ   (=  A[i,:]·q).
The bracket must satisfy  |u_assembly,i + δᵢ| ≤ 1.5 mm  at every support.

Diagram conventions
-------------------
  ·  Toroidal travel range :  light grey track, ±1.5 mm × MAG wide
  ·  Travel limit stops    :  thick red "|" bars (fixed to support structure)
  ·  VV bracket dot        :  blue→amber→red as bracket approaches its stop
  ·  Radial spokes         :  thin blue lines from VV centre to each support
                              attachment; rotate with VV, making Δθ visible.

Usage
-----
    python vv_viz.py                  # generate all outputs to docs/
    python vv_viz.py --no-gif         # skip (slow) GIF generation
"""

from __future__ import annotations
import argparse, io, base64, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D
from scipy.optimize import linprog

# ── geometry ──────────────────────────────────────────────────────────────────
MAG     = 500       # displacement magnification for display
N       = 9         # number of supports
R_M     = 8.0       # VV radius (m)
R_SLOT  = 8.5       # slot display radius — slightly outside VV (m)
DELTA_M = 0.0015    # half-gap (m) = 1.5 mm

ANGLES = np.array([np.pi / 2 - 2 * np.pi * i / N for i in range(N)])

# Constraint matrix A (9×3):  A @ [dx_m, dy_m, dθ_rad]  =  toroidal displacement
A = np.column_stack([-np.sin(ANGLES), np.cos(ANGLES), R_M * np.ones(N)])


# ── LP solver ─────────────────────────────────────────────────────────────────

def lp_rattle(u_m: np.ndarray, theta: float) -> tuple[float, np.ndarray]:
    """Max displacement in direction theta (rad) from assembly state u_m."""
    c   = -np.array([np.cos(theta), np.sin(theta), 0.0])
    Au  = np.vstack([A, -A])
    bu  = np.concatenate([DELTA_M - u_m, DELTA_M + u_m])
    res = linprog(c, A_ub=Au, b_ub=bu, bounds=[(None, None)] * 3, method="highs")
    if not res.success:
        return 0.0, np.zeros(3)
    return float(-res.fun), res.x


def max_rattle_mm(u_m: np.ndarray, nd: int = 36) -> float:
    """Maximum rattle range (mm) over all directions."""
    best = 0.0
    for th in np.linspace(0, np.pi, nd):
        fwd, _ = lp_rattle(u_m, th)
        bwd, _ = lp_rattle(u_m, th + np.pi)
        best = max(best, fwd + bwd)
    return best * 1000


# ── core plotter ──────────────────────────────────────────────────────────────

def plot_vv(ax, q_m_rad: np.ndarray, u_m: np.ndarray, title: str = "") -> None:
    """
    Draw the VV state on *ax*.

    Parameters
    ----------
    q_m_rad : [dx_m, dy_m, dθ_rad]  — true rigid-body displacement
    u_m     : (9,) assembly offsets in metres
    title   : optional subplot title
    """
    dx, dy, dth = q_m_rad
    # Magnified display quantities
    DX   = dx  * MAG
    DY   = dy  * MAG
    DTH  = dth * MAG   # magnified rotation angle

    phi  = np.linspace(0, 2 * np.pi, 721)
    GAP_D = DELTA_M * MAG   # half-gap in display metres (0.75 m)
    BAR_HW = 0.12           # half-height of slot limit bar (m, display)

    # -- reference ring (thin grey) --
    ax.plot(R_M * np.cos(phi), R_M * np.sin(phi),
            "-", color="#d8d8d8", lw=0.8, zorder=1)

    # -- displaced VV ring (thick blue) --
    vx = R_M * np.cos(phi + DTH) + DX
    vy = R_M * np.sin(phi + DTH) + DY
    ax.plot(vx, vy, "-", color="#1e4d9b", lw=3.5, zorder=3)

    # -- machine axis cross (stays fixed) --
    ax.plot(0, 0, "+", color="#222", ms=18, mew=2.2, zorder=8, clip_on=False)

    # -- VV centre dot (moves) --
    ax.plot(DX, DY, "o", color="#1e4d9b", ms=6, zorder=8)

    # -- radial spokes (move + rotate with VV) --
    for φi in ANGLES:
        ex = R_M * np.cos(φi + DTH) + DX
        ey = R_M * np.sin(φi + DTH) + DY
        ax.plot([DX, ex], [DY, ey], "-", color="#5a8fd4", lw=1.1,
                alpha=0.5, zorder=2)

    # -- support slots (fixed in machine frame) --
    for i, φi in enumerate(ANGLES):
        er = np.array([ np.cos(φi),  np.sin(φi)])   # radial unit vector
        et = np.array([-np.sin(φi),  np.cos(φi)])   # toroidal unit vector

        sc = R_SLOT * er   # slot centre (fixed)

        # track line
        ax.plot([sc[0] - GAP_D * et[0], sc[0] + GAP_D * et[0]],
                [sc[1] - GAP_D * et[1], sc[1] + GAP_D * et[1]],
                "-", color="#aaaaaa", lw=1.6, zorder=3)

        # limit bars ("|")
        for sign in (-1, +1):
            bc = sc + sign * GAP_D * et
            ax.plot([bc[0] - BAR_HW * er[0], bc[0] + BAR_HW * er[0]],
                    [bc[1] - BAR_HW * er[1], bc[1] + BAR_HW * er[1]],
                    "-", color="#cc2200", lw=3.0, zorder=5)

        # pin position (assembly offset + current VV toroidal displacement)
        delta_i  = A[i] @ q_m_rad
        pin_off_d = (u_m[i] + delta_i) * MAG   # magnified offset from slot centre
        pin_pos  = sc + pin_off_d * et

        # colour depends on gap usage fraction
        frac = abs(u_m[i] + delta_i) / DELTA_M
        if   frac > 0.90: pcol, pms = "#cc2200", 10
        elif frac > 0.70: pcol, pms = "#e07000",  8
        else:              pcol, pms = "#1a6ea8",  7

        ax.plot(pin_pos[0], pin_pos[1], "o",
                color=pcol, ms=pms, zorder=6,
                markeredgecolor="white", markeredgewidth=0.9)

        # faint radial connector (ring → pin)
        ax_i = R_M * np.cos(φi + DTH) + DX
        ay_i = R_M * np.sin(φi + DTH) + DY
        ax.plot([ax_i, pin_pos[0]], [ay_i, pin_pos[1]],
                "-", color="#999999", lw=0.5, zorder=2, alpha=0.35)

    ax.set_aspect("equal")
    ax.axis("off")
    pad = 1.3
    ax.set_xlim(-R_SLOT - pad, R_SLOT + pad)
    ax.set_ylim(-R_SLOT - pad, R_SLOT + pad)
    if title:
        ax.set_title(title, fontsize=9, color="#333", pad=5)


# ── GIF maker ─────────────────────────────────────────────────────────────────

def make_gif(u_m: np.ndarray, outpath: str, n_frames: int = 40,
             fps: int = 10, dpi: int = 80) -> tuple[str, float]:
    """
    Animate VV rocking along its principal rattle axis.
    Returns (outpath, rattle_mm).
    """
    best_fwd_m, best_dq_fwd, best_th = 0.0, np.zeros(3), 0.0
    for th in np.linspace(0, np.pi, 72):
        v, dq = lp_rattle(u_m, th)
        if v > best_fwd_m:
            best_fwd_m, best_dq_fwd, best_th = v, dq, th

    bwd_m, best_dq_bwd = lp_rattle(u_m, best_th + np.pi)
    rattle_mm = (best_fwd_m + bwd_m) * 1000

    t = np.concatenate([np.linspace(0, 1, n_frames // 2),
                        np.linspace(1, 0, n_frames // 2)])
    q_frames = [best_dq_bwd + s * (best_dq_fwd - best_dq_bwd) for s in t]

    fig, ax = plt.subplots(figsize=(6.5, 6.5), facecolor="white")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.93, bottom=0.01)

    def draw(k):
        ax.clear()
        plot_vv(ax, q_frames[k], u_m)
        ax.set_title(
            f"Rattle range {rattle_mm:.2f} mm  ·  ×{MAG} magnification",
            fontsize=8.5, color="#444", pad=3
        )

    anim = FuncAnimation(fig, draw, frames=n_frames, interval=100)
    anim.save(outpath, writer="pillow", fps=fps, dpi=dpi)
    plt.close(fig)
    return outpath, rattle_mm


# ── static figures ─────────────────────────────────────────────────────────────

def figure_three_panel(u_worst: np.ndarray) -> str:
    """Three-panel figure: nominal | worst-case | alternating. Returns base64 PNG."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 6), facecolor="white")
    fig.subplots_adjust(wspace=0.05, left=0.01, right=0.99, top=0.91, bottom=0.07)

    # Panel 1 — nominal assembly (u=0), VV at max +X rattle
    _, dq_nom = lp_rattle(np.zeros(9), 0.0)
    plot_vv(axes[0], dq_nom, np.zeros(9))
    axes[0].set_title("Nominal assembly (u = 0)\nShown at max +X rattle position",
                       fontsize=9, color="#333")

    # Panel 2 — worst MC sample
    _, dq_wc = lp_rattle(u_worst, 0.0)
    rattle_wc = max_rattle_mm(u_worst)
    plot_vv(axes[1], dq_wc, u_worst)
    axes[1].set_title(f"Worst MC sample (rattle = {rattle_wc:.2f} mm)\nShown at max rattle position",
                      fontsize=9, color="#333")

    # Panel 3 — alternating gaps → fully locked
    u_alt = np.array([DELTA_M if i % 2 == 0 else -DELTA_M for i in range(N)])
    plot_vv(axes[2], np.zeros(3), u_alt)
    axes[2].set_title("Alternating ±1.5 mm gaps\nRattle ≈ 0 — fully locked",
                      fontsize=9, color="#333")

    legend_els = [
        Line2D([0], [0], color="#d8d8d8", lw=1.5, label="Nominal VV ring"),
        Line2D([0], [0], color="#1e4d9b", lw=3.0, label=f"Displaced VV ring (×{MAG})"),
        Line2D([0], [0], color="#cc2200", lw=2.5, label="Slot limits (±1.5 mm)"),
        Line2D([0], [0], marker="o", color="#1a6ea8", ms=7, ls="none", label="Pin — slack"),
        Line2D([0], [0], marker="o", color="#e07000", ms=7, ls="none", label="Pin — near limit"),
        Line2D([0], [0], marker="o", color="#cc2200", ms=7, ls="none", label="Pin — at limit"),
    ]
    fig.legend(handles=legend_els, loc="lower center", ncol=6,
               fontsize=8, frameon=True, fancybox=False,
               edgecolor="#ccc", bbox_to_anchor=(0.5, 0.0))

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def figure_mc_dashboard(rattles_mm: np.ndarray) -> str:
    """MC histogram + CDF. Returns base64 PNG."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), facecolor="white")
    fig.subplots_adjust(left=0.08, right=0.97, top=0.88, bottom=0.12, wspace=0.3)

    # histogram
    ax1.hist(rattles_mm, bins=50, color="#2b5797", alpha=0.75,
             edgecolor="white", lw=0.4)
    # compute LP worst-case range for nominal u=0
    fwd_m, _ = lp_rattle(np.zeros(9), 0.0)
    bwd_m, _ = lp_rattle(np.zeros(9), np.pi)
    lp_range  = (fwd_m + bwd_m) * 1000
    for pct, col, ls in [(95, "#cc8800", "--"), (99, "#cc2200", "--")]:
        v = np.percentile(rattles_mm, pct)
        ax1.axvline(v, color=col, lw=2, ls=ls, label=f"P{pct} = {v:.2f} mm")
    ax1.axvline(rattles_mm.max(), color="#660000", lw=1.5, ls=":",
                label=f"Max = {rattles_mm.max():.2f} mm")
    ax1.axvline(lp_range, color="#888", lw=1.5, ls="-",
                label=f"LP worst case = {lp_range:.2f} mm")
    ax1.set_xlabel("Rattle range (mm)", fontsize=10)
    ax1.set_ylabel(f"Count  (n = {len(rattles_mm):,})", fontsize=10)
    ax1.set_title("MC Rattle Distribution", fontsize=11, color="#1a3a6e")
    ax1.legend(fontsize=8.5, frameon=False)
    ax1.spines[["top", "right"]].set_visible(False)

    # CDF
    s = np.sort(rattles_mm)
    p = np.linspace(0, 100, len(s))
    ax2.plot(s, p, "-", color="#2b5797", lw=2)
    ax2.fill_betweenx(p, s, alpha=0.12, color="#2b5797")
    for pct, col in [(50, "#228822"), (75, "#cc8800"), (95, "#cc5500"), (99, "#cc2200")]:
        v = np.percentile(rattles_mm, pct)
        ax2.axvline(v, color=col, lw=1.5, ls="--",
                    label=f"P{pct} = {v:.2f} mm")
    ax2.set_xlabel("Rattle range (mm)", fontsize=10)
    ax2.set_ylabel("Percentile", fontsize=10)
    ax2.set_title("Cumulative Distribution", fontsize=11, color="#1a3a6e")
    ax2.legend(fontsize=8.5, frameon=False)
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.set_ylim(0, 100)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def figure_partial_measurement(rattles_all: np.ndarray, u_ref: np.ndarray) -> str:
    """
    Show rattle CDFs for k = 0, 1, 2, 4 measured adjacent supports.
    Returns base64 PNG.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), facecolor="white")
    fig.subplots_adjust(left=0.07, right=0.97, top=0.88, bottom=0.12, wspace=0.32)
    ax_cdf, ax_bar = axes

    colours = {0: "#2b5797", 1: "#cc8800", 2: "#228822", 4: "#cc2200", 9: "#111111"}
    results: dict[int, np.ndarray] = {0: rattles_all}

    rng = np.random.default_rng(42)

    for k in [1, 2, 4]:
        fixed_idx = list(range(k))
        r_k = np.zeros(2000)
        for s in range(2000):
            u = rng.uniform(-DELTA_M, DELTA_M, N)
            for i in fixed_idx:
                u[i] = u_ref[i]
            r_k[s] = max_rattle_mm(u, nd=24)
        results[k] = r_k
    # k=9: deterministic
    results[9] = np.array([max_rattle_mm(u_ref, nd=72)])

    p95_vals = {k: float(np.percentile(v, 95)) if len(v) > 1 else float(v[0])
                for k, v in results.items()}

    for k, r in sorted(results.items()):
        if len(r) == 1:
            ax_cdf.axvline(r[0], color=colours[k], lw=2,
                           label=f"k = {k} (all measured): {r[0]:.2f} mm")
            continue
        s = np.sort(r)
        p = np.linspace(0, 100, len(s))
        ax_cdf.plot(s, p, "-", color=colours[k], lw=2,
                    label=f"k = {k} supports fixed  (P95 = {p95_vals[k]:.2f} mm)")

    ax_cdf.set_xlabel("Rattle range (mm)", fontsize=10)
    ax_cdf.set_ylabel("Percentile", fontsize=10)
    ax_cdf.set_title("Rattle CDF vs number of supports measured\n"
                     "(measured = adjacent supports 0 … k−1)", fontsize=9.5, color="#1a3a6e")
    ax_cdf.legend(fontsize=8.5, frameon=False)
    ax_cdf.set_ylim(0, 100)
    ax_cdf.spines[["top", "right"]].set_visible(False)

    # Bar chart: P95 reduction
    ks = sorted(p95_vals.keys())
    bar_c = [colours[k] for k in ks]
    ax_bar.bar(ks, [p95_vals[k] for k in ks], color=bar_c, alpha=0.8, width=0.7)
    ax_bar.set_xlabel("Supports measured (k)", fontsize=10)
    ax_bar.set_ylabel("P95 rattle range (mm)", fontsize=10)
    ax_bar.set_title("P95 reduction vs measurements taken", fontsize=9.5, color="#1a3a6e")
    ax_bar.set_xticks(ks)
    for k in ks:
        ax_bar.text(k, p95_vals[k] + 0.04, f"{p95_vals[k]:.2f}",
                    ha="center", va="bottom", fontsize=8, color="#333")
    ax_bar.spines[["top", "right"]].set_visible(False)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


# ── CLI entry point ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-gif", action="store_true", help="Skip GIF generation")
    parser.add_argument("--outdir", default="docs", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    anim_dir = os.path.join(args.outdir, "animations")
    os.makedirs(anim_dir, exist_ok=True)
    plots_dir = os.path.join(args.outdir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Load MC data
    try:
        rattles_mm = np.load("/tmp/rattle_mc_5k.npy")
        u_mc       = np.load("/tmp/u_mc_5k.npy")
        u_worst    = u_mc[rattles_mm.argmax()]
        print(f"Loaded MC data: {len(rattles_mm)} samples, max = {rattles_mm.max():.3f} mm")
    except FileNotFoundError:
        print("MC data not found at /tmp/ — running quick 1000-sample MC …")
        rng = np.random.default_rng(0)
        u_mc = rng.uniform(-DELTA_M, DELTA_M, (1000, N))
        rattles_mm = np.array([max_rattle_mm(u, nd=24) for u in u_mc])
        np.save("/tmp/rattle_mc_5k.npy", rattles_mm)
        np.save("/tmp/u_mc_5k.npy", u_mc)
        u_worst = u_mc[rattles_mm.argmax()]

    # Static figures
    print("Generating 3-panel state figure …")
    b64_3panel = figure_three_panel(u_worst)
    png_path = os.path.join(plots_dir, "vv_states.png")
    with open(png_path, "wb") as fh:
        fh.write(base64.b64decode(b64_3panel))
    print(f"  → {png_path}")

    print("Generating MC dashboard …")
    b64_mc = figure_mc_dashboard(rattles_mm)
    png_mc = os.path.join(plots_dir, "mc_dashboard.png")
    with open(png_mc, "wb") as fh:
        fh.write(base64.b64decode(b64_mc))
    print(f"  → {png_mc}")

    print("Generating partial-measurement figure (this may take ~2 min) …")
    b64_partial = figure_partial_measurement(rattles_mm, u_worst)
    png_partial = os.path.join(plots_dir, "partial_measurement.png")
    with open(png_partial, "wb") as fh:
        fh.write(base64.b64decode(b64_partial))
    print(f"  → {png_partial}")

    gif_nom_path = gif_wc_path = None
    if not args.no_gif:
        print("Generating nominal GIF …")
        gif_nom_path, nom_rattle = make_gif(
            np.zeros(N), os.path.join(anim_dir, "rattle_nominal.gif"))
        print(f"  → {gif_nom_path}  ({nom_rattle:.2f} mm)")
        print("Generating worst-case GIF …")
        gif_wc_path, wc_rattle = make_gif(
            u_worst, os.path.join(anim_dir, "rattle_worst_case.gif"))
        print(f"  → {gif_wc_path}  ({wc_rattle:.2f} mm)")

    print("\nAll outputs written.")
    return {
        "b64_3panel": b64_3panel,
        "b64_mc": b64_mc,
        "b64_partial": b64_partial,
        "gif_nom": gif_nom_path,
        "gif_wc": gif_wc_path,
        "rattles_mm": rattles_mm,
        "u_worst": u_worst,
        "u_mc": u_mc,
    }


if __name__ == "__main__":
    main()
