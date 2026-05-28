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

Rattle polytope
---------------
For a given assembly state u_m, the set of all reachable VV centre positions
(Δx, Δy) — with Δθ optimised freely — is a convex polygon (projection of the
3D LP feasible set onto the translation plane).  rattle_polytope_2d() computes
this polygon; plot_vv() draws it (×MAG) around the VV centre dot.

Note on rotation magnitude
--------------------------
At the translation LP optimum (max X), Δθ = 5.83 µrad → only 0.17° at ×500.
Invisible on a circle.  The dedicated rotation GIF shows max Δθ = 187.5 µrad
→ ±5.37° at ×500, where spoke 0 sweeps visibly.  In that mode ALL nine brackets
move synchronously (same R·Δθ = 1.5 mm) — the signature of pure rotation.

Usage
-----
    python vv_viz.py                  # generate all outputs to docs/
    python vv_viz.py --no-gif         # skip (slow) GIF generation
"""

from __future__ import annotations
import argparse, io, base64, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D
from scipy.optimize import linprog
from scipy.spatial import ConvexHull

# ── geometry ──────────────────────────────────────────────────────────────────
MAG     = 500       # displacement magnification for display (polytope-diagnostic)
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


def lp_vec(u_m: np.ndarray, cvec) -> tuple[float, np.ndarray]:
    """LP with arbitrary 3-vector objective."""
    Au  = np.vstack([A, -A])
    bu  = np.concatenate([DELTA_M - u_m, DELTA_M + u_m])
    res = linprog(-np.asarray(cvec, float), A_ub=Au, b_ub=bu,
                  bounds=[(None, None)] * 3, method="highs")
    if not res.success:
        return 0.0, np.zeros(3)
    return float(-res.fun), res.x


def max_rattle_mm(u_m: np.ndarray, nd: int = 36) -> float:
    """Maximum peak-to-peak rattle range (mm) — polytope diameter."""
    best = 0.0
    for th in np.linspace(0, np.pi, nd):
        fwd, _ = lp_rattle(u_m, th)
        bwd, _ = lp_rattle(u_m, th + np.pi)
        best = max(best, fwd + bwd)
    return best * 1000


def max_departure_mm(u_m: np.ndarray, nd: int = 72) -> float:
    """Maximum one-sided departure (mm) of the VV centre from the gravitational
    centre q=0. This is the metric for forced excursion from the self-centred
    rest position: the largest ||q[:2]|| reachable inside the feasible polytope.
    """
    best = 0.0
    for th in np.linspace(0, 2 * np.pi, nd, endpoint=False):
        _, q = lp_rattle(u_m, th)
        best = max(best, float(np.hypot(q[0], q[1])))
    return best * 1000


def rattle_polytope_2d(u_m: np.ndarray, nd: int = 360) -> np.ndarray | None:
    """
    Convex polygon of all reachable (Δx, Δy) positions from assembly state u_m.
    Δθ is treated as a free variable (LP uses it optimally for each direction).
    Returns (nv, 2) array of polygon vertices in metres, ordered CCW.
    Returns None if the polytope is degenerate (rattle ≈ 0).
    """
    pts = np.zeros((nd, 2))
    Au  = np.vstack([A, -A])
    bu  = np.concatenate([DELTA_M - u_m, DELTA_M + u_m])
    for k, th in enumerate(np.linspace(0, 2 * np.pi, nd, endpoint=False)):
        c = -np.array([np.cos(th), np.sin(th), 0.0])
        res = linprog(c, A_ub=Au, b_ub=bu, bounds=[(None, None)] * 3, method="highs")
        pts[k] = res.x[:2]
    # guard against degenerate (rattle ≈ 0) polygon
    span = np.ptp(pts, axis=0)
    if span.max() < 1e-9:
        return None
    try:
        hull   = ConvexHull(pts)
    except Exception:
        return None
    verts  = pts[hull.vertices]
    # sort CCW
    cx, cy = verts.mean(0)
    order  = np.argsort(np.arctan2(verts[:, 1] - cy, verts[:, 0] - cx))
    return verts[order]


# ── core plotter ──────────────────────────────────────────────────────────────

def plot_vv(ax, q_m_rad: np.ndarray, u_m: np.ndarray,
            polytope_m=None,
            title: str = "") -> None:
    """
    Draw the VV state on *ax*.

    Parameters
    ----------
    q_m_rad    : [dx_m, dy_m, dθ_rad]  — true rigid-body displacement
    u_m        : (9,) assembly offsets in metres
    polytope_m : (nv, 2) reachable polygon in metres, drawn at ×MAG around
                 the VV centre.  Pass rattle_polytope_2d(u_m) to show it.
    title      : optional subplot title
    """
    dx, dy, dth = q_m_rad
    DX   = dx  * MAG      # magnified X
    DY   = dy  * MAG      # magnified Y
    DTH  = dth * MAG      # magnified rotation angle (rad)

    phi   = np.linspace(0, 2 * np.pi, 721)
    GAP_D = DELTA_M * MAG   # half-gap display (0.75 m)
    BAR_HW = 0.12

    # ── reference ring ──
    ax.plot(R_M * np.cos(phi), R_M * np.sin(phi),
            "-", color="#d8d8d8", lw=0.8, zorder=1)

    # ── rattle polytope (fixed in machine frame, centred at origin) ──
    if polytope_m is not None:
        px = polytope_m[:, 0] * MAG
        py = polytope_m[:, 1] * MAG
        # close the polygon
        px = np.append(px, px[0])
        py = np.append(py, py[0])
        ax.fill(px, py, color="#2b5797", alpha=0.08, zorder=2)
        ax.plot(px, py, "-", color="#2b5797", lw=1.4, alpha=0.55, zorder=2)

    # ── displaced VV ring ──
    vx = R_M * np.cos(phi + DTH) + DX
    vy = R_M * np.sin(phi + DTH) + DY
    ax.plot(vx, vy, "-", color="#1e4d9b", lw=3.5, zorder=3)

    # ── machine axis ──
    ax.plot(0, 0, "+", color="#222", ms=18, mew=2.2, zorder=9, clip_on=False)

    # ── VV centre dot — bright red so it reads against the polytope ──
    ax.plot(DX, DY, "o", color="#cc2200", ms=8, zorder=9,
            markeredgecolor="white", markeredgewidth=1.2)

    # ── radial spokes: spoke 0 (top, φ=90°) orange as orientation marker ──
    for i, φi in enumerate(ANGLES):
        ex = R_M * np.cos(φi + DTH) + DX
        ey = R_M * np.sin(φi + DTH) + DY
        if i == 0:
            ax.plot([DX, ex], [DY, ey], "-", color="#e06000",
                    lw=2.5, alpha=0.9, zorder=4)   # orientation marker
            ax.plot(ex, ey, "o", color="#e06000", ms=7, zorder=4,
                    markeredgecolor="white", markeredgewidth=0.8)
        else:
            ax.plot([DX, ex], [DY, ey], "-", color="#5a8fd4",
                    lw=1.2, alpha=0.65, zorder=2)
            ax.plot(ex, ey, "o", color="#5a8fd4", ms=5, zorder=3,
                    markeredgecolor="white", markeredgewidth=0.6)

    # ── support linkages ──────────────────────────────────────────────────
    # Each four-bar linkage MOVES WITH THE VESSEL radially (radial motion is
    # free), so the toroidal slot is held at a constant radial distance from the
    # (displaced) vessel wall.  The pin slides within the ±1.5 mm slot — that
    # toroidal offset (u_i + δ_i) is the constrained quantity.
    for i, φi in enumerate(ANGLES):
        er = np.array([ np.cos(φi),  np.sin(φi)])
        et = np.array([-np.sin(φi),  np.cos(φi)])

        # displaced material point on the vessel wall at this support
        wall = np.array([R_M * np.cos(φi + DTH) + DX,
                         R_M * np.sin(φi + DTH) + DY])
        dr   = float(wall @ er) - R_M           # radial travel (free) of the wall here
        sc   = (R_SLOT + dr) * er               # slot tracks the vessel radially

        delta_i = A[i] @ q_m_rad
        pin     = sc + (u_m[i] + delta_i) * MAG * et   # pin = toroidal offset in slot

        # four-bar linkage / bracket: connects the moving wall to the pin
        ax.plot([wall[0], pin[0]], [wall[1], pin[1]],
                "-", color="#9a9a9a", lw=1.3, zorder=2, alpha=0.7)

        # toroidal slot channel (±1.5 mm travel) + red end-stops (ground stops)
        ax.plot([sc[0] - GAP_D * et[0], sc[0] + GAP_D * et[0]],
                [sc[1] - GAP_D * et[1], sc[1] + GAP_D * et[1]],
                "-", color="#9aa0a8", lw=2.4, zorder=3, solid_capstyle="round")
        for sign in (-1, +1):
            bc = sc + sign * GAP_D * et
            ax.plot([bc[0] - BAR_HW * er[0], bc[0] + BAR_HW * er[0]],
                    [bc[1] - BAR_HW * er[1], bc[1] + BAR_HW * er[1]],
                    "-", color="#cc2200", lw=3.0, zorder=5)

        frac = abs(u_m[i] + delta_i) / DELTA_M
        if   frac > 0.90: pcol, pms = "#cc2200", 10
        elif frac > 0.70: pcol, pms = "#e07000",  8
        else:              pcol, pms = "#1a6ea8",  7
        ax.plot(pin[0], pin[1], "o", color=pcol, ms=pms, zorder=6,
                markeredgecolor="white", markeredgewidth=0.9)

    ax.set_aspect("equal")
    ax.axis("off")
    pad = 1.3
    ax.set_xlim(-R_SLOT - pad, R_SLOT + pad)
    ax.set_ylim(-R_SLOT - pad, R_SLOT + pad)
    if title:
        ax.set_title(title, fontsize=9, color="#333", pad=5)


# ── GIF makers ────────────────────────────────────────────────────────────────

def make_gif(u_m: np.ndarray, outpath: str, n_frames: int = 40,
             fps: int = 10, dpi: int = 80) -> tuple[str, float]:
    """
    Animate VV rocking along its principal translation rattle axis.
    Draws the rattle polytope so the viewer sees where the centre travels.
    Returns (outpath, rattle_mm).
    """
    # animate along the MAX peak-to-peak RANGE (diameter) axis, not the max
    # forward-reach axis — the two differ for asymmetric (offset) polytopes.
    best_range, best_th = 0.0, 0.0
    for th in np.linspace(0, np.pi, 72):
        f, _ = lp_rattle(u_m, th)
        b, _ = lp_rattle(u_m, th + np.pi)
        if f + b > best_range:
            best_range, best_th = f + b, th
    best_fwd_m, best_dq_fwd = lp_rattle(u_m, best_th)
    bwd_m,      best_dq_bwd = lp_rattle(u_m, best_th + np.pi)
    rattle_mm = (best_fwd_m + bwd_m) * 1000

    polytope = rattle_polytope_2d(u_m, nd=180)

    t = np.concatenate([np.linspace(0, 1, n_frames // 2),
                        np.linspace(1, 0, n_frames // 2)])
    q_frames = [best_dq_bwd + s * (best_dq_fwd - best_dq_bwd) for s in t]

    fig, ax = plt.subplots(figsize=(6.5, 6.5), facecolor="white")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.93, bottom=0.01)

    def draw(k):
        ax.clear()
        plot_vv(ax, q_frames[k], u_m, polytope_m=polytope)
        # title intentionally empty — context is given in the figure caption

    anim = FuncAnimation(fig, draw, frames=n_frames, interval=100)
    anim.save(outpath, writer="pillow", fps=fps, dpi=dpi)
    plt.close(fig)
    return outpath, rattle_mm


def make_rotation_gif(u_m: np.ndarray, outpath: str, n_frames: int = 40,
                      fps: int = 10, dpi: int = 80) -> tuple[str, float]:
    """
    Animate VV rotating through its maximum rotation range (Δθ DOF).
    At ×500 the max rotation (187.5 µrad) becomes ±5.37° — clearly visible
    as spoke 0 sweeps around.  All 9 brackets move synchronously (pure rotation
    signature).
    Returns (outpath, rot_range_mrad).
    """
    fwd_m, dq_fwd = lp_vec(u_m, [0, 0, 1])
    bwd_m, dq_bwd = lp_vec(u_m, [0, 0, -1])
    rot_range_mrad = (fwd_m + bwd_m) * 1000  # µrad → mrad

    # Pure rotation does NOT translate the VV centre, so there is no 2-D
    # translation polytope to draw here — the rotation DOF is the 1-D interval
    # Δθ ∈ [−187.5, +187.5] µrad (the centre stays on the machine axis). The
    # synchronously sliding pins are the signature of this mode.
    polytope = None

    t = np.concatenate([np.linspace(0, 1, n_frames // 2),
                        np.linspace(1, 0, n_frames // 2)])
    q_frames = [dq_bwd + s * (dq_fwd - dq_bwd) for s in t]

    fig, ax = plt.subplots(figsize=(6.5, 6.5), facecolor="white")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.93, bottom=0.01)

    def draw(k):
        ax.clear()
        plot_vv(ax, q_frames[k], u_m, polytope_m=polytope)
        dth_disp_deg = np.degrees(q_frames[k][2] * MAG)
        ax.set_title(
            f"Pure rotation — centre fixed on axis  ·  Δθ = {q_frames[k][2]*1e6:+.1f} µrad  "
            f"(×{MAG} = {dth_disp_deg:+.2f}°)  ·  range {rot_range_mrad:.3f} mrad  ·  "
            f"all 9 pins slide synchronously",
            fontsize=7.2, color="#444", pad=3,
        )

    anim = FuncAnimation(fig, draw, frames=n_frames, interval=100)
    anim.save(outpath, writer="pillow", fps=fps, dpi=dpi)
    plt.close(fig)
    return outpath, rot_range_mrad


# ── key-frame strip (for PDF) ──────────────────────────────────────────────────

def make_keyframe_strip(u_m: np.ndarray, outpath: str,
                        mode: str = "translation",
                        n_frames: int = 5, dpi: int = 100) -> str:
    """
    Single-row strip of n_frames key frames along the principal rattle or
    rotation axis.  Suitable for embedding in a printed PDF.
    Returns base64-encoded PNG string.
    """
    if mode == "rotation":
        fwd_m, dq_fwd = lp_vec(u_m, [0, 0,  1])
        bwd_m, dq_bwd = lp_vec(u_m, [0, 0, -1])
        label = "Rotation"
        unit  = f"{(fwd_m+bwd_m)*1e6:.1f} µrad range"
    else:
        best_range, best_th = 0.0, 0.0
        for th in np.linspace(0, np.pi, 72):
            f, _ = lp_rattle(u_m, th)
            b, _ = lp_rattle(u_m, th + np.pi)
            if f + b > best_range:
                best_range, best_th = f + b, th
        best_fwd_m, dq_fwd = lp_rattle(u_m, best_th)
        bwd_m,      dq_bwd = lp_rattle(u_m, best_th + np.pi)
        label = "Translation"
        unit  = f"{(best_fwd_m+bwd_m)*1e3:.2f} mm range"

    polytope = rattle_polytope_2d(u_m, nd=180)
    t_vals = np.linspace(0, 1, n_frames)

    fig, axes = plt.subplots(1, n_frames,
                             figsize=(5 * n_frames, 5.2),
                             facecolor="white")
    fig.subplots_adjust(wspace=0.02, left=0.005, right=0.995,
                        top=0.88, bottom=0.005)

    for k, (ax, t) in enumerate(zip(axes, t_vals)):
        q = dq_bwd + t * (dq_fwd - dq_bwd)
        plot_vv(ax, q, u_m, polytope_m=polytope)
        pos_label = ["← max", "¾←", "centre", "¾→", "max →"][k] if n_frames == 5 \
                    else f"t={t:.2f}"
        ax.set_title(pos_label, fontsize=8, color="#555", pad=3)

    # Suptitle intentionally omitted — figure caption carries the context.

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    b64 = base64.b64encode(buf.getvalue()).decode()

    if outpath:
        with open(outpath, "wb") as fh:
            fh.write(base64.b64decode(b64))
    return b64


# ── static figures ─────────────────────────────────────────────────────────────

def _max_departure_state(u_m: np.ndarray, nd: int = 144) -> tuple[float, np.ndarray]:
    """Search nd directions for the LP-extreme that maximises ||q[:2]||.
    Returns (departure_mm, q*)."""
    best = 0.0
    best_q = np.zeros(3)
    for th in np.linspace(0, 2 * np.pi, nd, endpoint=False):
        _, q = lp_rattle(u_m, th)
        d = float(np.hypot(q[0], q[1]))
        if d > best:
            best, best_q = d, q
    return best * 1000.0, best_q


def figure_three_panel(u_worst: np.ndarray) -> str:
    """Two-panel state figure: nominal assembly | worst MC sample.

    In each panel the VV ring is drawn at its max-departure position; a red
    arrow from the machine axis (cross) to the displaced VV centre shows the
    direction and magnitude of that maximum displacement; the shaded polygon
    around the machine axis is the displacement polytope (the kinematic
    envelope of possible lateral positions). Returns base64 PNG."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), facecolor="white")
    fig.subplots_adjust(wspace=0.05, left=0.01, right=0.99, top=0.91, bottom=0.10)

    for ax, u, label in [(axes[0], np.zeros(N), "Nominal assembly (u = 0)"),
                         (axes[1], u_worst,    "Worst MC sample (offset assembly)")]:
        poly = rattle_polytope_2d(u, nd=120)
        dep_mm, q_star = _max_departure_state(u, nd=144)
        plot_vv(ax, q_star, u, polytope_m=poly)
        # arrow from machine axis (origin) to displaced VV centre — make it bold
        DX, DY = q_star[0] * MAG, q_star[1] * MAG
        ax.annotate(
            "", xy=(DX, DY), xytext=(0, 0),
            arrowprops=dict(arrowstyle="-|>,head_length=0.6,head_width=0.4",
                            color="#cc2200", lw=3.2, shrinkA=0, shrinkB=4),
            zorder=12,
        )
        # mm-scale label next to the arrow
        ax.text(DX * 0.55, DY * 0.55 + 0.25, f"{dep_mm:.2f} mm",
                color="#cc2200", fontsize=9, weight="bold", ha="left", va="bottom",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#cc2200", alpha=0.9, lw=0.8),
                zorder=13)
        ax.set_title(f"{label}\nMax displacement from centre = {dep_mm:.2f} mm",
                     fontsize=9.5, color="#333")

    legend_els = [
        Line2D([0], [0], color="#d8d8d8", lw=1.5, label="Reference ring"),
        Line2D([0], [0], color="#1e4d9b", lw=3.0, label=f"VV ring at max displacement (×{MAG})"),
        Line2D([0], [0], color="#cc2200", lw=2.2, label="Max-displacement vector (origin → VV centre)"),
        Line2D([0], [0], color="#e06000", lw=2.5, label="Spoke 0 (orientation)"),
        Line2D([0], [0], color="#cc2200", lw=2.5, label="Slot limits (±1.5 mm)"),
        mpatches.Patch(facecolor="#2b5797", alpha=0.15, edgecolor="#2b5797",
                       label="Displacement polytope"),
        Line2D([0], [0], marker="o", color="#1a6ea8", ms=7, ls="none", label="Pin — slack"),
        Line2D([0], [0], marker="o", color="#e07000", ms=7, ls="none", label="Pin — near limit"),
        Line2D([0], [0], marker="o", color="#cc2200", ms=7, ls="none", label="Pin — at limit"),
    ]
    fig.legend(handles=legend_els, loc="lower center", ncol=5,
               fontsize=7.5, frameon=True, fancybox=False,
               edgecolor="#ccc", bbox_to_anchor=(0.5, 0.0))

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def figure_mc_dashboard(rattles_mm: np.ndarray) -> str:
    """MC histogram + CDF. Returns base64 PNG."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), facecolor="white")
    fig.subplots_adjust(left=0.08, right=0.97, top=0.88, bottom=0.12, wspace=0.3)

    # histogram of departure-from-centre
    ax1.hist(rattles_mm, bins=50, color="#2b5797", alpha=0.75,
             edgecolor="white", lw=0.4)
    # nominal (u=0) departure = LP envelope half-width
    nom_dep = max_departure_mm(np.zeros(9), nd=72)
    # mode of the distribution
    h, e = np.histogram(rattles_mm, bins=50)
    mode = float(0.5 * (e[h.argmax()] + e[h.argmax() + 1]))
    med  = float(np.median(rattles_mm))
    p95  = float(np.percentile(rattles_mm, 95))
    p99  = float(np.percentile(rattles_mm, 99))
    rmax = float(rattles_mm.max())
    # Vertical reference lines, labelled DIRECTLY on the plot (no legend)
    h1_top = ax1.get_ylim()[1]
    # Cluster A (centred): mode / median / nominal — stagger heights
    for x, col, name, yf in [(mode,    "#228822", "Mode",    0.93),
                              (med,     "#1a6ea8", "Median",  0.78),
                              (nom_dep, "#666666", "Nominal", 0.63)]:
        ax1.axvline(x, color=col, lw=1.5, ls="--")
        ax1.text(x, h1_top * yf, f"{name}\n{x:.2f} mm",
                 ha="center", va="top", fontsize=8, color=col, weight="bold",
                 bbox=dict(boxstyle="round,pad=0.18", fc="white", ec=col, alpha=0.92, lw=0.8))
    # Cluster B (tail): P95 / P99 / Max
    for x, col, name, yf in [(p95,  "#cc8800", "P95",  0.93),
                              (p99,  "#cc2200", "P99",  0.78),
                              (rmax, "#660000", "Max",  0.63)]:
        ax1.axvline(x, color=col, lw=1.8, ls="--")
        ax1.text(x, h1_top * yf, f"{name}\n{x:.2f} mm",
                 ha="center", va="top", fontsize=8, color=col, weight="bold",
                 bbox=dict(boxstyle="round,pad=0.18", fc="white", ec=col, alpha=0.92, lw=0.8))
    ax1.set_xlabel("Max forced departure from gravitational centre (mm)", fontsize=10)
    ax1.set_ylabel(f"Count  (n = {len(rattles_mm):,})", fontsize=10)
    ax1.set_title("VV departure-from-centre — distribution",
                  fontsize=11, color="#1a3a6e")
    ax1.spines[["top", "right"]].set_visible(False)

    # CDF — direct labels at the percentile crossings (no legend)
    s = np.sort(rattles_mm)
    p = np.linspace(0, 100, len(s))
    ax2.plot(s, p, "-", color="#2b5797", lw=2)
    ax2.fill_betweenx(p, s, alpha=0.12, color="#2b5797")
    for pct, col in [(50, "#228822"), (75, "#1a6ea8"),
                     (95, "#cc8800"), (99, "#cc2200")]:
        v = float(np.percentile(rattles_mm, pct))
        ax2.axvline(v, color=col, lw=1.4, ls="--")
        ax2.axhline(pct, color=col, lw=0.6, ls=":", alpha=0.6)
        ax2.text(v, pct, f" P{pct}: {v:.2f} mm",
                 ha="left", va="bottom", fontsize=8.5, color=col, weight="bold",
                 bbox=dict(boxstyle="round,pad=0.18", fc="white", ec=col, alpha=0.92, lw=0.8))
    ax2.set_xlabel("Max forced departure from centre (mm)", fontsize=10)
    ax2.set_ylabel("Percentile", fontsize=10)
    ax2.set_title("Cumulative distribution", fontsize=11, color="#1a3a6e")
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.set_ylim(0, 100)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def figure_partial_measurement(rattles_all: np.ndarray, u_ref=None) -> str:
    """
    §8 figure under the departure-from-centre metric:
      Left  — the departure-from-centre distribution. Measuring all 9 gaps
              reveals ONE value from this distribution; centred assemblies
              give the nominal LP envelope (~1.55 mm), offset assemblies can
              reach FURTHER from the gravitational centre (up to ~2.9 mm).
      Right — conditional P95 vs number of gaps measured. Measuring more
              supports gradually narrows the conditional distribution; one
              sector landed (k=1) gives no meaningful change.
    Returns base64 PNG.
    """
    fig, (ax_d, ax_k) = plt.subplots(1, 2, figsize=(13, 5.5), facecolor="white")
    fig.subplots_adjust(left=0.07, right=0.97, top=0.85, bottom=0.13, wspace=0.27)

    r = rattles_all
    h, e = np.histogram(r, bins=50)
    mode = float(0.5 * (e[h.argmax()] + e[h.argmax() + 1]))
    med  = float(np.median(r))
    q95  = float(np.percentile(r, 95))
    nom  = max_departure_mm(np.zeros(N), nd=72)
    rmax = float(r.max())
    ax_d.hist(r, bins=50, color="#2b5797", alpha=0.75, edgecolor="white", lw=0.4)
    h_top = ax_d.get_ylim()[1]
    # Direct labels at each percentile / reference line (no legend)
    for x, c, name, yf in [(mode, "#228822", "Mode",    0.93),
                            (med,  "#1a6ea8", "Median",  0.78),
                            (nom,  "#666666", "Nominal", 0.63)]:
        ax_d.axvline(x, color=c, lw=1.5, ls="--")
        ax_d.text(x, h_top * yf, f"{name}\n{x:.2f} mm",
                  ha="center", va="top", fontsize=8, color=c, weight="bold",
                  bbox=dict(boxstyle="round,pad=0.18", fc="white", ec=c, alpha=0.92, lw=0.8))
    for x, c, name, yf in [(q95,  "#cc8800", "P95",  0.93),
                            (rmax, "#cc2200", "Max",  0.78)]:
        ax_d.axvline(x, color=c, lw=1.8, ls="--")
        ax_d.text(x, h_top * yf, f"{name}\n{x:.2f} mm",
                  ha="center", va="top", fontsize=8, color=c, weight="bold",
                  bbox=dict(boxstyle="round,pad=0.18", fc="white", ec=c, alpha=0.92, lw=0.8))
    ax_d.set_xlabel("Max forced departure from gravitational centre (mm)", fontsize=10)
    ax_d.set_ylabel(f"Count  (n = {len(r):,})", fontsize=10)
    ax_d.set_title("Displacement distribution (measuring all 9 reveals one value)",
                   fontsize=10, color="#1a3a6e")
    ax_d.spines[["top", "right"]].set_visible(False)

    # ── Right: conditional P95 vs number of gaps measured ──
    rng = np.random.default_rng(20260527)
    ks = [0, 1, 2, 3, 4, 6, 9]
    q_cent, q_typ = [], []
    for k in ks:
        idx = list(range(k))
        rc = np.empty(250)                       # measured = centred
        for s in range(250):
            u = rng.uniform(-DELTA_M, DELTA_M, N)
            for i in idx:
                u[i] = 0.0
            rc[s] = max_departure_mm(u, nd=18)
        q_cent.append(float(np.percentile(rc, 95)))
        if k == 0:
            q_typ.append(float(np.percentile(rc, 95)))
        else:
            qs = []
            for _ in range(4):
                vals = rng.uniform(-DELTA_M, DELTA_M, k)
                rt = np.empty(150)
                for s in range(150):
                    u = rng.uniform(-DELTA_M, DELTA_M, N)
                    for j, i in enumerate(idx):
                        u[i] = vals[j]
                    rt[s] = max_departure_mm(u, nd=18)
                qs.append(np.percentile(rt, 95))
            q_typ.append(float(np.mean(qs)))
    ax_k.plot(ks, q_cent, "o-", color="#228822", lw=2,
              label="measured = centred (lowest departure)")
    ax_k.plot(ks, q_typ, "s-", color="#cc8800", lw=2,
              label="measured = typical random values")
    ax_k.axhline(nom, color="#888", lw=1.2, ls=":",
                 label=f"nominal envelope (u=0) = {nom:.2f} mm")
    ax_k.annotate("one sector\nlanded (today)", xy=(1, q_typ[1]),
                  xytext=(1.8, 2.05), fontsize=8, color="#444",
                  arrowprops=dict(arrowstyle="->", color="#888"))
    ax_k.set_xlabel("Number of support gaps measured (k of 9)", fontsize=10)
    ax_k.set_ylabel("Conditional P95 departure from centre (mm)", fontsize=10)
    ax_k.set_title("Each measured gap pulls the polytope toward the centre —\n"
                   "the conditional bound tightens with every sector",
                   fontsize=9.5, color="#1a3a6e")
    ax_k.set_xticks(ks)
    ax_k.set_ylim(1.2, 2.6)
    ax_k.legend(fontsize=8, frameon=False, loc="upper right")
    ax_k.spines[["top", "right"]].set_visible(False)

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
    anim_dir  = os.path.join(args.outdir, "animations")
    plots_dir = os.path.join(args.outdir, "plots")
    strips_dir = os.path.join(args.outdir, "strips")
    os.makedirs(anim_dir,   exist_ok=True)
    os.makedirs(plots_dir,  exist_ok=True)
    os.makedirs(strips_dir, exist_ok=True)

    # Load canonical MC data (committed + reproducible — see vv_mc_generator.py).
    # Metric = peak-to-peak range (mm) = the n=1 lateral wall-displacement envelope.
    here     = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(here, "data")
    rpath    = os.path.join(data_dir, "rattle_mc_5k.npy")
    upath    = os.path.join(data_dir, "u_mc_5k.npy")
    if not (os.path.exists(rpath) and os.path.exists(upath)):
        print("Canonical MC data missing — regenerating via vv_mc_generator …")
        import vv_mc_generator
        vv_mc_generator.main()
    rattles_mm = np.load(rpath)
    u_mc       = np.load(upath)
    u_worst    = u_mc[rattles_mm.argmax()]
    # near-mode (typical as-built) sample — the value one is most likely to "land on"
    _h, _e  = np.histogram(rattles_mm, bins=60)
    _mode   = 0.5 * (_e[_h.argmax()] + _e[_h.argmax() + 1])
    i_mode  = int(np.argmin(np.abs(rattles_mm - _mode)))
    u_mode  = u_mc[i_mode]
    print(f"Loaded canonical MC: {len(rattles_mm)} samples, "
          f"max = {rattles_mm.max():.3f} mm (departure from centre); "
          f"mode ≈ {_mode:.3f} mm (sample {i_mode} = {rattles_mm[i_mode]:.3f} mm)")

    # Static figures
    print("Generating 3-panel state figure (with polytopes) …")
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

    # Key-frame strips
    print("Generating key-frame strips …")
    strip_nom_trans = os.path.join(strips_dir, "strip_nominal_translation.png")
    make_keyframe_strip(np.zeros(N), strip_nom_trans, mode="translation")
    print(f"  → {strip_nom_trans}")

    strip_wc_trans = os.path.join(strips_dir, "strip_worst_translation.png")
    make_keyframe_strip(u_worst, strip_wc_trans, mode="translation")
    print(f"  → {strip_wc_trans}")

    strip_mode_trans = os.path.join(strips_dir, "strip_mode_translation.png")
    make_keyframe_strip(u_mode, strip_mode_trans, mode="translation")
    print(f"  → {strip_mode_trans}")

    gif_nom_path = gif_wc_path = gif_mode_path = gif_rot_path = None
    if not args.no_gif:
        print("Generating nominal forced-excursion GIF (with polytope) …")
        gif_nom_path, nom_dep = make_gif(
            np.zeros(N), os.path.join(anim_dir, "rattle_nominal.gif"))
        print(f"  → {gif_nom_path}  ({nom_dep:.2f} mm)")

        print("Generating worst-case forced-excursion GIF (with polytope) …")
        gif_wc_path, wc_dep = make_gif(
            u_worst, os.path.join(anim_dir, "rattle_worst_case.gif"))
        print(f"  → {gif_wc_path}  ({wc_dep:.2f} mm)")

        print("Generating near-mode (typical as-built) GIF …")
        gif_mode_path, mode_dep = make_gif(
            u_mode, os.path.join(anim_dir, "rattle_mode.gif"))
        print(f"  → {gif_mode_path}  ({mode_dep:.2f} mm)")

    print("\nAll outputs written.")
    return {
        "b64_3panel": b64_3panel,
        "b64_mc": b64_mc,
        "b64_partial": b64_partial,
        "gif_nom":  gif_nom_path,
        "gif_wc":   gif_wc_path,
        "gif_mode": gif_mode_path,
        "gif_rot":  gif_rot_path,
        "rattles_mm": rattles_mm,
        "u_worst":  u_worst,
        "u_mc":     u_mc,
    }


if __name__ == "__main__":
    main()
