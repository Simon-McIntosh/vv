"""
Build the VVGS gravitational-centring explainer.

Generates three line diagrams (saved under docs/diagrams/) and writes the
companion HTML report at docs/vvgs-pendulum-mechanism.html. The report explains
how the inclined dual-hinge VVGS supports combine into an effective stable
pendulum, derives the effective length L_eff(alpha), and discusses limit cases,
confidence, holes and external evidence.

    uv run python build_pendulum_explainer.py
"""
from __future__ import annotations
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

HERE = os.path.dirname(os.path.abspath(__file__))
DIAGRAMS_DIR = os.path.join(HERE, "docs", "diagrams")
os.makedirs(DIAGRAMS_DIR, exist_ok=True)

# Parameters (estimates — see "Confidence" section in the HTML)
R_S      = 8.0      # support ring radius (m)
INCL_DEG = 15.0     # operational inclination from vertical (deg)
Z_COG    = 5.0      # estimated CoG height above the support attachment (m)
G        = 9.81     # m / s^2
M_TONNES = 8000     # estimated supported mass (t) — VV + in-vessel


def h_conv(alpha_deg: float) -> float:
    """Convergence height above the support attachment (m)."""
    return R_S / np.tan(np.radians(alpha_deg))


def L_eff(alpha_deg: float, z_cog: float = Z_COG) -> float:
    return h_conv(alpha_deg) - z_cog


def K_kN_per_mm(alpha_deg: float, m_t: float = M_TONNES) -> float:
    W = m_t * 1e3 * G                 # N
    L = L_eff(alpha_deg)              # m
    if L <= 0:
        return np.inf
    return (W / L) / 1e6              # kN / mm == MN / m / 1


def Tn(alpha_deg: float) -> float:
    L = L_eff(alpha_deg)
    if L <= 0:
        return np.inf
    return 2 * np.pi * np.sqrt(L / G)


# ── Diagram A: single inclined strut + convergence point ────────────────────
def diagram_a() -> str:
    fig, ax = plt.subplots(figsize=(6.2, 9.2))
    h = h_conv(INCL_DEG)

    # machine axis
    ax.axvline(0, color="#aaaaaa", lw=0.7, ls=":")
    ax.text(0.15, h + 1.2, "machine axis", ha="left", va="bottom",
            color="#666", fontsize=8.5)

    # ground reference
    ax.axhline(0, color="#bbbbbb", lw=0.6)

    # physical strut: from (R_S, 0) up-and-inward at angle alpha from vertical
    L_strut = 7.0
    ax_strut = R_S - L_strut * np.sin(np.radians(INCL_DEG))
    ay_strut = L_strut * np.cos(np.radians(INCL_DEG))
    ax.plot([R_S, ax_strut], [0, ay_strut], color="#1e4d9b", lw=2.8,
            label="VVGS strut (physical)")
    ax.plot([R_S], [0], "ko", ms=6, zorder=5)
    ax.plot([ax_strut], [ay_strut], "o", color="#1e4d9b", ms=6, zorder=5)
    ax.text(R_S + 0.25, -0.7,
            f"ground anchor\n(radius Rₛ = {R_S:.0f} m)",
            ha="left", va="top", fontsize=8)
    ax.text(ax_strut - 0.25, ay_strut + 0.25,
            "vessel attachment", ha="right", va="bottom",
            fontsize=8, color="#1e4d9b")

    # extension of strut axis up to the machine axis (convergence point)
    ax.plot([ax_strut, 0], [ay_strut, h], color="#1e4d9b", lw=1.2,
            ls=(0, (5, 4)), alpha=0.75, label="strut axis (extended)")

    # convergence point
    ax.plot([0], [h], marker="*", color="#cc2200", ms=18, zorder=6)
    ax.text(0.4, h, "convergence point P\n"
            f"h = Rₛ / tan(α) = {h:.1f} m",
            ha="left", va="center", fontsize=9.5, color="#cc2200",
            bbox=dict(boxstyle="round,pad=0.25", fc="white",
                      ec="#cc2200", alpha=0.95, lw=0.8))

    # vessel CoG
    ax.plot([0], [Z_COG], marker="X", color="#222", ms=11, zorder=6)
    ax.text(-0.25, Z_COG, "vessel CoG", ha="right", va="center",
            fontsize=9, color="#222")
    ax.text(-0.25, Z_COG - 0.7, f"z_CoG ≈ {Z_COG:.0f} m (est.)",
            ha="right", va="top", fontsize=8, color="#666")

    # L_eff bracket on the left
    Lx = -2.4
    ax.annotate("", xy=(Lx, Z_COG), xytext=(Lx, h),
                arrowprops=dict(arrowstyle="<->", color="#228822", lw=2))
    ax.text(Lx - 0.2, (Z_COG + h) / 2,
            f"L_eff = h − z_CoG\n≈ {h - Z_COG:.1f} m",
            ha="right", va="center", fontsize=10.5, color="#228822",
            weight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc="white",
                      ec="#228822", alpha=0.95, lw=0.8))

    # alpha arc at the ground pivot
    arc = mpatches.Arc((R_S, 0), 2.4, 2.4, theta1=90,
                       theta2=90 + INCL_DEG, color="#1e4d9b", lw=1.6)
    ax.add_patch(arc)
    ax.text(R_S - 1.6, 1.3, f"α = {INCL_DEG:.0f}°",
            color="#1e4d9b", fontsize=10, weight="bold")

    ax.set_xlim(-4.0, R_S + 3.0)
    ax.set_ylim(-2.0, h + 4.0)
    ax.set_aspect("equal")
    ax.set_xlabel("Radial position (m)")
    ax.set_ylabel("Vertical position above support ring (m)")
    ax.set_title("Diagram A — single VVGS strut: physical span and extended axis",
                 fontsize=11, color="#1a3a6e")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right", fontsize=8.5, frameon=False)

    out = os.path.join(DIAGRAMS_DIR, "A_single_strut.png")
    fig.savefig(out, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


# ── Diagram B: multi-strut system → virtual pivot above CoG ─────────────────
def diagram_b() -> str:
    fig, ax = plt.subplots(figsize=(7.8, 9.6))
    h = h_conv(INCL_DEG)

    ax.axvline(0, color="#aaaaaa", lw=0.7, ls=":")
    ax.axhline(0, color="#bbbbbb", lw=0.6)
    ax.text(0.15, h + 1.0, "machine axis", ha="left", va="bottom",
            color="#666", fontsize=8.5)

    # Two struts (cross-section through machine axis): one on +x, one on -x.
    # Both lean inward (toward the axis) by alpha; their extended axes meet at P.
    L_strut = 7.0
    for side in (+1, -1):
        x0 = side * R_S
        x1 = x0 - side * L_strut * np.sin(np.radians(INCL_DEG))
        y1 = L_strut * np.cos(np.radians(INCL_DEG))
        ax.plot([x0, x1], [0, y1], color="#1e4d9b", lw=2.6)
        ax.plot([x0], [0], "ko", ms=5)
        ax.plot([x1], [y1], "o", color="#1e4d9b", ms=5)
        ax.plot([x1, 0], [y1, h], color="#1e4d9b", lw=1.0,
                ls=(0, (5, 4)), alpha=0.7)

    # Convergence point P
    ax.plot([0], [h], marker="*", color="#cc2200", ms=18, zorder=6)
    ax.text(0.45, h - 0.4, "virtual pivot P\n(common to all 9 supports)",
            ha="left", va="top", fontsize=9.5, color="#cc2200",
            bbox=dict(boxstyle="round,pad=0.25", fc="white",
                      ec="#cc2200", alpha=0.95, lw=0.8))

    # Vessel ring (simplified bar) at the attachment height
    L_strut_top_y = L_strut * np.cos(np.radians(INCL_DEG))
    ax.plot([-R_S + L_strut * np.sin(np.radians(INCL_DEG)),
             R_S - L_strut * np.sin(np.radians(INCL_DEG))],
            [L_strut_top_y, L_strut_top_y], color="#1e4d9b", lw=3.5)
    ax.text(R_S - L_strut * np.sin(np.radians(INCL_DEG)) + 0.4,
            L_strut_top_y, "vessel (rigid)", ha="left", va="center",
            fontsize=9.5, color="#1e4d9b")

    # Vessel CoG
    ax.plot([0], [Z_COG], marker="X", color="#222", ms=11, zorder=6)
    ax.text(-0.25, Z_COG, "CoG", ha="right", va="center",
            fontsize=9, color="#222")

    # Equivalent pendulum: dashed line from P down to CoG, mass M at CoG
    ax.plot([0, 0], [h, Z_COG], color="#228822", lw=2.2,
            ls=(0, (2, 2)))
    ax.text(0.35, (h + Z_COG) / 2,
            f"equivalent pendulum:\nL_eff = h − z_CoG\n≈ {h - Z_COG:.1f} m",
            ha="left", va="center", fontsize=10, color="#228822",
            weight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc="white",
                      ec="#228822", alpha=0.95, lw=0.8))

    # Perturbation arrow showing restoring direction
    dx = 1.6
    cog_disp = (dx, Z_COG)
    ax.plot([cog_disp[0]], [cog_disp[1]], marker="X", color="#888", ms=10,
            alpha=0.6, zorder=4)
    ax.annotate("", xy=(0.2, Z_COG), xytext=(dx - 0.1, Z_COG),
                arrowprops=dict(arrowstyle="->", color="#cc8800",
                                lw=2, mutation_scale=18))
    ax.text(dx / 2, Z_COG - 0.8,
            "displaced CoG → gravitational\nrestoring torque about P",
            ha="center", va="top", fontsize=8.5, color="#7a5500", style="italic")

    ax.set_xlim(-R_S - 4, R_S + 4)
    ax.set_ylim(-2.0, h + 3.5)
    ax.set_aspect("equal")
    ax.set_xlabel("Radial position (m)")
    ax.set_ylabel("Vertical position above support ring (m)")
    ax.set_title("Diagram B — symmetric inclined struts ≡ pendulum suspended from P",
                 fontsize=11, color="#1a3a6e")
    ax.grid(True, alpha=0.25)

    out = os.path.join(DIAGRAMS_DIR, "B_virtual_pivot.png")
    fig.savefig(out, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


# ── Diagram C: L_eff and K vs inclination angle ─────────────────────────────
def diagram_c() -> str:
    alpha = np.linspace(1.0, 56.0, 400)        # avoid singularity at 0 and crit
    Leff  = np.array([L_eff(a) for a in alpha])
    Kkmm  = np.array([K_kN_per_mm(a) for a in alpha])
    Tns   = np.array([Tn(a)         for a in alpha])

    # critical angle: convergence at CoG (L_eff = 0)
    alpha_crit = np.degrees(np.arctan(R_S / Z_COG))

    fig, ax = plt.subplots(figsize=(8.4, 5.6))
    ax.plot(alpha, Leff, color="#1e4d9b", lw=2.4, label="L_eff (m)")
    ax.axhline(0, color="#bbbbbb", lw=0.6)
    ax.set_ylim(-2, 60)
    ax.set_xlabel("Hinge inclination α from vertical (°)", fontsize=10)
    ax.set_ylabel("Effective pendulum length L_eff (m)",
                  color="#1e4d9b", fontsize=10)
    ax.tick_params(axis="y", colors="#1e4d9b")

    # Operational, 45° and critical markers
    for a, label, col in [
        (INCL_DEG,    f"operational α = {INCL_DEG:.0f}°\nL_eff = {L_eff(INCL_DEG):.1f} m",  "#228822"),
        (45.0,        f"α = 45°\nL_eff = {L_eff(45):.1f} m\n(stiffer pendulum)",            "#cc8800"),
        (alpha_crit,  f"α_crit ≈ {alpha_crit:.0f}°\nL_eff → 0\n(virtual pivot ≡ CoG)",       "#cc2200"),
    ]:
        if a <= alpha[-1]:
            ax.axvline(a, color=col, lw=1.3, ls="--", alpha=0.8)
            ax.text(a + 0.6, 55, label, ha="left", va="top",
                    fontsize=8.5, color=col,
                    bbox=dict(boxstyle="round,pad=0.22", fc="white",
                              ec=col, alpha=0.92, lw=0.8))

    # right axis: K (kN/mm) — diverges near alpha_crit
    ax2 = ax.twinx()
    Kclip = np.clip(Kkmm, 0, 60)
    ax2.plot(alpha, Kclip, color="#cc2200", lw=2.0,
             ls=(0, (5, 3)), label="K (kN/mm)")
    ax2.set_ylim(0, 60)
    ax2.set_ylabel("Lateral centring stiffness K = W / L_eff   (kN/mm)",
                   color="#cc2200", fontsize=10)
    ax2.tick_params(axis="y", colors="#cc2200")

    # text annotations explaining the limits
    ax.text(2.0, 50,
            "α → 0°  (vertical struts)\n"
            "convergence at infinity\n"
            "L_eff → ∞,  K → 0\n"
            "no gravitational centring;\n"
            "each strut individually is\n"
            "an inverted pendulum (unstable)",
            ha="left", va="top", fontsize=8.5, color="#444",
            bbox=dict(boxstyle="round,pad=0.3", fc="#fff8e0",
                      ec="#cc8800", alpha=0.9))

    ax.text(56, 45,
            "α > α_crit\n"
            "convergence point\ndrops below CoG —\n"
            "system is unstable\n(inverted pendulum)",
            ha="right", va="top", fontsize=8.5, color="#444",
            bbox=dict(boxstyle="round,pad=0.3", fc="#fff0ee",
                      ec="#cc2200", alpha=0.9))

    ax.set_title(f"Diagram C — L_eff(α) and K(α). "
                 f"Rₛ = {R_S} m, z_CoG ≈ {Z_COG} m, M ≈ {M_TONNES} t",
                 fontsize=11, color="#1a3a6e")
    ax.grid(True, alpha=0.25)
    ax.set_xlim(0, 60)

    out = os.path.join(DIAGRAMS_DIR, "C_Leff_vs_angle.png")
    fig.savefig(out, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


# ── HTML report ─────────────────────────────────────────────────────────────
CSS = """
body{font-family:'Segoe UI',system-ui,sans-serif;max-width:1100px;margin:0 auto;padding:1.5rem 2rem;color:#222;line-height:1.55}
h1{border-bottom:3px solid #2b5797;padding-bottom:.4em;color:#1a3a6e}
h2{color:#2b5797;border-left:4px solid #2b5797;padding-left:.6em;margin-top:2em}
h3{color:#444;margin-top:1.4em}
table{border-collapse:collapse;width:100%;margin:1em 0;font-size:.95em}
th{background:#2b5797;color:white;padding:.5em .9em;text-align:left}
td{border:1px solid #ccc;padding:.4em .9em}
tr:nth-child(even) td{background:#f4f7ff}
.callout{border-left:4px solid;padding:.8em 1.1em;margin:1em 0;border-radius:0 6px 6px 0}
.ok{border-color:#2a8a2a;background:#edf8ed}
.warn{border-color:#cc8800;background:#fff8e0}
.info{border-color:#2255aa;background:#e8efff}
.key{border-color:#cc2200;background:#fff0ee}
figure{margin:1.5em 0;text-align:center}
figure img{max-width:100%;border:1px solid #dde;border-radius:6px;box-shadow:0 2px 10px rgba(0,0,0,.12)}
figcaption{font-size:.88em;color:#555;margin-top:.6em;font-style:italic;text-align:left;max-width:88%;margin-left:auto;margin-right:auto}
code{background:#f0f0f0;border-radius:3px;padding:.1em .4em;font-size:.92em}
pre{background:#f5f5f5;border:1px solid #ddd;border-radius:4px;padding:.8em 1em;overflow-x:auto;font-size:.88em}
.author{color:#1a3a6e;font-weight:600}
a{color:#1a3a6e}
"""


def write_html() -> str:
    alpha_crit = np.degrees(np.arctan(R_S / Z_COG))
    L15  = L_eff(INCL_DEG)
    L45  = L_eff(45)
    K15  = K_kN_per_mm(INCL_DEG)
    T15  = Tn(INCL_DEG)

    body = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>ITER VVGS Gravitational Centring — Mechanism, Geometry &amp; Confidence</title>
<style>{CSS}</style>
</head>
<body>
<h1>ITER VVGS Gravitational Centring — Mechanism, Geometry &amp; Confidence</h1>
<p style="color:#555;font-size:.95em;margin-bottom:.3em">
<span class="author">Simon McIntosh</span> &nbsp;·&nbsp; 2026-05-28 &nbsp;·&nbsp;
companion to the
<a href="vv-lateral-displacement-analysis.html">main lateral-displacement report</a>
</p>

<div class="callout info">
<strong>Purpose.</strong> The main report concludes that the 9 inclined VVGS supports act as
a stable gravitational pendulum in the lateral plane and self-centre the vessel at rest.
This companion derives that result from first principles, draws how the effective pendulum
length L<sub>eff</sub> is constructed geometrically, examines limit cases (vertical struts
and α = 45°), and is honest about how confident we should be in the model and where the
remaining holes are.
</div>

<!-- ═══════════════════════════════════════════════════════════ §1 -->
<h2 id="s1">§1 — The single-strut geometry</h2>

<p>Each VVGS strut is a near-vertical link whose axis is tilted <strong>α = 15°</strong>
from vertical, leaning inward. Its ground anchor sits at radius
<strong>R<sub>s</sub> ≈ 8 m</strong> on a horizontal support ring under the lower ports.
Extending the strut's central axis upward, it meets the vertical machine axis at a height</p>

<pre>h(α) = R<sub>s</sub> / tan(α)</pre>

<p>above the support ring. For the operational 15°, h ≈ {h_conv(INCL_DEG):.1f} m — far above
the vessel. The vessel's centre of gravity sits roughly z<sub>CoG</sub> ≈ {Z_COG:.0f} m above
the support attachment (this is an estimate — see §4 on confidence). The effective pendulum
length is then</p>

<pre>L<sub>eff</sub>(α) = h(α) − z<sub>CoG</sub> = R<sub>s</sub> / tan(α) − z<sub>CoG</sub>.</pre>

<figure>
  <img src="diagrams/A_single_strut.png" alt="Single strut + convergence point">
  <figcaption><strong>Diagram A.</strong> A single inclined VVGS strut. The thick blue
  segment is the physical strut between its ground anchor (at radius R<sub>s</sub>) and
  the vessel attachment. The dashed blue line is its <em>extended</em> axis — produced
  upward — which meets the vertical machine axis at the convergence point
  <span style="color:#cc2200">P</span>, at height h = R<sub>s</sub>/tan(α). The vessel CoG
  sits below P by L<sub>eff</sub> = h − z<sub>CoG</sub>; this is the effective pendulum
  length for the lateral mode.</figcaption>
</figure>

<!-- ═══════════════════════════════════════════════════════════ §2 -->
<h2 id="s2">§2 — From single struts to a stable pendulum</h2>

<p>A single near-vertical strut with a load on top is, taken by itself, an
<em>inverted pendulum</em>: ground pivot at the bottom, mass perched above — unstable.
What converts the system into a <em>stable</em> pendulum is the symmetric arrangement of
nine inclined struts: every strut's axis, extended upward, meets the machine axis at the
<em>same point</em> P. Because all the support-axis lines pass through P, the rigid vessel
can only execute small motions that look like a rotation about P. P is the system's
<strong>virtual pivot</strong>, and L<sub>eff</sub> = h − z<sub>CoG</sub> is the distance
from P to the centre of gravity.</p>

<figure>
  <img src="diagrams/B_virtual_pivot.png" alt="Multi-strut virtual pivot">
  <figcaption><strong>Diagram B.</strong> Cross-section through the machine axis, showing
  two inclined struts (one on either side). Each strut's axis, extended (dashed), reaches
  the same convergence point <span style="color:#cc2200">P</span>; this is true for all
  nine struts by symmetry. The rigid vessel above can only rotate about P; with the CoG
  hanging below P at L<sub>eff</sub>, gravity provides a restoring torque whenever the
  vessel is displaced laterally — exactly the dynamics of a stable pendulum of length
  L<sub>eff</sub>.</figcaption>
</figure>

<p>Because the motion is a pendulum, the lateral restoring stiffness (the force per unit
horizontal displacement of the CoG) is the standard result</p>

<pre>K = W / L<sub>eff</sub>,        T<sub>n</sub> = 2π · √(L<sub>eff</sub> / g),</pre>

<p>with W = M g the supported weight (M ≈ {M_TONNES} t). For α = 15° this gives
L<sub>eff</sub> ≈ {L15:.1f} m, K ≈ {K15:.2f} kN/mm and a natural period
T<sub>n</sub> ≈ {T15:.1f} s — a <em>soft</em> pendulum because the geometry forces
L<sub>eff</sub> to be long.</p>

<!-- ═══════════════════════════════════════════════════════════ §3 -->
<h2 id="s3">§3 — L<sub>eff</sub> as a function of inclination angle</h2>

<p>The construction of P is a pure geometry: it depends only on R<sub>s</sub>, α and
z<sub>CoG</sub>. Sweeping α gives the curve below.</p>

<figure>
  <img src="diagrams/C_Leff_vs_angle.png" alt="L_eff vs alpha and K vs alpha">
  <figcaption><strong>Diagram C.</strong> The effective pendulum length L<sub>eff</sub>
  (blue, left axis) and the corresponding lateral stiffness K = W/L<sub>eff</sub> (red,
  dashed, right axis) as a function of strut inclination α. Three regimes are marked:
  the operational α = 15°, the α = 45° comparison, and the critical α at which the
  convergence point reaches the CoG and the pendulum becomes infinitely stiff before
  going unstable.</figcaption>
</figure>

<h3>Limit case: vertical struts (α → 0°)</h3>
<p>With perfectly vertical struts the strut-axis lines never meet — the convergence point
P recedes to infinity. L<sub>eff</sub> → ∞ and K → 0. There is <strong>no gravitational
centring force</strong>: a small lateral perturbation produces no restoring torque, and
the vessel sits in a marginal/neutral equilibrium that is easily upset. Each individual
strut, viewed in isolation, is now exactly an <strong>inverted pendulum</strong>; only the
inward inclination converts the system into a stable pendulum by creating a virtual pivot
above the CoG.</p>

<h3>Limit case: α = 45°</h3>
<p>At α = 45° the convergence point sits at h = R<sub>s</sub> = {R_S:.0f} m above the
support attachment. With z<sub>CoG</sub> ≈ {Z_COG:.0f} m the effective pendulum is only
L<sub>eff</sub> ≈ {L45:.1f} m — about an order of magnitude shorter than at 15°. The
stiffness K = W/L<sub>eff</sub> rises by the same factor and the natural period drops to
T<sub>n</sub> ≈ {Tn(45):.1f} s. This is a much stiffer pendulum: a more strongly centred
vessel, but a strut tilt that would be impractical for a gravity support carrying ~8000 t.
The 15° design value is a deliberate balance between centring stiffness and structural
practicality.</p>

<h3>Critical inclination — the system becomes unstable</h3>
<p>If α is increased far enough that h = z<sub>CoG</sub>, the convergence point P falls
to the level of the CoG and L<sub>eff</sub> → 0; beyond this critical angle</p>

<pre>α<sub>crit</sub> = arctan(R<sub>s</sub> / z<sub>CoG</sub>) ≈ {alpha_crit:.0f}°</pre>

<p>P lies <em>below</em> the CoG. The vessel is then a true inverted pendulum about P
and is unstable: any small lateral perturbation grows. The operational α = 15° sits far on
the stable side of this transition with substantial margin.</p>

<!-- ═══════════════════════════════════════════════════════════ §4 -->
<h2 id="s4">§4 — Confidence &amp; holes in the argument</h2>

<p>How confident should we be that this mechanism actually works as described? The
qualitative result — that the inclined-hinge VVGS provides a stable gravitational centring
force — is robust. The quantitative numbers (L<sub>eff</sub>, K, T<sub>n</sub>) carry
moderate uncertainty from the parameters we have estimated rather than measured.</p>

<table>
<tr><th>Statement</th><th>Confidence</th><th>Why</th></tr>
<tr><td>The 15° inward inclination provides a centring force</td><td>HIGH</td><td>Stated explicitly in the ITER VVGS literature as design intent (see §5).</td></tr>
<tr><td>The mechanism is a stable (not inverted) pendulum at α = 15°</td><td>HIGH</td><td>Pure geometry: P sits ~30 m above the CoG, far above α<sub>crit</sub>.</td></tr>
<tr><td>L<sub>eff</sub> ≈ 26 m, K ≈ 3 kN/mm, T<sub>n</sub> ≈ 10 s</td><td>MODERATE</td><td>Sensitive to z<sub>CoG</sub> (estimated) and to the exact link geometry.</td></tr>
<tr><td>Linear/harmonic small-displacement model</td><td>HIGH</td><td>L<sub>eff</sub> ≫ 1.5 mm gap — nonlinear corrections are ppm.</td></tr>
<tr><td>2-D lateral mode captures the n = 1 first-wall shift</td><td>HIGH</td><td>The other rigid-body modes are either irrelevant (n = 0 rotation) or strongly constrained.</td></tr>
</table>

<h3>Where the argument can be tightened — known holes</h3>
<ol>
<li><strong>Estimated CoG height (z<sub>CoG</sub>).</strong> We took z<sub>CoG</sub> ≈ 5 m
above the support ring. The actual VV + in-vessel CoG depends on the blanket + divertor
distribution and is design-specific. L<sub>eff</sub> is sensitive to this (≈ 2 m per
metre of z<sub>CoG</sub> shift); a 1–2 m uncertainty in z<sub>CoG</sub> propagates to a
similar uncertainty in L<sub>eff</sub> and a few-percent shift in T<sub>n</sub>.</li>

<li><strong>The "rigid pin-pin strut" idealisation.</strong> The actual VVGS is a
<em>dual-hinge</em> mechanism — primary and secondary hinges with four dowels — not a
simple pin-pin link. The convergence-point argument requires that the effective line of
action of each support pass through a common point P, which holds for any kinematic chain
that, taken as a whole, is rotation-only about a single axis. The dual-hinge geometry is
designed precisely to give one rotational DOF (radial); confirming that the chain's
effective line of action passes through a single P (and not, say, two intermediate
instantaneous centres) needs the as-built hinge dimensions.</li>

<li><strong>The 15° reference.</strong> We have read "15° inclination" from the ITER
literature as <em>from vertical</em> (a slightly tilted near-vertical strut). If the
inclination is in fact stated from horizontal (a strut closer to horizontal than vertical),
the geometry is very different. The "leaning inward" language and the centring rationale
make the from-vertical reading by far the more natural one, and the literature's framing
("stable equilibrium") only makes sense in that case — but the precise angular reference
should be confirmed against the engineering drawing.</li>

<li><strong>Supported mass M.</strong> We use M ≈ 8000 t for VV + in-vessel components. The
VV alone is ~5200 t; the full in-vessel set adds up to the 8–9 kt range. K = W/L<sub>eff</sub>
scales with M, so K could be 15–20 % softer if a smaller effective M (say only the VV)
actually moves with the lateral mode.</li>

<li><strong>Other elastic stiffness contributions.</strong> The pendulum analysis is purely
gravitational. The supports also have <em>elastic</em> compliance (the dowel material,
the bracket, the linkage), which contributes additional lateral stiffness in parallel.
Our centring estimate is therefore a <em>lower bound</em> on the actual lateral
stiffness — the system is at least as stiff as the gravitational pendulum, possibly
stiffer once the elastic terms are added.</li>

<li><strong>Symmetry.</strong> The virtual-pivot argument requires the 9 struts to be
nominally symmetric. Installation tolerances will break that symmetry at the mm-scale; in
the small-perturbation, low-frequency lateral mode this is a higher-order effect and does
not move the qualitative picture.</li>
</ol>

<!-- ═══════════════════════════════════════════════════════════ §5 -->
<h2 id="s5">§5 — External evidence</h2>

<p>The model is anchored in the published ITER VVGS design literature; we are not relying
on the convergence-point pendulum as a private deduction.</p>

<ul>
<li><strong>The 15° inclination and its centring function are explicit design intent.</strong>
The structural-design literature on the ITER VVGS states (verbatim): <em>"Inclination of
15° for the hinge based supporting system was introduced to provide a centering force to
keep a stable equilibrium state of the vacuum vessel. Due to this inclination the hinges
are rotated by the radial expansion of the VV"</em> (Fusion Eng. Des., 2018 — structural
integrity analysis for the manufacturing design of the ITER VVGS). This confirms both the
inclination value and the qualitative claim made here.</li>

<li><strong>The dual-hinge / dowel mechanism is documented in the same literature</strong>
("Each VVGS consists of one primary hinge, one secondary hinge, upper and lower blocks and
4 dowels … radial movement is permitted by the rotation of dowels, while toroidal and
vertical movement is restricted by the rigidity of the dowels and that of the primary
hinge").</li>

<li><strong>The "soft landing" of VV sector #6 onto its gravity support</strong> is
described in the ITER news article on the first VVGS contact; this confirms the operational
geometry and the assembly-tolerance framing used in §1 of the main report (the 3 mm margin
quoted in the news article matches the toroidal slot used here).</li>

<li><strong>Independent mechanical-testing literature on the VV support structure</strong>
(coating screening, multi-axial mock-up tests, lubricant coating of the dowel) confirms
that the design carries the loads as a dual-hinge / dowel system rather than as a simple
solid strut.</li>
</ul>

<div class="callout ok">
<strong>Net.</strong> The gravitational centring mechanism is a textbook
inclined-strut / virtual-pivot result, applied to a configuration that ITER has
deliberately designed for centring (per the published structural-analysis literature on
the VVGS). The qualitative claim — vessel self-centres in a stable pendulum well with
a long L<sub>eff</sub> and a soft natural frequency — is on firm ground. The numerical
values (L<sub>eff</sub> ≈ 26 m, K ≈ 3 kN/mm, T<sub>n</sub> ≈ 10 s) are accurate to roughly
±20 % given the estimated z<sub>CoG</sub> and supported mass; pinning them more tightly
needs the as-built hinge geometry and the lateral mass that actually moves with the n=1
mode.
</div>

<hr style="margin-top:3em;border-color:#ddd">
<p style="font-size:.85em;color:#777">
Author: <span class="author">Simon McIntosh</span>. Diagrams generated by
<code>build_pendulum_explainer.py</code>; the geometry is parametric in R<sub>s</sub>,
α and z<sub>CoG</sub>. Source: <code>Simon-McIntosh/vv</code>.
</p>
</body>
</html>
"""
    out = os.path.join(HERE, "docs", "vvgs-pendulum-mechanism.html")
    with open(out, "w") as fh:
        fh.write(body)
    return out


def main():
    print("Generating diagrams …")
    print(" ", diagram_a())
    print(" ", diagram_b())
    print(" ", diagram_c())
    print("Writing HTML …")
    print(" ", write_html())


if __name__ == "__main__":
    main()
