"""
Build the Word version of the ITER VV lateral-displacement report.

This is a structural replica of docs/vv-lateral-displacement-analysis.html. Tables use
a minimal horizontal-rule style (line above and below the header, and a closing rule at
the bottom — no row shading, no vertical lines). Forced-excursion GIFs are placed
side-by-side (no strip images). All section content, callouts and figure captions mirror
the HTML.

    uv run python build_docx.py     ->  docs/vv-lateral-displacement-analysis.docx
"""
from __future__ import annotations
import os
import numpy as np
from docx import Document
from docx.shared import Pt, RGBColor, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

HERE = os.path.dirname(os.path.abspath(__file__))
DOCS = os.path.join(HERE, "docs")

# palette (matches the HTML CSS)
BLUE      = RGBColor(0x2b, 0x57, 0x97)
DARKBLUE  = RGBColor(0x1a, 0x3a, 0x6e)
GREY      = RGBColor(0x55, 0x55, 0x55)
WHITE     = RGBColor(0xff, 0xff, 0xff)
TABLE_RULE = "888888"        # gray for the horizontal table rules
CALLOUT = {                  # (left-border hex, fill hex) — matches the HTML
    "ok":   ("2a8a2a", "edf8ed"),
    "warn": ("cc8800", "fff8e0"),
    "info": ("2255aa", "e8efff"),
    "key":  ("cc2200", "fff0ee"),
}


# ── low-level OOXML helpers ─────────────────────────────────────────────────
def _shade(cell, fill_hex):
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:fill"), fill_hex)
    cell._tc.get_or_add_tcPr().append(shd)


def _set_cell_borders(cell, top=None, bottom=None, left=None, right=None):
    """Set per-edge borders. Each arg is (sz, color_hex) or None (= nil)."""
    tcPr = cell._tc.get_or_add_tcPr()
    borders = OxmlElement("w:tcBorders")
    for edge, spec in [("top", top), ("bottom", bottom),
                       ("left", left), ("right", right)]:
        el = OxmlElement(f"w:{edge}")
        if spec is None:
            el.set(qn("w:val"), "nil")
        else:
            sz, color = spec
            el.set(qn("w:val"), "single")
            el.set(qn("w:sz"), str(sz))
            el.set(qn("w:color"), color)
        borders.append(el)
    tcPr.append(borders)


def _set_borders_legacy(cell, edges, color="cccccc", sz=4):
    """Add borders to specific edges (preserves any existing nil settings on others)."""
    tcPr = cell._tc.get_or_add_tcPr()
    borders = OxmlElement("w:tcBorders")
    for edge in edges:
        el = OxmlElement(f"w:{edge}")
        el.set(qn("w:val"), "single")
        el.set(qn("w:sz"), str(sz))
        el.set(qn("w:color"), color)
        borders.append(el)
    tcPr.append(borders)


def _no_space(p):
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(2)


# ── content builders ────────────────────────────────────────────────────────
def heading2(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(16)
    p.paragraph_format.space_after = Pt(4)
    r = p.add_run(text)
    r.bold = True
    r.font.size = Pt(14)
    r.font.color.rgb = BLUE
    # blue bottom border on the heading paragraph
    pPr = p._p.get_or_add_pPr()
    pbdr = OxmlElement("w:pBdr")
    bottom = OxmlElement("w:bottom")
    bottom.set(qn("w:val"), "single"); bottom.set(qn("w:sz"), "6")
    bottom.set(qn("w:space"), "2"); bottom.set(qn("w:color"), "2b5797")
    pbdr.append(bottom); pPr.append(pbdr)
    return p


def heading3(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(10)
    p.paragraph_format.space_after = Pt(2)
    r = p.add_run(text)
    r.bold = True
    r.font.size = Pt(11.5)
    r.font.color.rgb = RGBColor(0x44, 0x44, 0x44)
    return p


def callout(doc, kind, runs, title=None):
    """Single-cell shaded table with a coloured left border — matches HTML callouts."""
    border_hex, fill_hex = CALLOUT[kind]
    tbl = doc.add_table(rows=1, cols=1)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    cell = tbl.cell(0, 0)
    _shade(cell, fill_hex)
    _set_borders_legacy(cell, ["left"], color=border_hex, sz=24)
    _set_borders_legacy(cell, ["top", "bottom", "right"], color=fill_hex, sz=4)
    p = cell.paragraphs[0]
    _no_space(p)
    if title:
        tr = p.add_run(title + "  ")
        tr.bold = True
        tr.font.color.rgb = RGBColor.from_string(border_hex.upper())
    for text, bold in runs:
        r = p.add_run(text)
        r.bold = bold
        r.font.size = Pt(10)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)
    return tbl


def body(doc, runs):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(6)
    for text, bold in runs:
        r = p.add_run(text)
        r.bold = bold
        r.font.size = Pt(10.5)
    return p


def pre(doc, text):
    """Monospace block (matches HTML <pre>)."""
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(6)
    p.paragraph_format.left_indent = Inches(0.25)
    r = p.add_run(text)
    r.font.name = "Consolas"
    r.font.size = Pt(9.5)
    r.font.color.rgb = RGBColor(0x33, 0x33, 0x33)
    return p


def table(doc, headers, rows):
    """Minimal-rule table: horizontal lines above/below the header and at the bottom only.
    No row shading, no vertical lines, no inter-row separators."""
    t = doc.add_table(rows=1, cols=len(headers))
    t.alignment = WD_TABLE_ALIGNMENT.CENTER

    # Header row — top + bottom horizontal rules, nothing else
    hdr = t.rows[0].cells
    for j, h in enumerate(headers):
        _set_cell_borders(hdr[j], top=(8, TABLE_RULE), bottom=(6, TABLE_RULE),
                          left=None, right=None)
        p = hdr[j].paragraphs[0]; _no_space(p)
        r = p.add_run(h); r.bold = True; r.font.color.rgb = DARKBLUE; r.font.size = Pt(10)

    n = len(rows)
    for i, row in enumerate(rows):
        cells = t.add_row().cells
        is_last = (i == n - 1)
        for j, val in enumerate(row):
            _set_cell_borders(
                cells[j],
                top=None,
                bottom=((8, TABLE_RULE) if is_last else None),
                left=None,
                right=None,
            )
            p = cells[j].paragraphs[0]; _no_space(p)
            r = p.add_run(str(val)); r.font.size = Pt(10)
            if j == 0:
                r.bold = True
    doc.add_paragraph().paragraph_format.space_after = Pt(4)
    return t


def figure(doc, img, caption, width=6.4):
    if not os.path.exists(img):
        body(doc, [(f"[missing figure: {img}]", False)])
        return
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run().add_picture(img, width=Inches(width))
    c = doc.add_paragraph(); c.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cr = c.add_run(caption); cr.italic = True; cr.font.size = Pt(8.5); cr.font.color.rgb = GREY
    c.paragraph_format.space_after = Pt(8)


def side_by_side_figures(doc, items, width_each=3.0):
    """items = [(img_path, caption_runs), ...]; caption_runs = list of (text, bold).
    Renders as a borderless 2-cell table with image + caption stacked in each cell."""
    t = doc.add_table(rows=1, cols=len(items))
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, (img, cap_runs) in enumerate(items):
        cell = t.cell(0, i)
        _set_cell_borders(cell, top=None, bottom=None, left=None, right=None)
        p = cell.paragraphs[0]; p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        if os.path.exists(img):
            p.add_run().add_picture(img, width=Inches(width_each))
        c = cell.add_paragraph(); c.alignment = WD_ALIGN_PARAGRAPH.CENTER
        for text, bold in cap_runs:
            cr = c.add_run(text)
            cr.italic = True; cr.bold = bold
            cr.font.size = Pt(8.5); cr.font.color.rgb = GREY
        c.paragraph_format.space_after = Pt(4)
    doc.add_paragraph().paragraph_format.space_after = Pt(8)


# ── assemble the document ───────────────────────────────────────────────────
def main():
    r = np.load(os.path.join(HERE, "data", "rattle_mc_5k.npy"))
    p95, p99, p50 = np.percentile(r, 95), np.percentile(r, 99), np.percentile(r, 50)
    rmax = float(r.max())
    h, e = np.histogram(r, bins=60); mode = 0.5 * (e[h.argmax()] + e[h.argmax() + 1])

    doc = Document()
    doc.styles["Normal"].font.name = "Segoe UI"
    doc.styles["Normal"].font.size = Pt(10.5)

    # ── title ───────────────────────────────────────────────────────────────
    t = doc.add_paragraph()
    tr = t.add_run("ITER Vacuum Vessel — Lateral Displacement Analysis")
    tr.bold = True; tr.font.size = Pt(19); tr.font.color.rgb = DARKBLUE
    pPr = t._p.get_or_add_pPr(); pbdr = OxmlElement("w:pBdr")
    b = OxmlElement("w:bottom"); b.set(qn("w:val"), "single"); b.set(qn("w:sz"), "18")
    b.set(qn("w:space"), "3"); b.set(qn("w:color"), "2b5797"); pbdr.append(b); pPr.append(pbdr)

    auth = doc.add_paragraph()
    ar = auth.add_run("Simon McIntosh"); ar.bold = True; ar.font.size = Pt(11); ar.font.color.rgb = DARKBLUE
    ar2 = auth.add_run("   ·   2026-05-28"); ar2.font.size = Pt(10); ar2.font.color.rgb = GREY

    sub = doc.add_paragraph()
    sr = sub.add_run("5,000-sample Monte Carlo  ·  scipy HiGHS LP  ·  3-DOF rigid-body model  ·  "
                     "metric: max lateral displacement of the VV centre from the gravitational "
                     "centre (n = 1 first-wall shift)")
    sr.font.size = Pt(8.5); sr.font.color.rgb = GREY

    # ── Summary of findings ─────────────────────────────────────────────────
    callout(doc, "info", [
        ("The vacuum vessel rests on 9 inward-inclined gravity supports whose 15° tilt "
         "provides a gravitational centring force in the lateral plane: ", False),
        ("the vessel self-centres at rest and departs from the machine axis only under an "
         "applied lateral force", True),
        (". The natural period of this pendulum is ≈ 10 s.\n\n", False),
        ("• The n = 1 first-wall metric is the maximum lateral displacement of the VV centre "
         "from the machine axis. Nominal envelope 1.55 mm; across the assembly population "
         f"(Monte Carlo, §5) P95 = {p95:.2f} mm, max = {rmax:.2f} mm; at-rest contribution = 0.\n",
         False),
        ("• Start-up (0 → 3 MA in 10 s) is favourable. Passive vessel currents pull the wall "
         "toward the offset toroidal-field magnetic centre; any motion during the ramp "
         "reduces the n ≤ 4 first-wall–field misalignment.\n", False),
        ("• Disruption is the bounding VVGS lateral load case. An MN-scale lateral impulse "
         "over ms gives ~12 mm/s impact velocity at the dowel stops and ~1 MN peak lateral "
         "force on the VVGS — much larger than steady operation.\n", False),
        ("• Gap measurements tighten the bound. Each measured u_i pulls the displacement "
         "polytope toward the centre; with all 9 measured and near-centred the bound "
         "collapses to the 1.55 mm nominal envelope.", False),
    ], title="Summary of findings:")

    # ── §1 — Problem & Assumptions ──────────────────────────────────────────
    heading2(doc, "1 — Problem & Assumptions")
    body(doc, [
        ("The vacuum vessel rests on ", False),
        ("nine gravity supports (VVGS)", True),
        (" equally spaced toroidally beneath the lower ports, on a ring of radius R_s ≈ 8 m. "
         "Each VVGS is a ", False),
        ("dual-hinge mechanism inclined at 15° from vertical", True),
        (", leaning inward toward the machine axis. Radial motion of the vessel is permitted "
         "by rotation of the dowels (this accommodates thermal expansion between assembly "
         "and operation/baking); toroidal and vertical motion are restrained by the dowels' "
         "rigidity. The toroidal restraint is a ", False),
        ("±1.5 mm (3 mm total) dowel slot", True),
        (".", False),
    ])
    table(doc, ["Parameter", "Value", "Note"], [
        ["Number of supports", "9", "Equally spaced toroidally"],
        ["Support ring radius R_s", "8 m", ""],
        ["Toroidal slot per support", "±1.5 mm (3 mm total)", "Assembly tolerance (see below)"],
        ["Hinge inclination from vertical", "15°", "Inward (centring)"],
        ["Supported mass M", "≈ 8000 t", "VV + in-vessel components"],
    ])

    heading3(doc, "Assembly assumption — the source of mobility")
    body(doc, [
        ("At each VVGS the connecting pin is oriented in the toroidal direction and runs "
         "through interleaved vessel-side and ground-side tabs. ", False),
        ("The holes are tight on the pin (no radial play); the toroidal clearance is the "
         "spacing between adjacent tabs along the pin.", True),
        (" That inter-tab spacing is required to bring the parts together so the tabs can "
         "be aligned and the pin can be inserted — without it the support cannot be "
         "assembled. The same spacing then permits the vessel-side tabs to slide along the "
         "pin (toroidally) once assembled, giving ", False),
        ("±1.5 mm (3 mm total)", True),
        (" of lateral mobility per support. If the joints are ", False),
        ("left unshimmed", True),
        (" — the case treated throughout this report — this assembly clearance is what "
         "drives the analysis.", False),
    ])
    body(doc, [
        ("We model the as-built state with the conservative prior that each support's pin "
         "offset u_i is ", False),
        ("independent across the 9 supports", True),
        (" and ", False),
        ("uniformly distributed on [−1.5 mm, +1.5 mm]", True),
        (". This is the prior used in the Monte Carlo of §5. (Shimming a support fixes its "
         "u_i to a measured value and removes that support's contribution to the mobility — "
         "see §6.)", False),
    ])

    heading3(doc, "Gravitational centring")
    body(doc, [
        ("The inward 15° inclination is deliberate. The nine support axes converge above the "
         "vessel centre of gravity; for the lateral (n = 1) mode the vessel therefore "
         "behaves as if suspended from that convergence point — an ", False),
        ("inclined-hinge gravitational pendulum", True),
        (" — and the lateral plane has a parabolic potential well centred on the nominal "
         "position. The effective pendulum length is L_eff ≈ R_s / tan(15°) − z_CoG ≈ 26 m "
         "(sensitive to angle: ~2 m per degree).", False),
    ])
    table(doc, ["Quantity", "Value"], [
        ["Lateral centring stiffness", "K ≈ 3.0 kN / mm"],
        ["Natural period", "T_n ≈ 10 s   (f ≈ 0.1 Hz)"],
        ["Force to push the vessel to a ±1.5 mm stop", "F_stop ≈ 4.5 kN"],
        ["Rest position with no lateral force", "q* = 0 (centred)"],
    ])
    callout(doc, "info", [
        ("Without an applied lateral force the vessel sits at the gravitational centre, "
         "regardless of the assembly offsets. The pins simply occupy their respective u_i "
         "positions in the slots; the vessel itself is on-axis.", False),
    ], title="Consequence.")

    # ── §2 — Constraint Model, Polytope & Linear Programming ────────────────
    heading2(doc, "2 — Constraint Model, Polytope & Linear Programming")
    body(doc, [
        ("We model the vacuum vessel as a rigid body with three in-plane degrees of "
         "freedom — two horizontal translations Δx, Δy and a rotation Δθ about the vertical "
         "axis (collected as ", False),
        ("q = [Δx, Δy, Δθ]", True),
        ("). Each of the nine supports imposes a single linear inequality constraint: the "
         "toroidal slide of the vessel relative to its dowel must stay within the ±1.5 mm "
         "slot. For support i at toroidal angle φ_i = π/2 − 2πi/9 on the support ring at "
         "radius R = 8 m, the slide is", False),
    ])
    pre(doc,
        "  δᵢ = −sin(φᵢ)·Δx + cos(φᵢ)·Δy + R·Δθ   ≡   A[i, :] · q ,\n"
        "      |uᵢ + δᵢ| ≤ 1.5 mm     for all i = 1 … 9,")
    body(doc, [
        ("so the displacement is bounded by ", False),
        ("18 linear inequalities", True),
        (" (one upper and one lower bound per support). The 9×3 constraint matrix has ", False),
        ("AᵀA = diag(4.5, 4.5, 576)", True),
        (" — exactly diagonal, so the three rigid-body DOFs are mutually uncorrelated.", False),
    ])
    callout(doc, "info", [
        ("The set of vessel positions that simultaneously satisfy all 18 inequalities is a "
         "convex region of (Δx, Δy, Δθ) space bounded by flat faces — geometrically a ", False),
        ("polytope", True),
        (" (the three-dimensional generalisation of a polygon). Projected onto the horizontal "
         "plane (Δx, Δy), with Δθ optimised at each direction, it is a closed convex polygon "
         "of feasible lateral VV-centre positions — the ", False),
        ("displacement polytope", True),
        (". Its shape depends on the assembly offsets and its diameter is set by the "
         "±1.5 mm slot.\n\n", False),
        ("A ", False),
        ("linear program (LP)", True),
        (" optimises a linear objective subject to linear inequality constraints — the "
         "canonical method for problems of exactly this kind. We solve, in each direction θ, "
         "the LP that maximises the projection of q on the unit vector (cos θ, sin θ, 0) "
         "subject to the 18 inequalities above; the optimum lies at the polytope vertex "
         "furthest along that direction. Sweeping θ traces the polytope boundary, and the "
         "maximum of ||q[:2]|| over the sweep is our metric: the ", False),
        ("maximum lateral displacement of the VV centre from the gravitational centre", True),
        (" that any applied force can drive the vessel to. We use the HiGHS solver "
         "(scipy.optimize.linprog); each LP solves in milliseconds.", False),
    ], title="Polytope & LP.")

    # ── §3 — State Diagrams ─────────────────────────────────────────────────
    heading2(doc, "3 — State Diagrams")
    body(doc, [
        ("Each panel shows the VV ring at its maximum-displacement position; the red arrow "
         "from the machine axis (black cross) to the displaced VV centre is the "
         "maximum-displacement vector — its direction and length show where and how far the "
         "vessel can be driven by an applied lateral force. The blue shaded polygon around "
         "the machine axis is the displacement polytope (the kinematic envelope of all "
         "reachable VV-centre positions). Displacements are magnified ×500.", False),
    ])
    figure(doc, os.path.join(DOCS, "plots", "vv_states.png"),
           "Left — nominal assembly (u = 0): the polytope is symmetric about the machine "
           "axis and the maximum displacement is 1.55 mm in any direction. Right — worst "
           "observed MC sample: the assembly offsets shift the polytope off-axis, so the "
           "maximum-displacement vector reaches further from the machine axis (2.98 mm). At "
           "each support the toroidal slot (grey, with red ±1.5 mm end-stops) is held at a "
           "constant radial distance from the vessel wall by the inclined linkage — radial "
           "motion is free; the coloured pin in the slot indicates how much of the ±1.5 mm "
           "toroidal travel is taken up (blue = slack, orange = near the limit, red = at the "
           "stop).")

    # ── §4 — Forced-Excursion Envelope ──────────────────────────────────────
    heading2(doc, "4 — Forced-Excursion Envelope")
    body(doc, [
        ("The animations sweep the VV centre through the polytope along its principal "
         "axis — the kinematic envelope of possible lateral motion about the centred rest "
         "position. With no applied force the vessel sits at the machine axis; sufficient "
         "applied force drives it through the envelope. ×500 magnification; the grey "
         "toroidal slot at each support tracks the vessel wall (radial motion is free), and "
         "the coloured pin shows the toroidal-constraint usage (blue = slack, orange = near "
         "limit, red = at stop).", False),
    ])
    side_by_side_figures(doc, [
        (os.path.join(DOCS, "animations", "rattle_worst_case.gif"),
         [("Worst observed MC sample. ", True),
          (f"Offset assembly; maximum forced departure from the machine axis ≈ "
           f"{rmax:.2f} mm — the kinematic ceiling for the worst random assembly drawn from "
           "the §1 uniform prior.", False)]),
        (os.path.join(DOCS, "animations", "rattle_mode.gif"),
         [("Typical as-built assembly ", True),
          (f"(drawn near the distribution mode). Maximum forced departure ≈ {mode:.2f} mm — "
           "the most likely value an as-built machine will actually exhibit.", False)]),
    ], width_each=3.0)

    # ── §5 — Monte Carlo Distribution ───────────────────────────────────────
    heading2(doc, "5 — Monte Carlo Distribution")
    body(doc, [
        ("Five thousand assemblies, with each support's offset drawn independently from "
         "Uniform(−1.5 mm, +1.5 mm). For each assembly the maximum lateral displacement of "
         "the VV centre from the gravitational centre is computed (72-direction LP). "
         "Reproducibly generated by vv_mc_generator.py with fixed seed 20260527.", False),
    ])
    figure(doc, os.path.join(DOCS, "plots", "mc_dashboard.png"),
           "Distribution (left) and CDF (right) of the maximum lateral displacement of the "
           "VV centre from the gravitational centre, over 5000 randomly-drawn assemblies. "
           "The nominal (centred-assembly) value is 1.55 mm; offset assemblies can reach "
           "further from the centre because the polytope is shifted off-axis.")
    table(doc, ["Statistic", "Max lateral displacement (mm)"], [
        ["Nominal (u = 0) — symmetric envelope half-width", "1.55"],
        ["Mode", f"{mode:.2f}"],
        ["Median (P50)", f"{p50:.2f}"],
        ["P90", "2.12"],
        ["P95", f"{p95:.2f}"],
        ["P99", f"{p99:.2f}"],
        ["Max observed (n = 5000)", f"{rmax:.2f}"],
    ])
    callout(doc, "info", [
        ("Quiescent (no lateral force) the vessel sits at the gravitational centre and "
         "contributes zero to the n = 1 first-wall shift. Under sufficient applied force the "
         "vessel can be displaced up to the polytope boundary; the MC quantifies that "
         "kinematic ceiling over the assembly distribution.", False),
    ])

    # ── §6 — Effect of Gap Measurements ─────────────────────────────────────
    heading2(doc, "6 — Effect of Gap Measurements")
    body(doc, [
        ("Each measured gap fixes one u_i to its true value. Because the polytope's offset "
         "from the gravitational centre is governed by the assembly offsets, ", False),
        ("every measured gap directly tightens the conditional displacement bound", True),
        (": each measurement pulls the polytope toward the centre. With all 9 measured and "
         "found near-centred, the conditional maximum displacement collapses to the 1.55 mm "
         "nominal envelope.", False),
    ])
    figure(doc, os.path.join(DOCS, "plots", "partial_measurement.png"),
           "Left — distribution of the maximum lateral displacement; measuring all 9 gaps "
           "reveals one value from this distribution. Right — conditional P95 vs number of "
           "measured sectors, for two scenarios of the measured values: centred (lowest "
           "residual bound) and typical random (mid-range). Because AᵀA is exactly diagonal "
           "the three rigid-body DOFs are uncorrelated, so only the number of measured "
           "sectors matters; adjacent vs spread-out is statistically identical.")
    table(doc, ["Measured sectors (k of 9)", "P95 — centred (mm)", "P95 — typical (mm)"], [
        ["0 (baseline)", f"{p95:.2f}", f"{p95:.2f}"],
        ["1", "≈ 2.26", "≈ 2.26"],
        ["3", "≈ 2.19", "≈ 2.22"],
        ["5", "≈ 2.05", "≈ 2.00"],
        ["9 (all)", "→ 1.55 (nominal env.)", "deterministic"],
    ])

    # ── §7 — Start-Up Lateral Loads: Passive Self-Alignment ─────────────────
    heading2(doc, "7 — Start-Up Lateral Loads: Passive Self-Alignment")
    body(doc, [
        ("During plasma current ramp-up (0 → ~3 MA over ~10 s), the time-changing plasma "
         "current induces image currents in the vessel. These currents interact with the "
         "toroidal-field asymmetry produced by TF vault closure tolerances (a "
         "long-wavelength n ≤ 4 perturbation) to produce a lateral n = 1 force on the "
         "vessel.", False),
    ])
    callout(doc, "key", [
        ("The lateral force on the induced vessel current is such that the vessel is "
         "pulled ", False),
        ("toward", True),
        (" the n = 1 magnetic axis of the (offset) toroidal field — passive vessel currents "
         "act to ", False),
        ("self-align", True),
        (" the wall with the field profile. ", False),
        ("Any vessel motion during the ramp therefore reduces the peak-to-peak misalignment "
         "between the n ≤ 4 first-wall and toroidal-field profiles", True),
        ("; it consumes none of the 6 mm budget, and in fact frees some of it back. The "
         "start-up ramp is a budget-relieving event for the n = 1 first-wall criterion, not "
         "a budget-eroding one.", False),
    ], title="Important: the start-up displacement direction is favourable.")

    heading3(doc, "Force scale & timescale")
    body(doc, [
        ("Order-of-magnitude: with vessel image current of O(10⁵ A) during the ramp, "
         "B_T ≈ 5 T, fractional n = 1 field error ε ~ 10⁻⁴–10⁻³, and a toroidal length scale "
         "~ 2πR, the lateral force F_ramp ~ ε · I_vv · B_T · L is of order ", False),
        ("1–10 kN", True),
        (" — comparable to the F_stop ≈ 4.5 kN centring threshold. The ramp duration (10 s) "
         "is comparable to the pendulum natural period (T_n ≈ 10 s), so the dynamic response "
         "factor for a single (non-periodic) ramp is ~1.3–1.5; the quasi-static deflection "
         "F/K = 0.3–3 mm becomes a peak transient excursion of order ", False),
        ("1–4 mm", True),
        (". Combined with the favourable direction above, a defensible peak start-up "
         "wall–field n = 1 mismatch from this source is below the kinematic ceiling — a "
         "detailed EM transient is required to pin the precise value.", False),
    ])

    # ── §8 — Disruption Loads & VVGS Impact Case ───────────────────────────
    heading2(doc, "8 — Disruption Loads & VVGS Impact Case")
    body(doc, [
        ("A current quench delivers an MN-scale lateral impulse to the vessel over ~10 ms — "
         "far faster than the 10 s vessel pendulum period and far larger than the kN-scale "
         "centring force. On this timescale the vessel responds essentially as a ", False),
        ("free mass", True),
        (":", False),
    ])
    pre(doc,
        "  impulse   J = ∫F dt   ~  (10 MN)·(10 ms)        ≈  10⁵ N·s\n"
        "  velocity  v = J / M  ~  10⁵ / 8×10⁶ kg          ≈  12  mm/s\n"
        "  amplitude (free)  A = v / ω  ~  v·Tₙ / 2π         ≈  20 mm   (≫ 1.5 mm gap)")
    callout(doc, "key", [
        ("For a rigid stop the peak lateral force on the VVGS is on the order of ", False),
        ("~1 MN", True),
        (" — much larger than the steady gravity-related lateral demand on the support. This "
         "is an ", False),
        ("impulsive load case for the VVGS lateral capacity", True),
        (", distinct from steady operation, and is potentially the ", False),
        ("bounding load case for VVGS lateral design", True),
        (". The soft (kN-scale) centring provides essentially no deceleration before impact.",
         False),
    ], title="The vessel impacts the dowel stops at ~12 mm/s with ~0.5 kJ of kinetic energy.")

    # ── §9 — The 6 mm n ≤ 4 First-Wall Budget ──────────────────────────────
    heading2(doc, "9 — The 6 mm n ≤ 4 First-Wall Budget")
    body(doc, [
        ("The start-up heat-load criterion limits the peak-to-peak difference between the "
         "n ≤ 4 filtered first-wall profile and the n ≤ 4 filtered toroidal-field profile "
         "to ", False),
        ("≤ 6 mm", True),
        (". The VV n = 1 lateral displacement, quantified in §5, is one contributor to the "
         "first-wall side of this inequality:", False),
    ])
    table(doc, ["Operating condition", "VV n = 1 contribution to the 6 mm budget"], [
        ["Quiescent / between shots (no lateral load)", "≈ 0 mm (self-centred)"],
        ["Start-up ramp (passive currents → wall follows TF)", "net reduces mismatch"],
        ["Steady asymmetric thermal/EM loads", "F/K — sub-mm"],
        ["Disruption transient (forced)", "up to gap (1.5 mm) + dynamic overshoot"],
        ["Kinematic ceiling (worst random assembly, fully forced)", f"≈ {rmax:.2f} mm"],
    ])
    callout(doc, "info", [
        ("Under nominal start-up conditions the VV n = 1 contribution is small and "
         "favourable (it closes the misalignment). The remaining ~3–6 mm of the budget is "
         "available for the toroidal-field side (mainly TF vault closure n ≤ 4) and any "
         "other first-wall n ≤ 4 contributors.", False),
    ])

    # ── footer ──────────────────────────────────────────────────────────────
    foot = doc.add_paragraph()
    fr = foot.add_run(
        "Author: Simon McIntosh. Monte Carlo data generated reproducibly by "
        "vv_mc_generator.py (scipy HiGHS LP, max-lateral-displacement-from-centre metric, "
        "N = 5000, fixed seed 20260527 → committed data/). Figures and animations by "
        "vv_viz.py; displacements magnified ×500. Centring parameters: M ≈ 8000 t, 15° "
        "inclined dual hinge → L_eff ≈ 26 m → K ≈ 3 kN/mm, T_n ≈ 10 s, F_stop ≈ 4.5 kN. "
        "Every number regenerates from the repository. Source: Simon-McIntosh/vv."
    )
    fr.font.size = Pt(8); fr.font.color.rgb = GREY

    out = os.path.join(DOCS, "vv-lateral-displacement-analysis.docx")
    doc.save(out)
    print(f"-> wrote {out}")


if __name__ == "__main__":
    main()
