"""
Build the Word version of the ITER VV lateral-rattle report.

Style-matched to docs/vv-lateral-displacement-analysis.html (blue #2b5797 / #1a3a6e palette,
left-accented callout boxes, blue-header tables) and embeds the worst-case and
near-mode rattle GIFs. Word renders an animated GIF as its first frame, so a
key-frame strip is placed beneath each GIF to convey the motion in print; the
animations play in the served HTML report.

    uv run python build_docx.py     # -> docs/vv-lateral-displacement-analysis.docx
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
CALLOUT = {  # (left-border hex, fill hex)
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


def _set_borders(cell, edges, color="cccccc", sz=4):
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


def callout(doc, kind, runs, title=None):
    """runs: list of (text, bold) tuples; rendered as one shaded, left-bordered cell."""
    border_hex, fill_hex = CALLOUT[kind]
    tbl = doc.add_table(rows=1, cols=1)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    cell = tbl.cell(0, 0)
    _shade(cell, fill_hex)
    _set_borders(cell, ["left"], color=border_hex, sz=24)
    _set_borders(cell, ["top", "bottom", "right"], color=fill_hex, sz=4)
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


def table(doc, headers, rows):
    t = doc.add_table(rows=1, cols=len(headers))
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    hdr = t.rows[0].cells
    for j, h in enumerate(headers):
        _shade(hdr[j], "2b5797")
        _set_borders(hdr[j], ["top", "bottom", "left", "right"], "2b5797", 4)
        p = hdr[j].paragraphs[0]; _no_space(p)
        r = p.add_run(h); r.bold = True; r.font.color.rgb = WHITE; r.font.size = Pt(9.5)
    for i, row in enumerate(rows):
        cells = t.add_row().cells
        fill = "f4f7ff" if i % 2 else "ffffff"
        for j, val in enumerate(row):
            _shade(cells[j], fill)
            _set_borders(cells[j], ["top", "bottom", "left", "right"], "cccccc", 4)
            p = cells[j].paragraphs[0]; _no_space(p)
            r = p.add_run(str(val)); r.font.size = Pt(9.5)
            if j == 0:
                r.bold = True
    doc.add_paragraph().paragraph_format.space_after = Pt(2)
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


# ── assemble the document ───────────────────────────────────────────────────
def main():
    # pull the canonical numbers so the docx stays in sync with the data
    r = np.load(os.path.join(HERE, "data", "rattle_mc_5k.npy"))
    p95, p99, p50 = np.percentile(r, 95), np.percentile(r, 99), np.percentile(r, 50)
    rmax = float(r.max())
    h, e = np.histogram(r, bins=60); mode = 0.5 * (e[h.argmax()] + e[h.argmax() + 1])

    doc = Document()
    doc.styles["Normal"].font.name = "Segoe UI"
    doc.styles["Normal"].font.size = Pt(10.5)

    # title
    t = doc.add_paragraph()
    tr = t.add_run("ITER Vacuum Vessel — Lateral Displacement Analysis")
    tr.bold = True; tr.font.size = Pt(19); tr.font.color.rgb = DARKBLUE
    pPr = t._p.get_or_add_pPr(); pbdr = OxmlElement("w:pBdr")
    b = OxmlElement("w:bottom"); b.set(qn("w:val"), "single"); b.set(qn("w:sz"), "18")
    b.set(qn("w:space"), "3"); b.set(qn("w:color"), "2b5797"); pbdr.append(b); pPr.append(pbdr)
    # author line
    auth = doc.add_paragraph()
    ar = auth.add_run("Simon McIntosh"); ar.bold = True; ar.font.size = Pt(11); ar.font.color.rgb = DARKBLUE
    ar2 = auth.add_run("   ·   2026-05-28"); ar2.font.size = Pt(10); ar2.font.color.rgb = GREY
    sub = doc.add_paragraph()
    sr = sub.add_run("5,000-sample Monte Carlo  ·  scipy HiGHS LP  ·  3-DOF rigid-body model  ·  "
                     "metric: max lateral displacement of the VV centre from the gravitational "
                     "centre (n = 1 first-wall shift)")
    sr.font.size = Pt(8.5); sr.font.color.rgb = GREY

    callout(doc, "info", [
        ("The VV rests on 9 inward-inclined gravity supports whose 15° tilt provides a "
         "gravitational centring force: the vessel self-centres at rest and departs from the "
         "machine axis only under an applied lateral force; pendulum period ≈ 10 s. The "
         "n = 1 first-wall metric is the ", False),
        ("max lateral displacement of the VV centre from the machine axis", True),
        (f". Nominal envelope 1.55 mm; across the assembly population P95 = {p95:.2f} mm, "
         f"max = {rmax:.2f} mm; at-rest contribution = 0. Start-up (0→3 MA / 10 s) is "
         "favourable: passive vessel currents pull the wall toward the offset toroidal-field "
         "magnetic centre and reduce the n ≤ 4 misalignment. Disruption is the bounding VVGS "
         "lateral load case (~12 mm/s impact, ~1 MN peak). Gap measurements tighten the bound "
         "— each measured sector pulls the polytope toward the centre.", False),
    ], title="Summary of findings:")

    heading2(doc, "1 — Problem & assumptions")
    body(doc, [
        ("9 VVGS, equally spaced toroidally on a ring R_s ≈ 8 m. Each is a dual-hinge mechanism "
         "inclined 15° from vertical (leaning inward). Radial motion is permitted by dowel "
         "rotation (thermal expansion); toroidal motion is restrained to a ±1.5 mm slot. The "
         "supported mass is M ≈ 8000 t (VV + in-vessel components). The slot allowance "
         "±1.5 mm (3 mm total) is confirmed against the GS design.", False),
    ])
    callout(doc, "warn", [
        ("The ±1.5 mm slot is an ", False), ("assembly tolerance", True),
        (": the as-built precision with which each VVGS pin lands in its dowel. If the gaps "
         "are left unshimmed (the case treated here), it is this tolerance that permits the "
         "vessel to be laterally mobile and is what drives this analysis. We model the as-built "
         "state by the conservative prior that each u_i is independent and uniformly "
         "distributed on [−1.5, +1.5] mm. Shimming a support pins its u_i to a measured value "
         "and removes that support's contribution to the mobility (see §6).", False),
    ], title="Assembly assumption — the source of mobility:")
    body(doc, [
        ("The inward 15° inclination puts the support axes' convergence point above the vessel "
         "CoG; for the lateral mode the vessel behaves as an ", False),
        ("inclined-hinge gravitational pendulum", True),
        (" — the lateral plane has a parabolic potential well centred on the nominal position.",
         False),
    ])
    table(doc, ["Quantity", "Value"], [
        ["Effective pendulum length L_eff", "≈ 26 m (sensitive to angle; ~2 m / °)"],
        ["Lateral centring stiffness K = W / L_eff", "≈ 3.0 kN / mm"],
        ["Natural period T_n", "≈ 10 s (f ≈ 0.1 Hz)"],
        ["Force to reach a ±1.5 mm stop", "≈ 4.5 kN"],
        ["Rest position (no lateral force)", "q* = 0 (gravitational PE minimum)"],
    ])

    heading2(doc, "2 — Constraint model")
    body(doc, [
        ("Toroidal slide at support i: δ_i = −sin(φ_i)·Δx + cos(φ_i)·Δy + R·Δθ, with |u_i + "
         "δ_i| ≤ 1.5 mm. The 9×3 constraint matrix has AᵀA = diag(4.5, 4.5, 576): the three "
         "rigid-body DOFs are mutually uncorrelated. The reachable (Δx, Δy) set is the "
         "displacement polytope; the metric is the maximum lateral displacement of the VV "
         "centre from the gravitational centre.", False),
    ])

    heading2(doc, "3 — State diagrams")
    figure(doc, os.path.join(DOCS, "plots", "vv_states.png"),
           "The VV sits at the gravitational centre in both panels (rest position). The shaded "
           "polygon around the centre is the displacement polytope — the kinematic envelope of "
           "possible lateral displacements under applied force. Left: nominal assembly "
           "(envelope half-width 1.55 mm). Right: worst observed MC sample — the offsets shift "
           "the polytope so the far corner lies up to 2.98 mm from the centre. ×500 magnification.")

    heading2(doc, "4 — Forced-excursion envelope")
    body(doc, [
        ("The animations sweep the VV centre through the polytope along its principal axis — "
         "the kinematic envelope of lateral motion about the centred rest position. With no "
         "applied force the vessel sits at the centre; sufficient applied force drives it "
         "through the envelope. ×500 magnification.", False),
    ])
    figure(doc, os.path.join(DOCS, "animations", "rattle_worst_case.gif"),
           f"Worst observed MC sample — max forced departure ≈ {rmax:.2f} mm. [GIF first frame]",
           width=3.6)
    figure(doc, os.path.join(DOCS, "strips", "strip_worst_translation.png"),
           "Worst-case key frames.")
    figure(doc, os.path.join(DOCS, "animations", "rattle_mode.gif"),
           f"Typical as-built assembly (near the distribution mode) — max forced departure ≈ "
           f"{mode:.2f} mm. [GIF first frame]", width=3.6)
    figure(doc, os.path.join(DOCS, "strips", "strip_mode_translation.png"),
           "Near-mode (typical) key frames.")

    heading2(doc, "5 — Monte Carlo distribution")
    table(doc, ["Statistic", "Max lateral displacement (mm)"], [
        ["Nominal (u = 0) — symmetric envelope half-width", "1.55"],
        ["Mode", f"{mode:.2f}"],
        ["Median (P50)", f"{p50:.2f}"],
        ["P95", f"{p95:.2f}"],
        ["P99", f"{p99:.2f}"],
        ["Max observed (n = 5000)", f"{rmax:.2f}"],
    ])
    figure(doc, os.path.join(DOCS, "plots", "mc_dashboard.png"),
           "Distribution (left) and CDF (right) of the maximum lateral displacement of the "
           "VV centre from the gravitational centre, over 5000 independent Uniform(±1.5 mm) "
           "assemblies. Nominal (centred-assembly) gives the symmetric envelope; offset "
           "assemblies reach further from the centre.")

    heading2(doc, "6 — Effect of gap measurements")
    body(doc, [
        ("Each measured gap fixes one u_i and pulls the polytope toward the gravitational "
         "centre: under this metric measurement directly tightens the conditional displacement "
         "bound. AᵀA is exactly diagonal so adjacency is irrelevant; only the ", False),
        ("number", True),
        (" of measured sectors matters. With all 9 measured and found near-centred, the "
         "conditional bound collapses to the 1.55 mm nominal envelope.", False),
    ])
    table(doc, ["Measured sectors (k of 9)", "P95 — centred (mm)", "P95 — typical (mm)"], [
        ["0 (baseline)", f"{p95:.2f}", f"{p95:.2f}"],
        ["1", "≈ 2.26", "≈ 2.26"],
        ["3", "≈ 2.19", "≈ 2.22"],
        ["5", "≈ 2.05", "≈ 2.00"],
        ["9 (all)", "→ 1.55 (nominal env.)", "deterministic"],
    ])
    figure(doc, os.path.join(DOCS, "plots", "partial_measurement.png"),
           "Left: the displacement distribution. Right: conditional P95 vs number of sectors "
           "measured, for two scenarios of the measured values (centred / typical random).")

    heading2(doc, "7 — Start-up lateral loads: passive self-alignment")
    body(doc, [
        ("During the 0 → ~3 MA, ~10 s plasma current ramp, the dI_p/dt induces image currents "
         "in the vessel. These currents interact with the n ≤ 4 toroidal-field asymmetry "
         "produced by TF vault closure tolerances to produce a lateral n = 1 force on the "
         "vessel.", False),
    ])
    callout(doc, "key", [
        ("The direction is favourable: the lateral force on the induced vessel current pulls "
         "the vessel TOWARD the n = 1 magnetic axis of the (offset) toroidal field — passive "
         "vessel currents act to ", False),
        ("self-align the wall with the field", True),
        (". Any vessel motion during the ramp therefore CLOSES the peak-to-peak misalignment "
         "between the n ≤ 4 filtered first-wall and toroidal-field profiles, rather than "
         "opening it. The start-up ramp is not a budget-eroding event — it is the opposite.",
         False),
    ], title="Start-up displacement direction:")
    body(doc, [
        ("Order-of-magnitude force: with vessel image current of O(10⁵ A) during the ramp, "
         "B_T ≈ 5 T, fractional n = 1 field error ε ~ 10⁻⁴–10⁻³, and a toroidal length scale "
         "~ 2πR, the lateral force F_ramp ~ ε · I_vv · B_T · L is of order 1–10 kN — comparable "
         "to the F_stop ≈ 4.5 kN centring threshold. The ramp duration (10 s) is comparable to "
         "T_n (~10 s), so the single-ramp dynamic response factor is ~1.3–1.5; F/K = 0.3–3 mm "
         "quasi-static becomes a peak excursion of order 1–4 mm — below the 1.5 mm gap when "
         "combined with the favourable direction. A defensible value needs the EM transient.",
         False),
    ])

    heading2(doc, "8 — Disruption loads &amp; VVGS impact case")
    body(doc, [
        ("A current quench delivers an MN-scale lateral impulse over ~10 ms — far faster than "
         "T_n ≈ 10 s and far larger than the kN-scale centring. On this timescale the vessel "
         "responds as a free mass:", False),
    ])
    table(doc, ["Quantity", "Estimate"], [
        ["Impulse J = ∫F dt", "(10 MN)·(10 ms) ≈ 10⁵ N·s"],
        ["Velocity v = J / M", "10⁵ / 8×10⁶ kg ≈ 12 mm/s"],
        ["Free-pendulum amplitude v·T_n / 2π", "≈ 20 mm  (≫ 1.5 mm gap)"],
    ])
    callout(doc, "key", [
        ("The vessel impacts the dowel stops at ~12 mm/s with ~0.5 kJ of kinetic energy. For "
         "a rigid stop, peak lateral force on the VVGS is of order ", False),
        ("~1 MN", True),
        (" — much larger than the steady gravity-related lateral demand. This is an "
         "impulsive lateral load case for VVGS design, distinct from steady operation, and is "
         "potentially the bounding load case for VVGS lateral capacity. The soft kN-scale "
         "centring provides essentially no deceleration before impact.", False),
    ], title="VVGS impulsive lateral load:")

    heading2(doc, "9 — The 6 mm n ≤ 4 first-wall budget")
    body(doc, [
        ("The start-up heat-load criterion limits the peak-to-peak difference between the "
         "n ≤ 4 filtered first-wall and toroidal-field profiles to ≤ 6 mm. The VV n = 1 "
         "contribution to the first-wall side is the lateral displacement quantified above.",
         False),
    ])
    table(doc, ["Operating condition", "VV n = 1 contribution"], [
        ["Quiescent / between shots (no lateral load)", "≈ 0 mm (self-centred)"],
        ["Start-up ramp (passive currents → wall follows TF)", "net REDUCES mismatch"],
        ["Steady asymmetric thermal/EM loads", "F/K — sub-mm"],
        ["Disruption transient (forced)", "up to gap (1.5 mm) + overshoot"],
        ["Kinematic ceiling (worst random assembly)", f"≈ {rmax:.2f} mm"],
    ])
    callout(doc, "info", [
        ("Under nominal start-up conditions the VV n = 1 contribution is small AND favourable "
         "(it closes the misalignment). The remaining 3–6 mm of the budget is available for the "
         "toroidal-field side (mainly TF vault closure n ≤ 4) and any other first-wall n ≤ 4 "
         "contributors.", False),
    ], title="Budget allocation:")

    foot = doc.add_paragraph()
    fr = foot.add_run("Author: Simon McIntosh. Monte Carlo data generated reproducibly by "
                      "vv_mc_generator.py (scipy HiGHS LP, max-lateral-displacement-from-centre "
                      "metric, N = 5000, fixed seed 20260527 → committed data/). Figures and "
                      "animations by vv_viz.py; displacements magnified ×500. Centring "
                      "parameters M ≈ 8000 t, 15° inclined dual hinge → L_eff ≈ 26 m → "
                      "K ≈ 3 kN/mm, T_n ≈ 10 s, F_stop ≈ 4.5 kN. Every number regenerates from "
                      "the repository. Source: Simon-McIntosh/vv.")
    fr.font.size = Pt(8); fr.font.color.rgb = GREY

    out = os.path.join(DOCS, "vv-lateral-displacement-analysis.docx")
    doc.save(out)
    print(f"-> wrote {out}")


if __name__ == "__main__":
    main()
