"""
Build the Word version of the ITER VV lateral-rattle report.

Style-matched to docs/vv-research-findings.html (blue #2b5797 / #1a3a6e palette,
left-accented callout boxes, blue-header tables) and embeds the worst-case and
near-mode rattle GIFs. Word renders an animated GIF as its first frame, so a
key-frame strip is placed beneath each GIF to convey the motion in print; the
animations play in the served HTML report.

    uv run python build_docx.py     # -> docs/vv-rattle-report.docx
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
    tr = t.add_run("ITER Vacuum Vessel — Lateral Rattle Analysis")
    tr.bold = True; tr.font.size = Pt(19); tr.font.color.rgb = DARKBLUE
    pPr = t._p.get_or_add_pPr(); pbdr = OxmlElement("w:pBdr")
    b = OxmlElement("w:bottom"); b.set(qn("w:val"), "single"); b.set(qn("w:sz"), "18")
    b.set(qn("w:space"), "3"); b.set(qn("w:color"), "2b5797"); pbdr.append(b); pPr.append(pbdr)
    sub = doc.add_paragraph()
    sr = sub.add_run("Revised 2026-05-28  ·  5,000-sample Monte Carlo  ·  scipy HiGHS LP  ·  "
                     "metric: max departure from gravitational centre (n=1 first-wall shift)")
    sr.font.size = Pt(8.5); sr.font.color.rgb = GREY

    callout(doc, "key", [
        ("The VV rests on 9 ", False), ("inward-inclined dual-hinge gravity supports", True),
        (" that act as a gravitational pendulum well in the lateral DOF — the vessel "
         "self-centres at rest. The relevant rattle metric is therefore the ", False),
        ("maximum forced departure from the gravitational centre", True),
        (f". With the toroidal slot at ±1.5 mm (confirmed), nominal envelope = 1.547 mm; MC "
         f"over independent assembly offsets gives P95 = {p95:.2f} mm, P99 = {p99:.2f} mm, "
         f"median {p50:.2f} mm, mode ≈ {mode:.2f} mm, max ≈ {rmax:.2f} mm. Two EM cases drive "
         "forced departure: (a) the 0→3 MA, 10 s start-up ramp (lateral force comparable to "
         "the centering, near-resonant with the ~10 s natural period — peak excursions "
         "O(1 mm)); and (b) disruptions (MN-scale impulse → vessel velocity → impact at the "
         "dowel stops — potentially a bounding VVGS lateral load case).", False),
    ], title="Key result:")

    callout(doc, "ok", [
        ("toroidal slot allowance ", False), ("±1.5 mm (3 mm total)", True),
        (", confirmed against the GS design. All numbers below use this value; they scale "
         "linearly with the slot.", False),
    ], title="Slot allowance — confirmed:")

    heading2(doc, "1 — Physical mechanism: the inclined-hinge VVGS")
    body(doc, [
        ("Each of the 9 gravity supports is a ", False),
        ("dual-hinge mechanism inclined at 15° from vertical", True),
        (", leaning inward. The dual hinges permit radial motion (thermal expansion via "
         "dowel rotation) while the dowels' rigidity restrains toroidal motion to the "
         "±1.5 mm slot. The inward inclination is deliberate: it puts the supports' "
         "common convergence point above the vessel CoG, so the vessel behaves as an "
         "inclined-hinge ", False),
        ("gravitational pendulum", True),
        (" — the radial/lateral DOF sits in a parabolic potential well centred on the "
         "nominal position.", False),
    ])
    table(doc, ["Quantity", "Value", "Notes"], [
        ["Effective pendulum length L_eff", "≈ 26 m", "= R_s / tan(15°) − z_CoG ; sensitive to angle (~2 m / °)"],
        ["Lateral stiffness K = W / L_eff", "≈ 3.0 kN / mm", "with M ≈ 8000 t (VV + in-vessel)"],
        ["Natural period T_n", "≈ 10 s (f ≈ 0.1 Hz)", "depends on L_eff only, not on M"],
        ["Force to reach the ±1.5 mm stop", "≈ 4.5 kN", "≈ 6×10⁻⁵ of the 78 MN weight"],
        ["Rest position (no lateral force)", "q* = 0 (centred)", "gravitational PE minimum"],
    ])

    heading2(doc, "2 — How the centering force reframes the analysis")
    callout(doc, "info", [
        ("At rest the vessel self-centres to q = 0 regardless of the toroidal assembly offsets — "
         "the pins simply sit at u_i in their ±1.5 mm slots while the vessel is centred. The "
         "relevant metric is no longer the peak-to-peak range but the ", False),
        ("max one-sided departure from the gravitational centre", True),
        (" under sufficient lateral force; this is bounded by the kinematic polytope.", False),
    ], title="Self-centred rest position:")

    heading2(doc, "3 — The number: max departure from centre (MC)")
    table(doc, ["Statistic", "Departure (mm)", "Notes"], [
        ["Nominal (u = 0)", "1.547", "Symmetric envelope half-width"],
        ["Mode", f"{mode:.2f}", "Most probable as-built value"],
        ["Median (P50)", f"{p50:.2f}", "Typical random assembly"],
        ["P95", f"{p95:.2f}", "95th percentile"],
        ["P99", f"{p99:.2f}", ""],
        ["Max observed", f"{rmax:.2f}", "Kinematic ceiling on forced departure"],
    ])
    figure(doc, os.path.join(DOCS, "plots", "mc_dashboard.png"),
           "Departure-from-centre distribution and CDF. Nominal (u=0) gives the symmetric "
           "envelope; offset assemblies reach further from the gravitational centre.")
    callout(doc, "info", [
        ("Quiescent (no significant lateral load) the vessel sits at the gravitational centre — "
         "n = 1 wall contribution ≈ 0. Under forced load the vessel departs from centre up to "
         "the polytope boundary — the MC bounds that kinematic ceiling.", False)],
        title="This is the n = 1 wall budget input:")

    heading2(doc, "4 — Forced excursion: worst &amp; typical assembly")
    body(doc, [("The animations show the VV rocked through its polytope along the principal "
                "axis — the kinematic envelope of possible forced motion about the centre. "
                "At rest the vessel sits at the mid-frame (q = 0); under sufficient lateral "
                "force it can be driven to the extremes. ×1500 magnification.", False)])
    callout(doc, "info", [
        ("Each grey toroidal slot stays at constant radial distance from the wall — the "
         "four-bar linkage moves radially with the vessel. The coloured pin shows the "
         "toroidal-constraint usage (blue = slack, orange = near limit, red = at stop).", False)],
        title="Reading the supports:")
    figure(doc, os.path.join(DOCS, "animations", "rattle_worst_case.gif"),
           f"Worst MC sample — max departure ≈ {rmax:.2f} mm (polytope diameter ≈ 3.09 mm). [GIF first frame]",
           width=3.6)
    figure(doc, os.path.join(DOCS, "strips", "strip_worst_translation.png"),
           "Worst-case key frames (×1500 magnification).")
    figure(doc, os.path.join(DOCS, "animations", "rattle_mode.gif"),
           f"Typical as-built assembly (near the distribution mode) — max departure ≈ "
           f"{mode:.2f} mm (the most likely value). [GIF first frame]",
           width=3.6)
    figure(doc, os.path.join(DOCS, "strips", "strip_mode_translation.png"),
           "Near-mode (typical assembly) key frames (×1500 magnification).")

    heading2(doc, "5 — Effect of gap measurements (now much more powerful)")
    body(doc, [("Important reversal from the previous diameter framing: the polytope diameter "
                "is set by the 3 mm slot and is essentially insensitive to the assembly "
                "offsets, but the ", False), ("departure from centre", True),
               (" is governed by how far the polytope is offset from q = 0 — exactly what the "
                "u_i set — so each measured gap directly tightens the conditional departure "
                "distribution. AᵀA is exactly diagonal so adjacency is irrelevant; only the ",
                False), ("number", True), (" of measurements matters.", False)])
    table(doc, ["Supports measured (k of 9)", "P95 — centred (mm)",
                "P95 — typical (mm)", "Effect"], [
        ["0 (none)", f"{p95:.2f}", f"{p95:.2f}", "baseline"],
        ["1  (one sector landed)", "≈ 2.2", "≈ 2.2", "small but real tightening"],
        ["3", "≈ 2.0", "≈ 2.1", "moderate"],
        ["5", "≈ 1.8", "≈ 2.0", "substantial"],
        ["9 (all)", "→ 1.55 (nominal env.)", "deterministic", "collapsed"],
    ])
    figure(doc, os.path.join(DOCS, "plots", "partial_measurement.png"),
           "Left: the departure distribution. Right: conditional P95 vs supports measured "
           "(centred vs typical random).")
    callout(doc, "key", [
        ("Each additional measured sector incrementally tightens the n = 1 budget — a clear "
         "path to bound reduction as landing progresses. This contrasts with the previous "
         "peak-to-peak diameter framing, which gave no relief from measurements.", False)],
        title="Each landed sector matters:")

    heading2(doc, "6 — Rotation (n = 0, separate)")
    body(doc, [
        ("The LP for max Δθ gives ", False),
        ("±187.5 µrad (0.375 mrad)", True),
        (" — all 9 pins slide synchronously by R·Δθ = ±1.5 mm. Rotating an axisymmetric "
         "torus about its own axis leaves the plasma–wall gap unchanged (", False),
        ("n = 0", True),
        ("), so this mode is local-alignment only and not part of the n = 1 budget.", False),
    ])

    heading2(doc, "7 — EM implications")
    callout(doc, "warn", [
        ("Plasma ramps 0→3 MA over ~10 s, driving vessel image currents that, with n ≤ 4 TF "
         "asymmetries (mainly TF vault closure), produce a lateral n = 1 force estimated at "
         "O(1–10 kN) — comparable to the F_stop ≈ 4.5 kN centering. The 10 s ramp ≈ T_n, so "
         "the dynamic response factor is ~1.3–1.5 (a single ramp, not periodic — no resonant "
         "accumulation). Quasi-static deflection F/K = 0.3–3 mm; dynamic peak departure ", False),
        ("O(1–4 mm)", True), (" — potentially close to the ±1.5 mm gap. A defensible value "
         "needs the actual n ≤ 4 force amplitude and damping.", False)],
        title="(a) Start-up plasma ramp (10 s):")
    callout(doc, "key", [
        ("A current quench delivers MN-scale lateral force over ms — far faster than T_n ≈ 10 s "
         "and far larger than the kN centering. On the disruption timescale the vessel responds "
         "as a free mass: impulse J = F·τ ~ 10⁵ N·s → v = J/M ~ 12 mm/s; the soft centering "
         "barely decelerates it. The vessel hits the dowel stops at O(10 mm/s) with O(500 J) of "
         "kinetic energy → peak impact force on the VVGS ~1 MN (rigid-stop estimate). This is "
         "an ", False),
        ("impulsive lateral load case for VVGS design", True),
        (" and could be bounding for the lateral capacity.", False),
    ], title="(b) Disruption / VDE (ms):")
    callout(doc, "info", [
        ("The start-up criterion compares the n ≤ 4 filtered first-wall and toroidal-field "
         "profiles; their peak-to-peak difference must remain within ", False),
        ("6 mm", True), (". The vessel n = 1 contribution is ≈ 0 quiescent (self-centred), "
         "~1–4 mm at the start-up peak (force-dependent), and up to the ~2.9 mm kinematic "
         "ceiling under disruption-class lateral forces. The remaining ~3–5 mm must cover the "
         "TF vault closure n ≤ 4 and any other first-wall n ≤ 4 sources.", False)],
        title="The 6 mm n ≤ 4 budget:")

    heading2(doc, "8 — Recommendations")
    table(doc, ["#", "Recommendation"], [
        ["1", "Use departure from gravitational centre as the n = 1 first-wall metric; "
              f"P95 = {p95:.2f} mm, MC max ≈ {rmax:.2f} mm. Quiescent contribution ≈ 0."],
        ["2", "Compute the start-up lateral force on the vessel from a 3 MA / 10 s ramp through "
              "the actual TF vault closure n ≤ 4 errors; expect O(1–4 mm) dynamic peak departure."],
        ["3", "Allocate the 6 mm n ≤ 4 budget between vessel n = 1 (above) and TF-side n ≤ 4."],
        ["4", "Disruption dynamic VVGS load case: model impulse → vessel velocity → end-stop "
              "impact (~MN peak lateral force). Potentially bounding for the VVGS."],
        ["5", "Gap metrology NOW meaningfully tightens the n = 1 budget under the departure "
              "metric — each landed sector incrementally pulls the polytope toward the "
              "gravitational centre. Report conditional P95 with every additional sector measured."],
        ["6", "Refine L_eff using the as-built hinge geometry — sensitive to ~2 m/° of "
              "inclination angle."],
        ["7", "Slot allowance ±1.5 mm (3 mm total) confirmed; linear scaling with the slot."],
        ["8", "Treat the ±187.5 µrad rotation (n = 0) as a separate local-alignment case for "
              "ports/penetrations."],
    ])

    foot = doc.add_paragraph()
    fr = foot.add_run("Monte Carlo generated reproducibly by vv_mc_generator.py "
                      "(scipy HiGHS LP, departure-from-centre metric, N = 5000, fixed seed "
                      "20260527 → committed data/). Centering parameters M ≈ 8000 t, 15° "
                      "incline → L_eff ≈ 26 m → K ≈ 3 kN/mm, T_n ≈ 10 s, F_stop ≈ 4.5 kN. "
                      "Every number regenerates from the repository. Source: Simon-McIntosh/vv.")
    fr.font.size = Pt(8); fr.font.color.rgb = GREY

    out = os.path.join(DOCS, "vv-rattle-report.docx")
    doc.save(out)
    print(f"-> wrote {out}")


if __name__ == "__main__":
    main()
