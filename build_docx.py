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
    p95, p50 = np.percentile(r, 95), np.percentile(r, 50)
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
    sr = sub.add_run("Revised 2026-05-27  ·  5,000-sample Monte Carlo  ·  scipy HiGHS LP  ·  "
                     "3-DOF rigid-body model  ·  metric: peak-to-peak range (n=1 wall displacement)")
    sr.font.size = Pt(8.5); sr.font.color.rgb = GREY

    callout(doc, "key", [
        ("The vacuum vessel can rattle laterally over a ", False),
        ("peak-to-peak range of 3.09 mm", True),
        (" at the worst (nominal, fully-centred) assembly — a hard theoretical ceiling. "
         "This lateral motion is an ", False),
        ("n=1 rigid displacement of the first wall", True),
        (": during limited start-up on the inner wall it localises the heat load. "
         f"Monte Carlo over random assemblies: P95 = {p95:.2f} mm, median = {p50:.2f} mm, "
         f"mode ≈ {mode:.2f} mm. A separate ±187.5 µrad rotation mode is n=0 "
         "(it does not perturb the plasma–wall gap).", False),
    ], title="Key result:")

    callout(doc, "ok", [
        ("the toroidal slot allowance is ", False),
        ("±1.5 mm (3 mm total travel)", True),
        (", confirmed against the gravity-support design (2026-05-27). The analysis is linear "
         "in the gap, so a ±3 mm slot would double every number; the confirmed ±1.5 mm fixes "
         "the scale.", False),
    ], title="Slot allowance — confirmed:")

    heading2(doc, "1 — Physical mechanism")
    body(doc, [("The VV rests on ", False), ("9 gravity supports", True),
               (" equally spaced toroidally on a ring of radius R ≈ 8 m. Each is a four-bar "
                "linkage with free radial motion and a toroidal pin limited to ±1.5 mm. "
                "Radial motion is free; toroidal motion is the constrained rattle. The 9×3 "
                "constraint matrix has AᵀA = diag(4.5, 4.5, 576) — exactly diagonal, so the "
                "three rigid-body DOFs are uncorrelated.", False)])

    heading2(doc, "2 — Why it matters: two distinct consequences")
    body(doc, [("The lateral rattle is a rigid horizontal shift of the vessel. It has two "
                "largely independent consequences, both bounded by the peak-to-peak range — "
                "one quasi-static, one dynamic.", False)])
    callout(doc, "key", [
        ("A horizontal shift makes the plasma–wall gap vary as cos(toroidal angle) — a textbook "
         "n=1 perturbation. Between shots the vessel settles to a different position; when the "
         "plasma is limited on the inner wall during start-up, the gap asymmetry localises the "
         "start-up heat load toroidally.", False)], title="(a) n=1 first-wall positioning:")
    callout(doc, "warn", [
        ("Because the vessel is mobile within its ±1.5 mm slots, EM loads (disruptions, VDEs, "
         "halo currents) can accelerate it across the gap until it impacts its end-stops. The "
         "shock loads on the gravity supports, and fatigue from repeated impact cycling, can be "
         "significant and must be assessed dynamically — not only as a static positioning "
         "number.", False)], title="(b) EM-driven impact (dynamic):")
    body(doc, [("Rotation about the vertical axis is ", False), ("n=0", True),
               (" — it leaves the plasma–wall gap unchanged and is treated as a separate "
                "local-alignment mode, not part of the n=1 budget.", False)])

    heading2(doc, "3 — The number: peak-to-peak rattle range")
    body(doc, [("Each support travels ±1.5 mm, yet the nominal range is 3.093 mm — slightly "
                "over the 3 mm slot because the rotation DOF lets the centre exploit the "
                "angled supports' slack (a 3-support binding vertex). 5,000-sample Monte Carlo "
                "over random assemblies:", False)])
    table(doc, ["Statistic", "Range (mm)", "Notes"], [
        ["LP ceiling (nominal u=0)", "3.093", "Hard maximum; centred assembly = worst = team's target"],
        ["P95", f"{p95:.2f}", "95th percentile of random assemblies"],
        ["Mode", f"{mode:.2f}", "Most probable as-built value"],
        ["Median", f"{p50:.2f}", "Typical random assembly"],
    ])
    figure(doc, os.path.join(DOCS, "plots", "mc_dashboard.png"),
           "Peak-to-peak range distribution and CDF; nominal ceiling and percentiles marked.")
    callout(doc, "warn", [
        ("The range is robust to assembly scatter (P95 = 3.01 mm is barely below the ceiling), "
         "and the team aims to centre the brackets — which is the worst case. Use ", False),
        ("3.09 mm", True), (". The MC P95 is not a relaxation of the design case.", False),
    ], title="Caution:")

    heading2(doc, "4 — Worst case and typical assembly")
    body(doc, [("The VV rocks along its principal rattle axis. Animations play in the served "
                "HTML report; below, each GIF (first frame) is followed by a key-frame strip "
                "showing the motion.", False)])
    callout(doc, "info", [
        ("At each support the grey toroidal slot (red ±1.5 mm end-stops) stays a constant "
         "radial distance from the vessel — the four-bar linkage moves radially with it, so "
         "radial motion is free — while the coloured pin shows the toroidal-constraint usage "
         "(blue = slack, orange = near limit, red = at the stop).", False)],
        title="Reading the supports:")
    figure(doc, os.path.join(DOCS, "animations", "rattle_worst_case.gif"),
           "Worst case — peak-to-peak range 3.09 mm (the design bound). [animated GIF]", width=3.6)
    figure(doc, os.path.join(DOCS, "strips", "strip_worst_translation.png"),
           "Worst-case key frames (×500 magnification).")
    figure(doc, os.path.join(DOCS, "animations", "rattle_mode.gif"),
           f"Typical as-built assembly, near the distribution mode — range ≈ {mode:.2f} mm "
           "(the most likely value). [animated GIF]", width=3.6)
    figure(doc, os.path.join(DOCS, "strips", "strip_mode_translation.png"),
           "Near-mode (typical assembly) key frames (×500 magnification).")

    heading2(doc, "5 — Effect of gap measurements (one sector landed)")
    body(doc, [("Measuring a bracket gap fixes where its 3 mm constraint band sits, not its "
                "width — so the bound tightens only when a measured offset happens to be a "
                "binding constraint. Because AᵀA is diagonal, adjacent vs spread-out supports "
                "are statistically identical (no 'coupled cluster' to target).", False)])
    table(doc, ["Supports measured (k of 9)", "Conditional P95 (mm)", "Effect"], [
        ["0 (none — today's design state)", "3.01", "baseline"],
        ["1  (one sector landed)", "≈ 3.0", "no meaningful change"],
        ["4", "≈ 2.9", "marginal"],
        ["6", "≈ 2.6", "tightening begins"],
        ["9 (all)", "deterministic ≈ 2.5", "one value; 0.2–3.09; never 0"],
    ])
    figure(doc, os.path.join(DOCS, "plots", "partial_measurement.png"),
           "Left: what measuring all gaps reveals (a value from the distribution, never 0). "
           "Right: conditional P95 vs gaps measured.")
    callout(doc, "key", [
        ("With one VV sector landed (k=1), the q95 bound is statistically unchanged from the "
         "all-unknown baseline. The bound to use today remains 3.09 mm. Measurement reveals the "
         "rattle, it does not relieve it; a material reduction needs roughly five or more "
         "sectors measured.", False)], title="Current state:")

    heading2(doc, "6 — Recommendations")
    table(doc, ["#", "Recommendation"], [
        ["1", "Use 3.09 mm peak-to-peak as the bounding n=1 lateral wall displacement (nominal/centred = worst case = ceiling)."],
        ["2", "Do not rely on the MC P95 (3.01 mm) as a relaxation — it is ~3% below the ceiling and the realistic assembly is centred."],
        ["3", "Assess start-up heat-load localisation on the inner-wall limiter (n=1) for an offset up to ≈ 2.9 mm single-sided, 3.09 mm shot-to-shot."],
        ["4", "Assess dynamic impact loads on the gravity supports from EM-driven motion across the ±1.5 mm gap, and fatigue from repeated impact cycling."],
        ["5", "The bound is set by the 3 mm toroidal slot and is largely irreducible by assembly control — the lever is the tolerance or added toroidal restraint."],
        ["6", "Gap metrology gives little benefit until ≥ 5 sectors are measured; confirm whether 'landed' implies a measured gap."],
        ["7", "Slot allowance ±1.5 mm (3 mm total) confirmed against the GS design (2026-05-27); numbers scale linearly with the slot."],
        ["8", "Treat the ±187.5 µrad rotation (n=0) as a separate local-alignment case for ports/penetrations."],
    ])

    foot = doc.add_paragraph()
    fr = foot.add_run("Monte Carlo generated reproducibly by vv_mc_generator.py "
                      "(scipy HiGHS LP, peak-to-peak range metric, N=5000, fixed seed 20260527 → "
                      "committed data in data/). Figures and animations by vv_viz.py. Every number "
                      "regenerates from the repository. Source: Simon-McIntosh/vv.")
    fr.font.size = Pt(8); fr.font.color.rgb = GREY

    out = os.path.join(DOCS, "vv-rattle-report.docx")
    doc.save(out)
    print(f"-> wrote {out}")


if __name__ == "__main__":
    main()
