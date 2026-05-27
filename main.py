"""
Reproduce the ITER VV lateral-rattle report end-to-end.

    uv run python main.py

Step 1 regenerates the canonical 5,000-sample peak-to-peak Monte-Carlo dataset
(fixed seed) into data/. Step 2 regenerates every figure, key-frame strip and
GIF embedded in docs/vv-research-findings.html. Run build_docx.py afterwards for
the Word report. (vv_rattle_mc.py is a supplementary LP / pseudoinverse
verification using the per-direction reach metric; it is not part of the report
pipeline.)
"""
import vv_mc_generator
import vv_viz


def main():
    vv_mc_generator.main()   # canonical peak-to-peak range MC -> data/
    vv_viz.main()            # figures, strips, GIFs -> docs/


if __name__ == "__main__":
    main()
