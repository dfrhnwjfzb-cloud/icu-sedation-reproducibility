"""Matplotlib panel alignment checks."""

import json
from pathlib import Path


def require_matplotlib_panel_alignment(
    fig,
    json_out,
    overlay_svg=None,
    tolerance_pt=1.5,
    gutter_tolerance_pt=1.5,
    strict=True,
    **kwargs,
):
    fig.canvas.draw()
    width, height = fig.get_size_inches() * 72
    axes = fig.axes
    rects = [
        {
            "left": a.get_position().x0 * width,
            "right": a.get_position().x1 * width,
            "bottom": a.get_position().y0 * height,
            "top": a.get_position().y1 * height,
        }
        for a in axes
    ]
    fail = []
    for i, a in enumerate(rects):
        for j, b in enumerate(rects[:i]):
            if abs(a["top"] - b["top"]) < tolerance_pt:
                if abs(a["bottom"] - b["bottom"]) > tolerance_pt:
                    fail.append([i, j, "same-row height"])
            if abs(a["left"] - b["left"]) < tolerance_pt:
                if abs(a["right"] - b["right"]) > tolerance_pt:
                    fail.append([i, j, "same-column width"])
    out = {
        "scope": "Axes rectangles only; does not replace original rendered glyph/collision QA",
        "status": "FAIL" if fail else "PASS" if len(rects) > 1 else "NOT_APPLICABLE",
        "rectangles_pt": rects,
        "failures": fail,
    }
    Path(json_out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    if overlay_svg:
        body = "".join(
            f'''<rect x="{r["left"]}" y="{height - r["top"]}" width="{r["right"] - r["left"]}" height="{r["top"] - r["bottom"]}" fill="none" stroke="black"/>'''
            for r in rects
        )
        Path(overlay_svg).write_text(
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}pt" height="{height}pt" viewBox="0 0 {width} {height}">{body}</svg>',
            encoding="utf-8",
        )
    if strict and fail:
        raise AssertionError("Plot-area alignment failed")
    return out
