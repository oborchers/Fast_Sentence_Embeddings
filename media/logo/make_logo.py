"""Generate the fse logo: two SVGs and their PNG exports, next to this file.

The wordmark is built from angular letters on a 512 grid; the crossbar of the e
leaves the word as an arrow. The SVGs contain only polygons, no font and no
raster. PNG exports need ``rsvg-convert`` (librsvg)::

    python media/logo/make_logo.py
"""

import subprocess
from pathlib import Path

HERE = Path(__file__).parent

S = 34  # stroke
TOP, BASE = 206, 382  # x-height band
ASC = 130  # f ascender top
C = 18  # 45 degree chamfer

Point = tuple[float, float]


def rect(x: float, y: float, w: float, h: float) -> list[Point]:
    return [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]


def letters(x0: float) -> tuple[list[list[Point]], float, float, float]:
    shapes, x = [], x0
    # f: stem with chamfered top-left, hook to the right, crossbar
    fw = 88
    sx = x + 22
    shapes.append(
        [
            (sx + C, ASC),
            (sx + fw - 22, ASC),
            (sx + fw - 22, ASC + S),
            (sx + S, ASC + S),
            (sx + S, BASE),
            (sx, BASE),
            (sx, ASC + C),
        ]
    )
    shapes.append(rect(x, TOP, fw, S))
    x += fw + 16
    # s: chamfered outer corners, short terminals so it never reads as a 5
    sw = 112
    mid = (TOP + BASE) / 2 - S / 2
    shapes.append(
        [
            (x + C, TOP),
            (x + sw, TOP),
            (x + sw, TOP + S + 14),
            (x + sw - S, TOP + S + 14),
            (x + sw - S, TOP + S),
            (x + S, TOP + S),
            (x + S, mid),
            (x, mid),
            (x, TOP + C),
        ]
    )
    shapes.append(rect(x, mid, sw, S))
    shapes.append(
        [
            (x + sw - S, mid),
            (x + sw, mid),
            (x + sw, BASE - C),
            (x + sw - C, BASE),
            (x, BASE),
            (x, BASE - S - 14),
            (x + S, BASE - S - 14),
            (x + S, BASE - S),
            (x + sw - S, BASE - S),
        ]
    )
    x += sw + 16
    # e: chamfered top-left, wide counter, open lower right
    ew = 128
    cb = TOP + 74  # crossbar top
    shapes.append(
        [
            (x + C, TOP),
            (x + ew, TOP),
            (x + ew, cb),
            (x + ew - S, cb),
            (x + ew - S, TOP + S),
            (x + S, TOP + S),
            (x + S, BASE - S),
            (x + ew - 6, BASE - S),
            (x + ew - 6, BASE),
            (x, BASE),
            (x, TOP + C),
        ]
    )
    return shapes, x, ew, cb


def arrow(x: float, ew: float, y: float) -> list[Point]:
    # The crossbar runs through the e and leaves it as a straight arrow.
    head = 40
    tail_end = x + ew + 18
    return [
        (x + S, y),
        (tail_end, y),
        (tail_end, y - 16),
        (tail_end + head, y + S / 2),
        (tail_end, y + S + 16),
        (tail_end, y + S),
        (x + S, y + S),
    ]


def polygon(points: list[Point], fill: str) -> str:
    return '<polygon fill="{}" points="{}"/>'.format(
        fill, " ".join(f"{x:g},{y:g}" for x, y in points)
    )


def svg(bg: str, ink: str, accent: str, x0: float = 46) -> str:
    shapes, ex, ew, ay = letters(x0)
    body = "\n    ".join(polygon(p, ink) for p in shapes)
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512" width="512" height="512" role="img" aria-label="fse">
  <title>fse</title>
  <rect width="512" height="512" fill="{bg}"/>
  <g>
    {body}
    {polygon(arrow(ex, ew, ay), accent)}
  </g>
</svg>
"""


VARIANTS = {
    # light: warm white, anthracite, forest green
    "fse-logo": ("#FAFAF8", "#1E1B18", "#2D6A4F"),
    # dark: warm near-black, off-white, light forest green
    "fse-logo-dark": ("#0F0D0A", "#E8E4DF", "#5B9E7D"),
}

if __name__ == "__main__":
    for name, (bg, ink, accent) in VARIANTS.items():
        source = HERE / f"{name}.svg"
        source.write_text(svg(bg, ink, accent))
        for px in (1024, 512, 64, 32):
            out = HERE / f"{name}-{px}.png"
            subprocess.run(
                ["rsvg-convert", "-w", str(px), "-h", str(px), str(source), "-o", str(out)],
                check=True,
            )
