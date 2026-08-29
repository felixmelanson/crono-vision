#!/usr/bin/env python3
"""
Build the PWA icons from assets/logo.png.

The source is a transparent-background mark (a copper aperture/pinwheel),
not square (960x920), so this pads it onto a square canvas before scaling.
iOS ignores alpha on home-screen icons — a transparent icon renders as
black — so every output here gets composited onto a solid background
first. The favicon in Artifact previews etc. can use the transparent
original directly; only these fixed-size app icons need the fill.

    python make_icons.py

Requires Pillow (`pip install pillow`) — this is the one script in the repo
that does, since the deployed app itself has zero image-processing needs.
"""

import os

from PIL import Image

SRC = "assets/logo.png"
BG = (10, 8, 6, 255)  # near-black with a hint of warmth, echoes the mark's copper

# How much of the icon canvas the logo itself occupies. iOS applies its own
# rounded-square mask and, on top of that, tends to crop corners more
# aggressively than you'd expect — leaving margin keeps the swirl from
# looking clipped.
FILL = 0.82


def build(size: int, mark: Image.Image) -> Image.Image:
    canvas = Image.new("RGBA", (size, size), BG)
    target = int(size * FILL)
    scaled = mark.resize((target, target), Image.LANCZOS)
    offset = ((size - target) // 2, (size - target) // 2)
    canvas.alpha_composite(scaled, offset)
    return canvas.convert("RGB")  # no alpha in the output — see module docstring


def main() -> None:
    logo = Image.open(SRC).convert("RGBA")
    w, h = logo.size
    side = max(w, h)

    # Pad the non-square source onto a transparent square first, so the
    # mark isn't stretched or off-center in a square icon.
    square = Image.new("RGBA", (side, side), (0, 0, 0, 0))
    square.alpha_composite(logo, ((side - w) // 2, (side - h) // 2))

    for size, name in ((180, "icon-180.png"), (192, "icon-192.png"), (512, "icon-512.png")):
        out = build(size, square)
        out.save(name)
        print(f"  {name}  {size}x{size}  {os.path.getsize(name):,} bytes")


if __name__ == "__main__":
    main()
