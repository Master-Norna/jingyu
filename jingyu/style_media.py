"""Deterministic material-oriented media distinct from the painterly style filters.

Each function receives a flat RGB render, scale, a paper multiplier ("L"), and a
seed. No global randomness, Blender state, or whole-image per-mark allocations.
"""

from __future__ import annotations

from typing import cast

from PIL import Image, ImageChops, ImageDraw, ImageEnhance, ImageFilter, ImageOps


def _on(image: Image.Image, paper: Image.Image) -> Image.Image:
    return ImageChops.multiply(image, Image.merge("RGB", (paper, paper, paper)))


def frottage(image: Image.Image, scale: float, paper: Image.Image, seed: int) -> Image.Image:
    """Rubbing: directional pigment catches raised-looking edges and paper ridges."""

    shade = ImageOps.invert(ImageOps.grayscale(image))
    relief = image.convert("L").filter(ImageFilter.FIND_EDGES)
    ink = ImageChops.lighter(
        shade.point(lambda value: value // 3),
        relief.point(lambda value: value * 3 // 5),
    )
    pitch = max(3, round(8 * scale))
    marks = Image.new("L", image.size, 0)
    draw = ImageDraw.Draw(marks)
    for y in range(0, image.height, pitch):
        for x in range(0, image.width, pitch * 2):
            key = ((x + 1) * 73856093 ^ (y + 1) * 19349663 ^ (seed + 197) * 83492791) & 0xFFFFFFFF
            px = min(image.width - 1, x + (key >> 8) % min(pitch, image.width - x))
            py = min(image.height - 1, y + (key >> 16) % min(pitch, image.height - y))
            pressure = cast(int, shade.getpixel((px, py)))
            if pressure < 30:
                continue
            length = pitch + (key >> 23) % (pitch + 1)
            draw.line(
                (px, py, min(image.width - 1, px + length), py),
                fill=min(255, pressure),
                width=max(1, round(1.4 * scale)),
            )
    ink = ImageChops.lighter(ink, marks)
    drawn = Image.composite(
        Image.new("RGB", image.size, (48, 45, 43)),
        Image.new("RGB", image.size, (246, 241, 229)),
        ink,
    )
    return _on(drawn, paper)


def embroidery(image: Image.Image, scale: float, paper: Image.Image, seed: int) -> Image.Image:
    """Form the colour image from separate diagonal thread stitches on cloth."""

    quantized = image.quantize(
        colors=8, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE
    )
    flat = quantized.convert("RGB")
    fabric = Image.new("RGB", image.size, (231, 223, 204))
    draw = ImageDraw.Draw(fabric)
    pitch = max(5, round(15 * scale))
    width = max(1, round(2.2 * scale))
    for y in range(0, image.height, pitch):
        for x in range(0, image.width, pitch):
            cx = min(image.width - 1, x + pitch // 2)
            cy = min(image.height - 1, y + pitch // 2)
            index = cast(int, quantized.getpixel((cx, cy)))
            colour = cast(tuple[int, int, int], flat.getpixel((cx, cy)))
            x0, y0 = x + 1, y + 1
            x1 = min(image.width - 1, x + pitch - 2)
            y1 = min(image.height - 1, y + pitch - 2)
            if x1 <= x0 or y1 <= y0:
                continue
            if (index + seed) & 1:
                segment = (x0, y1, x1, y0)
                highlight = (x0, y1 - 1, x1, max(0, y0 - 1))
            else:
                segment = (x0, y0, x1, y1)
                highlight = (x0, y0 - 1, x1, max(0, y1 - 1))
            dark = tuple(max(0, value * 4 // 5) for value in colour)
            bright = tuple(min(255, value + 34) for value in colour)
            draw.line(segment, fill=dark, width=width + 1)
            draw.line(segment, fill=colour, width=width)
            draw.line(highlight, fill=bright, width=1)
    return _on(fabric, paper)


def glaze(image: Image.Image, scale: float, paper: Image.Image, seed: int) -> Image.Image:
    """Glossy ceramic glaze: pooled colour near edges and separate bright runs."""

    smoothed = image.filter(ImageFilter.GaussianBlur(max(0.4, 1.1 * scale)))
    smoothed = ImageEnhance.Color(smoothed).enhance(1.20)
    transitions = ImageOps.grayscale(smoothed).filter(ImageFilter.FIND_EDGES)
    pools = transitions.filter(ImageFilter.GaussianBlur(max(0.6, scale)))
    pools = pools.point(lambda value: min(96, value * 2 // 5))
    dark = ImageEnhance.Brightness(smoothed).enhance(0.68)
    covered = Image.composite(dark, smoothed, pools)

    shine = Image.new("L", image.size, 0)
    drops = ImageDraw.Draw(shine)
    pitch = max(7, round(19 * scale))
    gray = ImageOps.grayscale(smoothed)
    for y in range(0, image.height, pitch):
        for x in range(0, image.width, pitch):
            key = ((x + 1) * 73856093 ^ (y + 1) * 19349663 ^ (seed + 211) * 83492791) & 0xFFFFFFFF
            px = min(image.width - 1, x + (key >> 8) % min(pitch, image.width - x))
            py = min(image.height - 1, y + (key >> 16) % min(pitch, image.height - y))
            light = cast(int, gray.getpixel((px, py)))
            if light < 65 or (key & 3) != 0:
                continue
            span = max(3, pitch // 2)
            drops.line(
                (px, py, px, min(image.height - 1, py + span)),
                fill=min(155, 55 + light // 3),
                width=max(1, round(scale)),
            )
    glossy = Image.composite(
        Image.new("RGB", image.size, (245, 248, 250)), covered, shine
    )
    return _on(glossy, paper)


__all__ = ["embroidery", "frottage", "glaze"]
