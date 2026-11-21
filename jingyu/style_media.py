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
                fill=min(255, pressure * (65 + ((key >> 19) & 63)) // 105),
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
    """Layer satin-style thread marks without bleeding colour across regions.

    Shared grayscale stroke masks let every thread pick up its *destination*
    pixel's palette colour. Unlike painting sampled RGB lines directly, this
    keeps small differently coloured shapes intact at stitch boundaries.
    """

    flat = image.quantize(
        colors=8, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE
    ).convert("RGB")
    fabric = Image.new("RGB", image.size, (231, 223, 204))
    pitch = max(6, round(12 * scale))
    stitch_width = max(1, round(2.1 * scale))
    stride = max(3, pitch // 3)
    width, height = image.size

    # Fine warp and weft remain visible between the laid threads.
    weave = ImageDraw.Draw(fabric)
    for y in range(0, height, max(3, pitch // 2)):
        weave.line((0, y, width - 1, y), fill=(224, 215, 196), width=1)
    for x in range(0, width, max(3, pitch // 2)):
        weave.line((x, 0, x, height - 1), fill=(226, 218, 199), width=1)

    under = Image.new("L", image.size, 0)
    core = Image.new("L", image.size, 0)
    ridge = Image.new("L", image.size, 0)
    under_draw, core_draw, ridge_draw = (
        ImageDraw.Draw(under),
        ImageDraw.Draw(core),
        ImageDraw.Draw(ridge),
    )

    for row, y in enumerate(range(0, height, stride)):
        for x in range(0, width, pitch):
            key = (
                (x + 1) * 73856093 ^ (y + 1) * 19349663 ^ (seed + 199) * 83492791
            ) & 0xFFFFFFFF
            cx = min(width - 1, x + pitch // 2)
            cy = min(height - 1, y + stride // 2)
            colour = cast(tuple[int, int, int], flat.getpixel((cx, cy)))
            # Preserve exposed fabric in light regions instead of filling a grid.
            if sum(colour) >= 705 and (key & 3) != 0:
                continue
            shift = ((key >> 12) % 3) - 1
            left = max(0, x + shift)
            right = min(width - 1, x + pitch - 2 + shift)
            if right <= left:
                continue
            top = max(0, y - stride // 2)
            bottom = min(height - 1, y + stride + stride // 2)
            segment = (left, bottom, right, top) if row & 1 else (left, top, right, bottom)
            under_draw.line(segment, fill=255, width=stitch_width + 2)
            core_draw.line(segment, fill=255, width=stitch_width)
            x0, y0, x1, y1 = segment
            ridge_draw.line(
                (x0, max(0, y0 - 1), x1, max(0, y1 - 1)),
                fill=255,
                width=1,
            )

    dark = flat.point(lambda value: value * 3 // 4)
    bright = flat.point(lambda value: min(255, value + 32))
    fabric = Image.composite(dark, fabric, under)
    fabric = Image.composite(flat, fabric, core)
    fabric = Image.composite(bright, fabric, ridge)
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
    glossy = Image.composite(Image.new("RGB", image.size, (245, 248, 250)), covered, shine)
    return _on(glossy, paper)


__all__ = ["embroidery", "frottage", "glaze"]
