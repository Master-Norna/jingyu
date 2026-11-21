"""Style presets: interpret physically lit renders using deterministic 2D drawing.

The render stays physically lit; a style re-interprets it the way a painter
would, with classic image operations only (no learned model):

* lines come from the exact id mask (where one object or part meets another)
  plus strong changes of value, so they follow the scene's real shapes;
* flat colour is value quantised in a few steps while hue is kept;
* oil strokes come from a Kuwahara filter (each pixel takes the calmest of four
  neighbouring windows), which keeps edges and melts texture into dabs;
* watercolour darkens at the edges of washes, granulates in the paper, and lets
  the white of the paper through;
* paper and canvas are deterministic procedural textures, so the same scene
  always gives the same picture.

Everything is Pillow; the photographic render is kept next to the styled image.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from functools import lru_cache
from typing import Any

from PIL import Image, ImageChops, ImageDraw, ImageEnhance, ImageFilter, ImageMath, ImageOps

from .geometry.noise import value_noise

PRESETS: dict[str, str] = {
    "none": "The render as it is.",
    "ink": "Pen lines along every edge over quiet, flattened colour on paper.",
    "watercolor": "Soft washes that darken at their edges, pigment settling in the paper grain.",
    "oil": "Dabs of paint that keep the big shapes and melt small detail, on canvas.",
    "cel": "Flat bands of light and shadow with bold outlines, like animation.",
    "charcoal": "Soft monochrome shading, smudged edges and dry pigment on warm paper.",
    "woodcut": "Dark relief blocks and carved hatching on warm printing paper.",
    "pencil": "Graphite values with layered diagonal and cross hatching on drawing paper.",
    "stippling": "Tonal drawing made of discrete ink dots whose density follows shadow.",
    "gouache": "Opaque matte colour planes with softened paint edges and fine pigment grain.",
}

#: Line widths and filter sizes are given for this image width and scaled with it.
REFERENCE_WIDTH = 1280


def apply_style(
    image: Image.Image, id_mask: Image.Image | None, spec: Mapping[str, Any]
) -> Image.Image:
    """The styled image; *id_mask* (same size, RGBA) provides the object outlines."""

    preset = spec.get("preset", "none")
    if preset == "none":
        return image
    rgb = image.convert("RGB")
    alpha = image.getchannel("A") if "A" in image.getbands() else None
    scale = rgb.width / REFERENCE_WIDTH
    seed = int(spec.get("seed", 0))
    line = float(spec.get("line_width", 1.5)) * scale
    paper = float(spec.get("paper", 0.5))
    if preset == "ink":
        styled = _ink(rgb, id_mask, line, paper, seed)
    elif preset == "watercolor":
        styled = _watercolor(rgb, id_mask, scale, paper, seed)
    elif preset == "oil":
        styled = _oil(rgb, scale, paper, seed)
    elif preset == "cel":
        styled = _cel(rgb, id_mask, line, int(spec.get("levels", 3)))
    elif preset == "charcoal":
        styled = _charcoal(rgb, id_mask, line, scale, paper, seed)
    elif preset == "woodcut":
        styled = _woodcut(rgb, id_mask, line, scale, paper, seed)
    elif preset == "pencil":
        styled = _pencil(rgb, id_mask, line, scale, paper, seed)
    elif preset == "stippling":
        styled = _stippling(rgb, scale, paper, seed)
    elif preset == "gouache":
        styled = _gouache(rgb, scale, paper, seed)
    else:
        raise ValueError(f"unknown style preset {preset!r}")
    strength = float(spec.get("strength", 1.0))
    result = styled if strength >= 1.0 else Image.blend(rgb, styled, strength)
    if alpha is not None:
        result.putalpha(alpha)
    return result


# ------------------------------------------------------------------- building blocks


def outlines(id_mask: Image.Image | None, size: tuple[int, int], width: float) -> Image.Image:
    """White where one object or part meets another (or the background), as "L"."""

    if id_mask is None or width <= 0:
        return Image.new("L", size, 0)
    mask = id_mask.convert("RGBA")
    edges = Image.new("L", mask.size, 0)
    for dx, dy in ((1, 0), (0, 1)):
        # A wrapped offset joins opposite image borders, inventing outlines.
        shifted = Image.new("RGBA", mask.size, (0, 0, 0, 0))
        shifted.paste(mask, (dx, dy))
        # Ids live in the low bits of the channels: any difference in any channel.
        for channel in ImageChops.difference(mask, shifted).split():
            edges = ImageChops.lighter(edges, channel.point(lambda v: 255 if v else 0))
    return _thicken(edges, width)


def value_edges(image: Image.Image, width: float, threshold: int) -> Image.Image:
    """White where the value changes sharply inside objects (creases, cast shadows)."""

    gray = ImageOps.grayscale(image).filter(ImageFilter.GaussianBlur(max(0.8, width)))
    found = gray.filter(ImageFilter.FIND_EDGES).point(lambda v: 255 if v > threshold else 0)
    return _thicken(found, width * 0.7)


def _thicken(lines: Image.Image, width: float) -> Image.Image:
    size = max(1, round(width))
    if size > 1:
        lines = lines.filter(ImageFilter.MaxFilter(size if size % 2 else size + 1))
    return lines.filter(ImageFilter.GaussianBlur(0.5))


def flatten(image: Image.Image, levels: int, softness: float) -> Image.Image:
    """Quantise value into *levels* steps, keeping hue and saturation."""

    h, s, v = image.convert("HSV").split()
    step = 255 / max(1, levels - 1) if levels > 1 else 255

    def band(value: int) -> int:
        position = value / step
        low = math.floor(position)
        t = position - low
        if softness > 0:
            t = min(1.0, max(0.0, (t - 0.5) / softness + 0.5))
        else:
            t = 1.0 if t >= 0.5 else 0.0
        return int(min(255, (low + t) * step))

    return Image.merge("HSV", (h, s, v.point(band))).convert("RGB")


@lru_cache(maxsize=8)
def _noise_tile(size: int, seed: int, feature: float) -> Image.Image:
    """A repeatable grey noise tile ("L"), features about *feature* pixels across."""

    data = bytearray(size * size)
    for y in range(size):
        for x in range(size):
            value = value_noise(x / feature, y / feature, 0.0, seed)
            data[y * size + x] = int(value * 255)
    return Image.frombytes("L", (size, size), bytes(data))


def paper_texture(size: tuple[int, int], seed: int, amount: float) -> Image.Image:
    """Paper: fine fibres and a soft mottle, as a multiplier image ("L", 255 = none)."""

    width, height = size
    grain = _noise_tile(256, seed, 1.6)
    tiled = Image.new("L", size)
    for y in range(0, height, 256):
        for x in range(0, width, 256):
            tiled.paste(grain, (x, y))
    mottle = _noise_tile(64, seed + 1, 8.0).resize(size, Image.Resampling.BICUBIC)
    fibres = Image.blend(tiled, mottle, 0.5)
    low = int(255 - 40 * amount)
    return fibres.point(lambda v: low + (255 - low) * v // 255)


def canvas_texture(size: tuple[int, int], amount: float, scale: float) -> Image.Image:
    """Woven canvas: a fine grid of threads, as a multiplier image."""

    width, height = size
    pitch = max(2.0, 3.0 * scale)
    low = int(255 - 35 * amount)
    column = bytes(
        low + int((255 - low) * (0.5 + 0.5 * math.cos(2 * math.pi * x / pitch)))
        for x in range(width)
    )
    rows = Image.frombytes("L", (width, 1), column).resize((width, height))
    row = bytes(
        low + int((255 - low) * (0.5 + 0.5 * math.cos(2 * math.pi * y / pitch)))
        for y in range(height)
    )
    cols = Image.frombytes("L", (1, height), row).resize((width, height))
    return ImageChops.multiply(rows, cols).point(lambda v: min(255, v + (255 - low) // 2))


def _shift_clamped(image: Image.Image, dx: int, dy: int) -> Image.Image:
    """Shift an image with edge extension, never joining opposite borders."""

    width, height = image.size
    dx = max(-width, min(width, dx))
    dy = max(-height, min(height, dy))
    shifted = image
    if dx:
        shifted = ImageChops.offset(shifted, dx, 0)
        left = dx > 0
        column = image.crop((0 if left else width - 1, 0, 1 if left else width, height))
        shifted.paste(column.resize((abs(dx), height)), (0 if left else width - abs(dx), 0))
    if dy:
        before = shifted
        shifted = ImageChops.offset(shifted, 0, dy)
        top = dy > 0
        row = before.crop((0, 0 if top else height - 1, width, 1 if top else height))
        shifted.paste(row.resize((width, abs(dy))), (0, 0 if top else height - abs(dy)))
    return shifted


def kuwahara(image: Image.Image, radius: int) -> Image.Image:
    """Each pixel takes the mean colour of the calmest of its four quadrant windows.

    Calm is measured as the mean absolute deviation of value in the window, which
    ranks windows like the variance does and stays within 8-bit images.
    """

    if radius < 1:
        return image
    half = radius // 2 + 1
    blur = ImageFilter.BoxBlur(half)
    luma = ImageOps.grayscale(image)
    deviation = ImageChops.difference(luma, luma.filter(blur)).filter(blur)
    means = image.filter(blur)
    best_spread: Image.Image | None = None
    best: Image.Image | None = None
    for dx, dy in ((-half, -half), (half, -half), (-half, half), (half, half)):
        spread = _shift_clamped(deviation, dx, dy)
        shifted = _shift_clamped(means, dx, dy)
        if best_spread is None or best is None:
            best_spread, best = spread, shifted
            continue
        calmer = ImageMath.lambda_eval(
            lambda a: (a["s"] < a["b"]) * 255, s=spread, b=best_spread
        ).convert("L")
        best = Image.composite(shifted, best, calmer)
        best_spread = ImageChops.darker(spread, best_spread)
    assert best is not None
    return best


# ---------------------------------------------------------------------- presets


def _ink(
    image: Image.Image, id_mask: Image.Image | None, line: float, paper: float, seed: int
) -> Image.Image:
    tone = ImageEnhance.Color(flatten(image, 4, 0.35)).enhance(0.55)
    cream = Image.new("RGB", image.size, (246, 241, 229))
    washed = Image.blend(tone, cream, 0.35)
    if line > 0:
        lines = ImageChops.lighter(
            outlines(id_mask, image.size, line), value_edges(image, line, 40)
        )
        ink = Image.new("RGB", image.size, (28, 26, 30))
        washed = Image.composite(ink, washed, lines)
    return _on(washed, paper_texture(image.size, seed, paper))


def _watercolor(
    image: Image.Image, id_mask: Image.Image | None, scale: float, paper: float, seed: int
) -> Image.Image:
    wash = flatten(image.filter(ImageFilter.GaussianBlur(2.5 * scale)), 6, 0.6)
    wash = ImageEnhance.Color(wash).enhance(0.85)
    wash = Image.blend(wash, Image.new("RGB", image.size, (250, 247, 240)), 0.18)
    rims = outlines(id_mask, image.size, 2.5 * scale).filter(ImageFilter.GaussianBlur(2.0 * scale))
    darker = ImageEnhance.Brightness(wash).enhance(0.72)
    edged = Image.composite(darker, wash, rims.point(lambda v: v * 3 // 5))
    grain = paper_texture(image.size, seed + 7, 0.6 + 0.4 * paper)
    return _on(edged, grain)


def _oil(image: Image.Image, scale: float, paper: float, seed: int) -> Image.Image:
    painted = kuwahara(image, max(2, round(5 * scale)))
    painted = kuwahara(painted, max(1, round(3 * scale)))
    painted = ImageEnhance.Color(painted).enhance(1.15)
    return _on(painted, canvas_texture(image.size, 0.4 + 0.6 * paper, scale))


def _cel(image: Image.Image, id_mask: Image.Image | None, line: float, levels: int) -> Image.Image:
    bands = ImageEnhance.Color(flatten(image, max(2, levels), 0.0)).enhance(1.2)
    lines = outlines(id_mask, image.size, line * 1.6)
    ink = Image.new("RGB", image.size, (20, 20, 24))
    return Image.composite(ink, bands, lines)


def _charcoal(
    image: Image.Image,
    id_mask: Image.Image | None,
    line: float,
    scale: float,
    paper: float,
    seed: int,
) -> Image.Image:
    """Smudged graphite values, dry grain and scene-aware sketch lines."""

    value = ImageOps.grayscale(image).filter(ImageFilter.GaussianBlur(max(0.45, 1.4 * scale)))
    pigment = ImageChops.multiply(
        ImageOps.invert(value), paper_texture(image.size, seed + 23, paper)
    )
    if line > 0:
        detail = ImageChops.lighter(
            outlines(id_mask, image.size, line), value_edges(image, line, 40)
        )
        pigment = ImageChops.lighter(pigment, detail.point(lambda v: v * 3 // 4))
    page = Image.new("RGB", image.size, (248, 245, 237))
    coal = Image.new("RGB", image.size, (34, 32, 30))
    return Image.composite(coal, page, pigment)


def _woodcut(
    image: Image.Image,
    id_mask: Image.Image | None,
    line: float,
    scale: float,
    paper: float,
    seed: int,
) -> Image.Image:
    """Relief-black shadows with directional cuts through the midtones."""

    shade = ImageOps.invert(
        ImageOps.grayscale(image).filter(ImageFilter.GaussianBlur(max(0.5, 0.8 * scale)))
    )
    solid = shade.point(lambda value: 255 if value >= 170 else 0)
    hatch = Image.new("L", image.size, 0)
    pitch = max(4, round(11 * scale))
    stroke = max(1, round(1.3 * scale))
    width, height = image.size
    draw = ImageDraw.Draw(hatch)
    for start in range(-height, width + height, pitch):
        draw.line((start, 0, start + height, height), fill=255, width=stroke)
    middle = shade.point(lambda value: min(255, max(0, (value - 45) * 3)))
    ink = ImageChops.lighter(solid, ImageChops.multiply(hatch, middle))
    if line > 0:
        ink = ImageChops.lighter(ink, outlines(id_mask, image.size, line * 1.6))
    paper_color = Image.new("RGB", image.size, (248, 242, 225))
    dark = Image.new("RGB", image.size, (36, 31, 26))
    pressed = Image.composite(dark, paper_color, ink)
    return _on(pressed, paper_texture(image.size, seed + 41, paper * 0.55))


def _diagonal_hatch(
    size: tuple[int, int], pitch: int, thickness: int, descending: bool
) -> Image.Image:
    """Small, consistent graphite hatches, with no wrapping at image borders."""

    width, height = size
    lines = Image.new("L", size, 0)
    draw = ImageDraw.Draw(lines)
    for offset in range(-height, width + height, pitch):
        if descending:
            draw.line((offset, 0, offset + height, height), fill=255, width=thickness)
        else:
            draw.line((offset, 0, offset - height, height), fill=255, width=thickness)
    return lines


def _pencil(
    image: Image.Image,
    id_mask: Image.Image | None,
    line: float,
    scale: float,
    paper: float,
    seed: int,
) -> Image.Image:
    """Build graphite shading from a light grey underdrawing and crossed hatches."""

    value = ImageOps.grayscale(image)
    shadow = ImageOps.invert(value.filter(ImageFilter.GaussianBlur(max(0.4, scale))))
    pitch = max(3, round(9 * scale))
    thickness = max(1, round(1.1 * scale))
    underdrawing = shadow.point(lambda v: v // 3)
    primary = ImageChops.multiply(
        _diagonal_hatch(image.size, pitch, thickness, True),
        shadow.point(lambda v: min(255, max(0, (v - 15) * 2))),
    )
    crossed = ImageChops.multiply(
        _diagonal_hatch(image.size, pitch, thickness, False),
        shadow.point(lambda v: min(255, max(0, (v - 120) * 3))),
    )
    graphite = ImageChops.lighter(underdrawing, ImageChops.lighter(primary, crossed))
    if line > 0:
        detail = ImageChops.lighter(
            outlines(id_mask, image.size, line), value_edges(image, line, 45)
        )
        graphite = ImageChops.lighter(graphite, detail.point(lambda v: v * 4 // 5))
    grain = paper_texture(image.size, seed + 61, paper)
    graphite = ImageChops.multiply(graphite, grain)
    page = Image.new("RGB", image.size, (249, 247, 240))
    dark = Image.new("RGB", image.size, (36, 35, 37))
    return Image.composite(dark, page, graphite)


def _stippling(image: Image.Image, scale: float, paper: float, seed: int) -> Image.Image:
    """Use distinct ink dots, not continuous hatches or whole-image pixel noise."""

    width, height = image.size
    pitch = max(4, round(9 * scale))
    radius = max(0.65, 1.9 * scale)
    shade = ImageOps.invert(ImageOps.grayscale(image))
    ink = Image.new("L", image.size, 0)
    dots = ImageDraw.Draw(ink)
    for y in range(0, height, pitch):
        for x in range(0, width, pitch):
            darkness = shade.getpixel(
                (min(width - 1, x + pitch // 2), min(height - 1, y + pitch // 2))
            )
            for mark in range(3):
                key = (
                    (x + 1) * 73856093 ^ (y + 1) * 19349663 ^ (seed + mark + 1) * 83492791
                ) & 0xFFFFFFFF
                if (key & 0xFFFF) / 65535 >= darkness / 255 * (0.92 - mark * 0.08):
                    continue
                px = min(width - 1, x + (key >> 16) % min(pitch, width - x))
                py = min(height - 1, y + (key >> 24) % min(pitch, height - y))
                dots.ellipse((px - radius, py - radius, px + radius, py + radius), fill=255)
    pigment = Image.new("RGB", image.size, (32, 31, 35))
    page = Image.new("RGB", image.size, (250, 247, 239))
    drawn = Image.composite(pigment, page, ink)
    return _on(drawn, paper_texture(image.size, seed + 71, paper * 0.6))


def _gouache(image: Image.Image, scale: float, paper: float, seed: int) -> Image.Image:
    """Opaque, softly broken matte colour, unlike translucent watercolor washes."""

    dabbed = kuwahara(image, max(2, round(5 * scale)))
    planes = flatten(dabbed, 5, 0.6)
    painted = Image.blend(dabbed, planes, 0.78)
    painted = ImageEnhance.Color(painted).enhance(1.13)
    matte = painted.filter(ImageFilter.GaussianBlur(max(0.2, 0.45 * scale)))
    return _on(matte, paper_texture(image.size, seed + 83, paper * 0.45))


def _on(image: Image.Image, texture: Image.Image) -> Image.Image:
    return ImageChops.multiply(image, Image.merge("RGB", (texture, texture, texture)))


__all__ = ["PRESETS", "REFERENCE_WIDTH", "apply_style", "flatten", "kuwahara", "outlines"]
