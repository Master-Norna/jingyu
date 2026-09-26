"""Style presets: turn a render into ink, watercolour, oil or cel painting.

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

from PIL import Image, ImageChops, ImageEnhance, ImageFilter, ImageMath, ImageOps

from .geometry.noise import value_noise

PRESETS: dict[str, str] = {
    "none": "The render as it is.",
    "ink": "Pen lines along every edge over quiet, flattened colour on paper.",
    "watercolor": "Soft washes that darken at their edges, pigment settling in the paper grain.",
    "oil": "Dabs of paint that keep the big shapes and melt small detail, on canvas.",
    "cel": "Flat bands of light and shadow with bold outlines, like animation.",
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
    else:
        raise ValueError(f"unknown style preset {preset!r}")
    strength = float(spec.get("strength", 1.0))
    return styled if strength >= 1.0 else Image.blend(rgb, styled, strength)


# ------------------------------------------------------------------- building blocks


def outlines(id_mask: Image.Image | None, size: tuple[int, int], width: float) -> Image.Image:
    """White where one object or part meets another (or the background), as "L"."""

    if id_mask is None:
        return Image.new("L", size, 0)
    mask = id_mask.convert("RGBA")
    edges = Image.new("L", mask.size, 0)
    for dx, dy in ((1, 0), (0, 1)):
        shifted = ImageChops.offset(mask, dx, dy)
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
        spread = ImageChops.offset(deviation, dx, dy)
        shifted = ImageChops.offset(means, dx, dy)
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
    lines = ImageChops.lighter(outlines(id_mask, image.size, line), value_edges(image, line, 40))
    ink = Image.new("RGB", image.size, (28, 26, 30))
    drawn = Image.composite(ink, washed, lines)
    return _on(drawn, paper_texture(image.size, seed, paper))


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


def _on(image: Image.Image, texture: Image.Image) -> Image.Image:
    return ImageChops.multiply(image, Image.merge("RGB", (texture, texture, texture)))


__all__ = ["PRESETS", "REFERENCE_WIDTH", "apply_style", "flatten", "kuwahara", "outlines"]
