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
from typing import Any, cast

from PIL import Image, ImageChops, ImageDraw, ImageEnhance, ImageFilter, ImageMath, ImageOps

from .geometry.noise import value_noise
from .style_media import embroidery, frottage, glaze
from .style_strokes import directional_hatch, local_tangent, tapered_brush

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
    "pastel": "Layered coloured chalk marks with soft smudging and broken paper coverage.",
    "etching": "Fine engraved ink lines and graduated crosshatching on printmaking paper.",
    "mosaic": "Discrete coloured tesserae separated by visible mortar lines.",
    "crayon": "Broken, saturated wax-pencil strokes over exposed toothy paper.",
    "screenprint": "Flat limited-colour ink plates with coarse halftone and registration fringe.",
    "papercut": "Layered cut-colour silhouettes with hard paper edges and cast shadows.",
    "sumi": "Layered monochrome ink washes, flowing brush marks and pale rice paper.",
    "stainedglass": "Backlit colour panes enclosed by dark lead-came boundaries.",
    "cyanotype": "Prussian-blue photographic values and pale contact-print silhouettes.",
    "frottage": "Graphite rubbing catches surface relief and directional paper ridges.",
    "embroidery": "Coloured thread stitches render forms on exposed woven fabric.",
    "glaze": "Glossy ceramic colour with pooled edges and reflected highlight runs.",
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
    elif preset == "pastel":
        styled = _pastel(rgb, scale, paper, seed)
    elif preset == "etching":
        styled = _etching(rgb, id_mask, line, scale, paper, seed)
    elif preset == "mosaic":
        styled = _mosaic(rgb, scale, paper, seed)
    elif preset == "crayon":
        styled = _crayon(rgb, scale, paper, seed)
    elif preset == "screenprint":
        styled = _screenprint(rgb, scale, paper, seed)
    elif preset == "papercut":
        styled = _papercut(rgb, scale, paper, seed)
    elif preset == "sumi":
        styled = _sumi(rgb, id_mask, line, scale, paper, seed)
    elif preset == "stainedglass":
        styled = _stainedglass(rgb, id_mask, line, paper, seed)
    elif preset == "cyanotype":
        styled = _cyanotype(rgb, id_mask, line, scale, paper, seed)
    elif preset == "frottage":
        styled = frottage(rgb, scale, paper_texture(rgb.size, seed + 223, paper), seed)
    elif preset == "embroidery":
        styled = embroidery(rgb, scale, paper_texture(rgb.size, seed + 227, paper * 0.4), seed)
    elif preset == "glaze":
        styled = glaze(rgb, scale, paper_texture(rgb.size, seed + 229, paper * 0.15), seed)
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


def _colour_edges(image: Image.Image, width: float, threshold: int) -> Image.Image:
    """Detect pigment boundaries even when neighbouring hues share luminance."""

    softened = image.filter(ImageFilter.GaussianBlur(max(0.35, 0.5 * width)))
    edges = Image.new("L", image.size, 0)
    for channel in softened.split():
        # Edge extension avoids false full-frame borders from FIND_EDGES.
        horizontal = ImageChops.difference(channel, _shift_clamped(channel, 1, 0))
        vertical = ImageChops.difference(channel, _shift_clamped(channel, 0, 1))
        edges = ImageChops.lighter(edges, ImageChops.lighter(horizontal, vertical))
    return _thicken(edges.point(lambda value: 255 if value > threshold else 0), width)


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
    """Paper grain has fixed pixel scale, regardless of the canvas dimensions.

    Fine fibres and broad mottling share image-space coordinates; changing the
    canvas size never stretches the texture already visible in a crop.
    """

    if amount <= 0:
        return Image.new("L", size, 255)
    width, height = size
    fibres = Image.blend(
        _noise_tile(256, seed, 1.6),
        _noise_tile(256, seed + 1, 8.0),
        0.5,
    )
    tiled = Image.new("L", size)
    for y in range(0, height, 256):
        for x in range(0, width, 256):
            tiled.paste(fibres, (x, y))
    low = max(0, min(255, round(255 - 40 * amount)))
    return tiled.point(lambda v: low + (255 - low) * v // 255)


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
    wash = flatten(image.filter(ImageFilter.GaussianBlur(max(0.55, 2.5 * scale))), 6, 0.6)
    wash = ImageEnhance.Color(wash).enhance(0.85)
    wash = Image.blend(wash, Image.new("RGB", image.size, (250, 247, 240)), 0.18)

    # Pigment also settles along internal colour boundaries, not only ID masks.
    # This catches adjacent hues with equal luminance (e.g. red and green).
    width = max(0.8, 2.5 * scale)
    rims = ImageChops.lighter(
        outlines(id_mask, image.size, width),
        _colour_edges(image, width, 24),
    )
    pools = rims.filter(ImageFilter.GaussianBlur(max(0.6, 1.5 * scale)))
    darker = ImageEnhance.Brightness(wash).enhance(0.70)
    edged = Image.composite(darker, wash, pools.point(lambda value: value * 3 // 5))
    grain = paper_texture(image.size, seed + 7, 0.6 + 0.4 * paper)
    return _on(edged, grain)


def _oil(image: Image.Image, scale: float, paper: float, seed: int) -> Image.Image:
    """Keep broad forms, then place small strokes tangent to local value changes."""

    painted = kuwahara(image, max(2, round(5 * scale)))
    painted = kuwahara(painted, max(1, round(3 * scale)))
    painted = ImageEnhance.Color(painted).enhance(1.15)
    source = painted.copy()
    value = ImageOps.grayscale(image)
    draw = ImageDraw.Draw(painted)
    pitch = max(5, round(14 * scale))
    length = max(3, round(9 * scale))
    thickness = max(1, round(2.1 * scale))
    for y in range(pitch // 2, image.height, pitch):
        for x in range(pitch // 2, image.width, pitch):
            key = ((x + 1) * 73856093 ^ (y + 1) * 19349663 ^ (seed + 233) * 83492791) & 0xFFFFFFFF
            px = min(image.width - 1, max(0, x + (key >> 8) % pitch - pitch // 2))
            py = min(image.height - 1, max(0, y + (key >> 16) % pitch - pitch // 2))
            left = cast(int, value.getpixel((max(0, px - 2), py)))
            right = cast(int, value.getpixel((min(image.width - 1, px + 2), py)))
            above = cast(int, value.getpixel((px, max(0, py - 2))))
            below = cast(int, value.getpixel((px, min(image.height - 1, py + 2))))
            dx, dy = float(above - below), float(right - left)
            norm = math.hypot(dx, dy)
            if norm < 4:
                dx, dy, norm = 1.0, (key % 5 - 2) * 0.16, 1.0
            radius = length * (0.35 + 0.15 * ((key >> 24) & 3)) / norm
            x0, y0 = px - dx * radius, py - dy * radius
            x1, y1 = px + dx * radius, py + dy * radius
            color = cast(tuple[int, int, int], source.getpixel((px, py)))
            p0 = (
                min(image.width - 1, max(0, round(x0))),
                min(image.height - 1, max(0, round(y0))),
            )
            p1 = (
                min(image.width - 1, max(0, round(x1))),
                min(image.height - 1, max(0, round(y1))),
            )
            edge_colors = (
                cast(tuple[int, int, int], image.getpixel(p0)),
                cast(tuple[int, int, int], image.getpixel(p1)),
            )
            if any(max(abs(edge[i] - color[i]) for i in range(3)) > 65 for edge in edge_colors):
                continue  # Do not drag a coloured stroke across an object edge.
            body = tuple(max(0, channel - 10) for channel in color)
            ridge = tuple(min(255, channel + 18) for channel in color)
            draw.line((x0, y0, x1, y1), fill=body, width=thickness + 1)
            draw.line((x0, y0 - 1, x1, y1 - 1), fill=ridge, width=thickness)
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
    """Directional charcoal rubbing with dry-pressure marks on textured paper.

    The smudge follows local tonal contours, not a global fixed direction.
    Short, seeded strokes restore broken dry pigment after the soft rub.
    Strong tonal boundaries keep their original value to limit cross-edge bleed.
    """

    shade = ImageOps.invert(ImageOps.grayscale(image))
    soft = shade.filter(ImageFilter.GaussianBlur(max(0.45, 1.1 * scale)))
    reach = max(1, round(4 * scale))
    left = _shift_clamped(soft, -reach, 0)
    right = _shift_clamped(soft, reach, 0)
    up = _shift_clamped(soft, 0, -reach)
    down = _shift_clamped(soft, 0, reach)
    # Stronger horizontal change -> smudge vertically along that edge, and
    # vice versa. Edge extension prevents marks from wrapping across the canvas.
    horizontal = Image.blend(left, right, 0.5)
    vertical = Image.blend(up, down, 0.5)
    gx = ImageChops.difference(left, right)
    gy = ImageChops.difference(up, down)
    along_vertical = ImageMath.lambda_eval(
        lambda values: (values["x"] > values["y"]) * 255, x=gx, y=gy
    ).convert("L")
    rubbed = Image.blend(soft, Image.composite(vertical, horizontal, along_vertical), 0.52)
    # Keep large silhouettes intact rather than smearing pigment through them.
    sharp = ImageChops.difference(shade, rubbed).point(
        lambda value: min(255, max(0, (value - 12) * 10))
    )
    rubbed = Image.composite(shade, rubbed, sharp)

    # The irregular paper tooth lifts pigment from the rubbing. With paper=0,
    # the procedural tooth vanishes; directional marks remain a separate layer.
    tooth = paper_texture(image.size, seed + 23, paper * 1.5)
    pigment = ImageChops.multiply(rubbed, tooth)

    marks = Image.new("L", image.size, 0)
    draw = ImageDraw.Draw(marks)
    pitch = max(5, round(12 * scale))
    step = max(1, round(3 * scale))
    for y in range(0, image.height, pitch):
        for x in range(0, image.width, pitch):
            key = ((x + 1) * 73856093 ^ (y + 1) * 19349663 ^ (seed + 31) * 83492791) & 0xFFFFFFFF
            px = min(image.width - 1, x + (key >> 8) % min(pitch, image.width - x))
            py = min(image.height - 1, y + (key >> 16) % min(pitch, image.height - y))
            darkness = cast(int, shade.getpixel((px, py)))
            if darkness < 45 or (key & 255) > darkness:
                continue
            vx, vy = local_tangent(soft, px, py, step)
            length = max(2, round((7 + ((key >> 24) & 7)) * scale))
            thickness = max(1, round((1.5 + ((key >> 20) & 3) * 0.45) * scale))
            pressure = min(95, darkness * (30 + ((key >> 27) & 15)) // 120)
            start = (px - vx * length / 2, py - vy * length / 2)
            end = (px + vx * length / 2, py + vy * length / 2)
            draw.line((*start, *end), fill=pressure, width=thickness)
    dry = marks.filter(ImageFilter.GaussianBlur(max(0.25, 0.38 * scale)))
    # A stroke can taper into shadow, but must never paint a blank white shape.
    dry = ImageChops.multiply(dry, shade.point(lambda value: min(255, value * 2)))
    pigment = ImageChops.add(pigment, dry)
    # A pure-white source area stays unmarked even beside a dark object.
    pigment = ImageChops.multiply(pigment, shade.point(lambda value: 255 if value > 8 else 0))
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
    """Carve white gouges through solid relief ink; shade midtones with short cuts."""

    shade = ImageOps.invert(
        ImageOps.grayscale(image).filter(ImageFilter.GaussianBlur(max(0.5, 0.8 * scale)))
    )
    pitch = max(4, round(11 * scale))
    stroke = max(1, round(1.3 * scale))
    solid = shade.point(lambda value: 255 if value >= 170 else 0)
    hatch = directional_hatch(shade, pitch, stroke, seed + 41)
    middle = shade.point(lambda value: min(255, max(0, (value - 45) * 3)))
    # A relief print needs exposed wood cuts inside its black block, not only
    # another stack of black hatch lines.
    gouges = directional_hatch(shade, pitch * 3, stroke, seed + 43, crossed=True)
    deepest = shade.point(lambda value: 255 if value >= 210 else 0)
    carved = ImageChops.subtract(solid, ImageChops.multiply(gouges, deepest))
    ink = ImageChops.lighter(carved, ImageChops.multiply(hatch, middle))
    if line > 0:
        ink = ImageChops.lighter(ink, outlines(id_mask, image.size, line * 1.6))
    paper_color = Image.new("RGB", image.size, (248, 242, 225))
    dark = Image.new("RGB", image.size, (36, 31, 26))
    pressed = Image.composite(dark, paper_color, ink)
    return _on(pressed, paper_texture(image.size, seed + 41, paper * 0.55))


def _pencil(
    image: Image.Image,
    id_mask: Image.Image | None,
    line: float,
    scale: float,
    paper: float,
    seed: int,
) -> Image.Image:
    """Layer graphite with locally guided pressure-varying hatch strokes."""

    value = ImageOps.grayscale(image)
    shadow = ImageOps.invert(value.filter(ImageFilter.GaussianBlur(max(0.4, scale))))
    pitch = max(3, round(9 * scale))
    thickness = max(1, round(1.1 * scale))
    underdrawing = shadow.point(lambda v: v // 3)
    primary = ImageChops.multiply(
        directional_hatch(value, pitch, thickness, seed + 61),
        shadow.point(lambda v: min(255, max(0, (v - 15) * 2))),
    )
    crossed = ImageChops.multiply(
        directional_hatch(value, pitch + max(1, pitch // 3), thickness, seed + 67, crossed=True),
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


def _stipple_hash(value: int) -> int:
    """Avalanche the coordinate key into stable independent-looking bits."""

    value &= 0xFFFFFFFF
    value ^= value >> 16
    value = value * 0x7FEB352D & 0xFFFFFFFF
    value ^= value >> 15
    value = value * 0x846CA68B & 0xFFFFFFFF
    return value ^ (value >> 16)


def _stippling_centers(shade: Image.Image, scale: float, seed: int) -> list[tuple[int, int]]:
    """Stratified, jittered dots with local exclusion; density follows actual value.

    The old three independent candidates per square produced visible columns
    and dot clumps. A staggered lattice proposes one candidate per cell; the
    darker pixels accept more candidates, and a deterministic local priority
    rejects pairs that are too close. There is no global RNG state.
    """

    width, height = shade.size
    pitch = max(3, round(6 * scale))
    radius = max(0.65, 1.55 * scale)
    distance_sq = (2 * radius + 0.35) ** 2
    candidates: dict[tuple[int, int], tuple[int, int, int]] = {}

    for row in range((height + pitch - 1) // pitch + 1):
        for col in range((width + pitch - 1) // pitch + 1):
            key = _stipple_hash((col + 1) * 73856093 ^ (row + 1) * 19349663 ^ (seed + 1) * 83492791)
            x = col * pitch + (pitch // 2 if row & 1 else 0)
            x += round(((key & 255) / 255 - 0.5) * 0.74 * pitch)
            y = row * pitch + round((((key >> 8) & 255) / 255 - 0.5) * 0.74 * pitch)
            if not (0 <= x < width and 0 <= y < height):
                continue
            darkness = cast(int, shade.getpixel((x, y)))
            density = max(0, darkness - 8) / 247
            if (_stipple_hash(key ^ 0x9E3779B9) & 0xFFFF) / 65535 >= density:
                continue
            candidates[(col, row)] = (x, y, _stipple_hash(key ^ 0x85EBCA6B))

    centers: list[tuple[int, int]] = []
    for (col, row), (x, y, rank) in candidates.items():
        blocked = False
        for neighbor_row in range(max(0, row - 1), row + 2):
            for neighbor_col in range(max(0, col - 1), col + 2):
                other = candidates.get((neighbor_col, neighbor_row))
                if other is None:
                    continue
                nx, ny, other_rank = other
                wins = other_rank > rank or (
                    other_rank == rank and (neighbor_col, neighbor_row) < (col, row)
                )
                if wins and (nx - x) ** 2 + (ny - y) ** 2 < distance_sq:
                    blocked = True
                    break
            if blocked:
                break
        if not blocked:
            centers.append((x, y))
    return centers


def _stippling(image: Image.Image, scale: float, paper: float, seed: int) -> Image.Image:
    """Ink points spread by local separation, with continuous tonal density."""

    shade = ImageOps.invert(ImageOps.grayscale(image))
    ink = Image.new("L", image.size, 0)
    dots = ImageDraw.Draw(ink)
    radius = max(0.65, 1.55 * scale)
    for x, y in _stippling_centers(shade, scale, seed):
        dots.ellipse((x - radius, y - radius, x + radius, y + radius), fill=255)

    # A dot near a silhouette must not spill into completely unshaded paper.
    # Apply this only to the final ink mask; the placement still uses local tone.
    ink = ImageChops.multiply(ink, shade.point(lambda value: 255 if value > 8 else 0))
    pigment = Image.new("RGB", image.size, (32, 31, 35))
    page = Image.new("RGB", image.size, (250, 247, 239))
    drawn = Image.composite(pigment, page, ink)
    return _on(drawn, paper_texture(image.size, seed + 71, paper * 0.6))


def _gouache(image: Image.Image, scale: float, paper: float, seed: int) -> Image.Image:
    """Opaque matte paint planes with seeded, colour-safe brush relief.

    The underpainting suppresses tiny photographic texture; raised strokes
    modulate the *destination* colour instead of carrying sampled pigment
    across boundaries. Sharp colour transitions retain their source hue.
    """

    dabbed = kuwahara(image, max(2, round(5 * scale)))
    planes = flatten(dabbed, 5, 0.6)
    underpaint = ImageEnhance.Color(Image.blend(dabbed, planes, 0.78)).enhance(1.13)
    value = ImageOps.grayscale(image)

    shadow = Image.new("L", image.size, 0)
    ridge = Image.new("L", image.size, 0)
    draw_shadow = ImageDraw.Draw(shadow)
    draw_ridge = ImageDraw.Draw(ridge)
    pitch = max(5, round(13 * scale))
    span = max(2, round(11 * scale))
    thickness = max(1, round(3.1 * scale))
    width, height = image.size

    for y in range(0, height, pitch):
        for x in range(0, width, pitch):
            key = ((x + 1) * 73856093 ^ (y + 1) * 19349663 ^ (seed + 193) * 83492791) & 0xFFFFFFFF
            px = min(width - 1, x + (key >> 8) % min(pitch, width - x))
            py = min(height - 1, y + (key >> 16) % min(pitch, height - y))
            vx, vy = local_tangent(value, px, py, max(1, round(3 * scale)))
            # Small seeded direction changes keep large flat regions from
            # looking like a uniform hatch, without changing local form.
            bend = (((key >> 24) & 7) - 3) * 0.045
            dx, dy = vx + bend * vy, vy - bend * vx
            reach = span * (0.34 + ((key >> 27) & 3) * 0.09)
            x0, y0 = px - dx * reach, py - dy * reach
            x1, y1 = px + dx * reach, py + dy * reach

            center = cast(tuple[int, int, int], image.getpixel((px, py)))
            endpoints = (
                (min(width - 1, max(0, round(x0))), min(height - 1, max(0, round(y0)))),
                (min(width - 1, max(0, round(x1))), min(height - 1, max(0, round(y1)))),
            )
            crosses_edge = False
            for endpoint in endpoints:
                other_colour = cast(tuple[int, int, int], image.getpixel(endpoint))
                if max(abs(center[i] - other_colour[i]) for i in range(3)) > 48:
                    crosses_edge = True
                    break
            if crosses_edge:
                continue  # A single paint dab must not bridge two coloured objects.

            pressure = 115 + ((key >> 19) & 63)
            draw_shadow.line((x0, y0, x1, y1), fill=pressure, width=thickness + 1)
            # One light raised edge and occasional breaks suggest loaded paint.
            if (key & 3) != 0:
                offset = 0.55 + ((key >> 22) & 3) * 0.12
                draw_ridge.line(
                    (
                        x0 - dy * offset, y0 + dx * offset,
                        x1 - dy * offset, y1 + dx * offset,
                    ),
                    fill=100 + ((key >> 17) & 63),
                    width=max(1, thickness // 2),
                )

    deeper = ImageEnhance.Brightness(underpaint).enhance(0.91)
    raised = ImageEnhance.Brightness(underpaint).enhance(1.11)
    painted = Image.composite(deeper, underpaint, shadow)
    painted = Image.composite(raised, painted, ridge)

    # Gouache makes a crisp, opaque colour boundary, not a translucent
    # watercolour bloom or a dark cartoon contour.
    boundaries = _colour_edges(image, max(0.8, 1.2 * scale), 24)
    painted = Image.composite(image, painted, boundaries)
    return _on(painted, paper_texture(image.size, seed + 83, paper * 0.35))


def _pastel(image: Image.Image, scale: float, paper: float, seed: int) -> Image.Image:
    """Smudge coloured chalk beneath short directional marks on toothy paper."""

    smudged = kuwahara(image, max(2, round(6 * scale)))
    softened = smudged.filter(ImageFilter.GaussianBlur(max(0.5, 1.8 * scale)))
    base = Image.blend(smudged, softened, 0.55)
    base = ImageEnhance.Color(base).enhance(1.25)
    base = Image.blend(base, Image.new("RGB", image.size, (253, 246, 235)), 0.07)

    pitch = max(5, round(12 * scale))
    span = max(2, round(7 * scale))
    width = max(1, round(2 * scale))
    marks = ImageDraw.Draw(base)
    for y in range(0, image.height, pitch):
        for x in range(0, image.width, pitch):
            key = ((x + 1) * 73856093 ^ (y + 1) * 19349663 ^ (seed + 101) * 83492791) & 0xFFFFFFFF
            px = min(image.width - 1, x + (key >> 8) % min(pitch, image.width - x))
            py = min(image.height - 1, y + (key >> 16) % min(pitch, image.height - y))
            sample = cast(tuple[int, int, int], image.getpixel((px, py)))
            chalk = tuple(min(255, round(channel * 0.88 + 237 * 0.12)) for channel in sample)
            direction = -1 if key & 1 else 1
            marks.line(
                (px, py, px + direction * span, py - max(1, span // 3)),
                fill=chalk,
                width=width,
            )
    return _on(base, paper_texture(image.size, seed + 107, paper * 0.9))


def _etching(
    image: Image.Image,
    id_mask: Image.Image | None,
    line: float,
    scale: float,
    paper: float,
    seed: int,
) -> Image.Image:
    """Use contour-guided engraved strokes, layered crosshatching and open paper."""

    value = ImageOps.grayscale(image)
    shade = ImageOps.invert(value.filter(ImageFilter.GaussianBlur(max(0.45, scale))))
    pitch = max(4, round(11 * scale))
    fine = max(1, round(1.2 * scale))
    primary = ImageChops.multiply(
        directional_hatch(value, pitch, fine, seed + 113),
        shade.point(lambda v: min(255, max(0, (v - 20) * 3))),
    )
    secondary = ImageChops.multiply(
        directional_hatch(value, pitch + max(1, pitch // 3), fine, seed + 117, crossed=True),
        shade.point(lambda v: min(255, max(0, (v - 95) * 3))),
    )
    # Short rather than page-spanning deep incisions keep a visible paper grain.
    deepest = ImageChops.multiply(
        directional_hatch(value, max(3, pitch // 2), fine, seed + 121),
        shade.point(lambda v: min(255, max(0, (v - 175) * 4))),
    )
    etched = ImageChops.lighter(primary, ImageChops.lighter(secondary, deepest))
    etched = ImageChops.lighter(etched, shade.point(lambda v: v // 10))
    if line > 0:
        contours = ImageChops.lighter(
            outlines(id_mask, image.size, line), value_edges(image, line, 58)
        )
        etched = ImageChops.lighter(etched, contours)
    page = Image.new("RGB", image.size, (250, 246, 234))
    plate_ink = Image.new("RGB", image.size, (33, 32, 34))
    pressed = Image.composite(plate_ink, page, etched)
    return _on(pressed, paper_texture(image.size, seed + 113, paper * 0.65))


def _mosaic(image: Image.Image, scale: float, paper: float, seed: int) -> Image.Image:
    """Set individually coloured tesserae with grout; no painterly blur."""

    pitch = max(6, round(23 * scale))
    grout = max(1, round(1.5 * scale))
    background = Image.new("RGB", image.size, (76, 72, 69))
    draw = ImageDraw.Draw(background)
    for y in range(0, image.height, pitch):
        for x in range(0, image.width, pitch):
            x0, y0 = x + grout, y + grout
            x1 = min(image.width - 1, x + pitch - grout - 1)
            y1 = min(image.height - 1, y + pitch - grout - 1)
            if x1 <= x0 or y1 <= y0:
                continue
            # Average each tile footprint: a single centre pixel aliases small objects.
            colour = cast(
                tuple[int, int, int],
                image.crop((x, y, min(image.width, x + pitch), min(image.height, y + pitch)))
                .resize((1, 1), Image.Resampling.BOX)
                .getpixel((0, 0)),
            )
            key = ((x + 1) * 73856093 ^ (y + 1) * 19349663 ^ (seed + 127) * 83492791) & 0xFFFFFFFF
            variation = ((key >> 8) % 13 - 6) * 0.75
            tile_colour = tuple(max(0, min(255, round(c + variation))) for c in colour)
            # Corners vary *inside* the grout clearance, never across neighbouring tiles.
            top = (key >> 16) & 1
            bottom = (key >> 18) & 1
            draw.polygon(
                [(x0 + top, y0), (x1, y0 + (1 - top)), (x1 - bottom, y1), (x0, y1 - (1 - bottom))],
                fill=tile_colour,
            )
    return _on(background, paper_texture(image.size, seed + 131, paper * 0.18))


def _crayon(image: Image.Image, scale: float, paper: float, seed: int) -> Image.Image:
    """Lay short waxy coloured marks with unpainted paper between strokes."""

    ground = Image.new("RGB", image.size, (250, 247, 234))
    strokes = ImageDraw.Draw(ground)
    pitch = max(5, round(12 * scale))
    span = max(3, round(10 * scale))
    thickness = max(2, round(2.8 * scale))
    for y in range(0, image.height, pitch):
        for x in range(0, image.width, pitch):
            for layer in range(2):
                key = (
                    (x + 1) * 73856093 ^ (y + 1) * 19349663 ^ (seed + 149 + layer) * 83492791
                ) & 0xFFFFFFFF
                px = min(image.width - 1, x + (key >> 8) % min(pitch, image.width - x))
                py = min(image.height - 1, y + (key >> 16) % min(pitch, image.height - y))
                sampled = cast(tuple[int, int, int], image.getpixel((px, py)))
                wax = tuple(max(0, min(255, round(v * 0.9 + 14))) for v in sampled)
                direction = -1 if layer else 1
                strokes.line(
                    (px, py, px + direction * span, py - max(1, span // 3)),
                    fill=wax,
                    width=thickness,
                )
    return _on(ground, paper_texture(image.size, seed + 151, paper * 0.8))


def _screenprint(image: Image.Image, scale: float, paper: float, seed: int) -> Image.Image:
    """Limit ink colours and print a separate, slightly misregistered halftone plate."""

    flat = image.quantize(colors=6, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    flat_rgb = flat.convert("RGB")
    sheet = Image.new("RGB", image.size, (251, 248, 239))
    printed = Image.blend(flat_rgb, sheet, 0.10)
    gray = ImageOps.grayscale(flat_rgb)
    dots = Image.new("L", image.size, 0)
    draw = ImageDraw.Draw(dots)
    pitch = max(5, round(11 * scale))
    jitter = seed % 3 - 1
    for y in range(0, image.height, pitch):
        for x in range(0, image.width, pitch):
            px = min(image.width - 1, max(0, x + pitch // 2 + jitter))
            py = min(image.height - 1, max(0, y + pitch // 2 - jitter))
            level = cast(int, gray.getpixel((px, py)))
            darkness = (255 - level) / 255
            if darkness < 0.12:
                continue
            radius = max(0.65, pitch * 0.42 * math.sqrt(darkness))
            draw.ellipse((px - radius, py - radius, px + radius, py + radius), fill=175)
    ink = Image.new("RGB", image.size, (35, 39, 48))
    printed = Image.composite(ink, printed, dots)

    # The second ink plate is shifted without wraparound, so the opposite
    # edges of the page cannot bleed together.
    edges = value_edges(flat_rgb, max(1.0, scale), 40)
    shifted = Image.new("L", image.size, 0)
    shifted.paste(edges, (1, 0))
    fringe = ImageChops.subtract(shifted, edges).point(lambda v: v * 2 // 5)
    printed = Image.composite(Image.new("RGB", image.size, (183, 64, 81)), printed, fringe)
    return _on(printed, paper_texture(image.size, seed + 157, paper * 0.45))


def _papercut(image: Image.Image, scale: float, paper: float, seed: int) -> Image.Image:
    """Assemble region-shaped paper silhouettes rather than a regular tile grid."""

    # Median-cut produces coherent colour regions; each index becomes one
    # paper shape with its own shadow. No stochastic palette training is used.
    indexed = image.quantize(colors=7, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    palette = indexed.getpalette()
    if palette is None:
        raise ValueError("quantized image has no palette")
    counts = indexed.histogram()
    # Large background papers go down first; smaller shapes layer on top.
    layers = sorted((i for i, count in enumerate(counts) if count), key=lambda i: -counts[i])
    page = Image.new("RGB", image.size, (245, 240, 226))
    shadow = Image.new("RGB", image.size, (80, 77, 74))
    offset = max(1, round(2 * scale))
    for index in layers:
        mask = indexed.point([255 if value == index else 0 for value in range(256)], mode="L")
        # A one-pixel clearance makes adjacent paper layers legible.
        if min(image.size) >= 3:
            mask = mask.filter(ImageFilter.MinFilter(3))
        shadow_mask = Image.new("L", image.size, 0)
        shadow_mask.paste(mask, (offset, offset))
        page = Image.composite(shadow, page, shadow_mask.point(lambda v: v // 3))
        channels = palette[3 * index : 3 * index + 3]
        colour = tuple(max(0, min(255, round(c * 0.91 + 15))) for c in channels)
        patch = Image.new("RGB", image.size, colour)
        page = Image.composite(patch, page, mask)
    return _on(page, paper_texture(image.size, seed + 163, paper * 0.35))


def _sumi(
    image: Image.Image,
    id_mask: Image.Image | None,
    line: float,
    scale: float,
    paper: float,
    seed: int,
) -> Image.Image:
    """Layer grey washes and locally directed brush marks with tapered ends."""

    value = ImageOps.grayscale(image)
    shade = ImageOps.invert(value)
    broad = shade.filter(ImageFilter.GaussianBlur(max(0.8, 2.2 * scale)))
    stepped = broad.point(lambda v: min(255, (v // 34) * 34 + v // 7))
    washes = Image.blend(broad, stepped, 0.40)
    marks = Image.new("L", image.size, 0)
    pitch = max(6, round(16 * scale))
    length = max(3, round(13 * scale))
    for y in range(0, image.height, pitch):
        for x in range(0, image.width, pitch):
            key = ((x + 1) * 73856093 ^ (y + 1) * 19349663 ^ (seed + 173) * 83492791) & 0xFFFFFFFF
            px = min(image.width - 1, x + (key >> 8) % min(pitch, image.width - x))
            py = min(image.height - 1, y + (key >> 16) % min(pitch, image.height - y))
            weight = cast(int, shade.getpixel((px, py)))
            if weight < 95 or (key & 7) > weight // 34:
                continue
            tapered_brush(
                marks,
                value,
                px,
                py,
                length,
                max(2, round(2.2 * scale)),
                min(255, 188 + weight // 5 + ((key >> 19) & 31)),
                key,
            )
    ink = ImageChops.lighter(washes, marks)
    if line > 0:
        edges = ImageChops.lighter(
            outlines(id_mask, image.size, line), value_edges(image, line, 48)
        )
        ink = ImageChops.lighter(ink, edges.point(lambda v: v * 3 // 4))
    page = Image.new("RGB", image.size, (249, 247, 239))
    pigment = Image.new("RGB", image.size, (25, 27, 29))
    return _on(
        Image.composite(pigment, page, ink),
        paper_texture(image.size, seed + 179, paper * 0.65),
    )


def _stainedglass(
    image: Image.Image,
    id_mask: Image.Image | None,
    line: float,
    paper: float,
    seed: int,
) -> Image.Image:
    """Enclose palette-limited illuminated panes in dark, solid leading."""

    palette_image = image.quantize(
        colors=10, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE
    )
    # Preserve palette *indices*. Converting P to L would turn them into
    # brightness values and could erase boundaries between different hues.
    indices = Image.frombytes("L", image.size, palette_image.tobytes())
    neighbourhood = indices.filter(ImageFilter.MaxFilter(3))
    opposite = indices.filter(ImageFilter.MinFilter(3))
    boundaries = ImageChops.difference(neighbourhood, opposite)
    boundaries = boundaries.point(lambda value: 255 if value else 0)
    if line > 0:
        boundaries = ImageChops.lighter(boundaries, outlines(id_mask, image.size, line))
    panes = ImageEnhance.Color(palette_image.convert("RGB")).enhance(1.3)
    panes = ImageEnhance.Brightness(panes).enhance(1.08)
    # No wraparound at image borders; thick leading remains a distinct layer.
    lead = boundaries.filter(ImageFilter.MaxFilter(3))
    glass = Image.composite(Image.new("RGB", image.size, (30, 32, 38)), panes, lead)
    return _on(glass, paper_texture(image.size, seed + 181, paper * 0.12))


def _cyanotype(
    image: Image.Image,
    id_mask: Image.Image | None,
    line: float,
    scale: float,
    paper: float,
    seed: int,
) -> Image.Image:
    """Photographic blue values with pale contours like contact-print edges."""

    exposure = ImageOps.grayscale(image).filter(ImageFilter.GaussianBlur(max(0.35, 0.75 * scale)))
    if line > 0:
        contours = outlines(id_mask, image.size, line)
        exposure = ImageChops.add(exposure, contours.point(lambda value: value // 5))
    blue = ImageOps.colorize(
        exposure,
        black=(11, 40, 98),
        white=(230, 244, 240),
        mid=(65, 123, 177),
    )
    return _on(blue, paper_texture(image.size, seed + 191, paper * 0.55))


def _on(image: Image.Image, texture: Image.Image) -> Image.Image:
    return ImageChops.multiply(image, Image.merge("RGB", (texture, texture, texture)))


__all__ = ["PRESETS", "REFERENCE_WIDTH", "apply_style", "flatten", "kuwahara", "outlines"]
