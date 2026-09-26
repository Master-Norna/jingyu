"""Derived views of a candidate for reviewing it the five ways.

The visual constitution (J7.1) asks every important review to look at an image
five ways: at a glance, squinting, in black and white, by subtraction and side
by side.  Four of them are image transforms provided here so a model does not
have to imagine squinting; subtraction is a re-render with objects hidden.
"""

from __future__ import annotations

import colorsys
import io
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from PIL import Image, ImageFilter, ImageOps

from .candidate import RENDER_NAME, Candidate
from .errors import JingyuError
from .lighting import LightMap
from .locate import IdMask

VIEWS: dict[str, str] = {
    "full": "The image (styled, when the scene has a style).",
    "render": "The photographic render before any style was applied.",
    "glance": "A small thumbnail: what the image says at first sight (J7.1 one).",
    "flip": "Mirrored left to right: fresh eyes for balance and habituation (J7.1 one).",
    "squint": "Details blurred away: only big masses of value and colour remain (J7.1 two).",
    "grayscale": "Luminance only: is the value structure clear without colour? (J7.1 three).",
    "values": "Luminance quantised into a few value steps: the value plan (J7.1 three).",
    "saturation": "Colour saturation as brightness: where the colour is strong or washed out.",
    "light": (
        "The render darkened except where one light (default: the environment's sun, else "
        "the first light) reaches directly: where the light lands and what it misses."
    ),
    "id_mask": "Each object in a flat distinct colour, with a legend.",
    "compare": "This candidate beside another one, same height (J7.1 five).",
}

GLANCE_SIZE = 192
SQUINT_FRACTION = 0.02


@dataclass(frozen=True)
class RenderedView:
    png: bytes
    width: int
    height: int
    meta: dict[str, Any] = field(default_factory=dict)


def _open_rgb(candidate: Candidate) -> Image.Image:
    with Image.open(candidate.image_path) as image:
        image.load()
        return image.convert("RGB")


def _open_render(candidate: Candidate) -> Image.Image:
    path = candidate.file(RENDER_NAME)
    if not path.is_file():
        return _open_rgb(candidate)
    with Image.open(path) as image:
        image.load()
        return image.convert("RGB")


def _fit(image: Image.Image, max_size: int) -> Image.Image:
    if max(image.size) <= max_size:
        return image
    fitted = image.copy()
    fitted.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    return fitted


def _encode(image: Image.Image, meta: dict[str, Any]) -> RenderedView:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return RenderedView(buffer.getvalue(), image.width, image.height, meta)


def _distinct_color(i: int) -> tuple[int, int, int]:
    hue = (i * 0.618033988749895) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.65, 0.95)
    return int(r * 255), int(g * 255), int(b * 255)


def render_view(
    candidate: Candidate,
    view: str,
    *,
    max_size: int = 1024,
    levels: int = 5,
    other: Candidate | None = None,
    light: str | None = None,
    region: Mapping[str, Any] | None = None,
) -> RenderedView:
    """Produce one view of *candidate* as PNG bytes, optionally cropped to *region*.

    A cropped view is enlarged to *max_size* so details can be inspected.
    """

    builders: dict[str, Callable[[], tuple[Image.Image, dict[str, Any]]]] = {
        "full": lambda: (_open_rgb(candidate), {}),
        "render": lambda: (_open_render(candidate), {}),
        "glance": lambda: (_fit(_open_rgb(candidate), GLANCE_SIZE), {}),
        "flip": lambda: (ImageOps.mirror(_open_rgb(candidate)), {}),
        "squint": lambda: _squint(candidate),
        "grayscale": lambda: (ImageOps.grayscale(_open_rgb(candidate)), {}),
        "values": lambda: _values(candidate, levels),
        "saturation": lambda: _saturation(candidate),
        "light": lambda: _light(candidate, light),
        "id_mask": lambda: _id_mask(candidate),
        "compare": lambda: _compare(candidate, other),
    }
    if view not in builders:
        raise JingyuError(
            "view.unknown_view", f"unknown view {view!r}", hint=f"views: {', '.join(VIEWS)}"
        )
    image, meta = builders[view]()
    limit = GLANCE_SIZE if view == "glance" else max_size
    if region is not None:
        if view in ("glance", "compare"):
            raise JingyuError("tool.invalid_arguments", f"the {view} view cannot be cropped")
        image, meta["region"] = _crop(image, region)
        scale = limit / max(image.size)
        if scale > 1:
            size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
            image = image.resize(size, Image.Resampling.LANCZOS)
    fitted = _fit(image, limit)
    return _encode(fitted, {"view": view, **meta})


def _crop(image: Image.Image, region: Mapping[str, Any]) -> tuple[Image.Image, dict[str, int]]:
    width, height = image.size
    if all(k in region for k in ("u0", "v0", "u1", "v1")):
        box = (
            math.floor(float(region["u0"]) * width),
            math.floor(float(region["v0"]) * height),
            round(float(region["u1"]) * width),
            round(float(region["v1"]) * height),
        )
    elif all(k in region for k in ("x0", "y0", "x1", "y1")):
        box = (int(region["x0"]), int(region["y0"]), int(region["x1"]), int(region["y1"]))
    else:
        raise JingyuError("tool.invalid_arguments", "a region needs x0/y0/x1/y1 or u0/v0/u1/v1")
    x0, y0, x1, y1 = box
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise JingyuError(
            "locate.out_of_bounds",
            f"region {box} is not inside the {width}x{height} image",
        )
    return image.crop(box), {"x0": x0, "y0": y0, "x1": x1, "y1": y1}


def _saturation(candidate: Candidate) -> tuple[Image.Image, dict[str, Any]]:
    hsv = _open_rgb(candidate).convert("HSV")
    saturation = hsv.getchannel("S")
    histogram = saturation.histogram()
    total = sum(histogram)
    mean = sum(value * count for value, count in enumerate(histogram)) / max(total, 1) / 255
    return saturation, {"mean_saturation": round(mean, 3)}


def _light(candidate: Candidate, light: str | None) -> tuple[Image.Image, dict[str, Any]]:
    lights = LightMap.load(candidate)
    ids = [entry["id"] for entry in lights.lights]
    if light is None:
        if not ids:
            raise JingyuError("light.unknown_light", "the scene has no lights to show")
        light = "environment.sun" if "environment.sun" in ids else ids[0]
    base = _open_rgb(candidate)
    mask = lights.lit_mask(light).resize(base.size, Image.Resampling.NEAREST)
    dim = Image.eval(base, lambda value: value // 4)
    shown = Image.composite(base, dim, mask)
    histogram = mask.histogram()
    return shown, {
        "light": light,
        "lights": ids,
        "reached_fraction": round(histogram[255] / max(sum(histogram), 1), 4),
    }


def _squint(candidate: Candidate) -> tuple[Image.Image, dict[str, Any]]:
    image = _open_rgb(candidate)
    radius = max(1.0, SQUINT_FRACTION * max(image.size))
    return image.filter(ImageFilter.GaussianBlur(radius)), {"blur_radius_px": radius}


def _values(candidate: Candidate, levels: int) -> tuple[Image.Image, dict[str, Any]]:
    if not 2 <= levels <= 16:
        raise JingyuError("tool.invalid_arguments", "levels must be between 2 and 16")
    gray = ImageOps.grayscale(_open_rgb(candidate))
    step = 256 / levels
    lut = [int(min(levels - 1, v // step) * 255 / (levels - 1)) for v in range(256)]
    histogram = [0] * levels
    for value, count in enumerate(gray.histogram()):
        histogram[int(min(levels - 1, value // step))] += count
    total = sum(histogram)
    return gray.point(lut), {
        "levels": levels,
        "value_distribution": [round(c / total, 4) for c in histogram],
    }


def _id_mask(candidate: Candidate) -> tuple[Image.Image, dict[str, Any]]:
    mask = IdMask.load(candidate)
    rgb = bytearray(bytes((40, 40, 40)) * (mask.width * mask.height))
    legend: dict[int, dict[str, Any]] = {}
    for y in range(mask.height):
        for x in range(mask.width):
            index = mask.index_at(x, y)
            if not index:
                continue
            if index not in legend:
                entry = mask.entry(index) or {"object": f"#{index}"}
                label = entry["object"] + (f" / {entry['part']}" if entry.get("part") else "")
                legend[index] = {"object": label, "color": _distinct_color(index)}
            offset = (y * mask.width + x) * 3
            rgb[offset : offset + 3] = bytes(legend[index]["color"])
    out = Image.frombytes("RGB", (mask.width, mask.height), bytes(rgb))
    return out, {
        "legend": [
            {"object": v["object"], "color": "#{:02x}{:02x}{:02x}".format(*v["color"])}
            for _, v in sorted(legend.items())
        ]
    }


def _compare(candidate: Candidate, other: Candidate | None) -> tuple[Image.Image, dict[str, Any]]:
    if other is None:
        raise JingyuError("tool.invalid_arguments", "the compare view needs other_candidate_id")
    left, right = _open_rgb(candidate), _open_rgb(other)
    height = min(left.height, right.height)
    left = left.resize((round(left.width * height / left.height), height), Image.Resampling.LANCZOS)
    right = right.resize(
        (round(right.width * height / right.height), height), Image.Resampling.LANCZOS
    )
    gap = max(4, height // 100)
    canvas = Image.new("RGB", (left.width + gap + right.width, height), (255, 255, 255))
    canvas.paste(left, (0, 0))
    canvas.paste(right, (left.width + gap, 0))
    return canvas, {"left": candidate.id, "right": other.id}


__all__ = ["VIEWS", "RenderedView", "render_view"]
