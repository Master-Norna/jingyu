"""Derived views of a candidate for reviewing it the five ways.

The visual constitution (J7.1) asks every important review to look at an image
five ways: at a glance, squinting, in black and white, by subtraction and side
by side.  Four of them are image transforms provided here so a model does not
have to imagine squinting; subtraction is a re-render with objects hidden.
"""

from __future__ import annotations

import colorsys
import io
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from PIL import Image, ImageFilter, ImageOps

from .candidate import Candidate
from .errors import JingyuError
from .locate import IdMask

VIEWS: dict[str, str] = {
    "full": "The render itself.",
    "glance": "A small thumbnail: what the image says at first sight (J7.1 one).",
    "flip": "Mirrored left to right: fresh eyes for balance and habituation (J7.1 one).",
    "squint": "Details blurred away: only big masses of value and colour remain (J7.1 two).",
    "grayscale": "Luminance only: is the value structure clear without colour? (J7.1 three).",
    "values": "Luminance quantised into a few value steps: the value plan (J7.1 three).",
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
) -> RenderedView:
    """Produce one view of *candidate* as PNG bytes."""

    builders: dict[str, Callable[[], tuple[Image.Image, dict[str, Any]]]] = {
        "full": lambda: (_open_rgb(candidate), {}),
        "glance": lambda: (_fit(_open_rgb(candidate), GLANCE_SIZE), {}),
        "flip": lambda: (ImageOps.mirror(_open_rgb(candidate)), {}),
        "squint": lambda: _squint(candidate),
        "grayscale": lambda: (ImageOps.grayscale(_open_rgb(candidate)), {}),
        "values": lambda: _values(candidate, levels),
        "id_mask": lambda: _id_mask(candidate),
        "compare": lambda: _compare(candidate, other),
    }
    if view not in builders:
        raise JingyuError(
            "view.unknown_view", f"unknown view {view!r}", hint=f"views: {', '.join(VIEWS)}"
        )
    image, meta = builders[view]()
    limit = GLANCE_SIZE if view == "glance" else max_size
    fitted = _fit(image, limit)
    return _encode(fitted, {"view": view, **meta})


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
                legend[index] = {"object": entry["object"], "color": _distinct_color(index)}
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
