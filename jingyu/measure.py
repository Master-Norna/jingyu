"""Measure what the renderer actually produced: colours of pixels and objects.

Material colours are reflectances.  What reaches the screen is that reflectance
times the light's colour, through the view transform, so an orange under warm low
sun can come out peach.  These measurements put the rendered colour next to the
authored one, so a model can see the drift instead of guessing it.
"""

from __future__ import annotations

import colorsys
from collections.abc import Iterable, Mapping
from typing import Any

from PIL import Image

from .conventions import linear_to_srgb
from .locate import Box, IdMask
from .materials.weathering import material_recipe


def hex_color(rgb: Iterable[float]) -> str:
    """``#rrggbb`` of an sRGB triple in [0, 1]."""

    return "#" + "".join(f"{round(min(1.0, max(0.0, c)) * 255):02x}" for c in rgb)


def describe_color(rgb: tuple[float, float, float]) -> dict[str, Any]:
    """Hex plus hue (degrees), saturation and value of an sRGB triple in [0, 1]."""

    hue, saturation, value = colorsys.rgb_to_hsv(*rgb)
    return {
        "hex": hex_color(rgb),
        "hue_deg": round(hue * 360.0, 1),
        "saturation": round(saturation, 3),
        "value": round(value, 3),
    }


def mean_color(pixels: Iterable[tuple[int, int, int]]) -> tuple[float, float, float] | None:
    """Mean of 8-bit sRGB pixels as floats in [0, 1]; None for no pixels."""

    total = [0, 0, 0]
    count = 0
    for r, g, b in pixels:
        total[0] += r
        total[1] += g
        total[2] += b
        count += 1
    if count == 0:
        return None
    return (total[0] / count / 255.0, total[1] / count / 255.0, total[2] / count / 255.0)


def material_color(scene: Mapping[str, Any], material_id: str | None) -> dict[str, Any] | None:
    """The authored base colour of a material (its reflectance), as displayed sRGB."""

    if material_id is None:
        return None
    entry = next((m for m in scene.get("materials", []) if m.get("id") == material_id), None)
    if entry is None:
        return None
    try:
        recipe = material_recipe(entry)
    except (KeyError, ValueError):
        return None
    srgb = tuple(linear_to_srgb(c) for c in recipe.base_color)
    return describe_color((srgb[0], srgb[1], srgb[2]))


def color_shift(rendered: Mapping[str, Any], authored: Mapping[str, Any]) -> dict[str, Any]:
    """How the rendered colour differs from the authored one."""

    hue = (float(rendered["hue_deg"]) - float(authored["hue_deg"]) + 180.0) % 360.0 - 180.0
    return {
        "hue_deg": round(hue, 1),
        "saturation": round(float(rendered["saturation"]) - float(authored["saturation"]), 3),
        "value": round(float(rendered["value"]) - float(authored["value"]), 3),
    }


def colors_by_object(
    mask: IdMask, image: Image.Image, box: Box | None = None
) -> dict[str, tuple[float, float, float]]:
    """Mean rendered sRGB colour of each object's pixels (all its parts), within *box*."""

    merged: dict[str, list[float]] = {}
    for index, (rgb, count) in _colors_by_index(mask, image, box).items():
        name = mask.object_of(index)
        if name is None:
            continue
        total = merged.setdefault(name, [0.0, 0.0, 0.0, 0.0])
        for k in range(3):
            total[k] += rgb[k] * count
        total[3] += count
    return {name: (t[0] / t[3], t[1] / t[3], t[2] / t[3]) for name, t in merged.items()}


def colors_by_index(
    mask: IdMask, image: Image.Image, box: Box | None = None
) -> dict[int, tuple[float, float, float]]:
    """Mean rendered sRGB colour per id mask entry (an object, or one of its parts)."""

    return {index: rgb for index, (rgb, _) in _colors_by_index(mask, image, box).items()}


def _colors_by_index(
    mask: IdMask, image: Image.Image, box: Box | None
) -> dict[int, tuple[tuple[float, float, float], int]]:
    rgb = image.convert("RGB")
    if rgb.size != (mask.width, mask.height):
        rgb = rgb.resize((mask.width, mask.height), Image.Resampling.NEAREST)
    data = rgb.tobytes()
    box = box or Box(0, 0, mask.width, mask.height)
    sums: dict[int, list[int]] = {}
    for y in range(box.y0, box.y1):
        for x in range(box.x0, box.x1):
            index = mask.index_at(x, y)
            if not index:
                continue
            offset = (y * mask.width + x) * 3
            s = sums.setdefault(index, [0, 0, 0, 0])
            s[0] += data[offset]
            s[1] += data[offset + 1]
            s[2] += data[offset + 2]
            s[3] += 1
    return {
        index: ((s[0] / s[3] / 255.0, s[1] / s[3] / 255.0, s[2] / s[3] / 255.0), s[3])
        for index, s in sums.items()
    }


def object_color_report(
    rendered: tuple[float, float, float], scene: Mapping[str, Any], material_id: str | None
) -> dict[str, Any]:
    """Rendered colour, authored material colour and the shift between them."""

    report: dict[str, Any] = {"rendered_color": describe_color(rendered)}
    authored = material_color(scene, material_id)
    if authored is not None:
        report["material_color"] = authored
        report["color_shift"] = color_shift(report["rendered_color"], authored)
    return report


__all__ = [
    "color_shift",
    "colors_by_index",
    "colors_by_object",
    "describe_color",
    "hex_color",
    "material_color",
    "mean_color",
    "object_color_report",
]
