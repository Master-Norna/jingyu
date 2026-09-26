"""Units, axes and colour conventions shared by every layer.

* Lengths are metres and angles are degrees.  Field names carry no unit suffix
  for these two; every other unit is spelled out in the name (``power_w``,
  ``lens_mm``).
* The world is right-handed and Z-up.  Rotations are XYZ Euler angles.
* Generated geometry stands on its origin: the origin is the bottom centre of
  the object's bounding box (flat shapes such as planes are centred).
* Colours in documents are sRGB hex strings (``#rrggbb``); renderers receive
  scene-linear values converted by :func:`srgb_hex_to_linear`.
* Ids are lowercase ASCII slugs matching :data:`ID_PATTERN`.

Standard library only: the Blender worker imports this module.
"""

from __future__ import annotations

import math
import re

LENGTH_UNIT = "m"
ANGLE_UNIT = "deg"
UP_AXIS = "Z"
HANDEDNESS = "right"
EULER_ORDER = "XYZ"
ORIGIN_RULE = "bottom-center"

# Blender truncates datablock names beyond 63 bytes, so ids stop at 63 characters.
ID_PATTERN = r"^[a-z][a-z0-9_-]{0,62}$"
COLOR_PATTERN = r"^#[0-9a-fA-F]{6}$"

_COLOR_RE = re.compile(COLOR_PATTERN)


def srgb_to_linear(channel: float) -> float:
    """Convert one sRGB-encoded channel in [0, 1] to scene-linear."""

    if channel <= 0.04045:
        return channel / 12.92
    return float(((channel + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(channel: float) -> float:
    """Convert one scene-linear channel in [0, 1] to sRGB encoding."""

    if channel <= 0.0031308:
        return channel * 12.92
    return float(1.055 * channel ** (1.0 / 2.4) - 0.055)


def parse_hex(color: str) -> tuple[float, float, float]:
    """Return the sRGB-encoded channels of ``#rrggbb`` as floats in [0, 1]."""

    if not _COLOR_RE.fullmatch(color):
        raise ValueError(f"colour must look like #rrggbb, got {color!r}")
    return tuple(int(color[i : i + 2], 16) / 255.0 for i in (1, 3, 5))  # type: ignore[return-value]


def srgb_hex_to_linear(color: str) -> tuple[float, float, float]:
    """Return the scene-linear RGB of an sRGB ``#rrggbb`` colour."""

    r, g, b = parse_hex(color)
    return (srgb_to_linear(r), srgb_to_linear(g), srgb_to_linear(b))


def radians(degrees: float) -> float:
    return math.radians(degrees)


def summary() -> dict[str, str]:
    """A compact, serialisable statement of the conventions for tools and docs."""

    return {
        "length_unit": LENGTH_UNIT,
        "angle_unit": ANGLE_UNIT,
        "up_axis": UP_AXIS,
        "handedness": HANDEDNESS,
        "euler_order": EULER_ORDER,
        "origin": ORIGIN_RULE,
        "color_format": "sRGB hex #rrggbb",
        "id_pattern": ID_PATTERN,
        "unit_suffix_rule": (
            "lengths (m) and angles (deg) have no suffix; other units are "
            "spelled out in the field name, e.g. power_w, lens_mm"
        ),
    }


__all__ = [
    "ANGLE_UNIT",
    "COLOR_PATTERN",
    "EULER_ORDER",
    "HANDEDNESS",
    "ID_PATTERN",
    "LENGTH_UNIT",
    "ORIGIN_RULE",
    "UP_AXIS",
    "linear_to_srgb",
    "parse_hex",
    "radians",
    "srgb_hex_to_linear",
    "srgb_to_linear",
    "summary",
]
