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

import itertools
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


MIN_TEMPERATURE_K = 1667.0
MAX_TEMPERATURE_K = 25000.0


def kelvin_to_linear(temperature_k: float) -> tuple[float, float, float]:
    """Scene-linear RGB (max channel 1) of a black body at *temperature_k*.

    Uses the Kim et al. cubic fit of the Planckian locus in CIE 1931 xy, valid from
    1667 K to 25000 K, then converts to linear Rec.709/sRGB primaries.
    """

    t = float(temperature_k)
    if not MIN_TEMPERATURE_K <= t <= MAX_TEMPERATURE_K:
        raise ValueError(f"colour temperature must be 1667..25000 K, got {t:g}")
    if t <= 4000.0:
        x = -0.2661239e9 / t**3 - 0.2343589e6 / t**2 + 0.8776956e3 / t + 0.179910
    else:
        x = -3.0258469e9 / t**3 + 2.1070379e6 / t**2 + 0.2226347e3 / t + 0.240390
    if t <= 2222.0:
        y = -1.1063814 * x**3 - 1.34811020 * x**2 + 2.18555832 * x - 0.20219683
    elif t <= 4000.0:
        y = -0.9549476 * x**3 - 1.37418593 * x**2 + 2.09137015 * x - 0.16748867
    else:
        y = 3.0817580 * x**3 - 5.87338670 * x**2 + 3.75112997 * x - 0.37001483
    big_x, big_y, big_z = x / y, 1.0, (1.0 - x - y) / y
    rgb = (
        3.2404542 * big_x - 1.5371385 * big_y - 0.4985314 * big_z,
        -0.9692660 * big_x + 1.8760108 * big_y + 0.0415560 * big_z,
        0.0556434 * big_x - 0.2040259 * big_y + 1.0572252 * big_z,
    )
    clipped = [max(0.0, c) for c in rgb]
    peak = max(clipped)
    return (clipped[0] / peak, clipped[1] / peak, clipped[2] / peak)


_SUN_TEMPERATURES = (
    (0.0, 2000.0),
    (5.0, 2600.0),
    (10.0, 3200.0),
    (20.0, 4200.0),
    (35.0, 5000.0),
    (60.0, 5600.0),
)


def sun_direction(elevation_deg: float, azimuth_deg: float) -> tuple[float, float, float]:
    """Unit vector from the scene toward the sun (azimuth counter-clockwise from +X)."""

    e, a = math.radians(elevation_deg), math.radians(azimuth_deg)
    return (math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e))


def sun_rotation_deg(elevation_deg: float, azimuth_deg: float) -> tuple[float, float, float]:
    """XYZ Euler rotation (degrees) that points a sun lamp's light away from the sun."""

    return (90.0 - float(elevation_deg), 0.0, float(azimuth_deg) + 90.0)


def sun_temperature_k(elevation_deg: float) -> float:
    """Typical colour temperature of direct sunlight at a given elevation."""

    e = float(elevation_deg)
    points = _SUN_TEMPERATURES
    if e <= points[0][0]:
        return points[0][1]
    for (e0, t0), (e1, t1) in itertools.pairwise(points):
        if e <= e1:
            return t0 + (t1 - t0) * (e - e0) / (e1 - e0)
    return points[-1][1]


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
    "MAX_TEMPERATURE_K",
    "MIN_TEMPERATURE_K",
    "ORIGIN_RULE",
    "UP_AXIS",
    "kelvin_to_linear",
    "linear_to_srgb",
    "parse_hex",
    "radians",
    "srgb_hex_to_linear",
    "srgb_to_linear",
    "summary",
    "sun_direction",
    "sun_rotation_deg",
    "sun_temperature_k",
]
