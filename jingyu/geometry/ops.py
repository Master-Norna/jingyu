"""The built-in geometry operators.

Primitives are thin wrappers around the lathe wherever the shape is
axially symmetric, so every round thing shares one tested code path.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from ..errors import JingyuError
from ..generator import GeneratorDef, ParamProblem, integer, number
from .curves import linspace, pchip
from .lathe import Point2, close_solid, revolve, shell
from .mesh import MeshData

_SEGMENTS = integer(
    "Number of segments around the axis; more is smoother and heavier.",
    default=64,
    minimum=3,
    maximum=1024,
)
_SHARP = number(
    "Profile corners turning more than this many degrees are shaded crisply.",
    default=30.0,
    minimum=0.0,
    maximum=180.0,
)


def _size3(description: str) -> dict[str, Any]:
    return {
        "type": "array",
        "description": description,
        "items": {"type": "number", "exclusiveMinimum": 0},
        "minItems": 3,
        "maxItems": 3,
    }


# ---------------------------------------------------------------- box / plane


def _box(p: Mapping[str, Any]) -> MeshData:
    x, y, z = (float(v) / 2.0 for v in p["size"])
    z *= 2.0
    vertices = (
        (-x, -y, 0.0), (x, -y, 0.0), (x, y, 0.0), (-x, y, 0.0),
        (-x, -y, z), (x, -y, z), (x, y, z), (-x, y, z),
    )  # fmt: skip
    faces = ((0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7))
    return MeshData(vertices, faces, smooth=False)


BOX = GeneratorDef[MeshData](
    name="box",
    summary="Axis-aligned box standing on its origin.",
    params={"size": _size3("Width (x), depth (y) and height (z) in metres.")},
    run=_box,
    examples=({"op": "box", "size": [1.2, 0.6, 0.75]},),
)


def _plane(p: Mapping[str, Any]) -> MeshData:
    x, y = (float(v) / 2.0 for v in p["size"])
    vertices = ((-x, -y, 0.0), (x, -y, 0.0), (x, y, 0.0), (-x, y, 0.0))
    return MeshData(vertices, ((0, 1, 2, 3),), smooth=False)


PLANE = GeneratorDef[MeshData](
    name="plane",
    summary="Flat rectangle in the XY plane, centred on its origin, facing +Z.",
    params={
        "size": {
            "type": "array",
            "description": "Width (x) and depth (y) in metres.",
            "items": {"type": "number", "exclusiveMinimum": 0},
            "minItems": 2,
            "maxItems": 2,
        }
    },
    run=_plane,
    examples=({"op": "plane", "size": [20, 20]},),
)


# ------------------------------------------------------- lathe-based primitives


def _sphere(p: Mapping[str, Any]) -> MeshData:
    r = float(p["radius"])
    rings = int(p["rings"])
    polyline: list[Point2] = []
    for k in range(rings + 1):
        phi = math.pi * k / rings
        polyline.append((0.0 if k in (0, rings) else r * math.sin(phi), r - r * math.cos(phi)))
    return revolve(polyline, int(p["segments"]), sharp_angle=180.0)


SPHERE = GeneratorDef[MeshData](
    name="sphere",
    summary="UV sphere resting on its origin (its lowest point).",
    params={
        "radius": number("Radius in metres.", exclusive_minimum=0),
        "segments": _SEGMENTS,
        "rings": integer("Number of rings from pole to pole.", default=32, minimum=2, maximum=512),
    },
    run=_sphere,
    examples=({"op": "sphere", "radius": 0.1},),
)


def _cylinder(p: Mapping[str, Any]) -> MeshData:
    r, h = float(p["radius"]), float(p["height"])
    return revolve([(0.0, 0.0), (r, 0.0), (r, h), (0.0, h)], int(p["segments"]))


CYLINDER = GeneratorDef[MeshData](
    name="cylinder",
    summary="Closed cylinder standing on its origin.",
    params={
        "radius": number("Radius in metres.", exclusive_minimum=0),
        "height": number("Height in metres.", exclusive_minimum=0),
        "segments": _SEGMENTS,
    },
    run=_cylinder,
    examples=({"op": "cylinder", "radius": 0.05, "height": 0.2},),
)


def _cone(p: Mapping[str, Any]) -> MeshData:
    r, h, top = float(p["radius"]), float(p["height"]), float(p["top_radius"])
    polyline: list[Point2] = [(0.0, 0.0), (r, 0.0)]
    polyline += [(top, h), (0.0, h)] if top > 0 else [(0.0, h)]
    return revolve(polyline, int(p["segments"]))


CONE = GeneratorDef[MeshData](
    name="cone",
    summary="Cone or truncated cone standing on its origin.",
    params={
        "radius": number("Bottom radius in metres.", exclusive_minimum=0),
        "height": number("Height in metres.", exclusive_minimum=0),
        "top_radius": number("Top radius in metres; 0 makes a point.", default=0.0, minimum=0),
        "segments": _SEGMENTS,
    },
    run=_cone,
    examples=({"op": "cone", "radius": 0.1, "height": 0.3},),
)


# ---------------------------------------------------------------------- lathe


def _profile_points(p: Mapping[str, Any]) -> list[Point2]:
    return [(float(r), float(z)) for r, z in p["profile"]]


def _lathe_mesh(p: Mapping[str, Any]) -> MeshData:
    profile = _profile_points(p)
    thickness = float(p["thickness"])
    polyline = shell(profile, thickness) if thickness > 0 else close_solid(profile)
    return revolve(polyline, int(p["segments"]), float(p["sharp_angle"]))


def _lathe_check(p: Mapping[str, Any]) -> list[ParamProblem]:
    try:
        _lathe_mesh({**p, "segments": 3})
    except JingyuError as exc:
        return [("profile", exc.message)]
    return []


LATHE = GeneratorDef[MeshData](
    name="lathe",
    summary=(
        "Revolve an (r, z) profile around the Z axis. thickness 0 makes a solid; "
        "thickness > 0 makes an open-topped shell with a closed floor."
    ),
    params={
        "profile": {
            "type": "array",
            "description": (
                "Points [r, z] in metres from the bottom outer edge upward. r >= 0. "
                "A solid is closed to the axis automatically at both ends."
            ),
            "items": {
                "type": "array",
                "prefixItems": [{"type": "number", "minimum": 0}, {"type": "number"}],
                "minItems": 2,
                "maxItems": 2,
            },
            "minItems": 2,
            "maxItems": 4096,
        },
        "thickness": number("Wall thickness in metres; 0 for a solid.", default=0.0, minimum=0),
        "segments": _SEGMENTS,
        "sharp_angle": _SHARP,
    },
    run=_lathe_mesh,
    check=_lathe_check,
    examples=(
        {"op": "lathe", "profile": [[0.04, 0], [0.05, 0.08], [0.045, 0.1]], "thickness": 0.003},
    ),
)


# --------------------------------------------------------------------- vessel


def vessel_profile(p: Mapping[str, Any]) -> list[Point2]:
    """The outer wall of a vessel: a monotone spline through four control rings."""

    h = float(p["height"])
    zs = [0.0, float(p["belly_at"]) * h, float(p["neck_at"]) * h, h]
    rs = [float(p[k]) for k in ("base_radius", "belly_radius", "neck_radius", "lip_radius")]
    samples = linspace(0.0, h, int(p["profile_samples"]))
    radii = pchip(zs, rs, samples)
    return list(zip(radii, samples, strict=True))


def _vessel(p: Mapping[str, Any]) -> MeshData:
    profile = vessel_profile(p)
    if p["open"]:
        polyline = shell(profile, float(p["thickness"]))
    else:
        polyline = close_solid(profile)
    return revolve(polyline, int(p["segments"]), float(p["sharp_angle"]))


def _vessel_check(p: Mapping[str, Any]) -> list[ParamProblem]:
    problems: list[ParamProblem] = []
    if not 0 < p["belly_at"] < p["neck_at"] < 1:
        problems.append(("neck_at", "require 0 < belly_at < neck_at < 1"))
        return problems
    try:
        _vessel({**p, "segments": 3})
    except JingyuError as exc:
        problems.append(("thickness", exc.message))
    return problems


VESSEL = GeneratorDef[MeshData](
    name="vessel",
    summary=(
        "Parametric vessel family (cup, bowl, vase, bottle, jar, pot): a lathe "
        "whose outer wall passes smoothly through base, belly, neck and lip rings."
    ),
    params={
        "height": number("Total height in metres.", default=0.25, exclusive_minimum=0),
        "base_radius": number("Radius at the foot, metres.", default=0.05, exclusive_minimum=0),
        "belly_radius": number("Radius at the belly, metres.", default=0.08, exclusive_minimum=0),
        "belly_at": number(
            "Height of the belly as a fraction of the height.",
            default=0.4,
            exclusive_minimum=0,
            maximum=1,
        ),
        "neck_radius": number("Radius at the neck, metres.", default=0.035, exclusive_minimum=0),
        "neck_at": number(
            "Height of the neck as a fraction of the height.",
            default=0.8,
            exclusive_minimum=0,
            maximum=1,
        ),
        "lip_radius": number("Radius at the rim, metres.", default=0.045, exclusive_minimum=0),
        "thickness": number(
            "Wall and floor thickness in metres (open vessels only).",
            default=0.004,
            exclusive_minimum=0,
        ),
        "open": {
            "type": "boolean",
            "description": "true: hollow with an open top; false: a closed solid.",
            "default": True,
        },
        "segments": integer(
            "Number of segments around the axis.", default=96, minimum=3, maximum=1024
        ),
        "profile_samples": integer(
            "Number of points sampled along the wall.", default=64, minimum=4, maximum=1024
        ),
        "sharp_angle": _SHARP,
    },
    run=_vessel,
    check=_vessel_check,
    examples=(
        {
            "op": "vessel",
            "height": 0.1,
            "base_radius": 0.032,
            "belly_radius": 0.037,
            "belly_at": 0.5,
            "neck_radius": 0.041,
            "neck_at": 0.9,
            "lip_radius": 0.042,
        },
        {
            "op": "vessel",
            "height": 0.3,
            "base_radius": 0.035,
            "belly_radius": 0.036,
            "belly_at": 0.55,
            "neck_radius": 0.012,
            "neck_at": 0.8,
            "lip_radius": 0.013,
        },
    ),
)

# ----------------------------------------------------------------------- wall

_OPENING = {
    "type": "object",
    "additionalProperties": False,
    "required": ["x", "sill", "width", "height"],
    "properties": {
        "x": {"type": "number", "description": "Centre of the opening along the wall (x), metres."},
        "sill": {
            "type": "number",
            "minimum": 0,
            "description": "Height of the opening's bottom edge above the wall base; 0 = door.",
        },
        "width": {"type": "number", "exclusiveMinimum": 0, "description": "Opening width."},
        "height": {"type": "number", "exclusiveMinimum": 0, "description": "Opening height."},
    },
}


def _wall_openings(p: Mapping[str, Any]) -> list[tuple[float, float, float, float]]:
    return [
        (
            float(o["x"]) - float(o["width"]) / 2.0,
            float(o["x"]) + float(o["width"]) / 2.0,
            float(o["sill"]),
            float(o["sill"]) + float(o["height"]),
        )
        for o in p["openings"]
    ]


def _wall(p: Mapping[str, Any]) -> MeshData:
    width, thickness, height = (float(v) for v in p["size"])
    x_min, x_max = -width / 2.0, width / 2.0
    y_front, y_back = -thickness / 2.0, thickness / 2.0
    holes = _wall_openings(p)
    xs = sorted({x_min, x_max, *(c for h in holes for c in h[:2] if x_min < c < x_max)})
    zs = sorted({0.0, height, *(c for h in holes for c in h[2:] if 0.0 < c < height)})

    def filled(i: int, j: int) -> bool:
        if not (0 <= i < len(xs) - 1 and 0 <= j < len(zs) - 1):
            return False
        cx, cz = (xs[i] + xs[i + 1]) / 2.0, (zs[j] + zs[j + 1]) / 2.0
        return not any(x0 < cx < x1 and z0 < cz < z1 for x0, x1, z0, z1 in holes)

    vertices: list[tuple[float, float, float]] = []
    index: dict[tuple[int, int, int], int] = {}

    def v(side: int, i: int, j: int) -> int:
        key = (side, i, j)
        if key not in index:
            index[key] = len(vertices)
            vertices.append((xs[i], y_back if side else y_front, zs[j]))
        return index[key]

    faces: list[tuple[int, ...]] = []
    for i in range(len(xs) - 1):
        for j in range(len(zs) - 1):
            if not filled(i, j):
                continue
            f00, f10, f11, f01 = v(0, i, j), v(0, i + 1, j), v(0, i + 1, j + 1), v(0, i, j + 1)
            b00, b10, b11, b01 = v(1, i, j), v(1, i + 1, j), v(1, i + 1, j + 1), v(1, i, j + 1)
            faces.append((f00, f10, f11, f01))
            faces.append((b10, b00, b01, b11))
            if not filled(i - 1, j):
                faces.append((b00, f00, f01, b01))
            if not filled(i + 1, j):
                faces.append((f10, b10, b11, f11))
            if not filled(i, j - 1):
                faces.append((f00, b00, b10, f10))
            if not filled(i, j + 1):
                faces.append((f01, f11, b11, b01))
    return MeshData(tuple(vertices), tuple(faces), smooth=False)


def _check_wall(p: Mapping[str, Any]) -> list[ParamProblem]:
    width, _, height = (float(v) for v in p["size"])
    problems: list[ParamProblem] = []
    for n, (x0, x1, _z0, z1) in enumerate(_wall_openings(p)):
        if x0 < -width / 2.0 - 1e-9 or x1 > width / 2.0 + 1e-9 or z1 > height + 1e-9:
            problems.append(("openings", f"opening {n} extends beyond the wall"))
    area = sum((x1 - x0) * (z1 - z0) for x0, x1, z0, z1 in _wall_openings(p))
    if not problems and area >= width * height - 1e-12:
        problems.append(("openings", "the openings remove the whole wall"))
    return problems


WALL = GeneratorDef[MeshData](
    name="wall",
    summary=(
        "Upright slab standing on its origin, spanning x, with rectangular openings "
        "cut through it (windows, doors). Light passes through the openings."
    ),
    params={
        "size": _size3("Width (x), thickness (y) and height (z) in metres."),
        "openings": {
            "type": "array",
            "items": _OPENING,
            "maxItems": 64,
            "default": [],
            "description": "Rectangular holes through the wall; x is measured from its centre.",
        },
    },
    run=_wall,
    check=_check_wall,
    examples=(
        {
            "op": "wall",
            "size": [4.0, 0.15, 2.6],
            "openings": [{"x": -0.6, "sill": 0.9, "width": 1.2, "height": 1.3}],
        },
    ),
)


ALL_OPS: tuple[GeneratorDef[MeshData], ...] = (
    BOX,
    PLANE,
    SPHERE,
    CYLINDER,
    CONE,
    LATHE,
    VESSEL,
    WALL,
)

__all__ = ["ALL_OPS", "vessel_profile"]
