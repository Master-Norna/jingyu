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
from .curves import linspace, smooth_monotone
from .lathe import Point2, close_solid, revolve, shell
from .mesh import MeshData
from .pieces import rounded_box

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
    return rounded_box(p["size"], float(p["bevel"]))


def _check_box(p: Mapping[str, Any]) -> list[ParamProblem]:
    if 2.0 * float(p["bevel"]) > min(float(v) for v in p["size"]):
        return [("bevel", "the rounding is larger than half the box's smallest side")]
    return []


BOX = GeneratorDef[MeshData](
    name="box",
    summary="Axis-aligned box standing on its origin, its edges optionally rounded.",
    params={
        "size": _size3("Width (x), depth (y) and height (z) in metres."),
        "bevel": number(
            "Radius of the rounded edges in metres; a few millimetres catch the light like a "
            "real, handled object. 0 is razor-sharp.",
            default=0.0,
            minimum=0.0,
        ),
    },
    run=_box,
    check=_check_box,
    examples=(
        {"op": "box", "size": [1.2, 0.6, 0.75]},
        {"op": "box", "size": [0.3, 0.2, 0.05], "bevel": 0.004},
    ),
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
    """The outer wall of a vessel: a smooth monotone curve through four control rings."""

    h = float(p["height"])
    zs = [0.0, float(p["belly_at"]) * h, float(p["neck_at"]) * h, h]
    rs = [float(p[k]) for k in ("base_radius", "belly_radius", "neck_radius", "lip_radius")]
    samples = linspace(0.0, h, int(p["profile_samples"]))
    radii = smooth_monotone(zs, rs, samples)
    return list(zip(radii, samples, strict=True))


def _vessel(p: Mapping[str, Any]) -> MeshData:
    profile = vessel_profile(p)
    if p["open"]:
        polyline = shell(profile, float(p["thickness"]))
    else:
        polyline = close_solid(profile)
    mesh = revolve(polyline, int(p["segments"]), float(p["sharp_angle"]))
    wobble = float(p["wobble"])
    return _hand_made(mesh, wobble, float(p["height"]), int(p["seed"])) if wobble > 0 else mesh


def _hand_made(mesh: MeshData, wobble: float, height: float, seed: int) -> MeshData:
    """The small irregularity of a thrown pot: slightly out of round, gently leaning.

    Every point moves by a smooth function of its angle and height only, so the
    inner and outer walls move together, the wall keeps its thickness and the
    foot stays flat on the ground.
    """

    golden = 2.399963229728653
    phases = [(seed * golden * (k + 1) + k * 1.618) % math.tau for k in range(4)]
    lean = 0.04 * wobble * height
    moved = []
    for x, y, z in mesh.vertices:
        h = z / height
        angle = math.atan2(y, x)
        round_ = 1.0 + 0.035 * wobble * (
            0.6 * math.sin(2.0 * angle + phases[0] + 2.5 * h)
            + 0.4 * math.sin(3.0 * angle + phases[1] - 1.7 * h)
        )
        shift = lean * h * h
        moved.append(
            (
                x * round_ + shift * math.cos(phases[2]),
                y * round_ + shift * math.sin(phases[2]),
                z,
            )
        )
    return MeshData(tuple(moved), mesh.faces, mesh.smooth, mesh.sharp_edges)


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
        "wobble": number(
            "Hand-made irregularity: 0 machine-perfect, 0.3 thrown on a wheel (slightly out "
            "of round, gently leaning), 1 rustic.",
            default=0.0,
            minimum=0.0,
            maximum=1.0,
        ),
        "seed": integer(
            "Variation of the irregularity: another number, another pot.",
            default=0,
            minimum=0,
            maximum=100000,
        ),
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
        {"op": "vessel", "height": 0.2, "belly_radius": 0.07, "wobble": 0.4, "seed": 7},
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
)

__all__ = ["ALL_OPS", "vessel_profile"]
