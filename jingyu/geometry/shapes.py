"""Free-form operators: extrude a floor plan, sweep a tube along a path, scatter
copies of a shape over an area.

Extrude and sweep keep the coordinates they are given: a profile or a path is
written where it belongs relative to the object's origin (a mug handle's path
runs from the mug's side to the mug's side), so they do not move it to stand
on its own bottom centre.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from itertools import pairwise
from typing import Any

from ..errors import JingyuError
from ..generator import GeneratorDef, ParamProblem, integer, number
from .mesh import MeshData, Vec3
from .noise import random
from .transform import apply, compose

Point2 = tuple[float, float]

#: The ``$id`` of the geometry union schema; a parameter that is itself a geometry
#: call refers to it.
GEOMETRY_SCHEMA_ID = "urn:jingyu:schema:geometry"


def _invalid(message: str) -> JingyuError:
    return JingyuError("geometry.invalid_profile", message)


# -------------------------------------------------------------------- extrude


def _signed_area(points: Sequence[Point2]) -> float:
    return 0.5 * sum(
        a[0] * b[1] - b[0] * a[1] for a, b in zip(points, [*points[1:], points[0]], strict=True)
    )


def _segments_cross(p: Point2, q: Point2, r: Point2, s: Point2) -> bool:
    def orient(a: Point2, b: Point2, c: Point2) -> float:
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    d1, d2 = orient(p, q, r), orient(p, q, s)
    d3, d4 = orient(r, s, p), orient(r, s, q)
    return d1 * d2 < 0 and d3 * d4 < 0


def _simple(points: Sequence[Point2]) -> bool:
    n = len(points)
    edges = [(points[i], points[(i + 1) % n]) for i in range(n)]
    for i in range(n):
        for j in range(i + 2, n):
            if i == 0 and j == n - 1:
                continue  # neighbours through the closing edge
            if _segments_cross(*edges[i], *edges[j]):
                return False
    return len({(round(x, 12), round(y, 12)) for x, y in points}) == n


def triangulate(points: Sequence[Point2]) -> list[tuple[int, int, int]]:
    """Ear clipping of a simple counter-clockwise polygon."""

    def inside(p: Point2, a: Point2, b: Point2, c: Point2) -> bool:
        d1 = (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
        d2 = (c[0] - b[0]) * (p[1] - b[1]) - (c[1] - b[1]) * (p[0] - b[0])
        d3 = (a[0] - c[0]) * (p[1] - c[1]) - (a[1] - c[1]) * (p[0] - c[0])
        return d1 >= 0 and d2 >= 0 and d3 >= 0

    remaining = list(range(len(points)))
    triangles: list[tuple[int, int, int]] = []
    guard = 0
    while len(remaining) > 3 and guard < 10 * len(points) ** 2:
        guard += 1
        for k in range(len(remaining)):
            i0, i1, i2 = remaining[k - 1], remaining[k], remaining[(k + 1) % len(remaining)]
            a, b, c = points[i0], points[i1], points[i2]
            if (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]) <= 1e-15:
                continue  # reflex or flat corner
            if any(inside(points[j], a, b, c) for j in remaining if j not in (i0, i1, i2)):
                continue
            triangles.append((i0, i1, i2))
            remaining.pop(k)
            break
        else:
            raise _invalid("the profile cannot be triangulated; is it self-intersecting?")
    triangles.append((remaining[0], remaining[1], remaining[2]))
    return triangles


def _profile(p: Mapping[str, Any]) -> list[Point2]:
    points = [(float(x), float(y)) for x, y in p["profile"]]
    if _signed_area(points) < 0:
        points.reverse()
    return points


def _extrude(p: Mapping[str, Any]) -> MeshData:
    points = _profile(p)
    height = float(p["height"])
    taper = float(p["taper"])
    n = len(points)
    cx = sum(x for x, _ in points) / n
    cy = sum(y for _, y in points) / n
    bottom = [(x, y, 0.0) for x, y in points]
    top = [(cx + (x - cx) * taper, cy + (y - cy) * taper, height) for x, y in points]
    faces: list[tuple[int, ...]] = []
    for i in range(n):
        j = (i + 1) % n
        faces.append((i, j, n + j, n + i))
    for a, b, c in triangulate(points):
        faces.append((n + a, n + b, n + c))
        faces.append((c, b, a))
    return MeshData(tuple(bottom + top), tuple(faces), smooth=False)


def _check_extrude(p: Mapping[str, Any]) -> list[ParamProblem]:
    points = [(float(x), float(y)) for x, y in p["profile"]]
    if abs(_signed_area(points)) < 1e-12 or not _simple(points):
        return [("profile", "the profile must be a simple polygon (no crossing edges)")]
    return []


EXTRUDE = GeneratorDef[MeshData](
    name="extrude",
    summary=(
        "A floor plan pulled straight up: any simple polygon (an L-shaped counter, a "
        "hexagonal tile, a slab of irregular stone), optionally narrowing to the top."
    ),
    params={
        "profile": {
            "type": "array",
            "description": "Corners [x, y] of the outline in metres, in order around it; "
            "kept where they are (not centred).",
            "items": {
                "type": "array",
                "prefixItems": [{"type": "number"}, {"type": "number"}],
                "minItems": 2,
                "maxItems": 2,
            },
            "minItems": 3,
            "maxItems": 1024,
        },
        "height": number("Height in metres.", exclusive_minimum=0),
        "taper": number(
            "Scale of the top outline about its centre: 1 straight walls, 0.5 half-size top.",
            default=1.0,
            exclusive_minimum=0,
            maximum=4,
        ),
    },
    run=_extrude,
    check=_check_extrude,
    examples=(
        {
            "op": "extrude",
            "profile": [[0, 0], [1.2, 0], [1.2, 0.6], [0.6, 0.6], [0.6, 1.4], [0, 1.4]],
            "height": 0.9,
        },
    ),
)


# ---------------------------------------------------------------------- sweep


def _vec(values: Sequence[float]) -> Vec3:
    return (float(values[0]), float(values[1]), float(values[2]))


def _lerp3(a: Vec3, b: Vec3, t: float) -> Vec3:
    return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t)


def _catmull(p0: Vec3, p1: Vec3, p2: Vec3, p3: Vec3, t: float) -> Vec3:
    """The Catmull-Rom point at *t* in [0, 1] between *p1* and *p2*."""

    t2, t3 = t * t, t * t * t

    def blend(k: int) -> float:
        return 0.5 * (
            2 * p1[k]
            + (p2[k] - p0[k]) * t
            + (2 * p0[k] - 5 * p1[k] + 4 * p2[k] - p3[k]) * t2
            + (3 * p1[k] - p0[k] - 3 * p2[k] + p3[k]) * t3
        )

    return (blend(0), blend(1), blend(2))


def _catmull_rom(points: Sequence[Vec3], samples: int) -> list[Vec3]:
    """A smooth curve through *points*, *samples* steps per span."""

    if len(points) == 2:
        return [_lerp3(points[0], points[1], i / samples) for i in range(samples + 1)]
    padded = [points[0], *points, points[-1]]
    out = [
        _catmull(padded[i - 1], padded[i], padded[i + 1], padded[i + 2], s / samples)
        for i in range(1, len(padded) - 2)
        for s in range(samples)
    ]
    out.append(points[-1])
    return out


def _sub(a: Sequence[float], b: Sequence[float]) -> Vec3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _scaled(v: Sequence[float], factor: float) -> Vec3:
    return (v[0] * factor, v[1] * factor, v[2] * factor)


def _cross(a: Sequence[float], b: Sequence[float]) -> Vec3:
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _unit(v: Sequence[float]) -> Vec3:
    length = math.sqrt(_dot(v, v))
    if length < 1e-15:
        raise _invalid("the path has two points in the same place")
    return (v[0] / length, v[1] / length, v[2] / length)


def tube(path: Sequence[Vec3], radii: Sequence[float], segments: int) -> MeshData:
    """A closed tube along *path* with a radius per path point.

    The cross-section is carried along by parallel transport, so the tube does
    not twist where the path turns.
    """

    tangents = [
        _unit(_sub(path[min(len(path) - 1, i + 1)], path[max(0, i - 1)])) for i in range(len(path))
    ]
    helper = (0.0, 0.0, 1.0) if abs(tangents[0][2]) < 0.9 else (1.0, 0.0, 0.0)
    normal = _unit(_cross(_cross(tangents[0], helper), tangents[0]))
    vertices: list[Vec3] = []
    for i, (centre, tangent) in enumerate(zip(path, tangents, strict=True)):
        if i:  # remove the normal's component along the new tangent
            normal = _unit(_sub(normal, _scaled(tangent, _dot(normal, tangent))))
        binormal = _cross(tangent, normal)
        for j in range(segments):
            angle = 2.0 * math.pi * j / segments
            c, s = math.cos(angle) * radii[i], math.sin(angle) * radii[i]
            vertices.append(
                (
                    centre[0] + c * normal[0] + s * binormal[0],
                    centre[1] + c * normal[1] + s * binormal[1],
                    centre[2] + c * normal[2] + s * binormal[2],
                )
            )
    faces: list[tuple[int, ...]] = []
    rings = len(path)
    for ring in range(rings - 1):
        for j in range(segments):
            first = ring * segments + j
            second = ring * segments + (j + 1) % segments
            faces.append((first, second, second + segments, first + segments))
    start = len(vertices)
    vertices.append(path[0])
    end = len(vertices)
    vertices.append(path[-1])
    last = (rings - 1) * segments
    for j in range(segments):
        k = (j + 1) % segments
        faces.append((start, k, j))
        faces.append((end, last + j, last + k))
    return MeshData(tuple(vertices), tuple(faces), smooth=True)


def _sweep(p: Mapping[str, Any]) -> MeshData:
    points = [_vec(point) for point in p["path"]]
    path = _catmull_rom(points, int(p["samples"]))
    lengths = [0.0]
    for a, b in pairwise(path):
        lengths.append(lengths[-1] + math.dist(a, b))
    total = lengths[-1] or 1.0
    r0, r1 = float(p["radius"]), float(p["radius_end"] or p["radius"])
    radii = [r0 + (r1 - r0) * length / total for length in lengths]
    return tube(path, radii, int(p["segments"]))


def _check_sweep(p: Mapping[str, Any]) -> list[ParamProblem]:
    points = [_vec(point) for point in p["path"]]
    if any(math.dist(a, b) < 1e-9 for a, b in pairwise(points)):
        return [("path", "two neighbouring path points are in the same place")]
    try:
        _sweep({**p, "segments": 3})
    except JingyuError as exc:
        return [("path", exc.message)]
    return []


SWEEP = GeneratorDef[MeshData](
    name="sweep",
    summary=(
        "A round tube along a smooth path through points: a mug handle, a pipe, a cable, "
        "a stem, a bent rod; it may narrow from one end to the other. The path is kept "
        "where it is written."
    ),
    params={
        "path": {
            "type": "array",
            "description": "Points [x, y, z] in metres the tube passes through, in order; "
            "the curve between them is smooth.",
            "items": {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3},
            "minItems": 2,
            "maxItems": 256,
        },
        "radius": number("Radius at the start in metres.", exclusive_minimum=0),
        "radius_end": number(
            "Radius at the end in metres; 0 keeps the start radius.", default=0.0, minimum=0
        ),
        "segments": integer("Segments around the tube.", default=24, minimum=3, maximum=256),
        "samples": integer(
            "Smooth steps between neighbouring path points.", default=8, minimum=1, maximum=64
        ),
    },
    run=_sweep,
    check=_check_sweep,
    examples=(
        {
            "op": "sweep",
            "path": [[0.04, 0, 0.03], [0.075, 0, 0.05], [0.04, 0, 0.08]],
            "radius": 0.006,
        },
        {
            "op": "sweep",
            "path": [[0, 0, 0], [0.1, 0.05, 0.3], [0.05, 0.2, 0.6]],
            "radius": 0.01,
            "radius_end": 0.004,
        },
    ),
)


# -------------------------------------------------------------------- scatter


def _positions(p: Mapping[str, Any]) -> list[tuple[float, float, int]]:
    """Poisson-disc dart throwing: at most count points, at least spacing apart."""

    width, depth = (float(v) for v in p["area"])
    disc = p["shape"] == "disc"
    spacing = float(p["spacing"])
    count = int(p["count"])
    seed = int(p["seed"])
    chosen: list[tuple[float, float, int]] = []
    for attempt in range(count * 40):
        if len(chosen) >= count:
            break
        u, v = random(attempt, seed, 1), random(attempt, seed, 2)
        x, y = (u - 0.5) * width, (v - 0.5) * depth
        if disc and (x / (width / 2)) ** 2 + (y / (depth / 2)) ** 2 > 1.0:
            continue
        if all((x - cx) ** 2 + (y - cy) ** 2 >= spacing * spacing for cx, cy, _ in chosen):
            chosen.append((x, y, attempt))
    return chosen


def _scatter(p: Mapping[str, Any]) -> MeshData:
    from . import GEOMETRY  # the registry holds this operator

    item = GEOMETRY.run(p["item"])
    seed = int(p["seed"])
    jitter = float(p["scale_jitter"])
    tilt = float(p["tilt"])
    vertices: list[Vec3] = []
    faces: list[tuple[int, ...]] = []
    sharp: list[tuple[int, int]] = []
    for x, y, index in _positions(p):
        scale = 1.0 + jitter * (random(index, seed, 3) * 2.0 - 1.0)
        turn = random(index, seed, 4) * 360.0 if p["rotate"] else 0.0
        lean = tilt * random(index, seed, 5)
        lean_dir = random(index, seed, 6) * 360.0
        matrix = compose((x, y, 0.0), (lean, 0.0, 0.0), (scale, scale, scale))
        matrix = _turned(matrix, lean_dir, turn, (x, y))
        offset = len(vertices)
        vertices.extend(apply(matrix, v) for v in item.vertices)
        faces.extend(tuple(i + offset for i in face) for face in item.faces)
        sharp.extend((a + offset, b + offset) for a, b in item.sharp_edges)
    if not faces:
        raise _invalid("nothing was placed: the area is too small for the spacing")
    return MeshData(tuple(vertices), tuple(faces), item.smooth, tuple(sharp))


def _turned(matrix: Any, lean_dir: float, turn: float, at: Point2) -> Any:
    """Rotate *matrix* about the vertical through *at*: lean first, then the heading."""

    from .transform import multiply

    x, y = at
    to_origin = compose((-x, -y, 0.0), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    back = compose((x, y, 0.0), (0.0, 0.0, lean_dir + turn), (1.0, 1.0, 1.0))
    return multiply(back, multiply(to_origin, matrix))


def _check_scatter(p: Mapping[str, Any]) -> list[ParamProblem]:
    from . import GEOMETRY

    item = p["item"]
    if item.get("op") == "scatter":
        return [("item", "a scatter cannot scatter scatters")]
    definition = GEOMETRY.get(str(item["op"]))
    params = definition.with_defaults({k: v for k, v in item.items() if k != "op"})
    problems = [("item", f"{item['op']}: {message}") for _, message in definition.check(params)]
    if not problems and not _positions(p):
        problems.append(("spacing", "nothing fits: the spacing is larger than the area"))
    return problems


SCATTER = GeneratorDef[MeshData](
    name="scatter",
    summary=(
        "Copies of one shape strewn over a rectangle or disc on the ground, each turned, "
        "sized and tilted a little differently: pebbles, fallen leaves, a pile of beads, "
        "tufts. One object, one material."
    ),
    params={
        "item": {
            "$ref": GEOMETRY_SCHEMA_ID,
            "description": "The shape to copy: any geometry call, e.g. a small sphere.",
        },
        "area": {
            "type": "array",
            "description": "Width (x) and depth (y) of the area in metres.",
            "items": {"type": "number", "exclusiveMinimum": 0},
            "minItems": 2,
            "maxItems": 2,
        },
        "shape": {
            "enum": ["rectangle", "disc"],
            "default": "rectangle",
            "description": "The area is a rectangle, or the disc (ellipse) inside it.",
        },
        "count": integer("How many copies at most.", default=20, minimum=1, maximum=5000),
        "spacing": number(
            "Least distance between copies' centres in metres; fewer than count fit when "
            "it is large.",
            default=0.0,
            minimum=0,
        ),
        "scale_jitter": number(
            "How much the size varies: 0 all alike, 0.3 up to 30 % larger or smaller.",
            default=0.2,
            minimum=0,
            maximum=0.9,
        ),
        "rotate": {"type": "boolean", "default": True, "description": "Turn each copy at random."},
        "tilt": number("Largest lean of a copy in degrees.", default=0.0, minimum=0, maximum=90),
        "seed": integer(
            "Another number, another arrangement.", default=0, minimum=0, maximum=100000
        ),
    },
    run=_scatter,
    check=_check_scatter,
    examples=(
        {
            "op": "scatter",
            "item": {"op": "sphere", "radius": 0.02, "segments": 16, "rings": 8},
            "area": [0.5, 0.3],
            "count": 30,
            "spacing": 0.045,
        },
    ),
)

SHAPES: tuple[GeneratorDef[MeshData], ...] = (EXTRUDE, SWEEP, SCATTER)

__all__ = ["EXTRUDE", "GEOMETRY_SCHEMA_ID", "SCATTER", "SHAPES", "SWEEP", "triangulate", "tube"]
