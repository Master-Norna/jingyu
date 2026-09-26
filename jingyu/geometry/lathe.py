"""The lathe operator: revolve a 2D profile around the Z axis.

Profiles are lists of ``(r, z)`` points in metres, ``r >= 0``.  Every
axially-symmetric shape (cylinders, cones, spheres, cups, bowls, vases,
bottles) reduces to a profile, which is why the lathe is the first operator in
the library.

The revolved polyline must start and end on the axis and walk counter-clockwise
around the solid in the ``(r, z)`` half-plane (outward along the bottom, up the
outside, back towards the axis at the top).  With that orientation the faces
produced here always point outward.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence

from ..errors import JingyuError
from .mesh import MeshData, Vec3

Point2 = tuple[float, float]

AXIS_EPS = 1e-9
MIN_RADIUS = 1e-6


def _invalid(message: str, **details: object) -> JingyuError:
    return JingyuError("geometry.invalid_profile", message, details=details)


def close_solid(profile: Sequence[Point2]) -> list[Point2]:
    """Close an arbitrary profile to the axis at both ends."""

    points = [(float(r), float(z)) for r, z in profile]
    if len(points) < 2:
        raise _invalid("a profile needs at least two points")
    if points[0][0] > AXIS_EPS:
        points.insert(0, (0.0, points[0][1]))
    else:
        points[0] = (0.0, points[0][1])
    if points[-1][0] > AXIS_EPS:
        points.append((0.0, points[-1][1]))
    else:
        points[-1] = (0.0, points[-1][1])
    return points


def shell(profile: Sequence[Point2], thickness: float) -> list[Point2]:
    """Turn an open outer wall into a closed, open-topped shell polyline.

    ``profile`` runs from the bottom outer edge up to the rim; every point must
    be off the axis.  The floor is closed at the bottom and the wall is offset
    inward by ``thickness``; the rim is left flat.
    """

    outer = [(float(r), float(z)) for r, z in profile]
    if len(outer) < 2:
        raise _invalid("a shell profile needs at least two points")
    if thickness <= 0:
        raise _invalid("shell thickness must be positive", thickness=thickness)
    for index, (r, _) in enumerate(outer):
        if r <= MIN_RADIUS:
            raise _invalid("a shell profile must stay off the axis", index=index, radius=r)

    wall = [(0.0, outer[0][1]), *outer]
    normals = [_segment_normal(a, b) for a, b in itertools.pairwise(wall)]
    inner: list[Point2] = []
    for i, (r, z) in enumerate(wall):
        if i == 0:
            nr, nz = normals[0]
            scale = 1.0
        elif i == len(wall) - 1:
            # Offset the rim horizontally so the lip stays flat at the rim height.
            nr, nz = 1.0, 0.0
            scale = 1.0
        else:
            (ar, az), (br, bz) = normals[i - 1], normals[i]
            sr, sz = ar + br, az + bz
            length = math.hypot(sr, sz)
            if length < 1e-12:
                raise _invalid("the profile folds back on itself", index=i - 1)
            nr, nz = sr / length, sz / length
            scale = 1.0 / max(nr * ar + nz * az, 0.25)
        inner.append((r - thickness * scale * nr, z - thickness * scale * nz))

    inner[0] = (0.0, inner[0][1])
    for index, (r, _) in enumerate(inner[1:], start=1):
        if r <= MIN_RADIUS:
            raise _invalid(
                "the wall is too thick for the profile's radius",
                index=index,
                thickness=thickness,
            )
    if inner[0][1] >= outer[-1][1]:
        raise _invalid("the floor thickness reaches the rim", thickness=thickness)
    return wall + list(reversed(inner))


def revolve(polyline: Sequence[Point2], segments: int, sharp_angle: float = 30.0) -> MeshData:
    """Revolve an axis-to-axis polyline into a closed mesh."""

    points = [(float(r), float(z)) for r, z in polyline]
    if len(points) < 3:
        raise _invalid("a revolved polyline needs at least three points")
    if segments < 3:
        raise _invalid("segments must be at least 3", segments=segments)
    if abs(points[0][0]) > AXIS_EPS or abs(points[-1][0]) > AXIS_EPS:
        raise _invalid("a revolved polyline must start and end on the axis")
    for index, (r, _) in enumerate(points[1:-1], start=1):
        if r <= MIN_RADIUS:
            raise _invalid("only the end points may touch the axis", index=index)
    for index, (a, b) in enumerate(itertools.pairwise(points)):
        if math.hypot(b[0] - a[0], b[1] - a[1]) < 1e-12:
            raise _invalid("consecutive profile points coincide", index=index)

    angles = [2.0 * math.pi * j / segments for j in range(segments)]
    trig = [(math.cos(a), math.sin(a)) for a in angles]

    vertices: list[Vec3] = [(0.0, 0.0, points[0][1])]
    for r, z in points[1:-1]:
        vertices.extend((r * c, r * s, z) for c, s in trig)
    last_pole = len(vertices)
    vertices.append((0.0, 0.0, points[-1][1]))

    def ring(k: int, j: int) -> int:
        """Vertex index of ring ``k`` (1-based interior point) at segment ``j``."""

        return 1 + (k - 1) * segments + (j % segments)

    faces: list[tuple[int, ...]] = []
    interior = len(points) - 2
    for j in range(segments):
        faces.append((0, ring(1, j + 1), ring(1, j)))
    for k in range(1, interior):
        for j in range(segments):
            faces.append((ring(k, j), ring(k, j + 1), ring(k + 1, j + 1), ring(k + 1, j)))
    for j in range(segments):
        faces.append((ring(interior, j), ring(interior, j + 1), last_pole))

    sharp: list[tuple[int, int]] = []
    for k in range(1, interior + 1):
        if _turn_degrees(points[k - 1], points[k], points[k + 1]) > sharp_angle:
            sharp.extend((ring(k, j), ring(k, j + 1)) for j in range(segments))

    return MeshData(tuple(vertices), tuple(faces), smooth=True, sharp_edges=tuple(sharp))


def _segment_normal(a: Point2, b: Point2) -> Point2:
    dr, dz = b[0] - a[0], b[1] - a[1]
    length = math.hypot(dr, dz)
    if length < 1e-12:
        raise _invalid("consecutive profile points coincide")
    return dz / length, -dr / length


def _turn_degrees(a: Point2, b: Point2, c: Point2) -> float:
    u = (b[0] - a[0], b[1] - a[1])
    v = (c[0] - b[0], c[1] - b[1])
    nu, nv = math.hypot(*u), math.hypot(*v)
    cosine = max(-1.0, min(1.0, (u[0] * v[0] + u[1] * v[1]) / (nu * nv)))
    return math.degrees(math.acos(cosine))


__all__ = ["Point2", "close_solid", "revolve", "shell"]
