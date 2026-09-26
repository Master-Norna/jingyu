"""Affine transforms in pure Python, matching Blender's conventions.

A transform is a row-major 4x4 matrix stored as a tuple of 16 floats.  Object
matrices are ``T @ R @ S`` with an XYZ Euler rotation in degrees (Blender's
``rotation_mode = 'XYZ'``: X is applied first, then Y, then Z), so the host can
compute the same world-space geometry the worker builds.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

Mat4 = tuple[float, ...]
Vec3 = tuple[float, float, float]

IDENTITY: Mat4 = (1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0)


def multiply(a: Mat4, b: Mat4) -> Mat4:
    """``a @ b``: apply *b* first, then *a*."""

    return tuple(
        sum(a[row * 4 + k] * b[k * 4 + col] for k in range(4))
        for row in range(4)
        for col in range(4)
    )


def rotation_xyz(degrees: Sequence[float]) -> Mat4:
    rx, ry, rz = (math.radians(float(a)) for a in degrees)
    cx, sx, cy, sy, cz, sz = (
        math.cos(rx),
        math.sin(rx),
        math.cos(ry),
        math.sin(ry),
        math.cos(rz),
        math.sin(rz),
    )
    # Rz @ Ry @ Rx
    return (
        cy * cz,
        sx * sy * cz - cx * sz,
        cx * sy * cz + sx * sz,
        0.0,
        cy * sz,
        sx * sy * sz + cx * cz,
        cx * sy * sz - sx * cz,
        0.0,
        -sy,
        sx * cy,
        cx * cy,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
    )


def compose(
    location: Sequence[float], rotation_deg: Sequence[float], scale: Sequence[float]
) -> Mat4:
    """``T @ R @ S`` for one node."""

    r = rotation_xyz(rotation_deg)
    sx, sy, sz = (float(s) for s in scale)
    tx, ty, tz = (float(t) for t in location)
    return (
        r[0] * sx,
        r[1] * sy,
        r[2] * sz,
        tx,
        r[4] * sx,
        r[5] * sy,
        r[6] * sz,
        ty,
        r[8] * sx,
        r[9] * sy,
        r[10] * sz,
        tz,
        0.0,
        0.0,
        0.0,
        1.0,
    )


def apply(m: Mat4, point: Sequence[float]) -> Vec3:
    x, y, z = point
    return (
        m[0] * x + m[1] * y + m[2] * z + m[3],
        m[4] * x + m[5] * y + m[6] * z + m[7],
        m[8] * x + m[9] * y + m[10] * z + m[11],
    )


def apply_all(m: Mat4, points: Iterable[Sequence[float]]) -> list[Vec3]:
    return [apply(m, p) for p in points]


def apply_direction(m: Mat4, vector: Sequence[float]) -> Vec3:
    """Transform a direction (ignores translation)."""

    x, y, z = vector
    return (
        m[0] * x + m[1] * y + m[2] * z,
        m[4] * x + m[5] * y + m[6] * z,
        m[8] * x + m[9] * y + m[10] * z,
    )


def solve_linear(m: Mat4, vector: Sequence[float]) -> Vec3:
    """Solve ``L @ x = vector`` for the 3x3 linear part ``L`` of *m* (Cramer's rule)."""

    a, b, c = m[0], m[1], m[2]
    d, e, f = m[4], m[5], m[6]
    g, h, i = m[8], m[9], m[10]
    det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
    if abs(det) < 1e-18:
        raise ValueError("the transform is singular")
    u, v, w = vector
    return (
        (u * (e * i - f * h) - b * (v * i - f * w) + c * (v * h - e * w)) / det,
        (a * (v * i - f * w) - u * (d * i - f * g) + c * (d * w - v * g)) / det,
        (a * (e * w - v * h) - b * (d * w - v * g) + u * (d * h - e * g)) / det,
    )


def look_rotation(direction: Sequence[float]) -> Mat4:
    """Rotation that points a camera or light (which looks along its local -Z) along
    *direction*, with its local +Y as close to world +Z as possible (no roll).

    Straight up or down, where "towards +Z" gives no heading, the local +X is kept
    on world +X (looking down) or -X (looking up), so north is up in the frame.
    """

    fx, fy, fz = (float(v) for v in direction)
    length = math.sqrt(fx * fx + fy * fy + fz * fz)
    if length == 0.0:
        raise ValueError("cannot aim along a zero-length direction")
    fx, fy, fz = fx / length, fy / length, fz / length
    horizontal = math.hypot(fx, fy)
    if horizontal < 1e-9:
        rx, ry, rz = (1.0 if fz < 0 else -1.0), 0.0, 0.0
    else:  # right = forward x world up
        rx, ry, rz = fy / horizontal, -fx / horizontal, 0.0
    # up = right x forward
    ux, uy, uz = ry * fz - rz * fy, rz * fx - rx * fz, rx * fy - ry * fx
    # Columns are the local axes in world space: +X right, +Y up, +Z back.
    return (rx, ux, -fx, 0.0, ry, uy, -fy, 0.0, rz, uz, -fz, 0.0, 0.0, 0.0, 0.0, 1.0)


def rotation_part(m: Mat4) -> Mat4:
    """The rotation of *m* with scale removed (its columns normalised)."""

    columns = []
    for col in range(3):
        x, y, z = m[col], m[4 + col], m[8 + col]
        length = math.sqrt(x * x + y * y + z * z)
        if length == 0.0:
            raise ValueError("the transform collapses an axis")
        columns.append((x / length, y / length, z / length))
    return (
        columns[0][0], columns[1][0], columns[2][0], 0.0,
        columns[0][1], columns[1][1], columns[2][1], 0.0,
        columns[0][2], columns[1][2], columns[2][2], 0.0,
        0.0, 0.0, 0.0, 1.0,
    )  # fmt: skip


def with_translation(m: Mat4, location: Sequence[float]) -> Mat4:
    """*m* with its translation replaced by *location*."""

    x, y, z = (float(v) for v in location)
    return (m[0], m[1], m[2], x, m[4], m[5], m[6], y, m[8], m[9], m[10], z, 0.0, 0.0, 0.0, 1.0)


def column(m: Mat4, index: int) -> Vec3:
    """Column *index* of the linear part: where local axis *index* points in world space."""

    return (m[index], m[4 + index], m[8 + index])


def translation(m: Mat4) -> Vec3:
    return (m[3], m[7], m[11])


__all__ = [
    "IDENTITY",
    "Mat4",
    "Vec3",
    "apply",
    "apply_all",
    "apply_direction",
    "column",
    "compose",
    "look_rotation",
    "multiply",
    "rotation_part",
    "rotation_xyz",
    "solve_linear",
    "translation",
    "with_translation",
]
