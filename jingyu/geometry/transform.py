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


__all__ = [
    "IDENTITY",
    "Mat4",
    "Vec3",
    "apply",
    "apply_all",
    "apply_direction",
    "compose",
    "multiply",
    "rotation_xyz",
    "solve_linear",
]
