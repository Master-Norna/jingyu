"""Shared, deterministic stroke direction and pressure for linear drawing media.

These are 2D image-space operations, not simulated pencil or brush physics.
The local image gradient defines a tangent direction; flat regions fall back to
opposed diagonal strokes. Hashes vary placement without using global RNG.
"""

from __future__ import annotations

import math
from typing import cast

from PIL import Image, ImageDraw


def local_tangent(
    value: Image.Image, x: int, y: int, step: int, crossed: bool = False
) -> tuple[float, float]:
    """Unit tangent along brightness contours, with a diagonal for flat areas."""

    width, height = value.size
    left = cast(int, value.getpixel((max(0, x - step), y)))
    right = cast(int, value.getpixel((min(width - 1, x + step), y)))
    up = cast(int, value.getpixel((x, max(0, y - step))))
    down = cast(int, value.getpixel((x, min(height - 1, y + step))))
    dx, dy = right - left, down - up
    if abs(dx) + abs(dy) < 8:
        return (0.70710678, 0.70710678 if crossed else -0.70710678)
    vx, vy = (dx, dy) if crossed else (-dy, dx)
    length = math.hypot(vx, vy)
    return vx / length, vy / length


def directional_hatch(
    value: Image.Image, pitch: int, thickness: int, seed: int, crossed: bool = False
) -> Image.Image:
    """Separated pressure-varying strokes that follow local brightness contours."""

    result = Image.new("L", value.size, 0)
    draw = ImageDraw.Draw(result)
    width, height = value.size
    jitter = max(1, pitch // 3)
    for y in range(0, height, pitch):
        for x in range(0, width, pitch):
            key = (
                (x + 1) * 73856093
                ^ (y + 1) * 19349663
                ^ (seed + (43 if crossed else 0) + 1) * 83492791
            ) & 0xFFFFFFFF
            px = min(width - 1, x + pitch // 2 + (key >> 8) % (jitter * 2 + 1) - jitter)
            py = min(height - 1, y + pitch // 2 + (key >> 16) % (jitter * 2 + 1) - jitter)
            px, py = max(0, px), max(0, py)
            vx, vy = local_tangent(value, px, py, max(2, pitch // 2), crossed)
            reach = pitch * (0.45 + ((key >> 22) & 3) * 0.08)
            ink = min(255, 176 + ((key >> 25) & 63))
            draw.line(
                (px - vx * reach, py - vy * reach, px + vx * reach, py + vy * reach),
                fill=ink,
                width=max(1, thickness),
            )
    return result


def tapered_brush(
    canvas: Image.Image,
    value: Image.Image,
    x: int,
    y: int,
    length: int,
    width: int,
    ink: int,
    seed_key: int,
) -> None:
    """Place one tapered, contour-following brush mark into an L mask."""

    vx, vy = local_tangent(value, x, y, max(2, length // 3))
    direction = -1 if seed_key & 1 else 1
    vx *= direction
    vy *= direction
    end_x = min(canvas.width - 1, max(0, round(x + length * vx)))
    end_y = min(canvas.height - 1, max(0, round(y + length * vy)))
    mid_x = round((x + end_x) / 2)
    mid_y = round((y + end_y) / 2)
    draw = ImageDraw.Draw(canvas)
    draw.line((x, y, mid_x, mid_y), fill=ink, width=max(1, width))
    draw.line(
        (mid_x, mid_y, end_x, end_y),
        fill=max(0, ink - 32),
        width=max(1, width // 2),
    )


__all__ = ["directional_hatch", "local_tangent", "tapered_brush"]
