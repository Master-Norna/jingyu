"""Deterministic value noise in pure Python, for shaping geometry.

Terrain, rocks and trees need the same kind of smooth randomness shaders use,
but on the host (so placement and physics see the final shape) and without
global state: every value is a pure function of the coordinates and a seed.
"""

from __future__ import annotations

import math


def _hash(ix: int, iy: int, iz: int, seed: int) -> float:
    """A repeatable pseudo-random value in [0, 1) for a lattice point."""

    h = (ix * 374761393 + iy * 668265263 + iz * 2147483647 + seed * 1274126177) & 0xFFFFFFFF
    h = ((h ^ (h >> 13)) * 1274126177) & 0xFFFFFFFF
    h ^= h >> 16
    return h / 4294967296.0


def _fade(t: float) -> float:
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def value_noise(x: float, y: float, z: float = 0.0, seed: int = 0) -> float:
    """Smooth noise in [0, 1] with features about one unit across."""

    ix, iy, iz = math.floor(x), math.floor(y), math.floor(z)
    fx, fy, fz = _fade(x - ix), _fade(y - iy), _fade(z - iz)
    total = 0.0
    for dz in (0, 1):
        wz = fz if dz else 1.0 - fz
        for dy in (0, 1):
            wy = fy if dy else 1.0 - fy
            for dx in (0, 1):
                wx = fx if dx else 1.0 - fx
                total += wx * wy * wz * _hash(ix + dx, iy + dy, iz + dz, seed)
    return total


def fbm(
    x: float,
    y: float,
    z: float = 0.0,
    *,
    octaves: int = 4,
    gain: float = 0.5,
    seed: int = 0,
    ridged: float = 0.0,
) -> float:
    """Fractal sum of value noise, centred on 0 and roughly within [-1, 1].

    *ridged* blends in ridged noise (1 - |2n - 1|), which forms crests like
    mountain ranges instead of rounded hills.
    """

    total = 0.0
    amplitude = 1.0
    norm = 0.0
    frequency = 1.0
    for octave in range(max(1, octaves)):
        n = value_noise(x * frequency, y * frequency, z * frequency, seed + octave * 101)
        smooth = n * 2.0 - 1.0
        crest = (1.0 - abs(smooth)) * 2.0 - 1.0
        total += amplitude * (smooth * (1.0 - ridged) + crest * ridged)
        norm += amplitude
        amplitude *= gain
        frequency *= 2.0
    return total / norm


def random(index: int, seed: int, salt: int = 0) -> float:
    """A repeatable pseudo-random number in [0, 1) for item *index*."""

    return _hash(index, salt, 17, seed)


__all__ = ["fbm", "random", "value_noise"]
