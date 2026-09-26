"""Small curve utilities for profile-based operators."""

from __future__ import annotations

import itertools
from collections.abc import Sequence


def pchip(xs: Sequence[float], ys: Sequence[float], samples: Sequence[float]) -> list[float]:
    """Monotone piecewise cubic Hermite interpolation (Fritsch-Carlson).

    Between two control points the curve never overshoots them, so a vessel
    profile stays within the radii its parameters promise.  ``xs`` must be
    strictly increasing.
    """

    n = len(xs)
    if n != len(ys) or n < 2:
        raise ValueError("pchip needs at least two matching control points")
    if any(b <= a for a, b in itertools.pairwise(xs)):
        raise ValueError("pchip control x values must be strictly increasing")

    h = [xs[i + 1] - xs[i] for i in range(n - 1)]
    delta = [(ys[i + 1] - ys[i]) / h[i] for i in range(n - 1)]
    m = [0.0] * n
    if n == 2:
        m[0] = m[1] = delta[0]
    else:
        for i in range(1, n - 1):
            if delta[i - 1] * delta[i] <= 0:
                m[i] = 0.0
            else:
                w1 = 2 * h[i] + h[i - 1]
                w2 = h[i] + 2 * h[i - 1]
                m[i] = (w1 + w2) / (w1 / delta[i - 1] + w2 / delta[i])
        m[0] = _end_slope(h[0], h[1], delta[0], delta[1])
        m[-1] = _end_slope(h[-1], h[-2], delta[-1], delta[-2])

    out: list[float] = []
    seg = 0
    for x in samples:
        if x <= xs[0]:
            out.append(float(ys[0]))
            continue
        if x >= xs[-1]:
            out.append(float(ys[-1]))
            continue
        while not xs[seg] <= x <= xs[seg + 1]:
            seg = seg + 1 if x > xs[seg + 1] else seg - 1
        t = (x - xs[seg]) / h[seg]
        t2, t3 = t * t, t * t * t
        out.append(
            (2 * t3 - 3 * t2 + 1) * ys[seg]
            + (t3 - 2 * t2 + t) * h[seg] * m[seg]
            + (-2 * t3 + 3 * t2) * ys[seg + 1]
            + (t3 - t2) * h[seg] * m[seg + 1]
        )
    return out


def _end_slope(h0: float, h1: float, d0: float, d1: float) -> float:
    slope = ((2 * h0 + h1) * d0 - h0 * d1) / (h0 + h1)
    if slope * d0 <= 0:
        return 0.0
    if d0 * d1 <= 0 and abs(slope) > abs(3 * d0):
        return 3 * d0
    return slope


def linspace(start: float, stop: float, count: int) -> list[float]:
    if count < 2:
        raise ValueError("linspace needs at least two samples")
    step = (stop - start) / (count - 1)
    return [start + i * step for i in range(count - 1)] + [stop]


__all__ = ["linspace", "pchip"]
