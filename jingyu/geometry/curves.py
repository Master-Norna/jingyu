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

    h, m = _pchip_slopes(xs, ys)
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


def _pchip_slopes(xs: Sequence[float], ys: Sequence[float]) -> tuple[list[float], list[float]]:
    """Segment widths and the Fritsch-Carlson slopes at the control points."""

    n = len(xs)
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
    return h, m


def smooth_monotone(
    xs: Sequence[float], ys: Sequence[float], samples: Sequence[float]
) -> list[float]:
    """Like :func:`pchip`, but with continuous curvature where it can be kept.

    A cubic through the control points is continuous in slope only: at each
    control point the curvature jumps, and on a glossy surface of revolution the
    jump shows as a crease in the highlight.  Here each segment is a quintic
    Hermite curve that shares the slope *and* the second derivative with its
    neighbour.  The shared second derivative is the smaller of the two one-sided
    cubic values (zero where they disagree in sign); it is reduced further where
    a segment would overshoot its control points, so the curve still stays within
    the radii its parameters promise.
    """

    n = len(xs)
    if n != len(ys) or n < 2:
        raise ValueError("pchip needs at least two matching control points")
    if any(b <= a for a, b in itertools.pairwise(xs)):
        raise ValueError("pchip control x values must be strictly increasing")
    h, m = _pchip_slopes(xs, ys)
    delta = [(ys[i + 1] - ys[i]) / h[i] for i in range(n - 1)]
    right = [(6 * delta[i] - 4 * m[i] - 2 * m[i + 1]) / h[i] for i in range(n - 1)]
    left = [(-6 * delta[i] + 2 * m[i] + 4 * m[i + 1]) / h[i] for i in range(n - 1)]
    a = [0.0] * n
    a[0], a[-1] = right[0], left[-1]
    for i in range(1, n - 1):
        lo, hi = left[i - 1], right[i]
        a[i] = 0.0 if lo * hi <= 0 else (lo if abs(lo) < abs(hi) else hi)
    cubic = [False] * (n - 1)
    for seg in range(n - 1):
        for _ in range(8):
            if not _overshoots(ys, h, m, a, seg):
                break
            a[seg] *= 0.5
            a[seg + 1] *= 0.5
        else:
            cubic[seg] = _overshoots(ys, h, m, a, seg)

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
        if cubic[seg]:
            out.append(_cubic(ys, h, m, seg, t))
        else:
            out.append(_quintic(ys, h, m, a, seg, t))
    return out


def _cubic(ys: Sequence[float], h: list[float], m: list[float], seg: int, t: float) -> float:
    t2, t3 = t * t, t * t * t
    return (
        (2 * t3 - 3 * t2 + 1) * ys[seg]
        + (t3 - 2 * t2 + t) * h[seg] * m[seg]
        + (-2 * t3 + 3 * t2) * ys[seg + 1]
        + (t3 - t2) * h[seg] * m[seg + 1]
    )


def _quintic(
    ys: Sequence[float], h: list[float], m: list[float], a: list[float], seg: int, t: float
) -> float:
    t2, t3 = t * t, t * t * t
    t4, t5 = t3 * t, t3 * t2
    w = h[seg]
    return (
        (1 - 10 * t3 + 15 * t4 - 6 * t5) * ys[seg]
        + (t - 6 * t3 + 8 * t4 - 3 * t5) * w * m[seg]
        + (0.5 * t2 - 1.5 * t3 + 1.5 * t4 - 0.5 * t5) * w * w * a[seg]
        + (0.5 * t3 - t4 + 0.5 * t5) * w * w * a[seg + 1]
        + (-4 * t3 + 7 * t4 - 3 * t5) * w * m[seg + 1]
        + (10 * t3 - 15 * t4 + 6 * t5) * ys[seg + 1]
    )


def _overshoots(
    ys: Sequence[float], h: list[float], m: list[float], a: list[float], seg: int
) -> bool:
    """Whether the quintic segment leaves its monotone band (or passes its extremum)."""

    lo, hi = min(ys[seg], ys[seg + 1]), max(ys[seg], ys[seg + 1])
    tolerance = 1e-9 * max(1.0, abs(lo), abs(hi))
    values = [_quintic(ys, h, m, a, seg, k / 64) for k in range(65)]
    if any(v < lo - tolerance or v > hi + tolerance for v in values):
        return True
    rising = ys[seg + 1] >= ys[seg]
    return any(
        (b < c - tolerance) if rising else (b > c + tolerance)
        for c, b in itertools.pairwise(values)
    )


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


__all__ = ["linspace", "pchip", "smooth_monotone"]
