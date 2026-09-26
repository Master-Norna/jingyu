"""Renderer-independent mesh data produced by geometry operators."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

Vec3 = tuple[float, float, float]
Mat4 = tuple[float, ...]


@dataclass(frozen=True)
class Portal:
    """A rectangular opening that light and sight pass through (a window, a door).

    The rectangle is ``centre + a * half_width + b * half_height`` for ``a`` and
    ``b`` in [-1, 1]; *key* locates the parameter that made it, relative to the
    geometry (``openings/0``).
    """

    key: str
    centre: Vec3
    half_width: Vec3
    half_height: Vec3

    def transformed(self, matrix: Mat4) -> Portal:
        return Portal(
            self.key,
            _apply(matrix, self.centre),
            _apply_direction(matrix, self.half_width),
            _apply_direction(matrix, self.half_height),
        )

    def moved(self, offset: Vec3) -> Portal:
        x, y, z = self.centre
        return Portal(
            self.key,
            (x + offset[0], y + offset[1], z + offset[2]),
            self.half_width,
            self.half_height,
        )

    def corners(self) -> tuple[Vec3, Vec3, Vec3, Vec3]:
        """Corners in order: bottom-left, bottom-right, top-right, top-left."""

        return (
            self.point(-1.0, -1.0),
            self.point(1.0, -1.0),
            self.point(1.0, 1.0),
            self.point(-1.0, 1.0),
        )

    def point(self, a: float, b: float) -> Vec3:
        c, u, v = self.centre, self.half_width, self.half_height
        return (
            c[0] + a * u[0] + b * v[0],
            c[1] + a * u[1] + b * v[1],
            c[2] + a * u[2] + b * v[2],
        )

    def crossing(self, origin: Vec3, direction: Vec3) -> tuple[float, float, float] | None:
        """Where a ray crosses the portal's plane: (distance, a, b), or None if parallel
        or behind the origin.  The ray passes through the opening when |a|, |b| <= 1."""

        u, v = self.half_width, self.half_height
        normal = (
            u[1] * v[2] - u[2] * v[1],
            u[2] * v[0] - u[0] * v[2],
            u[0] * v[1] - u[1] * v[0],
        )
        denominator = sum(n * d for n, d in zip(normal, direction, strict=True))
        if abs(denominator) < 1e-12:
            return None
        offset = tuple(c - o for c, o in zip(self.centre, origin, strict=True))
        distance = sum(n * o for n, o in zip(normal, offset, strict=True)) / denominator
        if distance <= 0.0:
            return None
        hit = tuple(
            o + d * distance - c for o, d, c in zip(origin, direction, self.centre, strict=True)
        )
        a = sum(h * x for h, x in zip(hit, u, strict=True)) / sum(x * x for x in u)
        b = sum(h * x for h, x in zip(hit, v, strict=True)) / sum(x * x for x in v)
        return distance, a, b


def _apply(m: Mat4, p: Vec3) -> Vec3:
    return (
        m[0] * p[0] + m[1] * p[1] + m[2] * p[2] + m[3],
        m[4] * p[0] + m[5] * p[1] + m[6] * p[2] + m[7],
        m[8] * p[0] + m[9] * p[1] + m[10] * p[2] + m[11],
    )


def _apply_direction(m: Mat4, p: Vec3) -> Vec3:
    return (
        m[0] * p[0] + m[1] * p[1] + m[2] * p[2],
        m[4] * p[0] + m[5] * p[1] + m[6] * p[2],
        m[8] * p[0] + m[9] * p[1] + m[10] * p[2],
    )


@dataclass(frozen=True)
class MeshData:
    """A polygon mesh with outward-facing winding (counter-clockwise from outside).

    ``sharp_edges`` lists vertex-index pairs whose shading must not be smoothed
    across; renderers keep them crisp even when ``smooth`` is true.  ``portals``
    are the openings cut through it.  An assembled mesh names its ``parts`` (a
    table's top and legs) and gives each face's part index in ``face_parts``; a
    mesh without parts is one piece.
    """

    vertices: tuple[Vec3, ...]
    faces: tuple[tuple[int, ...], ...]
    smooth: bool
    sharp_edges: tuple[tuple[int, int], ...] = ()
    portals: tuple[Portal, ...] = ()
    parts: tuple[str, ...] = ()
    face_parts: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.parts and len(self.face_parts) != len(self.faces):
            raise ValueError("face_parts must give a part for every face")

    def part_of(self, face: int) -> str | None:
        """The part a face belongs to, or None for a mesh without parts."""

        return self.parts[self.face_parts[face]] if self.parts else None

    def bounds(self) -> tuple[Vec3, Vec3]:
        xs, ys, zs = zip(*self.vertices, strict=True)
        return (min(xs), min(ys), min(zs)), (max(xs), max(ys), max(zs))

    def edge_use(self) -> Counter[tuple[int, int]]:
        """Count directed edges; a closed, consistently wound mesh uses each once."""

        use: Counter[tuple[int, int]] = Counter()
        for face in self.faces:
            for i, a in enumerate(face):
                b = face[(i + 1) % len(face)]
                use[(a, b)] += 1
        return use

    def is_closed_manifold(self) -> bool:
        """True when every edge is shared by exactly two faces with opposite winding."""

        use = self.edge_use()
        return all(n == 1 and use.get((b, a)) == 1 for (a, b), n in use.items())

    def signed_volume(self) -> float:
        """Divergence-theorem volume; positive when faces wind outward."""

        total = 0.0
        v = self.vertices
        for face in self.faces:
            a = v[face[0]]
            for i in range(1, len(face) - 1):
                b, c = v[face[i]], v[face[i + 1]]
                total += (
                    a[0] * (b[1] * c[2] - b[2] * c[1])
                    - a[1] * (b[0] * c[2] - b[2] * c[0])
                    + a[2] * (b[0] * c[1] - b[1] * c[0])
                )
        return total / 6.0


__all__ = ["MeshData", "Portal", "Vec3"]
