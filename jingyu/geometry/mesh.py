"""Renderer-independent mesh data produced by geometry operators."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

Vec3 = tuple[float, float, float]


@dataclass(frozen=True)
class MeshData:
    """A polygon mesh with outward-facing winding (counter-clockwise from outside).

    ``sharp_edges`` lists vertex-index pairs whose shading must not be smoothed
    across; renderers keep them crisp even when ``smooth`` is true.
    """

    vertices: tuple[Vec3, ...]
    faces: tuple[tuple[int, ...], ...]
    smooth: bool
    sharp_edges: tuple[tuple[int, int], ...] = ()

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


__all__ = ["MeshData", "Vec3"]
