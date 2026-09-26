"""World-space meshes and axis-aligned ray queries, in pure Python.

The host uses these to rest one object on another and to find objects that sink
into or float above each other, without starting Blender.  A query casts an
infinite line parallel to one coordinate axis and returns every surface it
crosses, with the side the surface faces; triangles are binned on a 2D grid so
a query only tests the few triangles near the line.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from .mesh import MeshData, Portal
from .transform import Mat4, Vec3, apply_all

#: Queries are nudged by these offsets so a line through a shared edge or vertex
#: is counted on exactly one side of it.
_JITTER = (1.3e-7, 2.9e-7)
_MAX_GRID = 256

#: For a line along axis ``a`` the grid uses the other two axes in cyclic order,
#: so a triangle with positive projected area faces the +a direction.
_PLANE_AXES = {0: (1, 2), 1: (2, 0), 2: (0, 1)}


@dataclass(frozen=True)
class Hit:
    """Where a line crosses a surface: its coordinate along the line and facing."""

    at: float
    facing: int  # +1 when the surface faces the +axis direction, -1 otherwise


@dataclass
class AxisIndex:
    """Triangles projected onto the plane perpendicular to one axis, binned on a grid."""

    axis: int
    triangles: list[tuple[float, float, float, float, float, float, float, float, float]]
    cells: dict[tuple[int, int], list[int]]
    origin: tuple[float, float]
    cell: float

    @classmethod
    def build(
        cls, vertices: Sequence[Vec3], triangles: Sequence[tuple[int, int, int]], axis: int
    ) -> AxisIndex:
        iu, iv = _PLANE_AXES[axis]
        projected: list[tuple[float, float, float, float, float, float, float, float, float]] = []
        for a, b, c in triangles:
            pa, pb, pc = vertices[a], vertices[b], vertices[c]
            projected.append(
                (pa[iu], pa[iv], pa[axis], pb[iu], pb[iv], pb[axis], pc[iu], pc[iv], pc[axis])
            )
        if not projected:
            return cls(axis, [], {}, (0.0, 0.0), 1.0)
        us = [v[iu] for v in vertices]
        vs = [v[iv] for v in vertices]
        u0, v0 = min(us), min(vs)
        su, sv = max(us) - u0, max(vs) - v0
        cell = math.sqrt(max(su * sv, 1e-12) / len(projected)) * 2.0
        cell = max(cell, su / _MAX_GRID, sv / _MAX_GRID, 1e-6)
        cells: dict[tuple[int, int], list[int]] = {}
        for index, t in enumerate(projected):
            i0 = math.floor((min(t[0], t[3], t[6]) - u0) / cell)
            i1 = math.floor((max(t[0], t[3], t[6]) - u0) / cell)
            j0 = math.floor((min(t[1], t[4], t[7]) - v0) / cell)
            j1 = math.floor((max(t[1], t[4], t[7]) - v0) / cell)
            for i in range(i0, i1 + 1):
                for j in range(j0, j1 + 1):
                    cells.setdefault((i, j), []).append(index)
        return cls(axis, projected, cells, (u0, v0), cell)

    def hits(self, point: Sequence[float]) -> list[Hit]:
        """Every crossing of the line through *point* parallel to the axis."""

        iu, iv = _PLANE_AXES[self.axis]
        u = float(point[iu]) + _JITTER[0]
        v = float(point[iv]) + _JITTER[1]
        key = (
            math.floor((u - self.origin[0]) / self.cell),
            math.floor((v - self.origin[1]) / self.cell),
        )
        found: list[Hit] = []
        for index in self.cells.get(key, ()):
            au, av, aw, bu, bv, bw, cu, cv, cw = self.triangles[index]
            area = (bu - au) * (cv - av) - (cu - au) * (bv - av)
            if abs(area) < 1e-18:
                continue
            wa = ((bu - u) * (cv - v) - (cu - u) * (bv - v)) / area
            wb = ((cu - u) * (av - v) - (au - u) * (cv - v)) / area
            wc = 1.0 - wa - wb
            if wa < 0.0 or wb < 0.0 or wc < 0.0:
                continue
            found.append(Hit(wa * aw + wb * bw + wc * cw, 1 if area > 0 else -1))
        return found


@dataclass
class WorldMesh:
    """A mesh in world space with lazily built per-axis indices."""

    vertices: list[Vec3]
    triangles: list[tuple[int, int, int]]
    source: MeshData
    #: The source face each triangle was cut from.
    triangle_faces: list[int] = field(default_factory=list, repr=False)
    #: The mesh's openings (windows, doors) in world space.
    portals: tuple[Portal, ...] = ()
    #: Bounding volume hierarchy for arbitrary rays (jingyu.geometry.raycast).
    bvh: Any = field(default=None, repr=False)
    _indices: dict[int, AxisIndex] = field(default_factory=dict, repr=False)
    _closed: bool | None = field(default=None, repr=False)
    _bounds: tuple[Vec3, Vec3] | None = field(default=None, repr=False)

    @classmethod
    def from_mesh(cls, mesh: MeshData, matrix: Mat4) -> WorldMesh:
        triangles: list[tuple[int, int, int]] = []
        faces: list[int] = []
        for index, face in enumerate(mesh.faces):
            for i in range(1, len(face) - 1):
                triangles.append((face[0], face[i], face[i + 1]))
                faces.append(index)
        portals = tuple(portal.transformed(matrix) for portal in mesh.portals)
        return cls(apply_all(matrix, mesh.vertices), triangles, mesh, faces, portals)

    @property
    def closed(self) -> bool:
        if self._closed is None:
            self._closed = self.source.is_closed_manifold()
        return self._closed

    def translated(self, offset: Sequence[float]) -> WorldMesh:
        dx, dy, dz = (float(o) for o in offset)
        moved = [(x + dx, y + dy, z + dz) for x, y, z in self.vertices]
        portals = tuple(p.moved((dx, dy, dz)) for p in self.portals)
        return WorldMesh(
            moved, self.triangles, self.source, self.triangle_faces, portals, _closed=self._closed
        )

    def index(self, axis: int) -> AxisIndex:
        if axis not in self._indices:
            self._indices[axis] = AxisIndex.build(self.vertices, self.triangles, axis)
        return self._indices[axis]

    def bounds(self) -> tuple[Vec3, Vec3]:
        if self._bounds is None:
            xs, ys, zs = zip(*self.vertices, strict=True)
            self._bounds = (min(xs), min(ys), min(zs)), (max(xs), max(ys), max(zs))
        return self._bounds

    def vertices_over(self, box: tuple[Vec3, Vec3]) -> list[Vec3]:
        """Vertices whose x and y fall inside the footprint of *box*."""

        (x0, y0, _), (x1, y1, _) = box
        return [v for v in self.vertices if x0 <= v[0] <= x1 and y0 <= v[1] <= y1]

    def depth_inside(self, point: Sequence[float]) -> float:
        """How far *point* is inside this closed mesh along the nearest axis; 0 outside.

        Inside-ness is the parity of crossings above the point on a vertical line;
        depth is the shortest distance to a surface along the three axis lines.
        """

        if not self.closed:
            return 0.0
        z = float(point[2])
        vertical = self.index(2).hits(point)
        if sum(1 for hit in vertical if hit.at > z) % 2 == 0:
            return 0.0
        depth = math.inf
        for axis in range(3):
            hits = vertical if axis == 2 else self.index(axis).hits(point)
            coordinate = float(point[axis])
            for hit in hits:
                depth = min(depth, abs(hit.at - coordinate))
        return 0.0 if math.isinf(depth) else depth


def overlaps(a: tuple[Vec3, Vec3], b: tuple[Vec3, Vec3], margin: float = 0.0) -> bool:
    """Whether two axis-aligned boxes overlap, each grown by *margin*."""

    return all(a[0][k] - margin <= b[1][k] and b[0][k] - margin <= a[1][k] for k in range(3))


def vertical_contacts(upper: WorldMesh, lower: WorldMesh) -> list[float]:
    """Heights by which *upper* must move up so each vertical contact pair just touches.

    A pair is a point of one mesh and a surface of the other on the same vertical
    line: a vertex of *upper* over an upward-facing surface of *lower*, or a
    vertex of *lower* under a downward-facing surface of *upper*.  A positive
    value means that contact currently overlaps; a negative one is a gap.
    """

    shifts: list[float] = []
    candidates = upper.vertices_over(lower.bounds())
    if candidates:
        below = lower.index(2)
        for vertex in candidates:
            shifts.extend(hit.at - vertex[2] for hit in below.hits(vertex) if hit.facing > 0)
    candidates = lower.vertices_over(upper.bounds())
    if candidates:
        above = upper.index(2)
        for vertex in candidates:
            shifts.extend(vertex[2] - hit.at for hit in above.hits(vertex) if hit.facing < 0)
    return shifts


def rest_shift(upper: WorldMesh, lower: WorldMesh) -> float | None:
    """Vertical move that sets *upper* on the topmost surface of *lower* beneath it.

    It is where *upper* would first touch *lower* when lowered from far above;
    None when no part of *lower* lies under *upper*.
    """

    shifts = vertical_contacts(upper, lower)
    return max(shifts) if shifts else None


def drop_gap(upper: WorldMesh, lower: WorldMesh, tolerance: float) -> float | None:
    """How far *upper* can fall before touching *lower*; None if nothing is below.

    Contacts that already overlap by more than *tolerance* are not "below" and
    are ignored (they are intersections, reported separately).
    """

    gaps = [-shift for shift in vertical_contacts(upper, lower) if shift <= tolerance]
    return max(min(gaps), 0.0) if gaps else None


__all__ = [
    "AxisIndex",
    "Hit",
    "WorldMesh",
    "drop_gap",
    "overlaps",
    "rest_shift",
    "vertical_contacts",
]
