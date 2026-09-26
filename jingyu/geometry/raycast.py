"""Rays in any direction against world-space meshes, in pure Python.

The axis-aligned queries in :mod:`jingyu.geometry.spatial` answer "what is below
this point"; aiming a camera or the sun needs "what does this line of sight or
this sunbeam hit first".  Each mesh gets a bounding volume hierarchy over its
triangles (built once, on first use), and a ray visits only the boxes it
crosses, so a few hundred rays against a room full of vessels stay fast.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from .spatial import WorldMesh
from .transform import Vec3

_LEAF_SIZE = 6
_EPSILON = 1e-12
_PAD = 1e-9


@dataclass(frozen=True)
class RayHit:
    """Where a ray first meets a mesh."""

    distance: float
    point: Vec3
    #: Unit normal of the triangle hit, facing the side its winding calls outside.
    normal: Vec3
    #: Index of the source face (a polygon of the generator's mesh).
    face: int


@dataclass(frozen=True)
class SceneHit:
    object: str
    hit: RayHit


class BVH:
    """A binary tree of axis-aligned boxes over a mesh's triangles."""

    def __init__(self, mesh: WorldMesh):
        self.vertices = mesh.vertices
        self.triangles = mesh.triangles
        order = list(range(len(self.triangles)))
        self._boxes: list[tuple[float, float, float, float, float, float]] = []
        #: For an inner node: (left child, right child); for a leaf: (-start - 1, count).
        self._links: list[tuple[int, int]] = []
        self._order: list[int] = []
        if order:
            bounds = [self._triangle_box(t) for t in self.triangles]
            centres = [((b[0] + b[3]) / 2, (b[1] + b[4]) / 2, (b[2] + b[5]) / 2) for b in bounds]
            self._build(order, bounds, centres)

    def _triangle_box(
        self, triangle: tuple[int, int, int]
    ) -> tuple[float, float, float, float, float, float]:
        a, b, c = (self.vertices[i] for i in triangle)
        return (
            min(a[0], b[0], c[0]),
            min(a[1], b[1], c[1]),
            min(a[2], b[2], c[2]),
            max(a[0], b[0], c[0]),
            max(a[1], b[1], c[1]),
            max(a[2], b[2], c[2]),
        )

    def _build(
        self,
        items: list[int],
        bounds: Sequence[tuple[float, float, float, float, float, float]],
        centres: Sequence[tuple[float, float, float]],
    ) -> int:
        # Boxes are padded so a ray grazing a triangle's edge is not lost to rounding.
        box = (
            min(bounds[i][0] for i in items) - _PAD,
            min(bounds[i][1] for i in items) - _PAD,
            min(bounds[i][2] for i in items) - _PAD,
            max(bounds[i][3] for i in items) + _PAD,
            max(bounds[i][4] for i in items) + _PAD,
            max(bounds[i][5] for i in items) + _PAD,
        )
        node = len(self._boxes)
        self._boxes.append(box)
        self._links.append((0, 0))
        if len(items) <= _LEAF_SIZE:
            start = len(self._order)
            self._order.extend(items)
            self._links[node] = (-start - 1, len(items))
            return node
        extent = [box[3] - box[0], box[4] - box[1], box[5] - box[2]]
        axis = extent.index(max(extent))
        items.sort(key=lambda i: centres[i][axis])
        middle = len(items) // 2
        left = self._build(items[:middle], bounds, centres)
        right = self._build(items[middle:], bounds, centres)
        self._links[node] = (left, right)
        return node

    def cast(
        self, origin: Sequence[float], direction: Sequence[float], t_max: float
    ) -> RayHit | None:
        """The nearest hit with ``0 < distance <= t_max``; *direction* must be unit length."""

        if not self._boxes:
            return None
        ox, oy, oz = (float(v) for v in origin)
        dx, dy, dz = (float(v) for v in direction)
        inv = _inverse((dx, dy, dz))
        best_t = t_max
        best: int | None = None
        stack = [0]
        while stack:
            node = stack.pop()
            if not _slab(self._boxes[node], (ox, oy, oz), inv, best_t):
                continue
            first, second = self._links[node]
            if first >= 0:
                stack.append(first)
                stack.append(second)
                continue
            start = -first - 1
            for index in self._order[start : start + second]:
                t = _intersect(self.vertices, self.triangles[index], ox, oy, oz, dx, dy, dz)
                if t is not None and t <= best_t:
                    best_t, best = t, index
        if best is None:
            return None
        a, b, c = (self.vertices[i] for i in self.triangles[best])
        normal = _unit(_cross(_sub(b, a), _sub(c, a)))
        point = (ox + dx * best_t, oy + dy * best_t, oz + dz * best_t)
        return RayHit(best_t, point, normal, best)


def _slab(
    box: tuple[float, float, float, float, float, float],
    origin: tuple[float, float, float],
    inv: tuple[float, ...],
    t_max: float,
) -> bool:
    t0, t1 = 0.0, t_max
    for axis in range(3):
        lo = (box[axis] - origin[axis]) * inv[axis]
        hi = (box[axis + 3] - origin[axis]) * inv[axis]
        if math.isnan(lo) or math.isnan(hi):  # a ray lying in the slab's plane
            if not box[axis] <= origin[axis] <= box[axis + 3]:
                return False
            continue
        if lo > hi:
            lo, hi = hi, lo
        t0 = max(t0, lo)
        t1 = min(t1, hi)
        if t0 > t1:
            return False
    return True


def _intersect(
    vertices: Sequence[Vec3],
    triangle: tuple[int, int, int],
    ox: float,
    oy: float,
    oz: float,
    dx: float,
    dy: float,
    dz: float,
) -> float | None:
    """Moller-Trumbore: distance along the ray to the triangle, both faces counted."""

    a, b, c = (vertices[i] for i in triangle)
    e1x, e1y, e1z = b[0] - a[0], b[1] - a[1], b[2] - a[2]
    e2x, e2y, e2z = c[0] - a[0], c[1] - a[1], c[2] - a[2]
    px, py, pz = dy * e2z - dz * e2y, dz * e2x - dx * e2z, dx * e2y - dy * e2x
    det = e1x * px + e1y * py + e1z * pz
    if abs(det) < _EPSILON:
        return None
    inv_det = 1.0 / det
    tx, ty, tz = ox - a[0], oy - a[1], oz - a[2]
    u = (tx * px + ty * py + tz * pz) * inv_det
    if u < 0.0 or u > 1.0:
        return None
    qx, qy, qz = ty * e1z - tz * e1y, tz * e1x - tx * e1z, tx * e1y - ty * e1x
    v = (dx * qx + dy * qy + dz * qz) * inv_det
    if v < 0.0 or u + v > 1.0:
        return None
    t = (e2x * qx + e2y * qy + e2z * qz) * inv_det
    return t if t > 1e-9 else None


def _sub(a: Sequence[float], b: Sequence[float]) -> Vec3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _cross(a: Sequence[float], b: Sequence[float]) -> Vec3:
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def _unit(v: Sequence[float]) -> Vec3:
    length = math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
    if length == 0.0:
        return (0.0, 0.0, 0.0)
    return (v[0] / length, v[1] / length, v[2] / length)


def bvh(mesh: WorldMesh) -> BVH:
    """The mesh's hierarchy, built on first use and kept on the mesh."""

    if mesh.bvh is None:
        mesh.bvh = BVH(mesh)
    tree = mesh.bvh
    assert isinstance(tree, BVH)
    return tree


def cast_mesh(
    mesh: WorldMesh, origin: Sequence[float], direction: Sequence[float], t_max: float = math.inf
) -> RayHit | None:
    """The first point where the ray from *origin* along unit *direction* meets *mesh*."""

    if not _slab(_box(mesh), _vec(origin), _inverse(direction), t_max):
        return None
    hit = bvh(mesh).cast(origin, direction, t_max)
    if hit is None:
        return None
    return RayHit(hit.distance, hit.point, hit.normal, mesh.triangle_faces[hit.face])


def cast(
    meshes: Mapping[str, WorldMesh],
    origin: Sequence[float],
    direction: Sequence[float],
    t_max: float = math.inf,
    *,
    skip: Iterable[str] = (),
) -> SceneHit | None:
    """The nearest object the ray meets among *meshes*, ignoring the ids in *skip*."""

    skipped = set(skip)
    best: SceneHit | None = None
    limit = t_max
    for object_id, mesh in meshes.items():
        if object_id in skipped:
            continue
        hit = cast_mesh(mesh, origin, direction, limit)
        if hit is not None and hit.distance <= limit:
            best, limit = SceneHit(object_id, hit), hit.distance
    return best


def _vec(v: Sequence[float]) -> tuple[float, float, float]:
    return (float(v[0]), float(v[1]), float(v[2]))


def _inverse(direction: Sequence[float]) -> tuple[float, ...]:
    return tuple(
        1.0 / float(d) if abs(float(d)) > _EPSILON else math.copysign(math.inf, float(d) or 1.0)
        for d in direction
    )


def _box(mesh: WorldMesh) -> tuple[float, float, float, float, float, float]:
    (x0, y0, z0), (x1, y1, z1) = mesh.bounds()
    return (x0 - _PAD, y0 - _PAD, z0 - _PAD, x1 + _PAD, y1 + _PAD, z1 + _PAD)


def unit(vector: Sequence[float]) -> Vec3:
    """*vector* scaled to unit length (the zero vector stays zero)."""

    return _unit(vector)


__all__ = ["BVH", "RayHit", "SceneHit", "bvh", "cast", "cast_mesh", "unit"]
