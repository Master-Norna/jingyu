"""Building blocks for assembled geometry: rounded boxes, pieces and parts.

An assembly (a table, a chair, a room) is one object built from several pieces,
each belonging to a named part that can carry its own material.  Pieces are
closed meshes placed by a transform; merging them keeps every piece's faces and
records which part each face belongs to.

Rounded boxes replace the razor-sharp edges of a plain box, the first thing that
gives a render away as computer-made: an edge of a few millimetres catches a
highlight and reads as a real, handled object.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from .lathe import revolve
from .mesh import MeshData, Portal, Vec3
from .transform import IDENTITY, Mat4, apply, compose

#: Steps across each quarter of a rounded edge.
BEVEL_SEGMENTS = 3


def _axis_samples(half: float, radius: float) -> list[float]:
    """Coordinates along one axis of a rounded box's surface grid.

    The rounded band is sampled at uniform angles (the projection of the band onto
    the flat face goes as tan), and a guard sample just inside the flat region keeps
    smooth shading from bending the whole face toward the edge.
    """

    if radius <= 0.0:
        return [-half, half]
    inner = half - radius
    band = [
        inner + radius * math.tan(math.radians(45.0 * k / BEVEL_SEGMENTS))
        for k in range(BEVEL_SEGMENTS + 1)
    ]
    values = set(band) | {-b for b in band}
    guard = min(radius * 0.05, inner * 0.5)
    if inner > 0.0 and guard > 0.0:
        values |= {inner - guard, -(inner - guard)}
    return sorted(values)


def rounded_box(size: Sequence[float], radius: float) -> MeshData:
    """A box of *size* standing on its origin, its edges rounded by *radius*.

    *radius* 0 gives a plain box with crisp, flat-shaded faces.
    """

    half = [float(s) / 2.0 for s in size]
    radius = max(0.0, min(float(radius), *half))
    samples = [_axis_samples(h, radius) for h in half]
    inner = [h - radius for h in half]
    vertices: list[Vec3] = []
    index: dict[tuple[int, int, int], int] = {}

    def vertex(key: tuple[int, int, int]) -> int:
        found = index.get(key)
        if found is not None:
            return found
        point = [samples[a][key[a]] for a in range(3)]
        if radius > 0.0:
            clamped = [max(-inner[a], min(inner[a], point[a])) for a in range(3)]
            offset = [point[a] - clamped[a] for a in range(3)]
            length = math.sqrt(sum(o * o for o in offset))
            point = [clamped[a] + radius * offset[a] / length for a in range(3)]
        index[key] = len(vertices)
        vertices.append((point[0], point[1], point[2] + half[2]))
        return index[key]

    faces: list[tuple[int, ...]] = []
    for axis in range(3):
        b, c = (axis + 1) % 3, (axis + 2) % 3
        for side in (0, len(samples[axis]) - 1):
            for i in range(len(samples[b]) - 1):
                for j in range(len(samples[c]) - 1):
                    corners = []
                    for di, dj in ((0, 0), (1, 0), (1, 1), (0, 1)):
                        key = [0, 0, 0]
                        key[axis], key[b], key[c] = side, i + di, j + dj
                        corners.append(vertex((key[0], key[1], key[2])))
                    faces.append(tuple(corners) if side else tuple(reversed(corners)))
    if radius == 0.0:
        return MeshData(tuple(vertices), tuple(faces), smooth=False)
    return MeshData(tuple(vertices), tuple(faces), smooth=True)


def cylinder(radius: float, height: float, segments: int = 32) -> MeshData:
    """A closed cylinder standing on its origin, crisp at its rims."""

    return revolve([(0.0, 0.0), (radius, 0.0), (radius, height), (0.0, height)], segments)


def tapered_leg(bottom: float, top: float, height: float, segments: int = 32) -> MeshData:
    """A round leg narrowing from *top* radius to *bottom* radius at the floor."""

    return revolve([(0.0, 0.0), (bottom, 0.0), (top, height), (0.0, height)], segments)


@dataclass(frozen=True)
class Piece:
    """A closed mesh, where it goes, and which part it belongs to."""

    mesh: MeshData
    part: str
    matrix: Mat4 = IDENTITY


def at(location: Sequence[float], rotation: Sequence[float] = (0.0, 0.0, 0.0)) -> Mat4:
    return compose(location, rotation, (1.0, 1.0, 1.0))


def assemble(
    pieces: Sequence[Piece], parts: Sequence[str], portals: Sequence[Portal] = ()
) -> MeshData:
    """Merge *pieces* into one mesh whose faces remember their part.

    Flat-shaded pieces keep their look inside the smooth-shaded whole: all their
    edges are marked sharp.
    """

    order = list(parts)
    vertices: list[Vec3] = []
    faces: list[tuple[int, ...]] = []
    face_parts: list[int] = []
    sharp: list[tuple[int, int]] = []
    for piece in pieces:
        offset = len(vertices)
        vertices.extend(apply(piece.matrix, v) for v in piece.mesh.vertices)
        part = order.index(piece.part)
        for face in piece.mesh.faces:
            faces.append(tuple(i + offset for i in face))
            face_parts.append(part)
            if not piece.mesh.smooth:
                sharp.extend(
                    (face[k] + offset, face[(k + 1) % len(face)] + offset) for k in range(len(face))
                )
        sharp.extend((a + offset, b + offset) for a, b in piece.mesh.sharp_edges)
    unique = sorted({(min(a, b), max(a, b)) for a, b in sharp})
    return MeshData(
        tuple(vertices),
        tuple(faces),
        smooth=True,
        sharp_edges=tuple(unique),
        portals=tuple(portals),
        parts=tuple(order),
        face_parts=tuple(face_parts),
    )


__all__ = [
    "BEVEL_SEGMENTS",
    "Piece",
    "assemble",
    "at",
    "cylinder",
    "rounded_box",
    "tapered_leg",
]
