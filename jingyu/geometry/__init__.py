"""Geometry operators: parametric generators that produce :class:`MeshData`.

Standard library only: the Blender worker imports this package and turns the
returned mesh data into Blender meshes.  Nothing here knows about Blender.
"""

from __future__ import annotations

from ..generator import GeneratorRegistry
from .mesh import MeshData
from .ops import ALL_OPS

GEOMETRY: GeneratorRegistry[MeshData] = GeneratorRegistry(
    kind="geometry operator", discriminator="op", unknown_code="geometry.unknown_op"
)
for _op in ALL_OPS:
    GEOMETRY.register(_op)

__all__ = ["GEOMETRY", "MeshData"]
