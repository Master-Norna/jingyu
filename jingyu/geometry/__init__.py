"""Geometry operators: parametric generators that produce :class:`MeshData`.

Standard library only: the Blender worker imports this package and turns the
returned mesh data into Blender meshes.  Nothing here knows about Blender.
"""

from __future__ import annotations

from ..generator import GeneratorRegistry
from .assemblies import ASSEMBLIES
from .mesh import MeshData
from .nature import NATURE
from .ops import ALL_OPS
from .shapes import GEOMETRY_SCHEMA_ID, SHAPES

GEOMETRY: GeneratorRegistry[MeshData] = GeneratorRegistry(
    kind="geometry operator",
    discriminator="op",
    unknown_code="geometry.unknown_op",
    schema_id=GEOMETRY_SCHEMA_ID,
)
for _op in (*ALL_OPS, *SHAPES, *ASSEMBLIES, *NATURE):
    GEOMETRY.register(_op)

__all__ = ["GEOMETRY", "MeshData"]
