"""Material families: parametric generators that expand into a :class:`Recipe`.

Standard library only: the Blender worker imports this package.
"""

from __future__ import annotations

from ..generator import GeneratorRegistry
from .families import ALL_FAMILIES, METALS
from .recipe import Recipe

MATERIALS: GeneratorRegistry[Recipe] = GeneratorRegistry(
    kind="material family", discriminator="family", unknown_code="material.unknown_family"
)
for _family in ALL_FAMILIES:
    MATERIALS.register(_family)

__all__ = ["MATERIALS", "METALS", "Recipe"]
