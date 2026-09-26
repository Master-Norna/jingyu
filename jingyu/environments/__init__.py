"""Environment families: parametric generators for the light around a scene.

Standard library only: the Blender worker imports this package.
"""

from __future__ import annotations

from ..generator import GeneratorRegistry
from .families import ALL_FAMILIES, SKY_STRENGTH_SCALE
from .recipe import EnvironmentRecipe, Sky, Sun

ENVIRONMENTS: GeneratorRegistry[EnvironmentRecipe] = GeneratorRegistry(
    kind="environment family",
    discriminator="family",
    unknown_code="environment.unknown_family",
)
for _family in ALL_FAMILIES:
    ENVIRONMENTS.register(_family)

__all__ = ["ENVIRONMENTS", "SKY_STRENGTH_SCALE", "EnvironmentRecipe", "Sky", "Sun"]
