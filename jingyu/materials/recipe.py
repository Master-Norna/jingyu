"""The renderer-facing material recipe every family expands into."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

RGB = tuple[float, float, float]


@dataclass(frozen=True)
class Recipe:
    """A physically based surface in scene-linear values.

    The fields mirror the principled BSDF that Blender (and most PBR renderers)
    implement; a family is a small parametric vocabulary that maps a few
    meaningful knobs (``gloss``, ``frost``, ``polish``) onto these fields.
    """

    base_color: RGB = (0.5, 0.5, 0.5)
    metallic: float = 0.0
    roughness: float = 0.5
    ior: float = 1.5
    transmission: float = 0.0
    coat_weight: float = 0.0
    coat_roughness: float = 0.03
    emission_color: RGB = (1.0, 1.0, 1.0)
    emission_strength: float = 0.0
    alpha: float = 1.0

    def __post_init__(self) -> None:
        for name in (
            "metallic",
            "roughness",
            "transmission",
            "coat_weight",
            "coat_roughness",
            "alpha",
        ):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"recipe {name} must lie in [0, 1], got {value}")
        if not 1.0 <= self.ior <= 4.0:
            raise ValueError(f"recipe ior must lie in [1, 4], got {self.ior}")
        if self.emission_strength < 0:
            raise ValueError("recipe emission_strength must be non-negative")

    @property
    def refractive(self) -> bool:
        return self.transmission > 0.0

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["base_color"] = list(self.base_color)
        data["emission_color"] = list(self.emission_color)
        return data


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


__all__ = ["RGB", "Recipe", "lerp"]
