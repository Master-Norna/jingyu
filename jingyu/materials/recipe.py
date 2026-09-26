"""The renderer-facing material recipe every family expands into.

A recipe is a physically based surface: constant principled channels, plus
optional *layers* that paint a region of the surface with other values and
*bumps* that give it relief.  Where a layer or a bump applies is a *field*: a
scalar in [0, 1] over the surface, taken from a classic procedural texture
(noise, cells, growth rings) or from the shape itself (surfaces facing up,
convex edges, creases).  Fields are 3D and read the object's own coordinates in
metres, so they need no UV unwrapping, have no seams and never tile.

Everything here is data; the Blender worker translates it into shader nodes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

RGB = tuple[float, float, float]

#: Kinds of field and what their raw value means before remapping.
FIELD_KINDS: dict[str, str] = {
    "noise": "fractal noise, 0..1, features about `size` metres across",
    "ridges": "1 where fractal noise crosses its middle: thin veins (marble)",
    "cells": "distance to the nearest random point of cells `size` across: 0 at the centres",
    "cracks": "distance to the nearest border between cells: 0 on the borders",
    "rings": "concentric rings around `axis`, `size` metres apart (wood grain)",
    "bands": "parallel bands across `axis`, `size` metres apart",
    "facing_up": "how much the surface faces up: 1 on a table top, 0 on walls and below",
    "edges": "1 on convex edges and corners (where things get worn), over `distance`",
    "cavity": "1 in creases and inside corners (where dirt collects), over `distance`",
    "low": "1 at the object's base, fading out over `distance` metres of height",
}
AXES = ("x", "y", "z")


@dataclass(frozen=True)
class Field:
    """A scalar over the surface: a raw texture value, remapped by a smooth step from
    ``low`` (0) to ``high`` (1) and optionally inverted."""

    kind: str
    size: float = 0.1
    detail: float = 2.0
    roughness: float = 0.5
    distortion: float = 0.0
    axis: str = "z"
    #: Scale of the coordinates along x, y, z before the texture: (0.05, 1, 1)
    #: stretches features twenty times along x (wood pores along the grain).
    stretch: tuple[float, float, float] = (1.0, 1.0, 1.0)
    distance: float = 0.01
    seed: int = 0
    low: float = 0.0
    high: float = 1.0
    invert: bool = False

    def __post_init__(self) -> None:
        if self.kind not in FIELD_KINDS:
            raise ValueError(f"unknown field kind {self.kind!r}")
        if self.axis not in AXES:
            raise ValueError(f"field axis must be one of {AXES}")
        if self.size <= 0 or self.distance <= 0 or min(self.stretch) <= 0:
            raise ValueError("field size, distance and stretch must be positive")
        if self.low == self.high:
            raise ValueError("field low and high must differ")


@dataclass(frozen=True)
class Layer:
    """Where the product of the ``mask`` fields (times ``amount``) is 1, the surface
    takes these values instead of the ones below; ``None`` leaves a channel alone."""

    mask: tuple[Field, ...]
    amount: float = 1.0
    color: RGB | None = None
    roughness: float | None = None
    metallic: float | None = None
    coat_weight: float | None = None

    def __post_init__(self) -> None:
        if not self.mask:
            raise ValueError("a layer needs at least one mask field")
        if not 0.0 <= self.amount <= 1.0:
            raise ValueError("layer amount must lie in [0, 1]")
        for name in ("roughness", "metallic", "coat_weight"):
            value = getattr(self, name)
            if value is not None and not 0.0 <= value <= 1.0:
                raise ValueError(f"layer {name} must lie in [0, 1]")


@dataclass(frozen=True)
class Bump:
    """Relief: the surface is pushed out ``distance`` metres where the field is 1."""

    field: Field
    distance: float
    strength: float = 1.0

    def __post_init__(self) -> None:
        if self.distance <= 0 or not 0.0 < self.strength <= 1.0:
            raise ValueError("bump distance must be positive and strength in (0, 1]")


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
    sheen_weight: float = 0.0
    emission_color: RGB = (1.0, 1.0, 1.0)
    emission_strength: float = 0.0
    alpha: float = 1.0
    #: A thin transparent sheet (a window pane): light passes straight through
    #: without bending, and shadows let it through; only the surface reflects.
    thin: bool = False
    layers: tuple[Layer, ...] = field(default=())
    bumps: tuple[Bump, ...] = field(default=())

    def __post_init__(self) -> None:
        for name in (
            "metallic",
            "roughness",
            "transmission",
            "coat_weight",
            "coat_roughness",
            "sheen_weight",
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

    @property
    def textured(self) -> bool:
        return bool(self.layers or self.bumps)

    def with_layers(self, *layers: Layer, bumps: tuple[Bump, ...] = ()) -> Recipe:
        """This recipe with more layers painted on top and more relief."""

        values = {k: getattr(self, k) for k in self.__dataclass_fields__}
        values["layers"] = (*self.layers, *layers)
        values["bumps"] = (*self.bumps, *bumps)
        return Recipe(**values)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["base_color"] = list(self.base_color)
        data["emission_color"] = list(self.emission_color)
        return data


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def mix(a: RGB, b: RGB, t: float) -> RGB:
    return (lerp(a[0], b[0], t), lerp(a[1], b[1], t), lerp(a[2], b[2], t))


def scale(color: RGB, factor: float) -> RGB:
    return (
        min(1.0, color[0] * factor),
        min(1.0, color[1] * factor),
        min(1.0, color[2] * factor),
    )


__all__ = ["AXES", "FIELD_KINDS", "RGB", "Bump", "Field", "Layer", "Recipe", "lerp", "mix", "scale"]
