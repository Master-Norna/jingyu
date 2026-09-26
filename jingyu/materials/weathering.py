"""Weathering: time on a surface, for any material.

A clean generator surface looks like a product render.  Real things are worn
where hands and other things rub them (convex edges), dusty where dust settles
(surfaces facing up), grimy where dirt collects (creases, near the floor),
marked by dried water (rings) and smudged by fingers (on glossy surfaces).
Each effect is a layer painted where a field says it belongs, so it follows the
shape of whatever it is applied to.  Weathering is written on a material entry,
next to its family, and works the same for every family.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..conventions import COLOR_PATTERN, srgb_hex_to_linear
from .recipe import RGB, Bump, Field, Layer, Recipe, mix

_DUST: RGB = (0.3, 0.28, 0.25)
_GRIME: RGB = (0.035, 0.028, 0.02)
_MINERAL: RGB = (0.62, 0.6, 0.55)


def _unit(description: str) -> dict[str, Any]:
    return {
        "type": "number",
        "minimum": 0,
        "maximum": 1,
        "default": 0.0,
        "description": description,
    }


WEATHERING_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "description": (
        "Time on the surface, for any material; every effect follows the object's shape. "
        "0 is factory-new. A little (0.1-0.3) is what makes a render read as a photograph."
    ),
    "properties": {
        "wear": _unit("Rubbed convex edges and corners: paint and glaze worn thin, gloss dulled."),
        "wear_color": {
            "type": "string",
            "pattern": COLOR_PATTERN,
            "description": "What shows where it is worn through (bare wood under paint, "
            "clay under glaze), sRGB hex; omitted: a paler, duller version of the surface.",
        },
        "dust": _unit("Matte grey dust settled on the surfaces facing up."),
        "grime": _unit("Dark dirt in creases, inside corners and near the base."),
        "stains": _unit("Dried water marks: pale rings of mineral deposit."),
        "fingerprints": _unit("Smudges that dull a glossy surface in patches."),
        "seed": {
            "type": "integer",
            "minimum": 0,
            "maximum": 100000,
            "default": 0,
            "description": "Variation: another number places the marks differently.",
        },
    },
}


def weather(recipe: Recipe, spec: Mapping[str, Any] | None) -> Recipe:
    """*recipe* with the weathering *spec* painted on top."""

    if not spec:
        return recipe
    seed = int(spec.get("seed", 0)) * 7 + 101
    layers: list[Layer] = []
    bumps: list[Bump] = []

    wear = float(spec.get("wear", 0.0))
    if wear > 0:
        under = spec.get("wear_color")
        color = (
            srgb_hex_to_linear(under) if under else mix(recipe.base_color, (0.5, 0.48, 0.45), 0.35)
        )
        # Occlusion inside the solid rises only to about 0.2 on an edge.
        edges = Field("edges", distance=0.004 + 0.01 * wear, low=0.02, high=0.12 - 0.04 * wear)
        chips = Field(
            "noise", size=0.01, detail=5, roughness=0.7, seed=seed, low=0.45 - 0.2 * wear, high=0.6
        )
        layers.append(
            Layer(
                (edges, chips),
                amount=min(1.0, 0.4 + wear),
                color=color,
                roughness=min(1.0, recipe.roughness + 0.3),
                coat_weight=0.0,
            )
        )

    stains = float(spec.get("stains", 0.0))
    if stains > 0:
        rims = Field("cracks", size=0.08, seed=seed + 1, low=0.03, high=0.0)
        where = Field("noise", size=0.2, detail=2, seed=seed + 2, low=0.7 - 0.3 * stains, high=0.75)
        layers.append(
            Layer((rims, where), amount=min(1.0, 0.3 + 0.6 * stains), color=_MINERAL, roughness=0.8)
        )

    grime = float(spec.get("grime", 0.0))
    if grime > 0:
        creases = Field("cavity", distance=0.02, low=0.05, high=0.6)
        base = Field("low", distance=0.08, low=0.0, high=1.0)
        mottle = Field("noise", size=0.05, detail=3, seed=seed + 3, low=0.3, high=0.8)
        layers.append(
            Layer((creases, mottle), amount=min(1.0, 1.2 * grime), color=_GRIME, roughness=0.9)
        )
        layers.append(Layer((base, mottle), amount=min(1.0, grime), color=_GRIME, roughness=0.9))

    dust = float(spec.get("dust", 0.0))
    if dust > 0:
        up = Field("facing_up", low=0.6, high=0.95)
        drifts = Field(
            "noise",
            size=0.03,
            detail=4,
            roughness=0.6,
            seed=seed + 4,
            low=0.55 - 0.35 * dust,
            high=0.75,
        )
        layers.append(
            Layer(
                (up, drifts),
                amount=min(1.0, 0.15 + 0.55 * dust),
                color=_DUST,
                roughness=1.0,
                metallic=0.0,
                coat_weight=0.0,
            )
        )
        bumps.append(Bump(Field("noise", size=0.0005, detail=2, seed=seed + 5), 0.00005, 0.3))

    prints = float(spec.get("fingerprints", 0.0))
    if prints > 0:
        smudges = Field(
            "noise", size=0.015, detail=3, seed=seed + 6, low=0.6 - 0.25 * prints, high=0.7
        )
        ridges = Field(
            "bands", size=0.0005, axis="x", distortion=8.0, seed=seed + 7, low=0.3, high=0.7
        )
        layers.append(
            Layer(
                (smudges, ridges), amount=0.8 * prints, roughness=min(1.0, recipe.roughness + 0.35)
            )
        )
    return recipe.with_layers(*layers, bumps=tuple(bumps))


def material_recipe(entry: Mapping[str, Any]) -> Recipe:
    """The recipe of a scene material entry: its family, then its weathering."""

    from . import MATERIALS  # the registry imports this module's package

    params = {k: v for k, v in entry.items() if k not in ("id", "weathering")}
    return weather(MATERIALS.run(params), entry.get("weathering"))


__all__ = ["WEATHERING_SCHEMA", "material_recipe", "weather"]
