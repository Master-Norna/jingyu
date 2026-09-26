"""The built-in material families.

A family names a kind of stuff (ceramic, glass, metal, plastic) and exposes the
few knobs that matter for it.  Two pieces of the same family differ only by
parameter values, never by a new file.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..conventions import COLOR_PATTERN, srgb_hex_to_linear
from ..generator import GeneratorDef, number
from .recipe import RGB, Recipe, lerp


def _color(description: str, default: str) -> dict[str, Any]:
    return {
        "type": "string",
        "pattern": COLOR_PATTERN,
        "description": description + " sRGB hex #rrggbb.",
        "default": default,
    }


def _unit(description: str, default: float) -> dict[str, Any]:
    return number(description, default=default, minimum=0.0, maximum=1.0)


def _principled(p: Mapping[str, Any]) -> Recipe:
    return Recipe(
        base_color=srgb_hex_to_linear(p["base_color"]),
        metallic=p["metallic"],
        roughness=p["roughness"],
        ior=p["ior"],
        transmission=p["transmission"],
        coat_weight=p["coat_weight"],
        coat_roughness=p["coat_roughness"],
        emission_color=srgb_hex_to_linear(p["emission_color"]),
        emission_strength=p["emission_strength"],
        alpha=p["alpha"],
    )


PRINCIPLED = GeneratorDef[Recipe](
    name="principled",
    summary="Direct physically based parameters; the escape hatch when no family fits.",
    params={
        "base_color": _color("Surface colour.", "#bfbfbf"),
        "metallic": _unit("0 dielectric, 1 metal.", 0.0),
        "roughness": _unit("0 mirror-smooth, 1 fully diffuse.", 0.5),
        "ior": number("Index of refraction.", default=1.5, minimum=1.0, maximum=4.0),
        "transmission": _unit("0 opaque, 1 fully transmissive (glass-like).", 0.0),
        "coat_weight": _unit("Strength of a clear coat layer.", 0.0),
        "coat_roughness": _unit("Roughness of the clear coat.", 0.03),
        "emission_color": _color("Emitted light colour.", "#ffffff"),
        "emission_strength": number(
            "Emitted radiance multiplier; 0 emits nothing.", default=0.0, minimum=0.0
        ),
        "alpha": _unit("Opacity; 1 is fully opaque.", 1.0),
    },
    run=_principled,
    examples=({"family": "principled", "base_color": "#808080", "roughness": 0.4},),
)


def _ceramic(p: Mapping[str, Any]) -> Recipe:
    gloss = float(p["gloss"])
    return Recipe(
        base_color=srgb_hex_to_linear(p["color"]),
        roughness=lerp(0.85, 0.3, gloss),
        ior=1.5,
        coat_weight=gloss,
        coat_roughness=lerp(0.25, 0.02, gloss),
    )


CERAMIC = GeneratorDef[Recipe](
    name="ceramic",
    summary="Fired clay: matte bisque (gloss 0) to a glassy glaze (gloss 1).",
    params={
        "color": _color("Body or glaze colour.", "#f2efe8"),
        "gloss": _unit("Glaze gloss from matte (0) to mirror-like (1).", 0.8),
    },
    run=_ceramic,
    examples=({"family": "ceramic", "color": "#6f93b3", "gloss": 0.9},),
)


def _glass(p: Mapping[str, Any]) -> Recipe:
    return Recipe(
        base_color=srgb_hex_to_linear(p["tint"]),
        roughness=lerp(0.0, 0.5, float(p["frost"])),
        ior=p["ior"],
        transmission=1.0,
        thin=bool(p["thin"]),
    )


GLASS = GeneratorDef[Recipe](
    name="glass",
    summary="Transparent refractive glass, clear or tinted, polished or frosted.",
    params={
        "tint": _color("Transmitted colour; white is clear.", "#ffffff"),
        "frost": _unit("Surface frosting from polished (0) to heavily frosted (1).", 0.0),
        "ior": number("Index of refraction.", default=1.5, minimum=1.0, maximum=4.0),
        "thin": {
            "type": "boolean",
            "default": False,
            "description": "true for a thin sheet such as a window pane: light and sunbeams "
            "pass straight through without bending, only the surface reflects. false for "
            "solid glass (a bottle, a paperweight) that bends what is seen through it.",
        },
    },
    run=_glass,
    examples=({"family": "glass", "tint": "#cfe8d4"}, {"family": "glass", "thin": True}),
)


#: Scene-linear specular reflectance (F0) of common metals.
METALS: dict[str, RGB] = {
    "aluminum": (0.913, 0.922, 0.924),
    "brass": (0.910, 0.778, 0.423),
    "chrome": (0.550, 0.556, 0.554),
    "copper": (0.955, 0.638, 0.538),
    "gold": (1.000, 0.766, 0.336),
    "iron": (0.562, 0.565, 0.578),
    "nickel": (0.660, 0.609, 0.526),
    "silver": (0.972, 0.960, 0.915),
    "titanium": (0.542, 0.497, 0.449),
}


def _metal(p: Mapping[str, Any]) -> Recipe:
    return Recipe(
        base_color=METALS[p["metal"]],
        metallic=1.0,
        roughness=lerp(0.6, 0.02, float(p["polish"])),
    )


METAL = GeneratorDef[Recipe](
    name="metal",
    summary="Conductive metal with measured reflectance, from brushed-dull to mirror-polished.",
    params={
        "metal": {
            "type": "string",
            "enum": sorted(METALS),
            "description": "Which metal; sets the physically measured reflectance colour.",
            "default": "silver",
        },
        "polish": _unit("From dull (0) to mirror-polished (1).", 0.8),
    },
    run=_metal,
    examples=({"family": "metal", "metal": "copper", "polish": 0.5},),
)


def _plastic(p: Mapping[str, Any]) -> Recipe:
    return Recipe(
        base_color=srgb_hex_to_linear(p["color"]),
        roughness=lerp(0.7, 0.05, float(p["gloss"])),
        ior=1.46,
    )


PLASTIC = GeneratorDef[Recipe](
    name="plastic",
    summary="Opaque dielectric such as plastic, paint or plaster, matte to glossy.",
    params={
        "color": _color("Surface colour.", "#d0d0d0"),
        "gloss": _unit("From matte (0) to glossy (1).", 0.3),
    },
    run=_plastic,
    examples=({"family": "plastic", "color": "#e8e4dc", "gloss": 0.1},),
)


ALL_FAMILIES: tuple[GeneratorDef[Recipe], ...] = (CERAMIC, GLASS, METAL, PLASTIC, PRINCIPLED)

__all__ = ["ALL_FAMILIES", "METALS"]
