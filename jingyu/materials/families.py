"""The built-in material families.

A family names a kind of stuff (ceramic, glass, metal, plastic, wood, stone,
fabric, fruit peel) and exposes the few knobs that matter for it.  Two pieces of
the same family differ only by parameter values, never by a new file.

Texture comes from classic procedural fields (see :mod:`.recipe`): growth rings
for wood, veins for marble, cells for granite and orange peel, fine noise for
clay and plaster.  A texture knob at 0 gives the plain, even surface.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..conventions import COLOR_PATTERN, srgb_hex_to_linear
from ..generator import GeneratorDef, integer, number
from .recipe import RGB, Bump, Field, Layer, Recipe, lerp, mix, scale


def _color(description: str, default: str) -> dict[str, Any]:
    return {
        "type": "string",
        "pattern": COLOR_PATTERN,
        "description": description + " sRGB hex #rrggbb.",
        "default": default,
    }


def _unit(description: str, default: float) -> dict[str, Any]:
    return number(description, default=default, minimum=0.0, maximum=1.0)


_SEED = integer(
    "Variation: another number gives another pattern of the same kind.",
    default=0,
    minimum=0,
    maximum=100000,
)


def _texture(description: str) -> dict[str, Any]:
    return _unit(description, 0.0)


# ------------------------------------------------------------------ principled


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


# --------------------------------------------------------------------- ceramic

_IRON_SPECK: RGB = (0.035, 0.025, 0.018)


def _ceramic(p: Mapping[str, Any]) -> Recipe:
    gloss = float(p["gloss"])
    seed = int(p["seed"])
    recipe = Recipe(
        base_color=srgb_hex_to_linear(p["color"]),
        roughness=lerp(0.85, 0.3, gloss),
        ior=1.5,
        coat_weight=gloss,
        coat_roughness=lerp(0.25, 0.02, gloss),
    )
    layers: list[Layer] = []
    bumps: list[Bump] = []
    speckle = float(p["speckle"])
    if speckle > 0:
        # Cells are 0 at their centres: a reversed ramp makes small round specks.
        spots = Field("cells", size=0.01, seed=seed, low=0.12 + 0.12 * speckle, high=0.08)
        layers.append(Layer((spots,), amount=min(1.0, 0.6 + 0.4 * speckle), color=_IRON_SPECK))
    texture = float(p["texture"])
    if texture > 0:
        grain = Field("noise", size=0.0015, detail=4, roughness=0.7, seed=seed + 1)
        bumps.append(Bump(grain, distance=0.0002 * texture, strength=min(1.0, 0.3 + texture)))
        mottle = Field("noise", size=0.03, detail=2, seed=seed + 2, low=0.35, high=0.75)
        layers.append(Layer((mottle,), amount=0.35 * texture, color=scale(recipe.base_color, 0.85)))
    return recipe.with_layers(*layers, bumps=tuple(bumps))


CERAMIC = GeneratorDef[Recipe](
    name="ceramic",
    summary="Fired clay: matte bisque (gloss 0) to a glassy glaze (gloss 1).",
    params={
        "color": _color("Body or glaze colour.", "#f2efe8"),
        "gloss": _unit("Glaze gloss from matte (0) to mirror-like (1).", 0.8),
        "speckle": _texture(
            "Dark iron specks of stoneware clay showing through the glaze, 0 none to 1 many."
        ),
        "texture": _texture(
            "Grain and mottling of the clay body, 0 smooth to 1 coarse; strongest on "
            "unglazed (low gloss) pieces."
        ),
        "seed": _SEED,
    },
    run=_ceramic,
    examples=(
        {"family": "ceramic", "color": "#6f93b3", "gloss": 0.9},
        {"family": "ceramic", "color": "#d9cbb2", "gloss": 0.35, "speckle": 0.6, "texture": 0.5},
    ),
)


# ----------------------------------------------------------------------- glass


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


# ----------------------------------------------------------------------- metal

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

#: What each metal turns into with age: (colour, roughness, metallic).  Iron rusts
#: to a rough orange-brown oxide; copper browns and, in the creases, greens.
TARNISH: dict[str, tuple[RGB, float, float]] = {
    "aluminum": ((0.35, 0.35, 0.34), 0.7, 0.3),
    "brass": ((0.09, 0.06, 0.025), 0.55, 0.4),
    "chrome": ((0.25, 0.25, 0.25), 0.5, 0.8),
    "copper": ((0.07, 0.03, 0.018), 0.55, 0.3),
    "gold": ((0.45, 0.33, 0.15), 0.4, 0.9),
    "iron": ((0.13, 0.04, 0.012), 0.9, 0.0),
    "nickel": ((0.2, 0.18, 0.12), 0.55, 0.6),
    "silver": ((0.05, 0.045, 0.04), 0.5, 0.5),
    "titanium": ((0.2, 0.18, 0.22), 0.5, 0.8),
}
_VERDIGRIS: RGB = (0.12, 0.33, 0.24)


def _metal(p: Mapping[str, Any]) -> Recipe:
    kind = p["metal"]
    seed = int(p["seed"])
    recipe = Recipe(
        base_color=METALS[kind],
        metallic=1.0,
        roughness=lerp(0.6, 0.02, float(p["polish"])),
    )
    layers: list[Layer] = []
    bumps: list[Bump] = []
    brushed = float(p["brushed"])
    if brushed > 0:
        streaks = Field("noise", size=0.004, detail=3, stretch=(0.02, 1.0, 1.0), seed=seed)
        layers.append(
            Layer((streaks,), amount=0.6 * brushed, roughness=min(1.0, recipe.roughness + 0.25))
        )
        bumps.append(Bump(streaks, distance=0.00003, strength=0.4 * brushed))
    tarnish = float(p["tarnish"])
    if tarnish > 0:
        color, rough, metallic = TARNISH[kind]
        # A film that darkens the whole surface as it thickens, uneven in soft clouds;
        # rust instead eats in with a harder edge until it covers nearly everything.
        soft = kind != "iron"
        start = 0.75 - 0.6 * tarnish
        patches = Field(
            "noise",
            size=0.05 if soft else 0.03,
            detail=4,
            roughness=0.6,
            seed=seed + 1,
            low=start - (0.35 if soft else 0.05),
            high=start + (0.15 if soft else 0.05),
        )
        creases = Field("cavity", distance=0.01, low=0.0, high=0.5)
        layers.append(
            Layer(
                (patches,),
                amount=min(1.0, 0.5 + tarnish),
                color=color,
                roughness=rough,
                metallic=metallic,
            )
        )
        layers.append(
            Layer((creases,), amount=tarnish, color=color, roughness=rough, metallic=metallic)
        )
        if kind == "copper" and tarnish > 0.4:
            green = Field("noise", size=0.02, detail=3, seed=seed + 2, low=0.6, high=0.7)
            layers.append(
                Layer(
                    (creases, green),
                    amount=(tarnish - 0.4) / 0.6,
                    color=_VERDIGRIS,
                    roughness=0.9,
                    metallic=0.0,
                )
            )
        if kind == "iron":
            bumps.append(
                Bump(Field("noise", size=0.003, detail=5, seed=seed + 3), 0.0004 * tarnish)
            )
    return recipe.with_layers(*layers, bumps=tuple(bumps))


METAL = GeneratorDef[Recipe](
    name="metal",
    summary=(
        "Conductive metal with measured reflectance, from brushed-dull to mirror-polished, "
        "clean or tarnished with age (iron rusts)."
    ),
    params={
        "metal": {
            "type": "string",
            "enum": sorted(METALS),
            "description": "Which metal; sets the physically measured reflectance colour.",
            "default": "silver",
        },
        "polish": _unit("From dull (0) to mirror-polished (1).", 0.8),
        "brushed": _texture("Fine parallel brush marks along x, 0 none to 1 strong."),
        "tarnish": _texture(
            "Age: 0 freshly cleaned; 0.3 a lived-in patina (clean copper looks pink, real "
            "copper is rarely clean); 1 heavily tarnished, or rusted for iron."
        ),
        "seed": _SEED,
    },
    run=_metal,
    examples=(
        {"family": "metal", "metal": "copper", "polish": 0.5},
        {"family": "metal", "metal": "copper", "polish": 0.6, "tarnish": 0.35},
        {"family": "metal", "metal": "iron", "polish": 0.3, "tarnish": 0.8},
    ),
)


# --------------------------------------------------------------------- plastic


def _plastic(p: Mapping[str, Any]) -> Recipe:
    recipe = Recipe(
        base_color=srgb_hex_to_linear(p["color"]),
        roughness=lerp(0.7, 0.05, float(p["gloss"])),
        ior=1.46,
    )
    texture = float(p["texture"])
    if texture <= 0:
        return recipe
    seed = int(p["seed"])
    relief = Field("noise", size=0.003, detail=3, roughness=0.6, seed=seed)
    blotch = Field("noise", size=0.25, detail=2, seed=seed + 1, low=0.3, high=0.8)
    return recipe.with_layers(
        Layer((blotch,), amount=0.25 * texture, color=scale(recipe.base_color, 0.9)),
        bumps=(Bump(relief, distance=0.0004 * texture, strength=min(1.0, 0.2 + texture)),),
    )


PLASTIC = GeneratorDef[Recipe](
    name="plastic",
    summary="Opaque dielectric such as plastic, paint or plaster, matte to glossy.",
    params={
        "color": _color("Surface colour.", "#d0d0d0"),
        "gloss": _unit("From matte (0) to glossy (1).", 0.3),
        "texture": _texture(
            "Surface relief and slight colour unevenness: 0 flawless moulded plastic, 0.3 "
            "painted wall, 1 rough plaster or stucco."
        ),
        "seed": _SEED,
    },
    run=_plastic,
    examples=(
        {"family": "plastic", "color": "#e8e4dc", "gloss": 0.1},
        {"family": "plastic", "color": "#e6e0d6", "gloss": 0.05, "texture": 0.6},
    ),
)


# ------------------------------------------------------------------------ wood


def _wood(p: Mapping[str, Any]) -> Recipe:
    seed = int(p["seed"])
    axis = p["grain_axis"]
    finish = float(p["finish"])
    base = srgb_hex_to_linear(p["color"])
    grain = srgb_hex_to_linear(p["grain_color"])
    along = {"x": (0.04, 1.0, 1.0), "y": (1.0, 0.04, 1.0), "z": (1.0, 1.0, 0.04)}[axis]
    rings = Field(
        "rings",
        size=float(p["ring_size"]),
        axis=axis,
        distortion=0.5 + 4.0 * float(p["figure"]),
        detail=2,
        seed=seed,
        low=0.3,
        high=0.95,
    )
    boards = Field("noise", size=0.4, detail=1, seed=seed + 1, low=0.3, high=0.7)
    pores = Field("noise", size=0.0008, detail=2, stretch=along, seed=seed + 2, low=0.55, high=0.8)
    recipe = Recipe(
        base_color=base,
        roughness=lerp(0.6, 0.35, finish),
        coat_weight=finish,
        coat_roughness=lerp(0.3, 0.04, finish),
    )
    open_pores = float(p["pores"])
    bumps = [Bump(rings, distance=0.00015, strength=0.3 * (1.0 - 0.7 * finish))]
    if open_pores > 0:
        bumps.append(Bump(pores, distance=0.0002, strength=open_pores * (1.0 - 0.6 * finish)))
    return recipe.with_layers(
        Layer((rings,), amount=0.7, color=grain, roughness=min(1.0, recipe.roughness + 0.05)),
        Layer((boards,), amount=0.35, color=mix(base, grain, 0.35)),
        Layer((pores,), amount=0.5 * open_pores, color=scale(grain, 0.7)),
        bumps=tuple(bumps),
    )


WOOD = GeneratorDef[Recipe](
    name="wood",
    summary=(
        "Wood with growth-ring grain running along an axis, raw and matte to lacquered. "
        "Oak, walnut, pine and birch are colours, ring spacing and figure."
    ),
    params={
        "color": _color("Colour of the light early wood.", "#b58658"),
        "grain_color": _color("Colour of the darker late-wood rings.", "#7a5230"),
        "ring_size": number(
            "Distance between growth rings in metres (pine ~0.006, oak ~0.003).",
            default=0.005,
            exclusive_minimum=0,
        ),
        "figure": _unit("How wavy the grain is: 0 straight, 1 wild burl-like.", 0.3),
        "grain_axis": {
            "enum": ["x", "y", "z"],
            "default": "x",
            "description": "The axis the grain runs along, in the object's own axes: the "
            "length of a board (x for a table top, z for a table leg).",
        },
        "pores": _unit("Open pores along the grain (oak, ash), 0 closed to 1 coarse.", 0.4),
        "finish": _unit("0 raw or oiled and matte, 1 lacquered and glossy.", 0.3),
        "seed": _SEED,
    },
    run=_wood,
    examples=(
        {"family": "wood"},
        {"family": "wood", "color": "#6b4a32", "grain_color": "#3e2a1c", "finish": 0.7},
    ),
)


# ----------------------------------------------------------------------- stone


def _stone(p: Mapping[str, Any]) -> Recipe:
    seed = int(p["seed"])
    kind = p["kind"]
    base = srgb_hex_to_linear(p["color"])
    accent = srgb_hex_to_linear(p["accent_color"])
    polish = float(p["polish"])
    recipe = Recipe(
        base_color=base,
        roughness=lerp(0.85, 0.08, polish),
        coat_weight=0.6 * polish,
        coat_roughness=0.05,
    )
    cloud = Field("noise", size=0.15, detail=4, seed=seed + 1, low=0.3, high=0.75)
    if kind == "marble":
        veins = Field(
            "ridges",
            size=0.12,
            detail=6,
            roughness=0.65,
            distortion=1.5,
            seed=seed,
            low=0.92,
            high=0.985,
        )
        return recipe.with_layers(
            Layer((cloud,), amount=0.3, color=mix(base, accent, 0.25)),
            Layer((veins,), amount=0.9, color=accent),
        )
    if kind == "granite":
        grains = Field("cells", size=0.006, seed=seed, low=0.35, high=0.15)
        light = Field("cells", size=0.009, seed=seed + 2, low=0.25, high=0.1)
        relief = (Bump(grains, 0.00005, 0.3 * (1.0 - polish)),) if polish < 1.0 else ()
        return recipe.with_layers(
            Layer((grains,), amount=0.9, color=accent),
            Layer((light,), amount=0.6, color=scale(base, 1.35)),
            bumps=relief,
        )
    # concrete: soft clouds and small pits
    pits = Field("cells", size=0.006, seed=seed, low=0.08, high=0.02)
    return recipe.with_layers(
        Layer((cloud,), amount=0.4, color=mix(base, accent, 0.5)),
        Layer((pits,), amount=0.7, color=scale(base, 0.6)),
        bumps=(
            Bump(pits, 0.0008, 0.6),
            Bump(Field("noise", size=0.002, detail=4, seed=seed), 0.0002, 0.5),
        ),
    )


STONE = GeneratorDef[Recipe](
    name="stone",
    summary="Stone: veined marble, speckled granite or poured concrete, rough to polished.",
    params={
        "kind": {
            "enum": ["marble", "granite", "concrete"],
            "default": "marble",
            "description": "marble: soft clouds crossed by thin veins; granite: a dense "
            "speckle of mineral grains; concrete: soft clouds and small pits.",
        },
        "color": _color("The main colour.", "#e9e6e0"),
        "accent_color": _color(
            "Veins (marble), dark grains (granite) or shading (concrete).", "#8f8a86"
        ),
        "polish": _unit("0 honed or raw, 1 polished to a mirror.", 0.5),
        "seed": _SEED,
    },
    run=_stone,
    examples=(
        {"family": "stone"},
        {"family": "stone", "kind": "granite", "color": "#9a948e", "accent_color": "#2c2926"},
        {"family": "stone", "kind": "concrete", "color": "#a9a59e", "polish": 0.0},
    ),
)


# ---------------------------------------------------------------------- fabric


def _fabric(p: Mapping[str, Any]) -> Recipe:
    seed = int(p["seed"])
    thread = float(p["thread_size"])
    base = srgb_hex_to_linear(p["color"])
    warp = Field("bands", size=thread, axis="x", distortion=0.3, seed=seed, low=0.2, high=0.8)
    weft = Field("bands", size=thread, axis="y", distortion=0.3, seed=seed, low=0.2, high=0.8)
    slub = Field(
        "noise",
        size=thread * 20,
        detail=2,
        stretch=(0.2, 1.0, 1.0),
        seed=seed + 1,
        low=0.35,
        high=0.75,
    )
    recipe = Recipe(
        base_color=base,
        roughness=0.85,
        sheen_weight=float(p["fuzz"]),
    )
    return recipe.with_layers(
        Layer((slub,), amount=0.4, color=scale(base, 0.82)),
        Layer((warp, weft), amount=0.3, color=scale(base, 0.75)),
        bumps=(Bump(warp, thread * 0.3, 0.5), Bump(weft, thread * 0.3, 0.5)),
    )


FABRIC = GeneratorDef[Recipe](
    name="fabric",
    summary="Woven cloth: linen, cotton, wool; a visible weave and a soft sheen.",
    params={
        "color": _color("Cloth colour.", "#c9bfae"),
        "thread_size": number(
            "Thread spacing in metres (fine cotton ~0.0005, linen ~0.001, burlap ~0.004).",
            default=0.001,
            exclusive_minimum=0,
        ),
        "fuzz": _unit("Soft sheen of loose fibres at grazing angles (velvet, wool).", 0.3),
        "seed": _SEED,
    },
    run=_fabric,
    examples=({"family": "fabric"}, {"family": "fabric", "color": "#3b4a63", "fuzz": 0.7}),
)


# ------------------------------------------------------------------------ peel


def _peel(p: Mapping[str, Any]) -> Recipe:
    seed = int(p["seed"])
    base = srgb_hex_to_linear(p["color"])
    blush = srgb_hex_to_linear(p["blush_color"])
    size = float(p["dimple_size"])
    dimples = Field("cells", size=size, seed=seed, low=0.0, high=0.5)
    patches = Field("noise", size=0.08, detail=1, seed=seed + 1, low=0.25, high=0.6)
    gloss = float(p["gloss"])
    recipe = Recipe(base_color=base, roughness=lerp(0.7, 0.2, gloss), coat_weight=0.3 * gloss)
    depth = float(p["dimples"])
    bumps = (Bump(dimples, distance=size * 0.25 * depth, strength=0.8),) if depth > 0 else ()
    return recipe.with_layers(
        Layer((patches,), amount=float(p["blush"]), color=blush),
        Layer((dimples,), amount=0.3, roughness=min(1.0, recipe.roughness + 0.15)),
        bumps=bumps,
    )


PEEL = GeneratorDef[Recipe](
    name="peel",
    summary=(
        "Fruit and vegetable skin: an orange's dimpled peel, an apple's waxy skin with a "
        "blush, a lemon, a tomato."
    ),
    params={
        "color": _color("Skin colour.", "#e5731b"),
        "blush_color": _color("Colour of patches and blush (an apple's red cheek).", "#c9561a"),
        "blush": _unit("How much of the skin carries the blush colour in patches.", 0.3),
        "dimples": _unit("Depth of the pores: 1 an orange, 0.3 a lemon, 0 an apple.", 0.8),
        "dimple_size": number(
            "Spacing of the pores in metres.", default=0.0015, exclusive_minimum=0
        ),
        "gloss": _unit("0 dull, 1 waxy and shiny.", 0.5),
        "seed": _SEED,
    },
    run=_peel,
    examples=(
        {"family": "peel"},
        {
            "family": "peel",
            "color": "#c8d45a",
            "blush_color": "#b3261e",
            "blush": 0.6,
            "dimples": 0.0,
            "gloss": 0.7,
        },
    ),
)


# ----------------------------------------------------------------------- water


def _water(p: Mapping[str, Any]) -> Recipe:
    seed = int(p["seed"])
    size = float(p["wave_size"])
    waves = float(p["waves"])
    swell = Field("noise", size=size, detail=3, roughness=0.5, stretch=(0.6, 1.0, 1.0), seed=seed)
    ripples = Field("noise", size=size * 0.15, detail=2, seed=seed + 1)
    recipe = Recipe(
        base_color=srgb_hex_to_linear(p["color"]),
        roughness=0.02 + 0.1 * float(p["murk"]),
        ior=1.333,
        transmission=1.0 - 0.6 * float(p["murk"]),
    )
    if waves <= 0:
        return recipe
    return recipe.with_layers(
        bumps=(
            Bump(swell, distance=size * 0.05 * waves, strength=min(1.0, 0.3 + waves)),
            Bump(ripples, distance=size * 0.01 * waves, strength=0.5),
        )
    )


WATER = GeneratorDef[Recipe](
    name="water",
    summary=(
        "A water surface for a pond, a lake, the sea or a glass of water: clear to murky, "
        "still to wavy. Put it on a plane at the water level."
    ),
    params={
        "color": _color("Colour the water takes in depth (white is perfectly clear).", "#d8ecee"),
        "murk": _unit("0 clear, 1 cloudy and opaque like a muddy pond.", 0.2),
        "waves": _unit("0 a still mirror, 1 a windy, broken surface.", 0.3),
        "wave_size": number(
            "Typical length of the waves in metres (a pond ~0.5, the sea ~5).",
            default=0.5,
            exclusive_minimum=0,
        ),
        "seed": _SEED,
    },
    run=_water,
    examples=(
        {"family": "water"},
        {"family": "water", "color": "#5e8a86", "murk": 0.6, "waves": 0.1},
    ),
)


ALL_FAMILIES: tuple[GeneratorDef[Recipe], ...] = (
    CERAMIC,
    GLASS,
    METAL,
    PLASTIC,
    WOOD,
    STONE,
    FABRIC,
    PEEL,
    WATER,
    PRINCIPLED,
)

__all__ = ["ALL_FAMILIES", "METALS", "TARNISH"]
