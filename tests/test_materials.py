from __future__ import annotations

from itertools import pairwise
from typing import Any

import pytest

from jingyu.conventions import srgb_hex_to_linear, srgb_to_linear
from jingyu.generator import GeneratorDef
from jingyu.materials import MATERIALS, METALS, Recipe
from jingyu.materials.recipe import lerp

UNIT_FIELDS = ("metallic", "roughness", "transmission", "coat_weight", "coat_roughness", "alpha")


def _defaults(definition: GeneratorDef[Any]) -> dict[str, Any]:
    return {k: s["default"] for k, s in definition.params.items() if "default" in s}


def _cases() -> list[Any]:
    cases = []
    for definition in MATERIALS:
        cases.append(pytest.param(definition.name, {}, id=f"{definition.name}-defaults"))
        for index, example in enumerate(definition.examples):
            params = {k: v for k, v in example.items() if k != "family"}
            cases.append(pytest.param(definition.name, params, id=f"{definition.name}-ex{index}"))
    return cases


def _recipe(family: str, **params: Any) -> Recipe:
    definition = MATERIALS.get(family)
    return definition.run({**_defaults(definition), **params})


def _assert_in_range(recipe: Recipe) -> None:
    for name in UNIT_FIELDS:
        assert 0.0 <= getattr(recipe, name) <= 1.0, name
    assert 1.0 <= recipe.ior <= 4.0
    assert recipe.emission_strength >= 0.0
    for channel in (*recipe.base_color, *recipe.emission_color):
        assert 0.0 <= channel <= 1.0


def test_every_family_has_an_example() -> None:
    assert all(definition.examples for definition in MATERIALS)


@pytest.mark.parametrize(("family", "params"), _cases())
def test_families_produce_in_range_recipes(family: str, params: dict[str, Any]) -> None:
    recipe = _recipe(family, **params)
    assert isinstance(recipe, Recipe)
    _assert_in_range(recipe)


def test_every_family_parameter_is_optional() -> None:
    # A material is always writable as just its family name.
    assert all(definition.required == () for definition in MATERIALS)


def test_ceramic_gloss_lowers_roughness_monotonically() -> None:
    roughness = [_recipe("ceramic", gloss=g / 10).roughness for g in range(11)]
    assert all(b < a for a, b in pairwise(roughness))
    coat = [_recipe("ceramic", gloss=g / 10).coat_weight for g in range(11)]
    assert coat == pytest.approx([g / 10 for g in range(11)])


def test_plastic_gloss_lowers_roughness_and_metal_polish_too() -> None:
    plastic = [_recipe("plastic", gloss=g / 4).roughness for g in range(5)]
    metal = [_recipe("metal", polish=p / 4).roughness for p in range(5)]
    for series in (plastic, metal):
        assert all(b < a for a, b in pairwise(series))


def test_glass_is_fully_transmissive_and_frost_roughens_it() -> None:
    clear = _recipe("glass")
    assert clear.transmission == 1.0
    assert clear.refractive
    assert clear.base_color == (1.0, 1.0, 1.0)
    frosted = _recipe("glass", frost=1.0)
    assert frosted.transmission == 1.0
    assert frosted.roughness > clear.roughness
    assert _recipe("glass", ior=1.33).ior == 1.33


@pytest.mark.parametrize("metal", sorted(METALS))
def test_every_metal_enum_value_is_a_conductor(metal: str) -> None:
    recipe = _recipe("metal", metal=metal)
    assert recipe.metallic == 1.0
    assert recipe.base_color == METALS[metal]
    assert not recipe.refractive
    _assert_in_range(recipe)


def test_metal_enum_matches_the_measured_table() -> None:
    assert MATERIALS.get("metal").params["metal"]["enum"] == sorted(METALS)


@pytest.mark.parametrize("family", ["ceramic", "plastic"])
def test_recipe_colours_are_scene_linear(family: str) -> None:
    assert _recipe(family, color="#ffffff").base_color == (1.0, 1.0, 1.0)
    assert _recipe(family, color="#000000").base_color == (0.0, 0.0, 0.0)
    grey = srgb_to_linear(128 / 255)
    assert _recipe(family, color="#808080").base_color == pytest.approx((grey, grey, grey))


def test_principled_passes_its_parameters_through() -> None:
    recipe = _recipe(
        "principled",
        base_color="#ff0000",
        metallic=0.25,
        roughness=0.75,
        emission_color="#00ff00",
        emission_strength=3.0,
        alpha=0.5,
    )
    assert recipe.base_color == (1.0, 0.0, 0.0)
    assert recipe.emission_color == srgb_hex_to_linear("#00ff00")
    assert (recipe.metallic, recipe.roughness, recipe.emission_strength, recipe.alpha) == (
        0.25,
        0.75,
        3.0,
        0.5,
    )


@pytest.mark.parametrize(
    "fields",
    [
        {"roughness": 1.5},
        {"metallic": -0.1},
        {"alpha": 2.0},
        {"ior": 0.9},
        {"ior": 4.5},
        {"emission_strength": -1.0},
    ],
)
def test_recipe_rejects_out_of_range_channels(fields: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        Recipe(**fields)  # type: ignore[arg-type]


def test_recipe_to_dict_uses_lists_for_colours() -> None:
    data = Recipe().to_dict()
    assert data["base_color"] == [0.5, 0.5, 0.5]
    assert data["emission_color"] == [1.0, 1.0, 1.0]
    assert set(data) == {
        "base_color",
        "metallic",
        "roughness",
        "ior",
        "transmission",
        "coat_weight",
        "coat_roughness",
        "emission_color",
        "emission_strength",
        "alpha",
        "thin",
        "sheen_weight",
        "layers",
        "bumps",
    }


def test_lerp_hits_both_ends() -> None:
    assert lerp(0.85, 0.3, 0.0) == 0.85
    assert lerp(0.85, 0.3, 1.0) == pytest.approx(0.3)
    assert lerp(0.0, 1.0, 0.25) == 0.25


# ------------------------------------------------------------------- texture


def test_fields_layers_and_bumps_reject_bad_values() -> None:
    from jingyu.materials.recipe import Bump, Field, Layer

    with pytest.raises(ValueError, match="kind"):
        Field("plaid")
    with pytest.raises(ValueError, match="axis"):
        Field("rings", axis="w")
    with pytest.raises(ValueError, match="differ"):
        Field("noise", low=0.5, high=0.5)
    with pytest.raises(ValueError, match="mask"):
        Layer(())
    with pytest.raises(ValueError, match="amount"):
        Layer((Field("noise"),), amount=1.5)
    with pytest.raises(ValueError, match="bump"):
        Bump(Field("noise"), distance=0.0)


@pytest.mark.parametrize(
    ("family", "params"),
    [
        ("wood", {}),
        ("stone", {"kind": "marble"}),
        ("stone", {"kind": "granite"}),
        ("stone", {"kind": "concrete"}),
        ("fabric", {}),
        ("peel", {}),
        ("ceramic", {"speckle": 0.5, "texture": 0.5}),
        ("plastic", {"texture": 0.5}),
        ("metal", {"metal": "copper", "tarnish": 0.6, "brushed": 0.5}),
        ("metal", {"metal": "iron", "tarnish": 0.8}),
    ],
)
def test_textured_families_paint_layers(family: str, params: dict[str, Any]) -> None:
    recipe = _recipe(family, **params)
    assert recipe.textured
    _assert_in_range(recipe)


def test_texture_knobs_at_zero_give_plain_surfaces() -> None:
    assert not _recipe("ceramic").textured
    assert not _recipe("plastic").textured
    assert not _recipe("metal").textured
    assert not _recipe("peel", dimples=0.0, blush=0.0).bumps


def test_rust_takes_the_colour_and_matte_of_oxide() -> None:
    rust = _recipe("metal", metal="iron", tarnish=0.8)
    oxide = rust.layers[0]
    assert oxide.metallic == 0.0 and oxide.roughness == pytest.approx(0.9)
    assert oxide.color is not None and oxide.color[0] > 2 * oxide.color[2]
    copper = _recipe("metal", metal="copper", tarnish=0.8)
    assert any(layer.metallic == 0.0 for layer in copper.layers)  # verdigris in the creases


def test_weathering_adds_one_effect_per_knob() -> None:
    from jingyu.materials.weathering import material_recipe, weather

    plain = _recipe("plastic")
    assert weather(plain, None) is plain
    assert weather(plain, {"dust": 0.0}).layers == ()
    worn = weather(plain, {"wear": 0.5, "wear_color": "#b08a5e"})
    assert [f.kind for f in worn.layers[0].mask] == ["edges", "noise"]
    assert worn.layers[0].color == pytest.approx(srgb_hex_to_linear("#b08a5e"))
    dusty = weather(plain, {"dust": 0.5})
    assert dusty.layers[0].mask[0].kind == "facing_up" and dusty.bumps
    grimy = weather(plain, {"grime": 0.5})
    assert {layer.mask[0].kind for layer in grimy.layers} == {"cavity", "low"}
    assert weather(plain, {"stains": 0.5}).layers[0].mask[0].kind == "cracks"
    assert weather(plain, {"fingerprints": 0.5}).layers[0].color is None
    entry = {"id": "paint", "family": "plastic", "weathering": {"dust": 0.3, "seed": 2}}
    assert material_recipe(entry).layers == weather(plain, {"dust": 0.3, "seed": 2}).layers
    assert material_recipe(entry).layers != weather(plain, {"dust": 0.3, "seed": 3}).layers


def test_weathering_is_validated_on_the_material_entry() -> None:
    from jingyu.scene import minimal_scene, validate_scene

    scene = minimal_scene()
    scene["materials"][0]["weathering"] = {"dust": 0.2, "wear": 0.1}
    result = validate_scene(scene)
    assert result.valid
    assert result.normalized["materials"][0]["weathering"]["grime"] == 0.0
    scene["materials"][0]["weathering"] = {"rust": 0.5}
    assert not validate_scene(scene).valid
