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
    }


def test_lerp_hits_both_ends() -> None:
    assert lerp(0.85, 0.3, 0.0) == 0.85
    assert lerp(0.85, 0.3, 1.0) == pytest.approx(0.3)
    assert lerp(0.0, 1.0, 0.25) == 0.25
