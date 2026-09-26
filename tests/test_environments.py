"""Environment families: consistent sun, sky, clouds and ground."""

from __future__ import annotations

import math
from typing import Any

import pytest

from jingyu.conventions import (
    kelvin_to_linear,
    sun_direction,
    sun_rotation_deg,
    sun_temperature_k,
)
from jingyu.environments import ENVIRONMENTS, EnvironmentRecipe
from jingyu.geometry.transform import apply_direction, rotation_xyz
from jingyu.scene import minimal_scene, validate_scene


def daylight(**params: Any) -> EnvironmentRecipe:
    spec = {"family": "daylight", **params}
    normalized = validate_scene(_scene_with(spec)).require_valid()
    return ENVIRONMENTS.run(normalized["world"]["environment"])


def _scene_with(environment: dict[str, Any]) -> dict[str, Any]:
    scene = minimal_scene()
    scene["lights"] = []
    scene["world"] = {"environment": environment}
    return scene


def test_clear_morning_has_a_warm_low_sun_and_a_sky() -> None:
    recipe = daylight(sun_elevation=8, sun_azimuth=200)
    assert recipe.sun is not None and recipe.sky is not None
    red, _, blue = recipe.sun.color
    assert red == 1.0 and blue < 0.4
    assert (recipe.sun.elevation, recipe.sun.azimuth) == (8, 200)
    assert recipe.ground is not None


def test_midday_sun_is_nearly_white() -> None:
    recipe = daylight(sun_elevation=70)
    assert recipe.sun is not None
    assert min(recipe.sun.color) > 0.75


def test_overcast_has_no_direct_sun_and_an_even_sky() -> None:
    recipe = daylight(cloud_cover=1)
    assert recipe.sun is None
    assert recipe.sky is None
    assert recipe.fill_strength > 0.1
    assert recipe.lit


def test_clouds_soften_and_dim_the_sun() -> None:
    clear, broken = daylight(), daylight(cloud_cover=0.5)
    assert clear.sun is not None and broken.sun is not None
    assert broken.sun.strength < clear.sun.strength
    assert broken.sun.angle > clear.sun.angle


def test_twilight_has_sky_but_no_sun() -> None:
    recipe = daylight(sun_elevation=-4)
    assert recipe.sun is None
    assert recipe.sky is not None and recipe.lit


def test_temperature_override_and_its_check() -> None:
    recipe = daylight(sun_temperature_k=6500)
    assert recipe.sun is not None
    assert recipe.sun.color == kelvin_to_linear(6500)
    result = validate_scene(_scene_with({"family": "daylight", "sun_temperature_k": 1000}))
    assert [(i.code, i.pointer) for i in result.errors] == [
        ("spec.invalid_parameter", "/world/environment/sun_temperature_k")
    ]


def test_uniform_environment_and_unlit_warning() -> None:
    assert validate_scene(_scene_with({"family": "uniform"})).issues == ()
    dark = validate_scene(_scene_with({"family": "uniform", "strength": 0}))
    assert [w.code for w in dark.warnings] == ["spec.scene_unlit"]
    no_sky = validate_scene(
        _scene_with({"family": "daylight", "sun_strength": 0, "sky_strength": 0})
    )
    assert [w.code for w in no_sky.warnings] == ["spec.scene_unlit"]


def test_unknown_family_is_a_schema_violation() -> None:
    result = validate_scene(_scene_with({"family": "moon"}))
    assert result.errors and result.errors[0].pointer.startswith("/world/environment")


def test_sun_rotation_points_the_lamp_away_from_the_sun() -> None:
    for elevation, azimuth in ((10, 0), (35, 135), (60, -90), (89, 10)):
        towards = sun_direction(elevation, azimuth)
        light = apply_direction(rotation_xyz(sun_rotation_deg(elevation, azimuth)), (0, 0, -1))
        assert all(math.isclose(a, -b, abs_tol=1e-9) for a, b in zip(light, towards, strict=True))


@pytest.mark.parametrize(("low", "high"), [(0, 5), (5, 20), (20, 60)])
def test_sun_warms_as_it_sinks(low: float, high: float) -> None:
    assert sun_temperature_k(low) < sun_temperature_k(high)


def test_mist_fills_the_air_and_zero_leaves_it_clear() -> None:
    from jingyu.environments.families import MIST_DENSITY

    clear = ENVIRONMENTS.run({"family": "daylight"})
    assert clear.air is None
    misty = ENVIRONMENTS.run({"family": "daylight", "mist": 0.5, "mist_color": "#ffe0c0"})
    assert misty.air is not None
    assert misty.air.density == pytest.approx(MIST_DENSITY * 0.25)
    assert misty.air.color[0] > misty.air.color[2]
    studio = ENVIRONMENTS.run({"family": "uniform", "mist": 0.2, "mist_glow": 0.0})
    assert studio.air is not None and studio.air.anisotropy == 0.0
