"""The built-in environment families.

An environment family names a kind of surrounding light (a flat studio sweep, a
day outdoors) and exposes the few knobs a person would describe it with: where
the sun is, how cloudy, how hazy.  The family derives everything that must agree
with those knobs: sky colour, sun colour temperature, shadow softness, how much
light the clouds let through.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from ..conventions import (
    COLOR_PATTERN,
    MIN_TEMPERATURE_K,
    kelvin_to_linear,
    srgb_hex_to_linear,
    sun_temperature_k,
)
from ..generator import GeneratorDef, ParamProblem, number
from .recipe import Air, EnvironmentRecipe, Sky, Sun

#: The physical sky is far brighter than jingyu's sun strengths (a few W/m^2), so
#: it is scaled to fill shadows at roughly a third of a clear midday sun
#: (measured in Blender 5.0: a white plane under the unscaled sky at 35 degrees
#: has about 7x the radiance it has under a strength-3 sun).
SKY_STRENGTH_SCALE = 0.04

#: Radiance of a white diffuse surface lit by the scaled clear sky alone, roughly
#: (measured in Blender 5.0 between 15 and 60 degrees of sun elevation).
CLEAR_SKY_RADIANCE = 0.14

#: Colour temperature of light from a fully overcast sky.
OVERCAST_TEMPERATURE_K = 6500.0


#: Scattering per metre at mist 1: thick fog, about 1.5 m of visibility.
MIST_DENSITY = 2.0

_AIR_PARAMS: dict[str, Any] = {
    "mist": number(
        "Dust or mist in the air, 0 clear to 1 thick fog. 0.1-0.3 indoors shows sunbeams "
        "through a window as shafts and softens the far wall; outdoors it hazes the distance.",
        default=0.0,
        minimum=0.0,
        maximum=1.0,
    ),
    "mist_color": {
        "type": "string",
        "pattern": COLOR_PATTERN,
        "description": "Colour of the particles in the air (sRGB hex): white mist, warm dust.",
        "default": "#ffffff",
    },
    "mist_glow": number(
        "How much the air glows looking toward the light, 0 evenly to 0.9 a bright halo "
        "with crisp shafts.",
        default=0.4,
        minimum=0.0,
        maximum=0.9,
    ),
}


def _air(p: Mapping[str, Any]) -> Air | None:
    mist = float(p["mist"])
    if mist <= 0.0:
        return None
    return Air(
        density=MIST_DENSITY * mist * mist,
        color=srgb_hex_to_linear(p["mist_color"]),
        anisotropy=float(p["mist_glow"]),
    )


def _uniform(p: Mapping[str, Any]) -> EnvironmentRecipe:
    return EnvironmentRecipe(
        fill=srgb_hex_to_linear(p["color"]), fill_strength=p["strength"], air=_air(p)
    )


UNIFORM = GeneratorDef[EnvironmentRecipe](
    name="uniform",
    summary="The same colour from every direction: a neutral studio or a plain backdrop.",
    params={
        "color": {
            "type": "string",
            "pattern": COLOR_PATTERN,
            "description": "Colour of the surrounding light and background. sRGB hex #rrggbb.",
            "default": "#404040",
        },
        "strength": number("Brightness multiplier.", default=1.0, minimum=0.0),
        **_AIR_PARAMS,
    },
    run=_uniform,
    examples=({"family": "uniform", "color": "#d8dde3", "strength": 0.6},),
)


def _smoothstep(edge0: float, edge1: float, x: float) -> float:
    t = min(max((x - edge0) / (edge1 - edge0), 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)


def _daylight(p: Mapping[str, Any]) -> EnvironmentRecipe:
    elevation = float(p["sun_elevation"])
    azimuth = float(p["sun_azimuth"])
    cloud = float(p["cloud_cover"])
    clear = 1.0 - cloud

    # Direct sun fades as clouds thicken and as the sun sinks through the horizon.
    horizon = _smoothstep(-0.5, 3.0, elevation)
    sun_strength = float(p["sun_strength"]) * clear**1.5 * horizon
    temperature = float(p["sun_temperature_k"])
    if temperature == 0.0:
        temperature = sun_temperature_k(max(elevation, 0.0))
        temperature += (OVERCAST_TEMPERATURE_K - temperature) * cloud
    sun = None
    if sun_strength > 0.0:
        sun = Sun(
            elevation=elevation,
            azimuth=azimuth,
            strength=sun_strength,
            angle=float(p["sun_angle"]) + 20.0 * cloud,
            color=kelvin_to_linear(temperature),
        )

    sky_strength = float(p["sky_strength"])
    sky = Sky(elevation, azimuth, float(p["haze"]), SKY_STRENGTH_SCALE * sky_strength * clear)
    # An overcast sky is a bright, even grey whose level follows the hidden sun.
    overcast_level = 0.45 * max(math.sin(math.radians(elevation)) + 0.18, 0.02)
    fill_strength = overcast_level * sky_strength * cloud
    # The ground outside reflects what falls on it: a diffuse surface's radiance is
    # its albedo times the irradiance over pi.
    sun_irradiance = sun_strength * max(math.sin(math.radians(elevation)), 0.0)
    sky_radiance = CLEAR_SKY_RADIANCE * sky_strength * clear + fill_strength
    light = [sun_irradiance / math.pi * c + sky_radiance for c in (sun.color if sun else (1, 1, 1))]
    albedo = srgb_hex_to_linear(p["ground_color"])
    return EnvironmentRecipe(
        fill=kelvin_to_linear(OVERCAST_TEMPERATURE_K),
        fill_strength=fill_strength,
        sky=sky if sky.strength > 0.0 else None,
        sun=sun,
        ground=(albedo[0] * light[0], albedo[1] * light[1], albedo[2] * light[2]),
        air=_air(p),
    )


def _check_daylight(p: Mapping[str, Any]) -> list[ParamProblem]:
    if 0.0 < float(p["sun_temperature_k"]) < MIN_TEMPERATURE_K:
        return [("sun_temperature_k", "must be 0 (automatic) or at least 1667 K")]
    return []


DAYLIGHT = GeneratorDef[EnvironmentRecipe](
    name="daylight",
    summary=(
        "Outdoor daylight from dawn to dusk, clear to overcast: a physical sky plus a sun "
        "whose direction, colour and softness agree with each other. Light through "
        "windows and doors comes from here too."
    ),
    params={
        "sun_elevation": number(
            "Sun height above the horizon in degrees. 2-10 early morning or late "
            "afternoon (long warm light), 25-45 mid-morning, 60+ midday; 0 to -6 is "
            "twilight with no direct sun.",
            default=35.0,
            minimum=-10.0,
            maximum=90.0,
        ),
        "sun_azimuth": number(
            "Direction toward the sun in degrees, counter-clockwise from +X seen from "
            "above (0 = +X, 90 = +Y, 180 = -X, 270 = -Y). Shadows point the other way.",
            default=135.0,
            minimum=-360.0,
            maximum=360.0,
        ),
        "cloud_cover": number(
            "0 clear sky with crisp shadows, 0.5 broken clouds, 1 overcast with no "
            "direct sun and soft, even light.",
            default=0.0,
            minimum=0.0,
            maximum=1.0,
        ),
        "haze": number(
            "Aerosols in the air, 0 crystal clear to 10 very hazy; more haze gives a paler sky.",
            default=1.0,
            minimum=0.0,
            maximum=10.0,
        ),
        "sun_strength": number(
            "Irradiance of the clear-sky sun in W/m^2 (Blender sun strength); clouds "
            "and a low sun reduce it automatically.",
            default=3.0,
            minimum=0.0,
            maximum=100.0,
        ),
        "sun_angle": number(
            "Apparent diameter of the clear-sky sun in degrees; clouds widen it.",
            default=0.53,
            minimum=0.0,
            maximum=45.0,
        ),
        "sun_temperature_k": number(
            "Sun colour temperature in kelvin; 0 means automatic, following the "
            "elevation (about 2600 K at 5 degrees, 5600 K at 60) and the clouds.",
            default=0.0,
            minimum=0.0,
            maximum=25000.0,
        ),
        "ground_color": {
            "type": "string",
            "pattern": COLOR_PATTERN,
            "description": (
                "Colour of the land below the horizon, seen through windows and "
                "reflected in glossy things (grass, earth, pavement, snow). sRGB hex #rrggbb."
            ),
            "default": "#6e675c",
        },
        "sky_strength": number(
            "Multiplier for light from the sky (blue fill in shadows).",
            default=1.0,
            minimum=0.0,
            maximum=100.0,
        ),
        **_AIR_PARAMS,
    },
    run=_daylight,
    check=_check_daylight,
    examples=(
        {"family": "daylight", "sun_elevation": 12, "sun_azimuth": 200},
        {"family": "daylight", "sun_elevation": 50, "cloud_cover": 1},
        {"family": "daylight", "sun_elevation": 20, "sun_azimuth": 160, "mist": 0.25},
    ),
)

ALL_FAMILIES = (UNIFORM, DAYLIGHT)
