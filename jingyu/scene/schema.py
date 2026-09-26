"""The scene description JSON Schema, assembled from the generator registries.

The geometry and material sections are generated from the registered
operators and families, so adding a generator automatically extends the schema,
the defaults and the catalogue.  ``schemas/scene.v1.schema.json`` at the
repository root is an exported copy kept in sync by a test.
"""

from __future__ import annotations

import copy
from functools import lru_cache
from typing import Any

from ..conventions import COLOR_PATTERN, ID_PATTERN
from ..environments import ENVIRONMENTS
from ..generator import discriminated_union
from ..geometry import GEOMETRY
from ..materials import MATERIALS
from .warnings import ACCEPTABLE_WARNINGS

SCENE_SCHEMA = "jingyu.scene.v1"
DIALECT = "https://json-schema.org/draft/2020-12/schema"

_ID = {"type": "string", "pattern": ID_PATTERN, "description": "Lowercase ASCII slug."}
_COLOR = {"type": "string", "pattern": COLOR_PATTERN, "description": "sRGB hex #rrggbb."}


def _vec3(description: str, default: list[float] | None = None) -> dict[str, Any]:
    schema: dict[str, Any] = {
        "type": "array",
        "description": description,
        "items": {"type": "number"},
        "minItems": 3,
        "maxItems": 3,
    }
    if default is not None:
        schema["default"] = default
    return schema


def _ref(name: str, **extra: Any) -> dict[str, Any]:
    return {"$ref": f"#/$defs/{name}", **extra}


_LOCATION = _vec3("Position in metres.")
_LOOK_AT = _vec3("Point in metres to aim at; the up direction stays +Z.")
_ROTATION = _vec3("XYZ Euler rotation in degrees.")
_AIM = {"oneOf": [{"required": ["look_at"]}, {"required": ["rotation"]}]}
_PARENT = {
    "$ref": "#/$defs/id",
    "description": (
        "Id of a group or an object this entry belongs to. Its location, rotation and "
        "aim are then relative to that parent, and it moves with it: fruit parented to "
        "its bowl follows the bowl wherever it goes."
    ),
}
_TEMPERATURE = {
    "type": "number",
    "minimum": 1667,
    "maximum": 25000,
    "description": (
        "Colour temperature in kelvin (e.g. 1900 candle, 2700 warm bulb, 5500 noon sun, "
        "7000 overcast); when given it replaces color."
    ),
}


def _light_branch(kind: str, properties: dict[str, Any], aimed: bool) -> dict[str, Any]:
    props: dict[str, Any] = {
        "id": _ref("id"),
        "kind": {"const": kind},
        "location": copy.deepcopy(_LOCATION),
        "color": _ref("color", default="#ffffff"),
        "temperature_k": copy.deepcopy(_TEMPERATURE),
        "parent": copy.deepcopy(_PARENT),
        **properties,
    }
    branch: dict[str, Any] = {
        "type": "object",
        "additionalProperties": False,
        "required": ["id", "kind", "location"],
        "properties": props,
    }
    if aimed:
        props["look_at"] = copy.deepcopy(_LOOK_AT)
        props["rotation"] = copy.deepcopy(_ROTATION)
        branch.update(copy.deepcopy(_AIM))
    return branch


def _number(description: str, default: float, **bounds: float) -> dict[str, Any]:
    schema: dict[str, Any] = {"type": "number", "description": description, "default": default}
    for key, value in bounds.items():
        schema[{"minimum": "minimum", "maximum": "maximum", "gt": "exclusiveMinimum"}[key]] = value
    return schema


def _light_schema() -> dict[str, Any]:
    power = _number("Emitted power in watts.", 1000.0, minimum=0.0)
    radius = _number("Size of the emitting sphere in metres; larger is softer.", 0.1, minimum=0.0)
    branches = {
        "point": _light_branch("point", {"power_w": power, "radius": radius}, aimed=False),
        "spot": _light_branch(
            "spot",
            {
                "power_w": power,
                "radius": radius,
                "spot_size": _number("Cone angle in degrees.", 45.0, gt=0.0, maximum=180.0),
                "blend": _number(
                    "Edge softness of the cone, 0 to 1.", 0.15, minimum=0.0, maximum=1.0
                ),
            },
            aimed=True,
        ),
        "area": _light_branch(
            "area",
            {
                "power_w": _number("Emitted power in watts.", 500.0, minimum=0.0),
                "size": _number(
                    "Width of the emitter in metres (and its depth unless size_y is given).",
                    1.0,
                    gt=0.0,
                ),
                "size_y": {
                    "type": "number",
                    "exclusiveMinimum": 0,
                    "description": (
                        "Depth of a rectangular emitter in metres, e.g. a window-shaped "
                        "soft light; omitted means square."
                    ),
                },
            },
            aimed=True,
        ),
        "sun": _light_branch(
            "sun",
            {
                "strength": _number(
                    "Irradiance in W/m^2 (Blender's sun strength).", 3.0, minimum=0.0
                ),
                "angle": _number(
                    "Apparent angular diameter in degrees; larger is softer.",
                    0.5,
                    minimum=0.0,
                    maximum=180.0,
                ),
            },
            aimed=True,
        ),
    }
    return discriminated_union("kind", branches, "Which kind of light.")


def _camera_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["id", "location"],
        **copy.deepcopy(_AIM),
        "properties": {
            "id": _ref("id"),
            "parent": copy.deepcopy(_PARENT),
            "location": copy.deepcopy(_LOCATION),
            "look_at": copy.deepcopy(_LOOK_AT),
            "rotation": copy.deepcopy(_ROTATION),
            "projection": {
                "enum": ["perspective", "orthographic"],
                "default": "perspective",
                "description": "Perspective or orthographic (parallel) projection.",
            },
            "lens_mm": _number("Focal length in millimetres (perspective).", 50.0, gt=0.0),
            "sensor_width_mm": _number("Sensor width in millimetres.", 36.0, gt=0.0),
            "ortho_scale": _number("Visible width in metres (orthographic).", 4.0, gt=0.0),
            "clip_start": _number("Nearest visible distance in metres.", 0.05, gt=0.0),
            "clip_end": _number("Farthest visible distance in metres.", 500.0, gt=0.0),
            "shift": {
                "type": "array",
                "description": (
                    "Lens shift [x, y] as a fraction of the frame; keeps verticals straight."
                ),
                "items": {"type": "number", "minimum": -2, "maximum": 2},
                "minItems": 2,
                "maxItems": 2,
                "default": [0.0, 0.0],
            },
        },
    }


def _accept_warnings(description: str) -> dict[str, Any]:
    return {
        "type": "array",
        "uniqueItems": True,
        "items": {"enum": list(ACCEPTABLE_WARNINGS)},
        "description": description,
    }


def _object_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["id", "geometry"],
        "properties": {
            "id": _ref("id"),
            "geometry": _ref("geometry"),
            "material": _ref(
                "id", description="Id of an entry in materials; omitted means neutral grey."
            ),
            "parent": copy.deepcopy(_PARENT),
            "accept_warnings": _accept_warnings(
                "Warning codes that are intended for this object only, e.g. "
                "frame.not_visible for a wall or ceiling that closes the room out of sight."
            ),
            "attached_to": {
                "type": "array",
                "uniqueItems": True,
                "items": _ref("id"),
                "description": (
                    "Ids of objects this one is fixed to on purpose: walls meeting at a "
                    "corner, a lamp hanging from a ceiling, a nail in a wall. Sinking into "
                    "them and hovering are then not reported."
                ),
            },
            "rest_on": _ref(
                "id",
                description=(
                    "Id of an object to rest on. The object keeps its x and y, and its "
                    "height (location z) is computed so its lowest points touch the top "
                    "surface of that object: a table top, the inside of a bowl."
                ),
            ),
            "location": _vec3("Position of the object's origin in metres.", [0.0, 0.0, 0.0]),
            "rotation": _vec3("XYZ Euler rotation in degrees.", [0.0, 0.0, 0.0]),
            "scale": {
                "type": "array",
                "description": "Per-axis scale factors.",
                "items": {"type": "number", "exclusiveMinimum": 0},
                "minItems": 3,
                "maxItems": 3,
                "default": [1.0, 1.0, 1.0],
            },
            "visible": {
                "type": "boolean",
                "default": True,
                "description": "false keeps the object in the scene but out of the render.",
            },
        },
    }


def _group_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["id"],
        "description": (
            "An invisible transform that objects, lights, cameras and other groups can "
            "belong to (parent); moving the group moves everything in it."
        ),
        "properties": {
            "id": _ref("id"),
            "parent": copy.deepcopy(_PARENT),
            "accept_warnings": _accept_warnings(
                "Warning codes that are intended for every object in this group, at any "
                "depth, e.g. frame.not_visible for the hidden legs and underside of a table."
            ),
            "location": _vec3("Position of the group's origin in metres.", [0.0, 0.0, 0.0]),
            "rotation": _vec3("XYZ Euler rotation in degrees.", [0.0, 0.0, 0.0]),
            "scale": {
                "type": "number",
                "exclusiveMinimum": 0,
                "description": "Uniform scale factor applied to everything in the group.",
                "default": 1.0,
            },
        },
    }


def _render_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["camera"],
        "properties": {
            "camera": _ref("id", description="Id of the camera to render through."),
            "resolution": {
                "type": "array",
                "description": "Image width and height in pixels.",
                "items": {"type": "integer", "minimum": 16, "maximum": 16384},
                "minItems": 2,
                "maxItems": 2,
                "default": [1280, 720],
            },
            "engine": {
                "enum": ["cycles", "eevee", "workbench"],
                "default": "cycles",
                "description": (
                    "cycles: path traced; eevee: real-time (needs a GPU); "
                    "workbench: preview shading."
                ),
            },
            "samples": {
                "type": "integer",
                "minimum": 1,
                "maximum": 65536,
                "default": 64,
                "description": "Samples per pixel.",
            },
            "denoise": {"type": "boolean", "default": True, "description": "Denoise (cycles)."},
            "device": {
                "enum": ["auto", "cpu", "gpu"],
                "default": "auto",
                "description": "Cycles device; auto uses a GPU when one is usable.",
            },
            "seed": {
                "type": "integer",
                "minimum": 0,
                "maximum": 2147483647,
                "default": 0,
                "description": "Sampling seed; fixed for reproducibility.",
            },
            "view_transform": {
                "enum": ["standard", "agx", "filmic", "khronos_pbr_neutral"],
                "default": "agx",
                "description": "Tone mapping from scene light to display.",
            },
            "exposure": _number("Exposure in stops.", 0.0, minimum=-10.0, maximum=10.0),
            "transparent_background": {
                "type": "boolean",
                "default": False,
                "description": "Render the world background as transparent.",
            },
        },
    }


@lru_cache(maxsize=1)
def _cached_schema() -> dict[str, Any]:
    return {
        "$schema": DIALECT,
        "$id": "urn:jingyu:schema:scene:v1",
        "title": "Jingyu scene description v1",
        "description": (
            "A declarative scene: generator calls for geometry and materials, plus "
            "lights, cameras and render settings. Lengths in metres, angles in "
            "degrees, Z up, colours as sRGB hex. Objects stand on their origin."
        ),
        "type": "object",
        "additionalProperties": False,
        "required": ["schema", "id", "cameras", "render"],
        "properties": {
            "schema": {"const": SCENE_SCHEMA, "description": "Format and version marker."},
            "id": _ref("id"),
            "title": {"type": "string", "maxLength": 200, "description": "Human title."},
            "intent": {
                "type": "string",
                "maxLength": 500,
                "description": (
                    "One sentence: what the picture must make a viewer see or feel. "
                    "Reviews judge the render against it; never affects rendering."
                ),
            },
            "notes": {
                "type": "string",
                "maxLength": 8000,
                "description": "Free-form context; never affects rendering.",
            },
            "accept_warnings": {
                "type": "array",
                "uniqueItems": True,
                "default": [],
                "items": {"enum": list(ACCEPTABLE_WARNINGS)},
                "description": (
                    "Warning codes that describe the picture as intended, e.g. "
                    "frame.underexposed for a deliberately dark image; they are no longer "
                    "reported. Prefer fixing the scene."
                ),
            },
            "world": {
                "type": "object",
                "additionalProperties": False,
                "default": {},
                "description": (
                    "The light around the scene. Without environment it is a uniform "
                    "color at strength; with one, the environment family decides and "
                    "color and strength are ignored."
                ),
                "properties": {
                    "color": _ref("color", default="#404040"),
                    "strength": _number("Background light multiplier.", 1.0, minimum=0.0),
                    "environment": _ref("environment"),
                },
            },
            "groups": {"type": "array", "items": _ref("group"), "maxItems": 4096, "default": []},
            "materials": {
                "type": "array",
                "items": _ref("material"),
                "maxItems": 4096,
                "default": [],
            },
            "objects": {
                "type": "array",
                "items": _ref("object"),
                "maxItems": 65536,
                "default": [],
            },
            "lights": {"type": "array", "items": _ref("light"), "maxItems": 1024, "default": []},
            "cameras": {"type": "array", "items": _ref("camera"), "minItems": 1, "maxItems": 256},
            "render": _ref("render"),
        },
        "$defs": {
            "id": _ID,
            "color": _COLOR,
            "geometry": GEOMETRY.union_schema(),
            "material": MATERIALS.union_schema(
                extra_properties={"id": _ID}, extra_required=("id",)
            ),
            "group": _group_schema(),
            "object": _object_schema(),
            "light": _light_schema(),
            "camera": _camera_schema(),
            "render": _render_schema(),
            "environment": ENVIRONMENTS.union_schema(),
        },
    }


def scene_schema() -> dict[str, Any]:
    """Return a fresh copy of the scene JSON Schema."""

    return copy.deepcopy(_cached_schema())


__all__ = ["DIALECT", "SCENE_SCHEMA", "scene_schema"]
