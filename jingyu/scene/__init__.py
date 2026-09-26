"""The scene description: schema, validation and normalisation (host side).

This package depends on ``jsonschema`` and is never imported by the Blender
worker, which receives scenes that were already validated and normalised here.
"""

from __future__ import annotations

from .examples import minimal_scene
from .schema import SCENE_SCHEMA, scene_schema
from .validate import (
    ValidationResult,
    validate_scene,
    validate_scene_file,
    validate_scene_text,
)

__all__ = [
    "SCENE_SCHEMA",
    "ValidationResult",
    "minimal_scene",
    "scene_schema",
    "validate_scene",
    "validate_scene_file",
    "validate_scene_text",
]
