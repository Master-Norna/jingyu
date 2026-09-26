"""A minimal, valid scene used by tools and documentation."""

from __future__ import annotations

import copy
from typing import Any

_MINIMAL: dict[str, Any] = {
    "schema": "jingyu.scene.v1",
    "id": "minimal-vase",
    "title": "One vase on a floor",
    "materials": [
        {"id": "glaze", "family": "ceramic", "color": "#6f93b3", "gloss": 0.9},
        {"id": "floor", "family": "plastic", "color": "#d9d6d0", "gloss": 0.05},
    ],
    "objects": [
        {"id": "floor", "geometry": {"op": "plane", "size": [6, 6]}, "material": "floor"},
        {
            "id": "vase",
            "geometry": {
                "op": "vessel",
                "height": 0.3,
                "base_radius": 0.045,
                "belly_radius": 0.085,
                "belly_at": 0.3,
                "neck_radius": 0.022,
                "neck_at": 0.9,
                "lip_radius": 0.03,
            },
            "material": "glaze",
        },
    ],
    "lights": [
        {
            "id": "key",
            "kind": "area",
            "location": [-1.2, -1.5, 1.8],
            "look_at": [0, 0, 0.15],
            "power_w": 400,
            "size": 1.5,
        }
    ],
    "cameras": [{"id": "main", "location": [0, -1.4, 0.45], "look_at": [0, 0, 0.15]}],
    "render": {"camera": "main", "resolution": [960, 720], "samples": 64},
}


def minimal_scene() -> dict[str, Any]:
    """Return a fresh copy of the minimal example scene."""

    return copy.deepcopy(_MINIMAL)


__all__ = ["minimal_scene"]
