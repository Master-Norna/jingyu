"""Host-side camera model and lines of sight through openings."""

from __future__ import annotations

import math
from typing import Any

import pytest

from jingyu.camera import CameraModel
from jingyu.scene import validate_scene
from jingyu.sightlines import background_openings


def _room(camera: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": "jingyu.scene.v1",
        "id": "room",
        "objects": [
            {
                "id": "wall",
                "geometry": {
                    "op": "wall",
                    "size": [4, 0.2, 3],
                    "openings": [
                        {"x": -1.0, "sill": 1.0, "width": 1.0, "height": 1.0},
                        {"x": 1.0, "sill": 0.0, "width": 0.8, "height": 2.0},
                    ],
                },
                "location": [0, 2, 0],
            }
        ],
        "cameras": [{"id": "cam", **camera}],
        "render": {"camera": "cam", "resolution": [200, 100]},
    }


def _model(scene: dict[str, Any]) -> tuple[dict[str, Any], CameraModel]:
    result = validate_scene(scene)
    normalized = result.require_valid()
    return normalized, CameraModel.from_scene(normalized, result.placement)


def test_projection_and_rays_are_inverse() -> None:
    scene = _room({"location": [0.3, -2, 1.4], "look_at": [0, 2, 1.2], "shift": [0.1, 0.05]})
    _, camera = _model(scene)
    for point in [(0.0, 2.0, 1.2), (-1.0, 1.0, 0.5), (0.7, 3.0, 2.0)]:
        projected = camera.project(point)
        assert projected is not None
        u, v, depth = projected
        origin, direction = camera.ray(u, v)
        back = tuple(
            o + d * depth / _dot(direction, camera.forward)
            for o, d in zip(origin, direction, strict=True)
        )
        assert all(math.isclose(a, b, abs_tol=1e-9) for a, b in zip(back, point, strict=True))
    # The aim point lands on the frame centre, moved by the lens shift.
    u, v, _ = camera.project((0, 2, 1.2)) or (0, 0, 0)
    assert u == pytest.approx(0.5 - 0.1)
    assert v == pytest.approx(0.5 + 0.05 * 2)  # shift is in frame widths; the frame is 2:1
    assert camera.project((0, -3, 1.4)) is None  # behind the camera


def _dot(a: Any, b: Any) -> float:
    return float(sum(x * y for x, y in zip(a, b, strict=True)))


def test_background_pixels_are_traced_to_the_opening_they_look_through() -> None:
    scene = _room({"location": [0, -2, 1.5], "look_at": [0, 2, 1.5], "lens_mm": 24})
    normalized, camera = _model(scene)
    window = camera.project((-1.0, 2.0, 1.5))
    door = camera.project((1.0, 2.0, 1.0))
    assert window is not None and door is not None
    pixels = [
        (int(window[0] * 200), int(window[1] * 100)),
        (int(window[0] * 200) + 1, int(window[1] * 100)),
        (int(door[0] * 200), int(door[1] * 100)),
    ]
    found = background_openings(normalized, 200, 100, pixels)
    assert [(f["object"], f["opening"], f["pixels"]) for f in found] == [
        ("wall", "openings/0", 2),
        ("wall", "openings/1", 1),
    ]
    assert found[0]["pointer"] == "/objects/0/geometry/openings/0"
    # A line of sight that misses both openings (over the wall) passes through none.
    assert background_openings(normalized, 200, 100, [(100, 0)]) == []
    assert background_openings(normalized, 200, 100, []) == []
