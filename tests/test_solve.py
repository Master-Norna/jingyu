"""Solving the camera from a framing and the sun from where its light should land."""

from __future__ import annotations

import copy
import math
from typing import Any

import pytest

from jingyu.camera import CameraModel
from jingyu.errors import JingyuError
from jingyu.geometry.raycast import cast, unit
from jingyu.scene import validate_scene
from jingyu.scene.patch import apply_patch
from jingyu.solve import FramingRequest, SunRequest, aim_sun, frame_subject
from jingyu.solve.common import frame_box, sample_points
from jingyu.tools import REGISTRY, ToolContext
from jingyu.workspace import Workspace


def _room() -> dict[str, Any]:
    """A table with a still life inside four walls; one window in the west wall."""

    return {
        "schema": "jingyu.scene.v1",
        "id": "room",
        "world": {"environment": {"family": "daylight", "sun_elevation": 30, "sun_azimuth": 90}},
        "groups": [{"id": "still", "location": [0.1, 0.0, 0.0]}],
        "objects": [
            {
                "id": "floor",
                "geometry": {"op": "box", "size": [6, 6, 0.1]},
                "location": [0, 0, -0.1],
            },
            {
                "id": "west",
                "geometry": {
                    "op": "wall",
                    "size": [4, 0.2, 3],
                    "openings": [{"x": 0.0, "sill": 1.0, "width": 1.2, "height": 1.2}],
                },
                "location": [-2, 0, 0],
                "rotation": [0, 0, 90],
            },
            {
                "id": "east",
                "geometry": {"op": "wall", "size": [4, 0.2, 3]},
                "location": [2, 0, 0],
                "rotation": [0, 0, 90],
            },
            {"id": "north", "geometry": {"op": "wall", "size": [4, 0.2, 3]}, "location": [0, 2, 0]},
            {"id": "table", "geometry": {"op": "box", "size": [1.2, 0.8, 0.75]}},
            {
                "id": "vase",
                "parent": "still",
                "geometry": {"op": "vessel", "height": 0.3},
                "location": [-0.2, 0.1, 1],
                "rest_on": "table",
            },
            {
                "id": "cup",
                "parent": "still",
                "geometry": {"op": "cylinder", "radius": 0.04, "height": 0.1},
                "location": [0.25, -0.1, 1],
                "rest_on": "table",
            },
        ],
        "cameras": [{"id": "cam", "location": [1.2, -1.6, 1.4], "look_at": [0, 0, 0.8]}],
        "render": {"camera": "cam", "resolution": [640, 480]},
    }


def _solve_frame(scene: dict[str, Any], **kwargs: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    result = validate_scene(scene)
    solution = frame_subject(result.require_valid(), result.placement, FramingRequest(**kwargs))
    edited = apply_patch(scene, solution["operations"])
    return solution, edited


def _subject_box(scene: dict[str, Any], ids: list[str]) -> dict[str, float]:
    result = validate_scene(scene)
    normalized = result.require_valid()
    camera = CameraModel.from_scene(normalized, result.placement)
    points = [p for i in ids for p in sample_points(result.placement.meshes[i])]
    box = frame_box(camera, points)
    assert box is not None
    return box


# --------------------------------------------------------------------- framing


@pytest.mark.parametrize(
    ("at", "size", "size_of"),
    [((0.5, 0.5), 0.5, "larger"), ((0.33, 0.6), 0.3, "height"), ((0.7, 0.4), 0.45, "width")],
)
def test_the_subject_lands_where_asked_and_as_big(
    at: tuple[float, float], size: float, size_of: str
) -> None:
    solution, edited = _solve_frame(_room(), subject=("still",), at=at, size=size, size_of=size_of)
    assert solution["subject"]["objects"] == ["vase", "cup"]
    box = _subject_box(edited, ["vase", "cup"])
    assert (box["u0"] + box["u1"]) / 2 == pytest.approx(at[0], abs=2e-3)
    assert (box["v0"] + box["v1"]) / 2 == pytest.approx(at[1], abs=2e-3)
    extent = {"width": box["u1"] - box["u0"], "height": box["v1"] - box["v0"]}
    extent["larger"] = max(extent.values())
    assert extent[size_of] == pytest.approx(size, abs=2e-3)
    assert solution["camera"]["lens_mm"] == 50.0  # the lens was kept
    assert solution["warnings"] == []


def test_the_view_direction_is_kept_unless_given() -> None:
    scene = _room()
    solution, _ = _solve_frame(scene, subject=("vase",), size=0.4)
    before = math.degrees(math.atan2(-1.6 - 0.0, 1.2 - 0.0))  # roughly where the camera was
    assert abs((solution["camera"]["azimuth"] - before + 180) % 360 - 180) < 10
    solution, edited = _solve_frame(scene, subject=("vase",), azimuth=270, elevation=20, size=0.4)
    assert solution["camera"]["azimuth"] == pytest.approx(270, abs=1e-6)
    assert solution["camera"]["elevation"] == pytest.approx(20, abs=1e-6)
    camera = edited["cameras"][0]
    # Seen from the south (azimuth 270), the camera stands straight south of its aim.
    assert camera["location"][0] == pytest.approx(camera["look_at"][0], abs=1e-3)
    assert camera["location"][1] < camera["look_at"][1]


def test_a_fixed_distance_solves_the_lens() -> None:
    solution, edited = _solve_frame(_room(), subject=("still",), size=0.6, distance=2.5)
    assert solution["camera"]["distance"] == pytest.approx(2.5, abs=1e-6)
    assert solution["camera"]["lens_mm"] != 50.0
    box = _subject_box(edited, ["vase", "cup"])
    assert max(box["u1"] - box["u0"], box["v1"] - box["v0"]) == pytest.approx(0.6, abs=2e-3)
    with pytest.raises(JingyuError) as info:
        _solve_frame(_room(), subject=("still",), distance=2.0, lens_mm=35)
    assert info.value.code == "tool.invalid_arguments"


def test_an_orthographic_camera_solves_its_scale() -> None:
    scene = _room()
    scene["cameras"][0].update(projection="orthographic", ortho_scale=3.0)
    solution, edited = _solve_frame(scene, subject=("vase",), at=(0.3, 0.5), size=0.5)
    assert "ortho_scale" in solution["camera"] and "lens_mm" not in solution["camera"]
    box = _subject_box(edited, ["vase"])
    assert (box["u0"] + box["u1"]) / 2 == pytest.approx(0.3, abs=2e-3)
    assert max(box["u1"] - box["u0"], box["v1"] - box["v0"]) == pytest.approx(0.5, abs=2e-3)


def test_a_rotated_camera_in_a_group_gets_a_local_location_and_look_at() -> None:
    scene = _room()
    scene["groups"].append({"id": "rig", "location": [0.5, 0, 0.2], "rotation": [0, 0, 30]})
    scene["cameras"][0] = {
        "id": "cam",
        "parent": "rig",
        "location": [0.5, -1.8, 1.2],
        "rotation": [70, 0, 10],
    }
    solution, edited = _solve_frame(scene, subject=("cup",), size=0.2, at=(0.6, 0.6))
    assert "rotation" not in edited["cameras"][0]
    assert solution["operations"][0]["value"]["rotation"] is None
    box = _subject_box(edited, ["cup"])
    assert (box["u0"] + box["u1"]) / 2 == pytest.approx(0.6, abs=2e-3)


def test_warnings_name_walls_the_camera_is_in_and_things_in_the_way() -> None:
    scene = _room()
    # From the north, a small subject framed from far away puts the camera in or
    # behind the north wall.
    solution, _ = _solve_frame(scene, subject=("cup",), size=0.05, azimuth=90, elevation=5)
    codes = {w["code"] for w in solution["warnings"]}
    assert "solve.subject_hidden" in codes or "solve.camera_inside" in codes
    assert any("north" in w["message"] for w in solution["warnings"])


def test_unknown_or_invisible_subjects_fail() -> None:
    with pytest.raises(JingyuError) as info:
        _solve_frame(_room(), subject=("nope",))
    assert info.value.code == "spec.unknown_reference"
    scene = _room()
    for obj in scene["objects"]:
        if obj["id"] in ("vase", "cup"):
            obj["visible"] = False
    with pytest.raises(JingyuError) as info:
        _solve_frame(scene, subject=("still",))
    assert info.value.code == "solve.no_solution"


# ------------------------------------------------------------------------- sun


def _solve_sun(scene: dict[str, Any], **kwargs: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    result = validate_scene(scene)
    solution = aim_sun(result.require_valid(), result.placement, SunRequest(**kwargs))
    return solution, apply_patch(scene, solution["operations"])


def _sun_reaches(scene: dict[str, Any], point: tuple[float, float, float]) -> bool:
    result = validate_scene(scene)
    environment = result.require_valid()["world"]["environment"]
    e, a = math.radians(environment["sun_elevation"]), math.radians(environment["sun_azimuth"])
    toward = (math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e))
    origin = tuple(p + d * 1e-4 for p, d in zip(point, toward, strict=True))
    return cast(result.placement.meshes, origin, toward) is None


def test_the_sun_is_aimed_through_the_window_onto_a_point() -> None:
    target = (0.0, 0.0, 0.75)
    solution, edited = _solve_sun(_room(), point=target)
    assert solution["through"]["object"] == "west"
    assert solution["through"]["pointer"] == "/objects/1/geometry/openings/0"
    assert solution["target"]["lit_fraction"] == 1.0
    assert _sun_reaches(edited, target)
    assert not _sun_reaches(_room(), target)  # the north wall shades it from the sun as given
    # The beam lands on the table around the target.
    table = next(p for p in solution["patch"] if p["object"] == "table")
    (x0, y0, _), (x1, y1, _) = table["world_box"]
    assert x0 - 0.05 <= target[0] <= x1 + 0.05 and y0 - 0.05 <= target[1] <= y1 + 0.05


def test_a_fixed_elevation_solves_the_azimuth_through_the_opening() -> None:
    target = (0.3, 0.2, 0.75)
    solution, edited = _solve_sun(_room(), point=target, elevation=25)
    assert solution["sun"]["elevation"] == 25
    assert _sun_reaches(edited, target)
    low, high = solution["through"]["azimuth_range"]
    assert low <= solution["sun"]["azimuth"] <= high
    elevations = solution["through"]["elevation_range"]
    assert elevations[0] < 25 < elevations[1]


def test_a_target_above_the_window_has_no_sun_through_it() -> None:
    with pytest.raises(JingyuError) as info:
        _solve_sun(_room(), point=(0.0, 0.0, 2.9), through=("west", "openings/0"))
    assert info.value.code == "solve.no_solution"


def test_an_object_target_counts_what_the_camera_sees_of_it() -> None:
    solution, _ = _solve_sun(_room(), object="table")
    assert solution["target"]["object"] == "table"
    assert solution["target"]["samples"] > 10
    assert solution["target"]["lit_fraction"] > 0.2


def test_a_spot_in_the_frame_is_a_target() -> None:
    scene = _room()
    result = validate_scene(scene)
    camera = CameraModel.from_scene(result.require_valid(), result.placement)
    u, v, _ = camera.project((0.4, -0.3, 0.75)) or (0, 0, 0)
    solution, edited = _solve_sun(scene, uv=(u, v))
    assert solution["target"]["object"] == "table"
    assert solution["target"]["point"] == pytest.approx([0.4, -0.3, 0.75], abs=1e-3)
    assert _sun_reaches(edited, (0.4, -0.3, 0.75))
    scene["cameras"][0]["look_at"] = [1.2, -1.6, 5.0]  # straight up at the open sky
    with pytest.raises(JingyuError) as info:
        _solve_sun(scene, uv=(0.5, 0.5))
    assert info.value.code == "solve.no_solution"


def test_a_sun_light_is_aimed_by_its_rotation() -> None:
    scene = _room()
    del scene["world"]
    scene["lights"] = [{"id": "sun", "kind": "sun", "location": [0, 0, 5], "look_at": [0, 0, 0]}]
    target = (0.0, 0.0, 0.75)
    solution, edited = _solve_sun(scene, point=target)
    assert solution["light"] == "sun"
    value = solution["operations"][0]["value"]
    assert value["look_at"] is None and len(value["rotation"]) == 3
    result = validate_scene(edited)
    matrix = result.placement.oriented["sun"]
    toward = unit((matrix[2], matrix[6], matrix[10]))
    assert (
        cast(
            result.placement.meshes,
            tuple(t + d * 1e-4 for t, d in zip(target, toward, strict=True)),
            toward,
        )
        is None
    )


def test_the_environment_must_have_a_sun() -> None:
    scene = _room()
    scene["world"] = {"environment": {"family": "uniform"}}
    with pytest.raises(JingyuError) as info:
        _solve_sun(scene, point=(0, 0, 0.75))
    assert info.value.code == "solve.no_solution"


def test_outdoors_the_current_sun_is_kept_when_it_already_reaches() -> None:
    scene = _room()
    scene["objects"] = [scene["objects"][0], scene["objects"][4]]  # floor and table only
    solution, _ = _solve_sun(scene, point=(0.0, 0.0, 0.75))
    assert solution["sun"] == {"elevation": 30.0, "azimuth": 90.0}
    assert "through" not in solution


def test_outdoors_a_blocked_sun_turns_to_the_nearest_open_azimuth() -> None:
    scene = _room()
    scene["objects"] = [scene["objects"][0], scene["objects"][4]]
    scene["objects"].append(
        {"id": "screen", "geometry": {"op": "box", "size": [2, 0.1, 2]}, "location": [0, 1.0, 0]}
    )
    solution, edited = _solve_sun(scene, point=(0.0, 0.0, 0.75))
    assert solution["sun"]["elevation"] == 30.0
    assert solution["sun"]["azimuth"] != 90.0
    assert _sun_reaches(edited, (0.0, 0.0, 0.75))


# ----------------------------------------------------------------------- tools


def test_the_tools_return_the_edit_and_save_it(tmp_path: Any) -> None:
    context = ToolContext(Workspace.at(tmp_path))
    framed = REGISTRY.invoke(
        "frame_subject",
        {
            "scene": _room(),
            "subject": ["still"],
            "at": {"u": 0.4, "v": 0.5},
            "size": 0.4,
            "save_as": "framed.json",
        },
        context,
    ).data
    assert framed["valid"] and framed["saved_to"] == "framed.json"
    sunny = REGISTRY.invoke(
        "aim_sun",
        {
            "scene_path": "framed.json",
            "target": {"point": [0, 0, 0.75]},
            "through": {"object": "west", "opening": 0},
        },
        context,
    ).data
    assert sunny["valid"]
    assert sunny["scene"]["cameras"] == framed["scene"]["cameras"]
    assert sunny["scene"]["world"]["environment"]["sun_azimuth"] == sunny["sun"]["azimuth"]
    original = copy.deepcopy(_room())
    assert original["world"] != sunny["scene"]["world"]
