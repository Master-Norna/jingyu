"""edit_scene's patch language and diff_candidates' scene diff."""

from __future__ import annotations

import copy
from typing import Any

import pytest

from jingyu.errors import JingyuError
from jingyu.scene import minimal_scene, validate_scene
from jingyu.scene.diff import diff_scenes
from jingyu.scene.patch import apply_patch, resolve_pointer


@pytest.fixture
def scene() -> dict[str, Any]:
    """A normalised scene, as edit_scene gets it from a candidate: every field exists."""

    return validate_scene(minimal_scene()).require_valid()


def test_id_segments_select_array_elements(scene: dict[str, Any]) -> None:
    vase_index = [o["id"] for o in scene["objects"]].index("vase")
    assert (
        resolve_pointer(scene, "/objects/@vase/location/2") == f"/objects/{vase_index}/location/2"
    )
    edited = apply_patch(
        scene, [{"op": "replace", "path": "/objects/@vase/location/0", "value": 0.5}]
    )
    assert edited["objects"][vase_index]["location"][0] == 0.5
    assert scene["objects"][vase_index]["location"][0] != 0.5  # input untouched


def test_all_operations(scene: dict[str, Any]) -> None:
    edited = apply_patch(
        scene,
        [
            {"op": "add", "path": "/objects/-", "value": {"id": "cup", "geometry": {"op": "box"}}},
            {"op": "copy", "from": "/objects/@cup", "path": "/objects/-"},
            {"op": "replace", "path": "/objects/@cup/id", "value": "cup_a"},
            {"op": "move", "from": "/objects/@cup_a", "path": "/objects/0"},
            {"op": "merge", "path": "/objects/@cup", "value": {"visible": False, "material": None}},
            {"op": "test", "path": "/objects/0/id", "value": "cup_a"},
            {"op": "remove", "path": "/objects/@floor"},
        ],
    )
    assert [o["id"] for o in edited["objects"]] == ["cup_a", "vase", "cup"]
    assert edited["objects"][2]["visible"] is False


def test_merge_null_removes_a_field(scene: dict[str, Any]) -> None:
    scene["objects"][1]["material"] = "glaze"
    edited = apply_patch(
        scene, [{"op": "merge", "path": "/objects/@vase", "value": {"material": None}}]
    )
    assert "material" not in edited["objects"][1]


@pytest.mark.parametrize(
    "operation",
    [
        {"op": "replace", "path": "/objects/@nope/location", "value": [0, 0, 0]},
        {"op": "replace", "path": "/objects/9/id", "value": "x"},
        {"op": "remove", "path": "/render/nope"},
        {"op": "test", "path": "/schema", "value": "other"},
        {"op": "test", "path": "/render/samples", "value": True},
        {"op": "add", "path": "objects", "value": 1},
        {"op": "move", "path": "/objects/0/geometry", "from": "/objects/0"},
        {"op": "copy", "path": "/objects/-"},
        {"op": "replace", "path": "/objects/01/id", "value": "x"},
        {"op": "frobnicate", "path": "/id"},
        {"op": "add", "path": "/objects/-"},
    ],
)
def test_failures_are_atomic_and_coded(scene: dict[str, Any], operation: dict[str, Any]) -> None:
    snapshot = copy.deepcopy(scene)
    with pytest.raises(JingyuError) as info:
        apply_patch(scene, [{"op": "replace", "path": "/id", "value": "changed"}, operation])
    assert info.value.code == "patch.failed"
    assert info.value.details["operation"] == 1
    assert scene == snapshot


def test_diff_matches_entities_by_id(scene: dict[str, Any]) -> None:
    edited = apply_patch(
        scene,
        [
            {"op": "replace", "path": "/objects/@vase/location", "value": [0.1, 0, 0]},
            {"op": "remove", "path": "/objects/@floor"},
            {"op": "add", "path": "/objects/0", "value": {"id": "cup", "geometry": {"op": "box"}}},
            {"op": "add", "path": "/intent", "value": "a quiet vase"},
        ],
    )
    changes = {(c["path"], c["change"]) for c in diff_scenes(scene, edited)}
    assert changes == {
        ("/objects/@floor", "removed"),
        ("/objects/@cup", "added"),
        (
            "/objects/@vase/location",
            "added" if "location" not in scene["objects"][1] else "changed",
        ),
        ("/intent", "added"),
    }
    assert diff_scenes(scene, copy.deepcopy(scene)) == []


def test_diff_reports_reordering_and_type_changes() -> None:
    before = {"objects": [{"id": "a"}, {"id": "b"}], "n": 1}
    after = {"objects": [{"id": "b"}, {"id": "a"}], "n": 1.0}
    assert diff_scenes(before, after) == [
        {"path": "/objects", "change": "reordered", "before": ["a", "b"], "after": ["b", "a"]}
    ]
    assert diff_scenes({"v": 1}, {"v": True}) == [
        {"path": "/v", "change": "changed", "before": 1, "after": True}
    ]
