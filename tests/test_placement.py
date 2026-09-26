"""Transforms, spatial queries, groups, rest_on and physics warnings (host side)."""

from __future__ import annotations

import copy
import math
from typing import Any

import pytest

from jingyu.geometry import GEOMETRY
from jingyu.geometry.spatial import WorldMesh, drop_gap, rest_shift
from jingyu.geometry.transform import (
    IDENTITY,
    apply,
    apply_direction,
    compose,
    multiply,
    rotation_xyz,
    solve_linear,
)
from jingyu.placement import resolve_placement
from jingyu.scene import validate_scene

TOL = 1e-9


def _close(a: Any, b: Any, tol: float = 1e-6) -> bool:
    return all(abs(x - y) <= tol for x, y in zip(a, b, strict=True))


# ------------------------------------------------------------------ transforms


def test_rotation_is_xyz_euler_x_first() -> None:
    # X first, then Z: +X is unaffected by the X turn, then Z turns it onto +Y.
    assert _close(apply_direction(rotation_xyz((90, 0, 90)), (1, 0, 0)), (0, 1, 0))
    # +Y: the X turn lifts it onto +Z, which Z leaves alone.
    assert _close(apply_direction(rotation_xyz((90, 0, 90)), (0, 1, 0)), (0, 0, 1))


def test_compose_is_translate_rotate_scale() -> None:
    m = compose((1, 2, 3), (0, 0, 90), (2, 1, 1))
    assert _close(apply(m, (1, 0, 0)), (1, 4, 3))
    assert _close(multiply(IDENTITY, m), m)


def test_solve_linear_inverts_the_linear_part() -> None:
    m = compose((5, -1, 2), (30, -20, 75), (1.5, 0.5, 2.0))
    x = (0.3, -0.7, 1.1)
    assert _close(solve_linear(m, apply_direction(m, x)), x, 1e-9)
    with pytest.raises(ValueError, match="singular"):
        solve_linear((0.0,) * 16, (1, 0, 0))


# ------------------------------------------------------------ spatial queries


def _mesh(spec: dict[str, Any], location: tuple[float, float, float] = (0, 0, 0)) -> WorldMesh:
    return WorldMesh.from_mesh(GEOMETRY.run(spec), compose(location, (0, 0, 0), (1, 1, 1)))


BOX = {"op": "box", "size": [1.0, 1.0, 1.0]}
BALL = {"op": "sphere", "radius": 0.1, "segments": 32, "rings": 16}


def test_depth_inside_a_closed_box() -> None:
    box = _mesh(BOX)
    assert box.closed
    assert math.isclose(box.depth_inside((0.0, 0.0, 0.9)), 0.1, abs_tol=1e-6)
    assert math.isclose(box.depth_inside((0.45, 0.0, 0.5)), 0.05, abs_tol=1e-6)
    assert box.depth_inside((0.0, 0.0, 1.2)) == 0.0
    assert box.depth_inside((2.0, 0.0, 0.5)) == 0.0


def test_open_meshes_have_no_inside() -> None:
    plane = _mesh({"op": "plane", "size": [2, 2]})
    assert not plane.closed
    assert plane.depth_inside((0, 0, -0.1)) == 0.0


def test_rest_shift_sets_one_mesh_on_another() -> None:
    ball = _mesh(BALL, (0.1, 0.2, 3.0))
    shift = rest_shift(ball, _mesh(BOX))
    assert shift is not None
    assert math.isclose(3.0 + shift, 1.0, abs_tol=1e-6)
    assert rest_shift(_mesh(BALL, (5, 5, 3)), _mesh(BOX)) is None


def test_drop_gap_measures_free_fall() -> None:
    assert math.isclose(drop_gap(_mesh(BALL, (0, 0, 1.25)), _mesh(BOX), 1e-3) or 0, 0.25)
    assert drop_gap(_mesh(BALL, (5, 0, 1.25)), _mesh(BOX), 1e-3) is None


# ------------------------------------------------------------------ placement


def still_life() -> dict[str, Any]:
    """A table top, a bowl in a rotated group resting on it, an orange in the bowl."""

    return {
        "schema": "jingyu.scene.v1",
        "id": "still",
        "groups": [{"id": "arrangement", "location": [0.2, 0.1, 0], "rotation": [0, 0, 40]}],
        "objects": [
            {"id": "floor", "geometry": {"op": "plane", "size": [6, 6]}},
            {"id": "top", "geometry": {"op": "box", "size": [1.2, 0.8, 0.05]}, "rest_on": "floor"},
            {
                "id": "bowl",
                "parent": "arrangement",
                "geometry": {
                    "op": "vessel",
                    "height": 0.08,
                    "base_radius": 0.06,
                    "belly_radius": 0.12,
                    "neck_radius": 0.13,
                    "lip_radius": 0.13,
                },
                "location": [0, 0, 3],
                "rest_on": "top",
            },
            {
                "id": "orange",
                "parent": "arrangement",
                "geometry": {"op": "sphere", "radius": 0.04},
                "location": [0.02, 0, 9],
                "rest_on": "bowl",
            },
        ],
        "lights": [{"id": "sun", "kind": "sun", "location": [0, 0, 5], "rotation": [30, 0, 0]}],
        "cameras": [{"id": "cam", "location": [0, -3, 1], "look_at": [0, 0, 0]}],
        "render": {"camera": "cam"},
    }


def test_rest_on_resolves_heights_in_support_order() -> None:
    result = validate_scene(still_life())
    assert result.valid, result.issues
    assert result.issues == ()
    placement = result.placement
    assert placement is not None
    assert placement.rested == {"top": "floor", "bowl": "top", "orange": "bowl"}
    assert math.isclose(placement.locations["top"][2], 0.0, abs_tol=TOL)
    assert math.isclose(placement.locations["bowl"][2], 0.05, abs_tol=1e-9)
    orange_bottom = placement.meshes["orange"].bounds()[0][2]
    # The orange sits on the bowl's inner floor, above the table top.
    assert 0.05 < orange_bottom < 0.05 + 0.02
    # x and y stay where they were given, in the rotated group.
    world = placement.world["orange"]
    expected = apply(compose((0.2, 0.1, 0), (0, 0, 40), (1, 1, 1)), (0.02, 0, 0))
    assert _close((world[3], world[7]), expected[:2])


def test_rest_on_in_a_tilted_group_moves_straight_down() -> None:
    scene = still_life()
    scene["groups"][0]["rotation"] = [10, 0, 0]
    scene["objects"][2]["location"] = [0, 0, 0.3]
    scene["objects"][3]["location"] = [0.02, 0, 0.4]
    placement = resolve_placement(validate_scene(scene).require_valid())
    before = apply(compose((0.2, 0.1, 0), (10, 0, 0), (1, 1, 1)), (0, 0, 0.3))
    world = placement.world["bowl"]
    assert _close((world[3], world[7]), before[:2], 1e-9)
    assert placement.meshes["bowl"].bounds()[0][2] == pytest.approx(0.05, abs=2e-3)


def test_children_follow_an_object_parent_wherever_rest_on_puts_it() -> None:
    scene = still_life()
    orange = scene["objects"][3]
    orange["parent"] = "bowl"
    orange["location"] = [0.02, 0, 0.3]
    scene["objects"][2]["location"] = [0, 0, 0.4]
    placement = resolve_placement(validate_scene(scene).require_valid())
    bowl, fruit = placement.world["bowl"], placement.world["orange"]
    # The bowl dropped onto the table; the orange kept its place relative to the bowl
    # (turned with the arrangement) and then settled onto the bowl's floor.
    offset = apply(compose((0, 0, 0), (0, 0, 40), (1, 1, 1)), (0.02, 0, 0))
    assert _close((fruit[3] - bowl[3], fruit[7] - bowl[7]), offset[:2], 1e-9)
    assert 0.05 < placement.meshes["orange"].bounds()[0][2] < 0.07
    assert placement.parent_world["orange"] == bowl
    # Moving the bowl moves the orange with it.
    scene["objects"][2]["location"] = [0.1, 0, 0.4]
    moved = resolve_placement(validate_scene(scene).require_valid())
    step = apply(compose((0, 0, 0), (0, 0, 40), (1, 1, 1)), (0.1, 0, 0))
    assert _close(
        (moved.world["orange"][3] - fruit[3], moved.world["orange"][7] - fruit[7]), step[:2], 1e-9
    )


def test_lights_and_cameras_can_ride_on_objects() -> None:
    scene = still_life()
    scene["lights"].append(
        {"id": "lamp", "kind": "point", "parent": "top", "location": [0, 0, 0.5]}
    )
    placement = resolve_placement(validate_scene(scene).require_valid())
    top = placement.world["top"]
    lamp = placement.oriented["lamp"]
    assert _close((lamp[3], lamp[7], lamp[11]), apply(top, (0, 0, 0.5)))


def test_a_loop_through_parents_and_supports_fails() -> None:
    scene = still_life()
    scene["objects"][2]["parent"] = "orange"  # the bowl is carried by its own orange
    result = validate_scene(scene)
    assert any(i.code == "spec.placement_failed" and "cycle" in i.message for i in result.errors), (
        result.issues
    )


def test_rest_on_with_nothing_below_fails() -> None:
    scene = still_life()
    scene["objects"][2]["location"] = [5, 5, 3]
    result = validate_scene(scene)
    assert not result.valid
    assert [(i.code, i.pointer) for i in result.errors] == [
        ("spec.placement_failed", "/objects/2/rest_on")
    ]


def test_rest_on_cycles_fail() -> None:
    scene = still_life()
    scene["objects"][1]["rest_on"] = "orange"
    result = validate_scene(scene)
    assert "spec.placement_failed" in {i.code for i in result.errors}
    assert any("cycle" in i.message for i in result.errors)


@pytest.mark.parametrize(
    ("mutate", "code", "pointer"),
    [
        (
            lambda s: s["objects"][2].update(parent="nope"),
            "spec.unknown_reference",
            "/objects/2/parent",
        ),
        (
            lambda s: s["objects"][2].update(parent="sun"),
            "spec.unknown_reference",
            "/objects/2/parent",
        ),
        (
            lambda s: s["objects"][2].update(parent="bowl"),
            "spec.unknown_reference",
            "/objects/2/parent",
        ),
        (
            lambda s: (
                s["objects"][1].update(parent="bowl"),
                s["objects"][2].update(parent="top"),
            ),
            "spec.parent_cycle",
            "/objects/1/parent",
        ),
        (
            lambda s: s["objects"][2].update(rest_on="bowl"),
            "spec.unknown_reference",
            "/objects/2/rest_on",
        ),
        (
            lambda s: s["objects"][2].update(rest_on="sun"),
            "spec.unknown_reference",
            "/objects/2/rest_on",
        ),
        (
            lambda s: s["cameras"][0].update(parent="ghost"),
            "spec.unknown_reference",
            "/cameras/0/parent",
        ),
        (lambda s: s["groups"].append({"id": "floor"}), "spec.duplicate_id", "/objects/0/id"),
        (
            lambda s: s["groups"].extend([{"id": "a", "parent": "b"}, {"id": "b", "parent": "a"}]),
            "spec.parent_cycle",
            "/groups/1/parent",
        ),
    ],
)
def test_hierarchy_errors(mutate: Any, code: str, pointer: str) -> None:
    scene = still_life()
    mutate(scene)
    result = validate_scene(scene)
    assert (code, pointer) in {(i.code, i.pointer) for i in result.errors}


def test_duplicate_ids_across_groups_and_objects() -> None:
    scene = still_life()
    scene["groups"].append({"id": "orange"})
    codes = {(i.code, i.pointer) for i in validate_scene(scene).errors}
    assert ("spec.duplicate_id", "/objects/3/id") in codes


# -------------------------------------------------------------------- physics


def _warnings(scene: dict[str, Any]) -> list[tuple[str, str]]:
    result = validate_scene(scene)
    assert result.valid, result.errors
    return [(w.code, w.pointer) for w in result.warnings]


def test_sinking_is_reported_once_per_pair() -> None:
    scene = still_life()
    orange = scene["objects"][3]
    del orange["rest_on"]
    del orange["parent"]
    orange["location"] = [0.4, 0.3, 0.03]
    assert _warnings(scene) == [("physics.intersection", "/objects/3/location")]


def test_hovering_is_reported_with_the_support_below() -> None:
    scene = still_life()
    orange = scene["objects"][3]
    del orange["rest_on"]
    del orange["parent"]
    orange["location"] = [0.4, 0.3, 0.2]
    result = validate_scene(scene)
    [warning] = result.warnings
    assert (warning.code, warning.pointer) == ("physics.floating", "/objects/3/location")
    assert "'top'" in warning.message and "150 mm" in warning.message


def test_hidden_objects_are_not_checked() -> None:
    scene = still_life()
    orange = scene["objects"][3]
    del orange["rest_on"]
    orange["location"] = [0.0, 0.0, 0.5]
    orange["visible"] = False
    assert _warnings(scene) == []


def test_example_scenes_have_no_physics_warnings() -> None:
    from helpers import REPO_ROOT

    from jingyu.scene import validate_scene_file

    for path in sorted((REPO_ROOT / "examples" / "scenes").glob("*.json")):
        result = validate_scene_file(path)
        assert result.valid and result.warnings == [], (path.name, result.issues)


def test_placement_does_not_change_the_scene() -> None:
    scene = validate_scene(still_life()).require_valid()
    snapshot = copy.deepcopy(scene)
    resolve_placement(scene)
    assert scene == snapshot


def test_a_reflector_card_may_hover() -> None:
    scene = still_life()
    scene["objects"].append(
        {
            "id": "card",
            "geometry": {"op": "box", "size": [0.4, 0.01, 0.4]},
            "location": [0.8, 0, 0.5],
            "rotation": [0, 0, 30],
            "camera_visible": False,
        }
    )
    assert _warnings(scene) == []
    scene["objects"][-1]["camera_visible"] = True
    assert [code for code, _ in _warnings(scene)] == ["physics.floating"]


def test_fill_lights_have_their_own_defaults() -> None:
    scene = still_life()
    scene["lights"].append(
        {"id": "fill", "kind": "fill", "location": [1, -1, 1], "look_at": [0, 0, 0]}
    )
    light = validate_scene(scene).require_valid()["lights"][-1]
    assert (light["power_w"], light["size"]) == (100.0, 1.0)
    del scene["lights"][-1]["look_at"]
    assert not validate_scene(scene).valid  # a fill light must be aimed
