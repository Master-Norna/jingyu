from __future__ import annotations

import copy
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from helpers import REPO_ROOT, synthetic_scene_document

from jingyu.canonical_json import canonical_sha256
from jingyu.errors import JingyuError
from jingyu.scene import (
    SCENE_SCHEMA,
    minimal_scene,
    validate_scene,
    validate_scene_file,
    validate_scene_text,
)

EXAMPLE_SCENES = sorted((REPO_ROOT / "examples" / "scenes").glob("*.json"))

Mutation = Callable[[dict[str, Any]], None]


def _mutated(mutate: Mutation) -> dict[str, Any]:
    scene = minimal_scene()
    mutate(scene)
    return scene


def _vase(scene: dict[str, Any]) -> dict[str, Any]:
    return scene["objects"][1]


def test_minimal_scene_is_valid_without_warnings() -> None:
    result = validate_scene(minimal_scene())
    assert result.valid
    assert result.issues == ()
    assert result.scene_sha256 == canonical_sha256(result.normalized)


def test_minimal_scene_is_a_fresh_copy() -> None:
    scene = minimal_scene()
    scene["objects"].clear()
    assert minimal_scene()["objects"]


def test_synthetic_test_scene_is_valid() -> None:
    assert validate_scene(synthetic_scene_document()).issues == ()


def test_there_are_example_scenes() -> None:
    assert EXAMPLE_SCENES


@pytest.mark.parametrize("path", EXAMPLE_SCENES, ids=lambda p: p.name)
def test_example_scenes_are_valid_without_warnings(path: Path) -> None:
    result = validate_scene_file(path)
    assert result.valid, [i.to_dict() for i in result.issues]
    assert result.issues == ()


def test_defaults_are_filled_in() -> None:
    normalized = validate_scene(minimal_scene()).require_valid()
    assert normalized["world"] == {"color": "#404040", "strength": 1.0}
    floor, vase = normalized["objects"]
    for obj in (floor, vase):
        assert obj["location"] == [0.0, 0.0, 0.0]
        assert obj["rotation"] == [0.0, 0.0, 0.0]
        assert obj["scale"] == [1.0, 1.0, 1.0]
        assert obj["visible"] is True
    geometry = vase["geometry"]
    assert (geometry["thickness"], geometry["open"], geometry["segments"]) == (0.004, True, 96)
    assert geometry["profile_samples"] == 64
    assert normalized["lights"][0]["color"] == "#ffffff"
    camera = normalized["cameras"][0]
    assert camera["projection"] == "perspective"
    assert (camera["lens_mm"], camera["clip_start"], camera["clip_end"]) == (50.0, 0.05, 500.0)
    assert camera["shift"] == [0.0, 0.0]
    render = normalized["render"]
    assert render["engine"] == "cycles"
    assert render["device"] == "auto"
    assert render["view_transform"] == "agx"
    assert (render["seed"], render["denoise"], render["transparent_background"]) == (0, True, False)


def test_defaults_follow_the_selected_branch() -> None:
    scene = minimal_scene()
    scene["lights"] = [
        {"id": "p", "kind": "point", "location": [0, 0, 2]},
        {"id": "s", "kind": "sun", "location": [0, 0, 5], "rotation": [10, 0, 0]},
    ]
    scene["materials"].append({"id": "m", "family": "metal"})
    normalized = validate_scene(scene).require_valid()
    point, sun = normalized["lights"]
    assert (point["power_w"], point["radius"]) == (1000.0, 0.1)
    assert "strength" not in point
    assert (sun["strength"], sun["angle"]) == (3.0, 0.5)
    assert "power_w" not in sun
    assert normalized["materials"][2] == {
        "id": "m",
        "family": "metal",
        "metal": "silver",
        "polish": 0.8,
    }


def test_normalisation_does_not_touch_the_input() -> None:
    scene = minimal_scene()
    before = copy.deepcopy(scene)
    validate_scene(scene)
    assert scene == before


def test_integral_floats_become_ints_where_the_schema_says_integer() -> None:
    scene = minimal_scene()
    scene["render"].update(resolution=[960.0, 720.0], samples=64.0, seed=7.0)
    _vase(scene)["geometry"]["segments"] = 48.0
    normalized = validate_scene(scene).require_valid()
    render = normalized["render"]
    assert render["resolution"] == [960, 720]
    assert all(type(v) is int for v in (*render["resolution"], render["samples"], render["seed"]))
    assert type(normalized["objects"][1]["geometry"]["segments"]) is int
    # Numbers typed "number" keep their spelling.
    assert type(normalized["objects"][1]["geometry"]["height"]) is float


def test_identity_ignores_spelling_key_order_and_explicit_defaults() -> None:
    base = validate_scene(minimal_scene()).scene_sha256
    spelled = minimal_scene()
    spelled["render"]["resolution"] = [960.0, 720.0]
    spelled["render"]["engine"] = "cycles"
    spelled["objects"][0]["visible"] = True
    reordered = json.loads(json.dumps(minimal_scene(), sort_keys=True))
    assert validate_scene(spelled).scene_sha256 == base
    assert validate_scene(reordered).scene_sha256 == base
    changed = minimal_scene()
    changed["render"]["samples"] = 65
    assert validate_scene(changed).scene_sha256 != base


def test_ids_may_repeat_across_the_material_and_node_namespaces() -> None:
    scene = minimal_scene()
    assert {m["id"] for m in scene["materials"]} & {o["id"] for o in scene["objects"]} == {"floor"}
    scene["materials"].append({"id": "vase", "family": "glass"})
    assert validate_scene(scene).valid


INVALID_CASES: list[tuple[str, Mutation, str, str]] = [
    ("unknown top-level field", lambda s: s.update(colour="red"), "spec.schema_violation", ""),
    (
        "unknown object field",
        lambda s: _vase(s).update(colour="red"),
        "spec.schema_violation",
        "/objects/1",
    ),
    (
        "unknown geometry parameter",
        lambda s: _vase(s)["geometry"].update(waist=0.1),
        "spec.schema_violation",
        "/objects/1/geometry",
    ),
    (
        "bad render engine",
        lambda s: s["render"].update(engine="octane"),
        "spec.schema_violation",
        "/render/engine",
    ),
    (
        "unknown geometry op",
        lambda s: _vase(s).update(geometry={"op": "torus"}),
        "spec.schema_violation",
        "/objects/1/geometry/op",
    ),
    (
        "unknown material family",
        lambda s: s["materials"].append({"id": "oak", "family": "wood"}),
        "spec.schema_violation",
        "/materials/2/family",
    ),
    (
        "material parameter out of range",
        lambda s: s["materials"][0].update(gloss=2),
        "spec.schema_violation",
        "/materials/0/gloss",
    ),
    (
        "missing required geometry parameter",
        lambda s: s["objects"][0]["geometry"].pop("size"),
        "spec.schema_violation",
        "/objects/0/geometry",
    ),
    (
        "uppercase id",
        lambda s: _vase(s).update(id="Vase"),
        "spec.schema_violation",
        "/objects/1/id",
    ),
    (
        "id with a trailing newline",
        lambda s: _vase(s).update(id="vase\n"),
        "spec.schema_violation",
        "/objects/1/id",
    ),
    (
        "colour with a trailing newline",
        lambda s: s["materials"][0].update(color="#ffffff\n"),
        "spec.schema_violation",
        "/materials/0/color",
    ),
    (
        "camera with both look_at and rotation",
        lambda s: s["cameras"][0].update(rotation=[90, 0, 0]),
        "spec.schema_violation",
        "/cameras/0",
    ),
    (
        "light without an aim",
        lambda s: s["lights"][0].pop("look_at"),
        "spec.schema_violation",
        "/lights/0",
    ),
    (
        "resolution below the minimum",
        lambda s: s["render"].update(resolution=[8, 720]),
        "spec.schema_violation",
        "/render/resolution/0",
    ),
    (
        "duplicate object id",
        lambda s: s["objects"].append({"id": "vase", "geometry": {"op": "box", "size": [1, 1, 1]}}),
        "spec.duplicate_id",
        "/objects/2/id",
    ),
    (
        "light reuses an object id",
        lambda s: s["lights"][0].update(id="vase"),
        "spec.duplicate_id",
        "/lights/0/id",
    ),
    (
        "duplicate material id",
        lambda s: s["materials"].append({"id": "glaze", "family": "glass"}),
        "spec.duplicate_id",
        "/materials/2/id",
    ),
    (
        "unknown material reference",
        lambda s: _vase(s).update(material="celadon"),
        "spec.unknown_reference",
        "/objects/1/material",
    ),
    (
        "unknown render camera",
        lambda s: s["render"].update(camera="side"),
        "spec.unknown_reference",
        "/render/camera",
    ),
    (
        "vessel belly above its neck",
        lambda s: _vase(s)["geometry"].update(belly_at=0.9, neck_at=0.5),
        "spec.invalid_parameter",
        "/objects/1/geometry/neck_at",
    ),
    (
        "vessel wall thicker than its neck",
        lambda s: _vase(s)["geometry"].update(thickness=0.05),
        "spec.invalid_parameter",
        "/objects/1/geometry/thickness",
    ),
    (
        "lathe profile touching the axis",
        lambda s: _vase(s).update(
            geometry={"op": "lathe", "profile": [[0.05, 0], [0, 0.05], [0.05, 0.1]]}
        ),
        "spec.invalid_parameter",
        "/objects/1/geometry/profile",
    ),
    (
        "camera looking at its own location",
        lambda s: s["cameras"][0].update(look_at=[0, -1.4, 0.45]),
        "spec.degenerate_aim",
        "/cameras/0/look_at",
    ),
    (
        "light looking at its own location",
        lambda s: s["lights"][0].update(look_at=[-1.2, -1.5, 1.8]),
        "spec.degenerate_aim",
        "/lights/0/look_at",
    ),
    (
        "clip start beyond clip end",
        lambda s: s["cameras"][0].update(clip_start=10, clip_end=5),
        "spec.invalid_range",
        "/cameras/0/clip_start",
    ),
    (
        "clip start equal to clip end",
        lambda s: s["cameras"][0].update(clip_start=5, clip_end=5),
        "spec.invalid_range",
        "/cameras/0/clip_start",
    ),
    (
        "wrong schema string",
        lambda s: s.update(schema="jingyu.scene.v2"),
        "spec.unsupported_schema",
        "/schema",
    ),
    ("missing schema string", lambda s: s.pop("schema"), "spec.unsupported_schema", "/schema"),
    (
        "NaN in an already-parsed document",
        lambda s: s["cameras"][0].update(location=[0, float("nan"), 0.45]),
        "json.non_finite_number",
        "/cameras/0/location/1",
    ),
    (
        "infinity in an already-parsed document",
        lambda s: _vase(s).update(location=[float("inf"), 0, 0]),
        "json.non_finite_number",
        "/objects/1/location/0",
    ),
]


@pytest.mark.parametrize(
    ("mutate", "code", "pointer"),
    [pytest.param(m, c, p, id=label) for label, m, c, p in INVALID_CASES],
)
def test_invalid_scenes_report_code_and_pointer(mutate: Mutation, code: str, pointer: str) -> None:
    result = validate_scene(_mutated(mutate))
    assert not result.valid
    assert result.scene_sha256 is None
    found = [(i.code, i.pointer) for i in result.errors]
    assert (code, pointer) in found, found
    # Each case breaks one thing, so it produces exactly one error.
    assert len(found) == 1, found


@pytest.mark.parametrize(
    ("mutate", "code", "pointer"),
    [
        pytest.param(
            lambda s: s.update(lights=[], world={"strength": 0}),
            "spec.scene_unlit",
            "/lights",
            id="no lights and a dark world",
        ),
        pytest.param(
            lambda s: s.update(lights=[], world={"color": "#000000"}),
            "spec.scene_unlit",
            "/lights",
            id="no lights and a black world",
        ),
        pytest.param(
            lambda s: [o.update(visible=False) for o in s["objects"]],
            "spec.empty_scene",
            "/objects",
            id="no visible objects",
        ),
        pytest.param(
            lambda s: s.update(objects=[]), "spec.empty_scene", "/objects", id="no objects"
        ),
    ],
)
def test_warnings_do_not_invalidate_the_scene(mutate: Mutation, code: str, pointer: str) -> None:
    result = validate_scene(_mutated(mutate))
    assert result.valid
    assert result.errors == []
    assert [(w.code, w.pointer, w.severity) for w in result.warnings] == [
        (code, pointer, "warning")
    ]
    assert result.scene_sha256 is not None


def test_scene_lit_by_the_world_or_an_emitter_needs_no_light() -> None:
    assert validate_scene(_mutated(lambda s: s.update(lights=[]))).issues == ()
    emissive = _mutated(lambda s: s.update(lights=[], world={"strength": 0}))
    emissive["materials"].append({"id": "lamp", "family": "principled", "emission_strength": 5})
    assert validate_scene(emissive).issues == ()


def test_schema_issues_carry_hints() -> None:
    result = validate_scene(_mutated(lambda s: s["render"].update(engine="octane")))
    assert result.errors[0].hint == "allowed values: cycles, eevee, workbench"
    result = validate_scene(_mutated(lambda s: s["cameras"][0].update(rotation=[0, 0, 0])))
    assert result.errors[0].hint == "give exactly one of look_at or rotation"


def test_semantic_checks_report_every_problem_at_once() -> None:
    def break_twice(scene: dict[str, Any]) -> None:
        scene["render"]["camera"] = "side"
        _vase(scene)["material"] = "celadon"

    result = validate_scene(_mutated(break_twice))
    assert {i.pointer for i in result.errors} == {"/render/camera", "/objects/1/material"}


@pytest.mark.parametrize("document", [[], "scene", 3, None])
def test_non_object_documents_are_schema_violations(document: Any) -> None:
    result = validate_scene(document)
    assert [(i.code, i.pointer) for i in result.issues] == [("spec.schema_violation", "")]
    assert result.normalized is None


def test_require_valid_raises_spec_invalid_with_all_issues() -> None:
    result = validate_scene(_mutated(lambda s: s["render"].update(camera="side")))
    with pytest.raises(JingyuError) as info:
        result.require_valid()
    error = info.value
    assert error.code == "spec.invalid"
    issues = error.details["issues"]
    assert [(i["code"], i["pointer"]) for i in issues] == [
        ("spec.unknown_reference", "/render/camera")
    ]
    assert "camera 'side' is not defined" in error.message


def test_require_valid_returns_the_normalised_scene() -> None:
    result = validate_scene(minimal_scene())
    assert result.require_valid() is result.normalized


@pytest.mark.parametrize(
    ("text", "code"),
    [
        ('{"schema": "jingyu.scene.v1", "schema": "jingyu.scene.v1"}', "json.duplicate_key"),
        ('{"schema": "jingyu.scene.v1", "id": NaN}', "json.non_finite_number"),
        ('{"schema": "jingyu.scene.v1", "exposure": 1e999}', "json.non_finite_number"),
        ('{"schema": ', "json.invalid"),
    ],
)
def test_validate_scene_text_reports_json_codes(text: str, code: str) -> None:
    result = validate_scene_text(text)
    assert [i.code for i in result.issues] == [code]
    assert not result.valid


def test_validate_scene_text_accepts_a_valid_document() -> None:
    assert validate_scene_text(json.dumps(minimal_scene())).valid


def test_validate_scene_file_tolerates_a_bom(tmp_path: Path) -> None:
    path = tmp_path / "scene.json"
    path.write_bytes(b"\xef\xbb\xbf" + json.dumps(minimal_scene()).encode("utf-8"))
    assert validate_scene_file(path).valid


def test_validate_scene_file_reports_json_codes(tmp_path: Path) -> None:
    path = tmp_path / "scene.json"
    path.write_text('{"a": 1, "a": 2}', encoding="utf-8")
    assert [i.code for i in validate_scene_file(path).issues] == ["json.duplicate_key"]


def test_schema_marker_constant() -> None:
    assert minimal_scene()["schema"] == SCENE_SCHEMA == "jingyu.scene.v1"
