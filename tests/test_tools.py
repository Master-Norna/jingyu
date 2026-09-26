from __future__ import annotations

import io
import json
import re
from typing import Any

import pytest
from helpers import CandidateFactory, synthetic_scene_document
from jsonschema import Draft202012Validator
from PIL import Image

from jingyu import constitution, doctor
from jingyu.candidate import Candidate
from jingyu.environments import ENVIRONMENTS
from jingyu.errors import ERROR_CODES, JingyuError
from jingyu.geometry import GEOMETRY
from jingyu.materials import MATERIALS
from jingyu.scene import minimal_scene, scene_schema
from jingyu.tools import REGISTRY, Tool, ToolContext, ToolRegistry, ToolResult
from jingyu.views import VIEWS
from jingyu.workspace import Workspace

TOOLS = [pytest.param(tool, id=tool.name) for tool in REGISTRY]


@pytest.fixture
def ctx(workspace: Workspace) -> ToolContext:
    return ToolContext(workspace)


def _invoke(ctx: ToolContext, name: str, arguments: dict[str, Any] | None = None) -> ToolResult:
    return REGISTRY.invoke(name, arguments, ctx)


def _code(ctx: ToolContext, name: str, arguments: dict[str, Any] | None = None) -> str:
    with pytest.raises(JingyuError) as info:
        _invoke(ctx, name, arguments)
    return info.value.code


# ------------------------------------------------------------- the registry


@pytest.mark.parametrize("tool", TOOLS)
def test_tool_schemas_are_valid_draft_2020_12(tool: Tool) -> None:
    for schema in (tool.input_schema, tool.output_schema):
        Draft202012Validator.check_schema(schema)
        assert schema["type"] == "object"
    assert tool.input_schema["additionalProperties"] is False


@pytest.mark.parametrize("tool", TOOLS)
def test_tools_are_documented(tool: Tool) -> None:
    assert tool.title.strip()
    assert tool.description.strip()


def test_tool_names_are_unique_snake_case() -> None:
    names = REGISTRY.names()
    assert len(names) == len(set(names))
    assert all(re.fullmatch(r"[a-z][a-z0-9]*(_[a-z0-9]+)*", name) for name in names), names
    assert [t.name for t in REGISTRY] == names


def test_only_render_scene_writes() -> None:
    writers = [t.name for t in REGISTRY if not t.read_only]
    assert writers == ["render_scene"]
    assert not any(t.destructive for t in REGISTRY)
    assert {t.name for t in REGISTRY if t.returns_images} == {"render_scene", "view_candidate"}


def test_unknown_tool(ctx: ToolContext) -> None:
    with pytest.raises(JingyuError) as info:
        _invoke(ctx, "draw_a_cat")
    assert info.value.code == "tool.unknown"
    assert info.value.hint and "get_guide" in info.value.hint


@pytest.mark.parametrize(
    ("name", "arguments", "pointer"),
    [
        ("get_guide", {"verbose": True}, ""),
        ("list_generators", {"kind": "lights"}, "/kind"),
        ("validate_scene", {}, ""),
        ("validate_scene", {"scene": minimal_scene(), "scene_path": "a.json"}, ""),
        ("validate_scene", {"scene": "not an object"}, "/scene"),
        ("list_candidates", {"limit": 0}, "/limit"),
        ("get_candidate", {}, ""),
        ("view_candidate", {"candidate_id": "c_x", "view": "sideways"}, "/view"),
        ("locate_in_candidate", {"candidate_id": "c_x"}, ""),
        ("locate_in_candidate", {"candidate_id": "c_x", "point": {"x": -1, "y": 0}}, "/point/x"),
        ("get_constitution", {"ids": ["J1", "J1"]}, "/ids"),
        ("get_constitution", {"ids": ["X1"]}, "/ids/0"),
        ("render_scene", {"scene": minimal_scene(), "quality": "best"}, "/quality"),
    ],
)
def test_bad_arguments_are_rejected_before_the_handler(
    ctx: ToolContext, name: str, arguments: dict[str, Any], pointer: str
) -> None:
    with pytest.raises(JingyuError) as info:
        _invoke(ctx, name, arguments)
    assert info.value.code == "tool.invalid_arguments"
    assert pointer in [e["pointer"] for e in info.value.details["errors"]]


def test_more_than_twelve_constitution_ids_are_rejected(ctx: ToolContext) -> None:
    ids = [f"J1.{n:02d}" for n in range(1, constitution.MAX_IDS + 2)]
    assert _code(ctx, "get_constitution", {"ids": ids}) == "tool.invalid_arguments"


def _toy_registry(handler: Any) -> ToolRegistry:
    registry = ToolRegistry()
    registry.add(
        Tool(
            name="toy",
            title="Toy",
            description="A toy tool.",
            input_schema={"type": "object", "additionalProperties": False, "properties": {}},
            output_schema={
                "type": "object",
                "required": ["n"],
                "properties": {"n": {"type": "integer"}},
            },
            handler=handler,
        )
    )
    return registry


def test_output_is_checked_against_the_output_schema(ctx: ToolContext) -> None:
    good = _toy_registry(lambda c, a: ToolResult({"n": 1}))
    assert good.invoke("toy", None, ctx).data == {"n": 1}
    bad = _toy_registry(lambda c, a: ToolResult({"n": "one"}))
    with pytest.raises(JingyuError) as info:
        bad.invoke("toy", {}, ctx)
    assert info.value.code == "tool.invalid_output"
    assert "/n" in info.value.message


def test_registry_add_enforces_its_rules() -> None:
    registry = _toy_registry(lambda c, a: ToolResult({"n": 1}))
    base = registry.get("toy")
    with pytest.raises(ValueError, match="registered twice"):
        registry.add(base)
    for name in ("Toy", "t", "toy-tool", "1toy"):
        with pytest.raises(ValueError, match="invalid tool name"):
            registry.add(Tool(**{**base.__dict__, "name": name}))
    open_input = {"type": "object", "properties": {}}
    with pytest.raises(ValueError, match="additionalProperties false"):
        registry.add(Tool(**{**base.__dict__, "name": "other", "input_schema": open_input}))
    with pytest.raises(ValueError, match="must describe an object"):
        registry.add(Tool(**{**base.__dict__, "name": "other", "output_schema": {"type": "array"}}))


# ------------------------------------------------- tools that need no Blender


def test_get_guide(ctx: ToolContext) -> None:
    data = _invoke(ctx, "get_guide").data
    assert data["workflow"]
    assert data["conventions"]["up_axis"] == "Z"
    assert [t["name"] for t in data["tools"]] == REGISTRY.names()
    assert data["workspace"] == str(ctx.workspace.root)


def test_get_scene_schema(ctx: ToolContext) -> None:
    data = _invoke(ctx, "get_scene_schema").data
    assert data["schema_id"] == "jingyu.scene.v1"
    assert data["json_schema"] == scene_schema()
    assert data["example"] == minimal_scene()


@pytest.mark.parametrize(
    ("kind", "keys"),
    [
        ("all", {"geometry", "materials", "environments"}),
        ("geometry", {"geometry"}),
        ("materials", {"materials"}),
        ("environments", {"environments"}),
    ],
)
def test_list_generators(ctx: ToolContext, kind: str, keys: set[str]) -> None:
    data = _invoke(ctx, "list_generators", {"kind": kind}).data
    assert set(data) == keys
    if "geometry" in data:
        assert [g["name"] for g in data["geometry"]] == GEOMETRY.names()
    if "materials" in data:
        assert [m["name"] for m in data["materials"]] == MATERIALS.names()
    if "environments" in data:
        assert [e["name"] for e in data["environments"]] == ENVIRONMENTS.names()
    assert set(_invoke(ctx, "list_generators").data) == {"geometry", "materials", "environments"}


def test_validate_scene_inline(ctx: ToolContext) -> None:
    data = _invoke(ctx, "validate_scene", {"scene": minimal_scene()}).data
    assert data["valid"] is True
    assert data["issues"] == []
    assert len(data["scene_sha256"]) == 64
    assert "normalized" not in data
    full = _invoke(ctx, "validate_scene", {"scene": minimal_scene(), "include_normalized": True})
    assert full.data["normalized"]["render"]["engine"] == "cycles"


def test_validate_scene_reports_issues(ctx: ToolContext) -> None:
    scene = minimal_scene()
    scene["render"]["camera"] = "side"
    data = _invoke(ctx, "validate_scene", {"scene": scene, "include_normalized": True}).data
    assert data["valid"] is False
    assert data["scene_sha256"] is None
    assert [(i["code"], i["pointer"]) for i in data["issues"]] == [
        ("spec.unknown_reference", "/render/camera")
    ]
    # An invalid scene still has a normalised form to inspect.
    assert "normalized" in data


def test_validate_scene_from_a_workspace_path(ctx: ToolContext) -> None:
    (ctx.workspace.root / "scenes").mkdir()
    (ctx.workspace.root / "scenes" / "a.json").write_text(
        json.dumps(minimal_scene()), encoding="utf-8"
    )
    assert _invoke(ctx, "validate_scene", {"scene_path": "scenes/a.json"}).data["valid"]
    assert _code(ctx, "validate_scene", {"scene_path": "../a.json"}) == "workspace.path_escape"
    assert _code(ctx, "validate_scene", {"scene_path": "nope.json"}) == "workspace.not_found"
    (ctx.workspace.root / "dup.json").write_text('{"a": 1, "a": 2}', encoding="utf-8")
    assert _code(ctx, "validate_scene", {"scene_path": "dup.json"}) == "json.duplicate_key"
    assert _code(ctx, "validate_scene", {"scene_path": "scenes"}) == "io.unreadable"


def test_list_candidates(ctx: ToolContext, make_candidate: CandidateFactory) -> None:
    assert _invoke(ctx, "list_candidates").data == {"candidates": []}
    first, second = make_candidate(), make_candidate()
    listed = _invoke(ctx, "list_candidates").data["candidates"]
    assert [c["candidate_id"] for c in listed] == [second.id, first.id]
    assert listed[0] == second.summary()
    limited = _invoke(ctx, "list_candidates", {"limit": 1}).data["candidates"]
    assert [c["candidate_id"] for c in limited] == [second.id]


def test_get_constitution(ctx: ToolContext) -> None:
    index = _invoke(ctx, "get_constitution").data
    assert index["version"] == constitution.VERSION
    assert "entries" not in index
    assert index["index"]["clauses"]
    entries = _invoke(ctx, "get_constitution", {"ids": ["J1.01", "J7.1"]}).data["entries"]
    assert [e["id"] for e in entries] == ["J1.01", "J7.1"]
    assert _code(ctx, "get_constitution", {"ids": ["J99.99"]}) == "constitution.unknown_clause"


def test_list_error_codes(ctx: ToolContext) -> None:
    codes = _invoke(ctx, "list_error_codes").data["codes"]
    assert {c["code"]: c["meaning"] for c in codes} == ERROR_CODES


def test_diagnose_environment_reports_a_missing_blender(
    ctx: ToolContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    def not_found() -> None:
        raise JingyuError("blender.not_found", "no Blender runtime found", hint="install it")

    monkeypatch.setattr(doctor, "discover_runtime", not_found)
    data = _invoke(ctx, "diagnose_environment").data
    assert data["ok"] is False
    assert data["runtime"] is None
    assert [p["code"] for p in data["problems"]] == ["blender.not_found"]
    assert data["dependencies"]["jsonschema"]


# ------------------------------------- candidate tools on a synthetic candidate


def test_get_candidate(ctx: ToolContext, candidate: Candidate) -> None:
    data = _invoke(ctx, "get_candidate", {"candidate_id": candidate.id}).data
    assert data["path"] == f"candidates/{candidate.id}"
    assert data["receipt"] == candidate.receipt
    assert data["integrity"] == {"ok": True, "problems": []}
    assert _code(ctx, "get_candidate", {"candidate_id": "nope"}) == "candidate.invalid_id"
    unknown = "c_20260101T000000Z_00000000_0000"
    assert _code(ctx, "get_candidate", {"candidate_id": unknown}) == "candidate.not_found"


@pytest.mark.parametrize("view", [v for v in VIEWS if v != "compare"])
def test_view_candidate_returns_an_image(ctx: ToolContext, candidate: Candidate, view: str) -> None:
    result = _invoke(ctx, "view_candidate", {"candidate_id": candidate.id, "view": view})
    assert result.data["description"] == VIEWS[view]
    (image,) = result.images
    assert image.mime_type == "image/png"
    assert view in image.caption
    with Image.open(io.BytesIO(image.png)) as decoded:
        assert decoded.size == (result.data["width"], result.data["height"])


def test_view_candidate_compare(ctx: ToolContext, make_candidate: CandidateFactory) -> None:
    left, right = make_candidate(), make_candidate()
    arguments = {"candidate_id": left.id, "view": "compare"}
    assert _code(ctx, "view_candidate", arguments) == "tool.invalid_arguments"
    result = _invoke(ctx, "view_candidate", {**arguments, "other_candidate_id": right.id})
    assert result.data["meta"]["right"] == right.id


def test_locate_in_candidate(ctx: ToolContext, candidate: Candidate) -> None:
    point = _invoke(
        ctx, "locate_in_candidate", {"candidate_id": candidate.id, "point": {"x": 40, "y": 20}}
    ).data
    assert point["mode"] == "point"
    assert point["result"]["hit"]["pointer"] == "/objects/1"
    region = _invoke(
        ctx,
        "locate_in_candidate",
        {"candidate_id": candidate.id, "region": {"u0": 0, "v0": 0, "u1": 1, "v1": 1}},
    ).data
    assert region["mode"] == "region"
    assert region["result"]["objects"][0]["object"] == "floor"
    both = {"candidate_id": candidate.id, "point": {"x": 1, "y": 1}, "region": {"x0": 0}}
    assert _code(ctx, "locate_in_candidate", both) == "tool.invalid_arguments"
    outside = {"candidate_id": candidate.id, "point": {"x": 500, "y": 1}}
    assert _code(ctx, "locate_in_candidate", outside) == "locate.out_of_bounds"


def test_describe_layout(ctx: ToolContext, candidate: Candidate) -> None:
    layout = _invoke(ctx, "describe_layout", {"candidate_id": candidate.id}).data["layout"]
    assert [o["object"] for o in layout["objects"]] == ["floor", "vase", "crate"]
    assert layout["not_in_frame"] == ["far"]


# ------------------------------------ render_scene failures before any Blender


def test_render_scene_rejects_invalid_scenes_before_rendering(ctx: ToolContext) -> None:
    scene = minimal_scene()
    scene["render"]["camera"] = "side"
    with pytest.raises(JingyuError) as info:
        _invoke(ctx, "render_scene", {"scene": scene})
    assert info.value.code == "spec.invalid"
    assert info.value.details["issues"][0]["pointer"] == "/render/camera"
    assert not ctx.workspace.candidates_dir.exists()


def test_render_scene_rejects_hiding_unknown_objects(ctx: ToolContext) -> None:
    arguments = {"scene": synthetic_scene_document(), "hide": ["vase", "unicorn"]}
    with pytest.raises(JingyuError) as info:
        _invoke(ctx, "render_scene", arguments)
    assert info.value.code == "spec.unknown_reference"
    assert "unicorn" in info.value.message
    assert not ctx.workspace.candidates_dir.exists()
