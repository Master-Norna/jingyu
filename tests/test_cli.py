from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from helpers import CandidateFactory

from jingyu import __version__, doctor
from jingyu.candidate import Candidate
from jingyu.cli import main
from jingyu.errors import JingyuError
from jingyu.scene import minimal_scene, scene_schema
from jingyu.tools import REGISTRY
from jingyu.workspace import Workspace


def _run(capsys: pytest.CaptureFixture[str], workspace: Workspace, *argv: str) -> tuple[int, Any]:
    code = main(["--workspace", str(workspace.root), *argv])
    out = capsys.readouterr().out
    return code, json.loads(out)


def _scene_file(tmp_path: Path, scene: Any, name: str = "scene.json") -> str:
    path = tmp_path / name
    path.write_text(json.dumps(scene), encoding="utf-8")
    return str(path)


def test_tools_lists_the_registry(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["tools"]) == 0
    listed = json.loads(capsys.readouterr().out)
    assert [t["name"] for t in listed] == REGISTRY.names()
    assert all({"name", "title", "description", "arguments"} <= set(t) for t in listed)
    render = next(t for t in listed if t["name"] == "render_scene")
    assert render["arguments"]["quality"] == '"preview" | "final" = "preview"'
    assert render["arguments"]["hide"] == "array of string = []"
    assert render["exactly_one_of"] == ["scene", "scene_path", "from_candidate"]
    locate = next(t for t in listed if t["name"] == "locate_in_candidate")
    assert locate["arguments"]["candidate_id"] == "string (required)"


def test_tools_with_a_name_prints_its_schemas(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["tools", "view_candidate"]) == 0
    shown = json.loads(capsys.readouterr().out)
    tool = REGISTRY.get("view_candidate")
    assert shown["input_schema"] == tool.input_schema
    assert shown["output_schema"] == tool.output_schema


def test_tools_with_an_unknown_name_fails(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["tools", "no_such_tool"]) == 1
    assert json.loads(capsys.readouterr().out)["error"]["code"] == "tool.unknown"


def test_schema_prints_the_scene_schema(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["schema"]) == 0
    assert json.loads(capsys.readouterr().out) == scene_schema()


def test_schema_write_matches_the_exported_format(tmp_path: Path) -> None:
    target = tmp_path / "scene.schema.json"
    assert main(["schema", "--write", str(target)]) == 0
    expected = json.dumps(scene_schema(), ensure_ascii=False, indent=2) + "\n"
    assert target.read_text(encoding="utf-8") == expected


def test_validate_a_good_scene(
    capsys: pytest.CaptureFixture[str], workspace: Workspace, tmp_path: Path
) -> None:
    code, data = _run(capsys, workspace, "validate", _scene_file(tmp_path, minimal_scene()))
    assert code == 0
    assert data["valid"] is True
    assert "normalized" not in data
    code, data = _run(
        capsys, workspace, "validate", _scene_file(tmp_path, minimal_scene()), "--normalized"
    )
    assert code == 0
    assert data["normalized"]["world"]["color"] == "#404040"


def test_validate_a_bad_scene(
    capsys: pytest.CaptureFixture[str], workspace: Workspace, tmp_path: Path
) -> None:
    scene = minimal_scene()
    scene["cameras"][0]["clip_start"] = 1000
    code, data = _run(capsys, workspace, "validate", _scene_file(tmp_path, scene))
    assert code == 1
    assert data["valid"] is False
    assert [(i["code"], i["pointer"]) for i in data["issues"]] == [
        ("spec.invalid_range", "/cameras/0/clip_start")
    ]


def test_validate_reports_json_errors(
    capsys: pytest.CaptureFixture[str], workspace: Workspace, tmp_path: Path
) -> None:
    path = tmp_path / "dup.json"
    path.write_text('{"schema": 1, "schema": 2}', encoding="utf-8")
    code, data = _run(capsys, workspace, "validate", str(path))
    assert code == 1
    assert data == {
        "error": {
            "code": "json.duplicate_key",
            "message": "duplicate JSON object key 'schema'",
            "hint": "Each key may appear once per object.",
        }
    }


@pytest.mark.parametrize("argv", [["validate", "{missing}"], ["render", "{missing}"]])
def test_missing_scene_files_are_unreadable(
    capsys: pytest.CaptureFixture[str], workspace: Workspace, tmp_path: Path, argv: list[str]
) -> None:
    missing = str(tmp_path / "missing.json")
    code, data = _run(capsys, workspace, *(missing if a == "{missing}" else a for a in argv))
    assert code == 1
    assert data["error"]["code"] == "io.unreadable"


def test_non_utf8_scene_files_are_invalid_json(
    capsys: pytest.CaptureFixture[str], workspace: Workspace, tmp_path: Path
) -> None:
    path = tmp_path / "latin1.json"
    path.write_bytes(b'{"title": "caf\xe9"}')
    code, data = _run(capsys, workspace, "validate", str(path))
    assert code == 1
    assert data["error"]["code"] == "json.invalid"


def test_call_get_guide(capsys: pytest.CaptureFixture[str], workspace: Workspace) -> None:
    code, data = _run(capsys, workspace, "call", "get_guide")
    assert code == 0
    assert data["workspace"] == str(workspace.root)
    assert data["jingyu_version"] == __version__


def test_call_with_json_arguments(
    capsys: pytest.CaptureFixture[str], workspace: Workspace, tmp_path: Path
) -> None:
    arguments = json.dumps({"ids": ["J1.01"]})
    code, data = _run(capsys, workspace, "call", "get_constitution", "--args", arguments)
    assert code == 0
    assert [e["id"] for e in data["entries"]] == ["J1.01"]
    args_file = tmp_path / "args.json"
    args_file.write_text(json.dumps({"kind": "materials"}), encoding="utf-8")
    code, data = _run(capsys, workspace, "call", "list_generators", "--args", f"@{args_file}")
    assert code == 0
    assert set(data) == {"materials"}


@pytest.mark.parametrize(
    ("raw", "code"),
    [
        ("{not json", "json.invalid"),
        ('{"kind": "all", "kind": "all"}', "json.duplicate_key"),
        ("[1, 2]", "tool.invalid_arguments"),
        ('{"kind": "everything"}', "tool.invalid_arguments"),
    ],
)
def test_call_with_bad_arguments(
    capsys: pytest.CaptureFixture[str], workspace: Workspace, raw: str, code: str
) -> None:
    exit_code, data = _run(capsys, workspace, "call", "list_generators", "--args", raw)
    assert exit_code == 1
    assert data["error"]["code"] == code


def test_call_with_a_missing_arguments_file(
    capsys: pytest.CaptureFixture[str], workspace: Workspace, tmp_path: Path
) -> None:
    missing = tmp_path / "missing.json"
    code, data = _run(capsys, workspace, "call", "list_generators", "--args", f"@{missing}")
    assert code == 1
    assert data["error"]["code"] == "io.unreadable"


def test_call_an_unknown_tool(capsys: pytest.CaptureFixture[str], workspace: Workspace) -> None:
    code, data = _run(capsys, workspace, "call", "paint")
    assert code == 1
    assert data["error"]["code"] == "tool.unknown"


def test_render_reports_invalid_scenes_without_blender(
    capsys: pytest.CaptureFixture[str], workspace: Workspace, tmp_path: Path
) -> None:
    scene = minimal_scene()
    scene["objects"][1]["material"] = "celadon"
    code, data = _run(capsys, workspace, "render", _scene_file(tmp_path, scene))
    assert code == 1
    assert data["error"]["code"] == "spec.invalid"


def test_view_writes_the_image(
    capsys: pytest.CaptureFixture[str], workspace: Workspace, candidate: Candidate, tmp_path: Path
) -> None:
    image_dir = tmp_path / "views"
    code, data = _run(
        capsys, workspace, "--image-dir", str(image_dir), "view", candidate.id, "grayscale"
    )
    assert code == 0
    (path,) = data["images"]
    assert Path(path) == image_dir / f"{candidate.id}-grayscale.png"
    assert Path(path).read_bytes().startswith(b"\x89PNG")


def test_call_names_cropped_views_by_their_region_in_reading_order(
    capsys: pytest.CaptureFixture[str], workspace: Workspace, candidate: Candidate
) -> None:
    arguments = {"candidate_id": candidate.id, "view": "full"}
    for region, suffix in (
        ({"x0": 10, "y0": 5, "x1": 40, "y1": 30}, "crop-10-5-40-30"),
        ({"u0": 0.1, "v0": 0.2, "u1": 0.5, "v1": 0.6}, "crop-0.1-0.2-0.5-0.6"),
    ):
        code, data = _run(
            capsys,
            workspace,
            "call",
            "view_candidate",
            "--args",
            json.dumps({**arguments, "region": region}),
        )
        assert code == 0
        assert Path(data["images"][0]).name == f"{candidate.id}-full-{suffix}.png"


def test_view_defaults_to_the_workspace_views_directory(
    capsys: pytest.CaptureFixture[str], workspace: Workspace, make_candidate: CandidateFactory
) -> None:
    left, right = make_candidate(), make_candidate()
    code, data = _run(capsys, workspace, "view", left.id, "compare", "--other", right.id)
    assert code == 0
    assert Path(data["images"][0]).parent == workspace.root / "views"


def test_locate_and_layout(
    capsys: pytest.CaptureFixture[str], workspace: Workspace, candidate: Candidate
) -> None:
    code, data = _run(capsys, workspace, "locate", candidate.id, "--xy", "40", "20")
    assert code == 0
    assert data["result"]["hit"]["object"] == "vase"
    code, data = _run(capsys, workspace, "locate", candidate.id, "--uv", "0.5", "0.9")
    assert data["result"]["hit"]["object"] == "floor"
    code, data = _run(capsys, workspace, "locate", candidate.id, "--box", "0", "0", "80", "60")
    assert data["mode"] == "region"
    code, data = _run(capsys, workspace, "layout", candidate.id)
    assert code == 0
    assert data["layout"]["not_in_frame"] == ["far"]
    code, data = _run(capsys, workspace, "locate", candidate.id, "--xy", "999", "0")
    assert code == 1
    assert data["error"]["code"] == "locate.out_of_bounds"


def test_doctor_exits_1_when_blender_is_missing(
    capsys: pytest.CaptureFixture[str], workspace: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    def not_found() -> None:
        raise JingyuError("blender.not_found", "no Blender runtime found")

    monkeypatch.setattr(doctor, "discover_runtime", not_found)
    code, data = _run(capsys, workspace, "doctor")
    assert code == 1
    assert data["ok"] is False


def test_version_and_usage_errors(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as info:
        main(["--version"])
    assert info.value.code == 0
    assert capsys.readouterr().out.strip() == f"jingyu {__version__}"
    with pytest.raises(SystemExit) as info:
        main([])
    assert info.value.code == 2
    with pytest.raises(SystemExit) as info:
        main(["locate", "c_x"])
    assert info.value.code == 2
