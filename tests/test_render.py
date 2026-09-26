"""The host side of rendering (validate, run the worker, commit), with a fake worker."""

from __future__ import annotations

import itertools
import json
import textwrap
from pathlib import Path
from typing import Any

import pytest
from helpers import script_runtime
from PIL import Image

from jingyu import render
from jingyu.candidate import CANDIDATE_ID_RE, ID_MAP_NAME, CandidateStore
from jingyu.errors import JingyuError
from jingyu.locate import IdMask
from jingyu.render import PREVIEW_MAX_SAMPLES, quality_overrides, render_scene
from jingyu.scene import minimal_scene
from jingyu.workspace import Workspace

#: Writes an image, an id mask covering the frame with object 1, and a response.
FAKE_RENDER = textwrap.dedent(
    """
    import json, sys
    from pathlib import Path
    from PIL import Image

    request = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    out = Path(request["output_dir"])
    scene = request["scene"]
    width, height = request["overrides"].get("resolution", scene["render"]["resolution"])
    Image.new("RGB", (width, height), (90, 120, 150)).save(out / "image.png")
    outputs = {"image": "image.png"}
    id_map = None
    if request["passes"]["id_mask"] and "NO_MASK" not in scene.get("notes", ""):
        Image.new("RGBA", (width, height), (0, 0, 1, 255)).save(out / "id_mask.png")
        outputs["id_mask"] = "id_mask.png"
        first = scene["objects"][0]
        id_map = {
            "schema": "jingyu.id-map.v1",
            "encoding": "rgb24-index",
            "objects": {
                "1": {
                    "object": first["id"],
                    "pointer": "/objects/0",
                    "material": first.get("material"),
                }
            },
        }
    if "NO_IMAGE_OUTPUT" in scene.get("notes", ""):
        outputs = {}
    print("fake render done")
    response = {
        "protocol": request["protocol"],
        "action": "render",
        "ok": True,
        "blender": {"version": "fake"},
        "render": {"engine": scene["render"]["engine"], "resolution": [width, height]},
        "outputs": outputs,
        "id_map": id_map,
        "timings": {"render_ms": 1, "total_ms": 2},
        "warnings": [{"code": "blender.render_failed", "message": "worker warning"}],
    }
    Path(sys.argv[2]).write_text(json.dumps(response), encoding="utf-8")
    """
)


@pytest.fixture(autouse=True)
def sequential_candidate_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    # Real ids carry 16 random bits per second, so two quick renders of one scene
    # could collide; tests use sequential ids in the same format instead.
    counter = itertools.count(1)

    def new_id(scene_sha256: str, now: object = None) -> str:
        n = next(counter)
        return f"c_20260926T1200{n:02d}Z_{scene_sha256[:8]}_{n:04x}"

    monkeypatch.setattr(render, "new_candidate_id", new_id)


def _render(workspace: Workspace, scene: Any, **kwargs: Any) -> Any:
    return render_scene(workspace, scene, runtime=script_runtime(FAKE_RENDER), **kwargs)


def test_a_colliding_candidate_id_is_drawn_again(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    first_id, second_id = "c_20260926T120000Z_0123abcd_0001", "c_20260926T120000Z_0123abcd_0002"
    drawn = iter([first_id, first_id, second_id])
    monkeypatch.setattr(render, "new_candidate_id", lambda sha, now=None: next(drawn))
    assert _render(workspace, minimal_scene()).candidate.id == first_id
    assert _render(workspace, minimal_scene()).candidate.id == second_id


def test_quality_overrides() -> None:
    render = {"resolution": [1280, 721], "samples": 256}
    assert quality_overrides(render, "final") == {}
    assert quality_overrides(render, "preview") == {
        "resolution": [640, 360],
        "samples": PREVIEW_MAX_SAMPLES,
    }
    assert quality_overrides({"resolution": [20, 16], "samples": 4}, "preview") == {
        "resolution": [16, 16],
        "samples": 4,
    }


def test_render_commits_a_candidate(workspace: Workspace) -> None:
    scene = minimal_scene()
    scene["lights"] = []
    scene["world"] = {"strength": 0}
    outcome = _render(workspace, scene)
    candidate = outcome.candidate
    assert CandidateStore(workspace).verify(candidate) == []
    assert set(candidate.receipt["files"]) == {
        "image.png",
        "id_mask.png",
        "id_map.json",
        "scene.json",
        "blender.log",
    }
    receipt = candidate.receipt
    assert receipt["quality"] == "preview"
    assert receipt["scene"]["id"] == "minimal-vase"
    assert receipt["render"]["resolution"] == [480, 360]
    assert receipt["blender"]["runtime"]["source"] == "test-script"
    assert receipt["timings_ms"]["render"] == 1
    assert receipt["timings_ms"]["host_total"] >= 0
    # Scene warnings come first, then checks on the frame, then the worker's.
    codes = [w["code"] for w in receipt["warnings"]]
    assert codes[0] == "spec.scene_unlit"
    assert codes[-1] == "blender.render_failed"
    assert set(codes[1:-1]) <= {"frame.not_visible", "frame.underexposed"}
    assert [w.code for w in outcome.warnings] == codes[:-1]
    # The committed scene is the normalised one.
    assert candidate.scene()["render"]["engine"] == "cycles"
    assert "fake render done" in candidate.file("blender.log").read_text(encoding="utf-8")
    with Image.open(candidate.image_path) as image:
        assert image.size == (480, 360)
    assert IdMask.load(candidate).locate_point({"x": 0, "y": 0})["hit"]["object"] == "floor"
    assert list(workspace.staging_dir.iterdir()) == []


def test_a_style_paints_the_image_and_keeps_the_render(workspace: Workspace) -> None:
    from jingyu.views import render_view

    scene = minimal_scene()
    scene["render"]["style"] = {"preset": "ink"}
    candidate = _render(workspace, scene).candidate
    assert CandidateStore(workspace).verify(candidate) == []
    assert "render.png" in candidate.receipt["files"]
    with (
        Image.open(candidate.image_path) as styled,
        Image.open(candidate.file("render.png")) as photo,
    ):
        assert styled.size == photo.size
        assert styled.tobytes() != photo.convert(styled.mode).tobytes()
    assert render_view(candidate, "render").png != render_view(candidate, "full").png
    plain = _render(workspace, minimal_scene()).candidate
    assert "render.png" not in plain.receipt["files"]


def test_final_quality_keeps_the_scene_resolution(workspace: Workspace) -> None:
    scene = minimal_scene()
    scene["render"]["resolution"] = [64, 48]
    candidate = _render(workspace, scene, quality="final").candidate
    assert candidate.receipt["quality"] == "final"
    assert candidate.receipt["render"]["resolution"] == [64, 48]


def test_each_render_is_a_new_candidate(workspace: Workspace) -> None:
    first = _render(workspace, minimal_scene()).candidate
    second = _render(workspace, minimal_scene()).candidate
    assert first.id != second.id
    assert all(CANDIDATE_ID_RE.fullmatch(c.id) for c in (first, second))
    assert first.receipt["scene"]["sha256"] == second.receipt["scene"]["sha256"]
    assert [c.id for c in CandidateStore(workspace).list()] == [second.id, first.id]


def test_a_render_without_a_mask_has_no_id_map(workspace: Workspace) -> None:
    scene = minimal_scene()
    scene["notes"] = "NO_MASK"
    candidate = _render(workspace, scene).candidate
    assert not candidate.has_id_mask
    assert ID_MAP_NAME not in candidate.receipt["files"]


def test_invalid_scenes_never_reach_the_worker(workspace: Workspace) -> None:
    scene = minimal_scene()
    scene["render"]["camera"] = "side"
    with pytest.raises(JingyuError) as info:
        render_scene(workspace, scene, runtime=script_runtime("raise SystemExit(99)"))
    assert info.value.code == "spec.invalid"
    assert not workspace.candidates_dir.exists()


def test_missing_outputs_are_bad_responses_and_leave_no_trace(workspace: Workspace) -> None:
    scene = minimal_scene()
    scene["notes"] = "NO_IMAGE_OUTPUT"
    with pytest.raises(JingyuError) as info:
        _render(workspace, scene)
    assert info.value.code == "blender.bad_response"
    assert list(workspace.staging_dir.iterdir()) == []
    assert CandidateStore(workspace).list() == []


def test_worker_failures_propagate_and_leave_no_trace(workspace: Workspace) -> None:
    failing = (
        "import json, sys\n"
        "error = {'code': 'blender.gpu_unavailable', 'message': 'no GPU'}\n"
        "json.dump({'protocol': 'jingyu.worker.v1', 'ok': False, 'error': error},"
        " open(sys.argv[2], 'w'))\n"
    )
    with pytest.raises(JingyuError) as info:
        render_scene(workspace, minimal_scene(), runtime=script_runtime(failing))
    assert info.value.code == "blender.gpu_unavailable"
    assert list(workspace.staging_dir.iterdir()) == []


def test_the_worker_receives_a_normalised_request(workspace: Workspace, tmp_path: Path) -> None:
    capture = tmp_path / "request.json"
    script = (
        f"import shutil, sys\nshutil.copy(sys.argv[1], {str(capture)!r})\nraise SystemExit(1)\n"
    )
    with pytest.raises(JingyuError):
        render_scene(workspace, minimal_scene(), runtime=script_runtime(script))
    request = json.loads(capture.read_text(encoding="utf-8"))
    assert request["protocol"] == "jingyu.worker.v1"
    assert request["action"] == "render"
    assert request["passes"] == {"id_mask": True, "light": True}
    assert request["overrides"] == {"resolution": [480, 360], "samples": 16}
    assert request["scene"]["objects"][1]["geometry"]["segments"] == 96
