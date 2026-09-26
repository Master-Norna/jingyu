"""Integration tests against a real Blender runtime (executable or bpy module).

They are skipped when no runtime is found.  A runtime that is found but broken
makes them fail: a skip must never hide a broken Blender.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from PIL import Image

from jingyu import doctor
from jingyu.bridge import BlenderRuntime, discover_runtime, run_worker
from jingyu.candidate import ID_MAP_NAME, ID_MASK_NAME, IMAGE_NAME, SCENE_NAME, CandidateStore
from jingyu.canonical_json import load_file_strict
from jingyu.errors import JingyuError
from jingyu.locate import IdMask
from jingyu.render import RenderOutcome, render_scene
from jingyu.scene import minimal_scene
from jingyu.tools import REGISTRY, ToolContext
from jingyu.views import render_view
from jingyu.workspace import Workspace

pytestmark = pytest.mark.blender

TIMEOUT_S = 600.0
#: minimal_scene renders at 960 x 720; preview quality halves it.
PREVIEW_SIZE = (480, 360)
CENTRE = {"u0": 0.45, "v0": 0.4, "u1": 0.55, "v1": 0.6}


@pytest.fixture(scope="module")
def runtime() -> BlenderRuntime:
    try:
        return discover_runtime()
    except JingyuError as exc:
        assert exc.code == "blender.not_found"
        pytest.skip(f"{exc.message}: {exc.hint}")


@pytest.fixture(scope="module")
def blender_workspace(tmp_path_factory: pytest.TempPathFactory) -> Workspace:
    return Workspace.at(tmp_path_factory.mktemp("blender-workspace"))


@pytest.fixture(scope="module")
def rendered(runtime: BlenderRuntime, blender_workspace: Workspace) -> RenderOutcome:
    return render_scene(
        blender_workspace, minimal_scene(), quality="preview", runtime=runtime, timeout_s=TIMEOUT_S
    )


def test_worker_probe_reports_a_supported_blender(runtime: BlenderRuntime, tmp_path: Path) -> None:
    run = run_worker(runtime, {"action": "probe"}, tmp_path, timeout_s=TIMEOUT_S)
    assert run.exit_code == 0
    assert run.log_path.is_file()
    response = run.response
    assert response["ok"] is True
    assert tuple(response["blender"]["version_tuple"]) >= (4, 2, 0)
    assert "cycles" in response["engines_registered"]
    assert isinstance(response["gpu_devices"], list)


def test_worker_reports_bad_requests_with_a_code(runtime: BlenderRuntime, tmp_path: Path) -> None:
    with pytest.raises(JingyuError) as info:
        run_worker(runtime, {"action": "explode"}, tmp_path, timeout_s=TIMEOUT_S)
    assert info.value.code == "worker.bad_request"
    assert "log_tail" in info.value.details


def test_doctor_finds_the_runtime(runtime: BlenderRuntime) -> None:
    report = doctor.diagnose(timeout_s=TIMEOUT_S)
    assert report["ok"] is True, report["problems"]
    assert report["runtime"] == runtime.to_dict()
    assert report["blender"]["engines_registered"]


def test_render_commits_a_verified_candidate(
    rendered: RenderOutcome, blender_workspace: Workspace
) -> None:
    candidate = rendered.candidate
    store = CandidateStore(blender_workspace)
    assert store.verify(candidate) == []
    assert store.open(candidate.id) == candidate
    assert set(candidate.receipt["files"]) == {
        IMAGE_NAME,
        ID_MASK_NAME,
        ID_MAP_NAME,
        SCENE_NAME,
        "blender.log",
    }
    receipt = candidate.receipt
    assert receipt["quality"] == "preview"
    assert receipt["render"]["engine"] == "cycles"
    assert receipt["render"]["resolution"] == list(PREVIEW_SIZE)
    assert receipt["render"]["samples"] == 16
    assert receipt["warnings"] == []
    assert rendered.warnings == ()
    # Staging is cleaned up by the atomic publish.
    assert list(blender_workspace.staging_dir.iterdir()) == []


def test_image_and_id_mask_have_the_preview_size(rendered: RenderOutcome) -> None:
    for name in (IMAGE_NAME, ID_MASK_NAME):
        with Image.open(rendered.candidate.file(name)) as image:
            assert image.size == PREVIEW_SIZE, name
    with Image.open(rendered.candidate.file(ID_MASK_NAME)) as mask:
        assert mask.mode == "RGBA"


def test_id_mask_decodes_exactly_and_finds_the_vase(rendered: RenderOutcome) -> None:
    id_map = load_file_strict(rendered.candidate.file(ID_MAP_NAME))
    assert id_map["objects"] == {
        "1": {"object": "floor", "pointer": "/objects/0", "material": "floor"},
        "2": {"object": "vase", "pointer": "/objects/1", "material": "glaze"},
    }
    mask = IdMask.load(rendered.candidate)
    layout = mask.layout()
    assert layout["unassigned_pixels"] == 0
    assert layout["not_in_frame"] == []
    assert {o["object"] for o in layout["objects"]} == {"floor", "vase"}
    centre = mask.locate_region(CENTRE)
    assert centre["objects"][0]["object"] == "vase"
    assert centre["objects"][0]["material_pointer"] == "/materials/0"
    assert mask.locate_point({"u": 0.5, "v": 0.5})["hit"]["pointer"] == "/objects/1"


def test_hide_removes_an_object_from_the_render(
    rendered: RenderOutcome, blender_workspace: Workspace
) -> None:
    context = ToolContext(blender_workspace, render_timeout_s=TIMEOUT_S)
    data = REGISTRY.invoke(
        "render_scene", {"scene": minimal_scene(), "hide": ["vase"], "return_image": False}, context
    ).data
    store = CandidateStore(blender_workspace)
    hidden = store.open(data["candidate_id"])
    assert store.verify(hidden) == []
    assert data["scene_sha256"] != rendered.candidate.receipt["scene"]["sha256"]
    assert hidden.scene()["objects"][1]["visible"] is False
    id_map = load_file_strict(hidden.file(ID_MAP_NAME))
    assert [entry["object"] for entry in id_map["objects"].values()] == ["floor"]
    mask = IdMask.load(hidden)
    assert "vase" not in {o["object"] for o in mask.locate_region(CENTRE)["objects"]}
    # A hidden object is not reported as missing from the frame.
    assert mask.layout()["not_in_frame"] == []

    # Subtraction review: the two candidates side by side.
    listed = [c.id for c in store.list()]
    assert listed[:2] == [hidden.id, rendered.candidate.id]
    compare = render_view(rendered.candidate, "compare", other=hidden)
    assert compare.height == PREVIEW_SIZE[1]


@pytest.mark.gpu_engines
@pytest.mark.skipif(
    os.environ.get("JINGYU_TEST_GPU_ENGINES") != "1",
    reason="set JINGYU_TEST_GPU_ENGINES=1 to render with EEVEE and Workbench",
)
@pytest.mark.parametrize("engine", ["eevee", "workbench"])
def test_gpu_engines_render(runtime: BlenderRuntime, tmp_path: Path, engine: str) -> None:
    workspace = Workspace.at(tmp_path / "workspace")
    scene = minimal_scene()
    scene["render"]["engine"] = engine
    outcome = render_scene(
        workspace, scene, quality="preview", runtime=runtime, timeout_s=TIMEOUT_S
    )
    candidate = outcome.candidate
    assert candidate.receipt["render"]["engine"] == engine
    assert CandidateStore(workspace).verify(candidate) == []
    with Image.open(candidate.image_path) as image:
        assert image.size == PREVIEW_SIZE
    # The id mask pass always renders with Cycles, so it is exact for every engine.
    assert IdMask.load(candidate).layout()["unassigned_pixels"] == 0
