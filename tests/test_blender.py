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
from jingyu.candidate import (
    ID_MAP_NAME,
    ID_MASK_NAME,
    IMAGE_NAME,
    LIGHT_MAP_NAME,
    LIGHT_MASK_NAME,
    SCENE_NAME,
    CandidateStore,
)
from jingyu.canonical_json import load_file_strict
from jingyu.errors import JingyuError
from jingyu.lighting import LightMap
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
        LIGHT_MASK_NAME,
        LIGHT_MAP_NAME,
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


def test_daylight_groups_and_rest_on_render(runtime: BlenderRuntime, tmp_path: Path) -> None:
    """The worker builds the placement the host resolved, lit by a daylight environment."""

    from test_placement import still_life

    workspace = Workspace.at(tmp_path / "workspace")
    scene = still_life()
    scene["world"] = {"environment": {"family": "daylight", "sun_elevation": 20}}
    scene["lights"] = [
        {
            "id": "window",
            "kind": "area",
            "location": [0, -1, 1.5],
            "look_at": [0, 0, 0],
            "size": 1.0,
            "size_y": 0.4,
            "temperature_k": 5600,
            "parent": "arrangement",
        }
    ]
    scene["cameras"] = [
        {"id": "cam", "location": [0.9, -0.9, 0.7], "look_at": [0.2, 0.1, 0.08], "lens_mm": 50}
    ]
    scene["render"]["resolution"] = [320, 240]
    outcome = render_scene(workspace, scene, quality="final", runtime=runtime, timeout_s=TIMEOUT_S)
    candidate = outcome.candidate
    assert CandidateStore(workspace).verify(candidate) == []
    layout = IdMask.load(candidate).layout()
    # Given at z = 9, the orange only shows up if rest_on put it in the bowl.
    assert layout["not_in_frame"] == []
    assert {o["object"] for o in layout["objects"]} >= {"top", "bowl", "orange"}
    assert "frame.underexposed" not in {w.code for w in outcome.warnings}
    lights = LightMap.load(candidate)
    assert [light["id"] for light in lights.lights] == ["environment.sun", "window"]
    assert set(lights.document["in_view"]) >= {"top", "bowl", "orange"}
    # Nothing above the table: the camera sees open sky upward.
    assert lights.document["openness"]["open_upward_fraction"] > 0.9
    # The orange sits in the bowl and the window light reaches most of what is seen of it.
    assert lights.document["objects"]["orange"]["lit_fraction"]["window"] > 0.3


def test_light_pass_finds_shadows(rendered: RenderOutcome) -> None:
    lights = LightMap.load(rendered.candidate)
    (key,) = lights.lights
    with Image.open(rendered.candidate.file(LIGHT_MASK_NAME)) as mask:
        assert max(mask.size) == 240
    assert set(lights.document["in_view"]) == {"floor", "vase"}
    shadows = {(s["caster"], s["receiver"]) for s in lights.document["shadows"]}
    assert ("vase", "floor") in shadows
    # The vase's side toward the lamp is lit; the floor far behind it is not all lit.
    assert 0.0 < lights.document["objects"]["floor"]["lit_fraction"][key["id"]] < 1.0


_MARKERS = [(-0.4, 0.3, 0.1), (0.35, -0.2, 0.0), (0.0, 0.5, 0.45), (0.25, 0.25, 0.2)]


@pytest.mark.parametrize(
    "camera",
    [
        {
            "location": [1.4, -1.6, 0.9],
            "look_at": [0.1, 0.1, 0.1],
            "lens_mm": 35,
            "shift": [0.1, -0.05],
        },
        {"location": [0.2, -0.1, 2.4], "look_at": [0.2, -0.1, 0.0], "lens_mm": 28},
        {"location": [0.5, -2.0, 0.6], "rotation": [80, 0, 10], "parent": "rig"},
        {
            "location": [0.0, -3.0, 1.2],
            "look_at": [0.0, 0.0, 0.2],
            "projection": "orthographic",
            "ortho_scale": 1.8,
        },
    ],
    ids=["look-at-shifted", "straight-down", "rotated-in-a-group", "orthographic"],
)
def test_host_projection_matches_the_render(
    runtime: BlenderRuntime, tmp_path: Path, camera: dict[str, object]
) -> None:
    """Where the host camera model puts a point is where Blender draws it."""

    from jingyu.camera import CameraModel
    from jingyu.scene import validate_scene

    radius = 0.02
    scene = {
        "schema": "jingyu.scene.v1",
        "id": "markers",
        "world": {"color": "#808080"},
        "groups": [{"id": "rig", "location": [0.1, 0.0, 0.1], "rotation": [0, 0, 15]}],
        "objects": [
            {"id": f"m{i}", "geometry": {"op": "sphere", "radius": radius}, "location": list(p)}
            for i, p in enumerate(_MARKERS)
        ],
        "cameras": [{"id": "cam", **camera}],
        "render": {"camera": "cam", "resolution": [320, 240], "samples": 1},
    }
    workspace = Workspace.at(tmp_path / "workspace")
    outcome = render_scene(workspace, scene, quality="final", runtime=runtime, timeout_s=TIMEOUT_S)
    layout = {o["object"]: o for o in IdMask.load(outcome.candidate).layout()["objects"]}
    result = validate_scene(scene)
    normalized = result.require_valid()
    model = CameraModel.from_scene(normalized, result.placement)
    for i, (x, y, z) in enumerate(_MARKERS):
        projected = model.project((x, y, z + radius))
        assert projected is not None
        u, v, _ = projected
        seen = layout[f"m{i}"]["centroid_uv"]
        assert abs(seen["u"] - u) * 320 < 1.2, (i, seen, (u, v))
        assert abs(seen["v"] - v) * 240 < 1.2, (i, seen, (u, v))


def test_solved_framing_and_sun_hold_in_the_render(runtime: BlenderRuntime, tmp_path: Path) -> None:
    """frame_subject and aim_sun predict what Blender then renders."""

    from test_solve import _room

    workspace = Workspace.at(tmp_path / "workspace")
    context = ToolContext(workspace, render_timeout_s=TIMEOUT_S)
    framed = REGISTRY.invoke(
        "frame_subject",
        {"scene": _room(), "subject": ["still"], "at": {"u": 0.4, "v": 0.55}, "size": 0.4},
        context,
    ).data
    target = [0.35, -0.2, 0.75]
    sunny = REGISTRY.invoke(
        "aim_sun", {"scene": framed["scene"], "target": {"point": target}}, context
    ).data
    scene = sunny["scene"]
    scene["render"]["resolution"] = [320, 240]
    candidate = render_scene(
        workspace, scene, quality="final", runtime=runtime, timeout_s=TIMEOUT_S
    ).candidate
    layout = IdMask.load(candidate).layout()
    boxes = [o["bbox_uv"] for o in layout["objects"] if o["object"] in ("vase", "cup")]
    u0, u1 = min(b["u0"] for b in boxes), max(b["u1"] for b in boxes)
    v0, v1 = min(b["v0"] for b in boxes), max(b["v1"] for b in boxes)
    assert (u0 + u1) / 2 == pytest.approx(0.4, abs=2 / 320)
    assert (v0 + v1) / 2 == pytest.approx(0.55, abs=2 / 240)
    assert max(u1 - u0, v1 - v0) == pytest.approx(0.4, abs=3 / 240)
    from jingyu.camera import CameraModel
    from jingyu.scene import validate_scene

    result = validate_scene(scene)
    u, v, _ = CameraModel.from_scene(result.require_valid(), result.placement).project(target) or (
        0,
        0,
        0,
    )
    assert "environment.sun" in LightMap.load(candidate).at_uv(u, v)["lit_by"]


def test_fill_light_and_reflector_card(runtime: BlenderRuntime, tmp_path: Path) -> None:
    """A fill light lifts the shadow side without a shadow; a card lights but is unseen."""

    workspace = Workspace.at(tmp_path / "workspace")
    scene = minimal_scene()
    scene["render"]["resolution"] = [240, 180]
    scene["lights"][0]["power_w"] = 300.0
    card = {
        "id": "card",
        "geometry": {"op": "box", "size": [0.4, 0.02, 0.5]},
        "location": [0.0, -0.9, 0.0],
        "camera_visible": False,
    }
    fill = {
        "id": "fill",
        "kind": "fill",
        "location": [1.5, -1.5, 0.6],
        "look_at": [0, 0, 0.15],
        "power_w": 400.0,
        "size": 1.5,
    }
    plain = render_scene(workspace, scene, quality="final", runtime=runtime, timeout_s=TIMEOUT_S)
    scene["objects"].append(card)
    scene["lights"].append(fill)
    helped = render_scene(workspace, scene, quality="final", runtime=runtime, timeout_s=TIMEOUT_S)
    mask = IdMask.load(helped.candidate)
    layout = mask.layout()
    # The card stands right in front of the vase, yet the camera sees the vase through it.
    assert "card" not in {o["object"] for o in layout["objects"]}
    assert "card" not in layout["not_in_frame"]
    assert layout == IdMask.load(plain.candidate).layout()
    assert "frame.not_visible" not in {w.code for w in helped.warnings}
    lights = LightMap.load(helped.candidate)
    assert [light["id"] for light in lights.lights][-1] == "fill"
    assert not any(s["light"] == "fill" for s in lights.document["shadows"])

    # The fill brightens the frame overall.
    def mean(candidate_path: Path) -> float:
        with Image.open(candidate_path) as image:
            gray = image.convert("L")
            return sum(v * c for v, c in enumerate(gray.histogram())) / (gray.width * gray.height)

    assert mean(helped.candidate.image_path) > mean(plain.candidate.image_path) * 1.05


def _furnished_room() -> dict[str, object]:
    return {
        "schema": "jingyu.scene.v1",
        "id": "furnished",
        "world": {"environment": {"family": "daylight", "sun_elevation": 30, "sun_azimuth": 90}},
        "materials": [
            {"id": "plaster", "family": "plastic", "color": "#e6e0d6", "gloss": 0.05},
            {"id": "oak", "family": "plastic", "color": "#9a7048", "gloss": 0.3},
            {"id": "steel", "family": "metal", "metal": "iron", "polish": 0.4},
        ],
        "objects": [
            {
                "id": "room",
                "geometry": {
                    "op": "room",
                    "size": [4, 4, 2.6],
                    "openings": [
                        {
                            "wall": "left",
                            "x": 0.0,
                            "sill": 0.8,
                            "width": 1.2,
                            "height": 1.3,
                            "glass": True,
                        }
                    ],
                },
                "material": "plaster",
                "part_materials": {"floor": "oak"},
            },
            {
                "id": "table",
                "geometry": {"op": "table", "size": [1.2, 0.7, 0.75]},
                "material": "oak",
                "part_materials": {"base": "steel"},
                "rest_on": "room",
            },
            {
                "id": "chair",
                "geometry": {"op": "chair"},
                "location": [0.2, -0.6, 0],
                "rotation": [0, 0, 180],
                "material": "oak",
                "rest_on": "room",
            },
            {
                "id": "shelf",
                "geometry": {"op": "shelf", "size": [0.8, 0.3, 1.8], "shelves": 4},
                "location": [0.8, 1.8, 0],
                "material": "oak",
                "rest_on": "room",
            },
            {
                "id": "box",
                "geometry": {"op": "box", "size": [0.2, 0.15, 0.1], "bevel": 0.004},
                "location": [0.8, 1.8, 0.9],
                "rest_on": "shelf",
            },
        ],
        "cameras": [
            {"id": "cam", "location": [1.6, -1.6, 1.5], "look_at": [0, 0.3, 0.8], "lens_mm": 24}
        ],
        "render": {"camera": "cam", "resolution": [320, 240], "samples": 8},
    }


def test_assemblies_parts_and_glass_render(runtime: BlenderRuntime, tmp_path: Path) -> None:
    """Rooms, furniture and parts: materials per part, ids per part, light through glass."""

    from jingyu.camera import CameraModel
    from jingyu.scene import validate_scene

    workspace = Workspace.at(tmp_path / "workspace")
    context = ToolContext(workspace, render_timeout_s=TIMEOUT_S)
    target = [0.1, 0.0, 0.75]
    sunny = REGISTRY.invoke(
        "aim_sun",
        {
            "scene": _furnished_room(),
            "target": {"point": target},
            "through": {"object": "room", "opening": 0},
        },
        context,
    ).data
    scene = sunny["scene"]
    result = validate_scene(scene)
    placement = result.placement
    # The box was put between two boards and fell onto the one below it.
    box_bottom = placement.meshes["box"].bounds()[0][2]
    assert 0.4 < box_bottom < 0.9
    candidate = render_scene(
        workspace, scene, quality="final", runtime=runtime, timeout_s=TIMEOUT_S
    ).candidate
    id_map = load_file_strict(candidate.file(ID_MAP_NAME))
    table_parts = {e.get("part") for e in id_map["objects"].values() if e["object"] == "table"}
    assert table_parts == {"top", "base"}
    mask = IdMask.load(candidate)
    layout = {o["object"]: o for o in mask.layout()["objects"]}
    assert {p["part"] for p in layout["table"]["parts"]} == {"top", "base"}
    assert {p["material"] for p in layout["table"]["parts"]} == {"oak", "steel"}
    camera = CameraModel.from_scene(result.require_valid(), placement, resolution=(320, 240))
    u, v, _ = camera.project(target) or (0, 0, 0)
    hit = mask.locate_point({"u": u, "v": v})["hit"]
    assert hit["object"] == "table" and hit["part"] == "top"
    assert hit["part_pointer"] == "/objects/1/part_materials/top"
    # The sun reaches the table top through the glass pane.
    assert "environment.sun" in LightMap.load(candidate).at_uv(u, v)["lit_by"]


def test_texture_weathering_and_mist_render(runtime: BlenderRuntime, tmp_path: Path) -> None:
    """Procedural texture varies across a surface; the mist box stays out of the passes."""

    workspace = Workspace.at(tmp_path / "workspace")
    scene = {
        "schema": "jingyu.scene.v1",
        "id": "textures",
        "world": {
            "environment": {"family": "uniform", "color": "#c8cdd4", "strength": 0.6, "mist": 0.15}
        },
        "materials": [
            {"id": "oak", "family": "wood", "ring_size": 0.01},
            {"id": "flat", "family": "plastic", "color": "#b58658", "gloss": 0.3},
            {
                "id": "paint",
                "family": "plastic",
                "color": "#3d6b8c",
                "weathering": {"wear": 0.7, "dust": 0.5},
            },
        ],
        "objects": [
            {
                "id": "wood",
                "geometry": {"op": "box", "size": [0.3, 0.3, 0.3], "bevel": 0.01},
                "location": [-0.2, 0, 0],
                "material": "oak",
            },
            {
                "id": "plain",
                "geometry": {"op": "box", "size": [0.3, 0.3, 0.3], "bevel": 0.01},
                "location": [0.2, 0, 0],
                "material": "flat",
            },
            {
                "id": "old",
                "geometry": {"op": "vessel", "wobble": 0.5},
                "location": [0, 0.4, 0],
                "material": "paint",
            },
        ],
        "lights": [
            {
                "id": "key",
                "kind": "area",
                "location": [-1, -1.5, 1.5],
                "look_at": [0, 0, 0.1],
                "power_w": 300,
            }
        ],
        "cameras": [
            {"id": "cam", "location": [0, -1.2, 0.5], "look_at": [0, 0.1, 0.15], "lens_mm": 40}
        ],
        "render": {"camera": "cam", "resolution": [320, 240], "samples": 16},
    }
    candidate = render_scene(
        workspace, scene, quality="final", runtime=runtime, timeout_s=TIMEOUT_S
    ).candidate
    mask = IdMask.load(candidate)
    layout = mask.layout()
    assert {o["object"] for o in layout["objects"]} == {"wood", "plain", "old"}
    assert layout["unassigned_pixels"] == 0
    assert set(LightMap.load(candidate).document["in_view"]) == {"wood", "plain", "old"}

    def spread(name: str) -> float:
        box = next(o for o in layout["objects"] if o["object"] == name)["bbox"]
        inner = (box["x0"] + 8, box["y0"] + 8, box["x1"] - 8, box["y1"] - 8)
        with Image.open(candidate.image_path) as image:
            gray = image.convert("L").crop(inner)
            values = [gray.getpixel((x, y)) for y in range(gray.height) for x in range(gray.width)]
        mean = sum(values) / len(values)
        return (sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5

    assert spread("wood") > spread("plain") + 3


def test_a_resident_worker_serves_renders_and_recovers(
    runtime: BlenderRuntime, tmp_path: Path
) -> None:
    from jingyu.bridge.resident import WorkerPool

    workspace = Workspace.at(tmp_path / "workspace")
    pool = WorkerPool(tmp_path / "workers")
    scene = minimal_scene()
    scene["render"]["resolution"] = [96, 72]
    try:
        first = render_scene(workspace, scene, runtime=runtime, timeout_s=TIMEOUT_S, workers=pool)
        worker = next(iter(pool._workers.values()))
        process = worker._process
        second = render_scene(workspace, scene, runtime=runtime, timeout_s=TIMEOUT_S, workers=pool)
        assert worker._process is process and worker.served == 2
        store = CandidateStore(workspace)
        assert store.verify(first.candidate) == [] and store.verify(second.candidate) == []
        # Each candidate keeps only its own part of the worker's output.
        for outcome in (first, second):
            log = outcome.candidate.file("blender.log").read_text(
                encoding="utf-8", errors="replace"
            )
            assert log.count("jingyu worker: request") == 1
        # A dead worker is replaced by the next request.
        assert process is not None
        process.kill()
        process.wait()
        third = render_scene(workspace, scene, runtime=runtime, timeout_s=TIMEOUT_S, workers=pool)
        assert store.verify(third.candidate) == [] and worker._process is not process
        # An overrun kills the worker; the next render starts a fresh one.
        with pytest.raises(JingyuError) as info:
            render_scene(workspace, scene, runtime=runtime, timeout_s=0.01, workers=pool)
        assert info.value.code == "blender.timeout"
        fourth = render_scene(workspace, scene, runtime=runtime, timeout_s=TIMEOUT_S, workers=pool)
        assert store.verify(fourth.candidate) == []
    finally:
        pool.close()
