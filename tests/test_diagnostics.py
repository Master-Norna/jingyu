"""Seeing the result clearly: light pass queries, colour measurement, warning control."""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import pytest
from helpers import (
    HEIGHT,
    SYNTHETIC_ID_MAP_OBJECTS,
    WIDTH,
    CandidateFactory,
    synthetic_light_map,
    synthetic_mask,
    synthetic_scene_document,
)
from PIL import Image
from test_placement import still_life

from jingyu.candidate import Candidate
from jingyu.cli import main
from jingyu.errors import JingyuError
from jingyu.frame import check_frame
from jingyu.idmask import ENCODING, ID_MAP_SCHEMA
from jingyu.lighting import LightMap
from jingyu.measure import (
    color_shift,
    describe_color,
    hex_color,
    material_color,
    mean_color,
    object_color_report,
)
from jingyu.scene import validate_scene
from jingyu.scene.warnings import ACCEPTABLE_WARNINGS, drop_accepted
from jingyu.tools import REGISTRY, ToolContext
from jingyu.views import render_view
from jingyu.workspace import Workspace

# ------------------------------------------------------------------ measure


def test_hex_and_describe_color() -> None:
    assert hex_color((1.0, 0.5, 0.0)) == "#ff8000"
    assert hex_color((2.0, -1.0, 0.0)) == "#ff0000"
    orange = describe_color((1.0, 0.5, 0.0))
    assert orange == {"hex": "#ff8000", "hue_deg": 30.0, "saturation": 1.0, "value": 1.0}


def test_mean_color() -> None:
    assert mean_color([]) is None
    assert mean_color([(255, 0, 0), (0, 0, 255)]) == pytest.approx((0.5, 0.0, 0.5))


def test_material_color_is_the_authored_srgb() -> None:
    scene = validate_scene(synthetic_scene_document()).require_valid()
    glaze = material_color(scene, "glaze")
    assert glaze is not None and glaze["hex"] == "#6f93b3"
    assert material_color(scene, "nope") is None
    assert material_color(scene, None) is None


def test_color_shift_wraps_hue() -> None:
    a = {"hue_deg": 350.0, "saturation": 0.5, "value": 0.5}
    b = {"hue_deg": 10.0, "saturation": 0.8, "value": 0.4}
    assert color_shift(a, b) == {"hue_deg": -20.0, "saturation": -0.3, "value": 0.1}


def test_object_color_report_puts_rendered_next_to_authored() -> None:
    scene = validate_scene(synthetic_scene_document()).require_valid()
    report = object_color_report((1.0, 0.5, 0.0), scene, "glaze")
    assert report["rendered_color"]["hex"] == "#ff8000"
    assert report["material_color"]["hex"] == "#6f93b3"
    assert report["color_shift"]["hue_deg"] < 0  # the render is warmer than the glaze
    assert set(object_color_report((1.0, 0.5, 0.0), scene, None)) == {"rendered_color"}


# ---------------------------------------------------------------- light map


def test_light_map_queries(candidate: Candidate) -> None:
    lights = LightMap.load(candidate)
    assert [light["id"] for light in lights.lights] == ["environment.sun", "key"]
    # The vase is at x 30..50; the sun reaches the left half of the image.
    assert lights.at_uv(35 / WIDTH, 30 / HEIGHT) == {
        "surface": True,
        "lit_by": ["environment.sun", "key"],
        "not_reached_by": [],
    }
    assert lights.at_uv(45 / WIDTH, 30 / HEIGHT)["lit_by"] == ["key"]
    assert lights.at_uv(0.5, 0.0) == {"surface": False, "lit_by": [], "not_reached_by": []}
    region = lights.in_region(30 / WIDTH, 15 / HEIGHT, 50 / WIDTH, 50 / HEIGHT)
    assert region["surface_samples"] == 20 * 35
    assert region["lit_fraction"] == {"environment.sun": 0.5, "key": 1.0}
    assert lights.lit_mask("key").getbbox() == (30, 15, 50, 50)
    with pytest.raises(JingyuError) as info:
        lights.lit_mask("moon")
    assert info.value.code == "light.unknown_light"


def test_candidates_without_a_light_pass(make_candidate: CandidateFactory) -> None:
    old = make_candidate(with_light=False)
    with pytest.raises(JingyuError) as info:
        LightMap.load(old)
    assert info.value.code == "candidate.missing_pass"


# -------------------------------------------------------------------- views


def _pixels(png: bytes) -> Image.Image:
    image = Image.open(io.BytesIO(png))
    image.load()
    return image.convert("RGB")


def test_light_view_darkens_where_the_light_misses(candidate: Candidate) -> None:
    sun = render_view(candidate, "light")
    assert sun.meta["light"] == "environment.sun"
    key = render_view(candidate, "light", light="key")
    image = _pixels(key.png)
    with Image.open(candidate.image_path) as original:
        source = original.convert("RGB")
    lit, unlit = (40, 30), (70, 50)  # vase; floor away from the lamp
    assert image.getpixel(lit) == source.getpixel(lit)
    assert sum(image.getpixel(unlit)) < sum(source.getpixel(unlit))
    with pytest.raises(JingyuError) as info:
        render_view(candidate, "light", light="moon")
    assert info.value.code == "light.unknown_light"


def test_saturation_view_is_greyscale(candidate: Candidate) -> None:
    view = render_view(candidate, "saturation")
    assert Image.open(io.BytesIO(view.png)).mode == "L"
    assert 0.0 <= view.meta["mean_saturation"] <= 1.0


def test_region_crops_and_enlarges(candidate: Candidate) -> None:
    view = render_view(candidate, "full", region={"x0": 30, "y0": 15, "x1": 50, "y1": 50})
    assert view.meta["region"] == {"x0": 30, "y0": 15, "x1": 50, "y1": 50}
    assert max(view.width, view.height) > 35  # enlarged to fill the view
    assert view.width / view.height == pytest.approx(20 / 35, rel=0.05)
    by_uv = render_view(candidate, "full", region={"u0": 0.0, "v0": 0.0, "u1": 0.5, "v1": 0.5})
    assert by_uv.meta["region"] == {"x0": 0, "y0": 0, "x1": 40, "y1": 30}
    for region in ({"x0": 10, "y0": 10, "x1": 10, "y1": 20}, {"x0": 0, "y0": 0, "x1": 99, "y1": 9}):
        with pytest.raises(JingyuError) as info:
            render_view(candidate, "full", region=region)
        assert info.value.code == "locate.out_of_bounds"


# -------------------------------------------------------------------- frame


def _gray(tmp_path: Path) -> Path:
    path = tmp_path / "image.png"
    Image.new("RGB", (WIDTH, HEIGHT), (120, 120, 120)).save(path)
    return path


def test_objects_outside_the_view_are_not_reported_missing(tmp_path: Path) -> None:
    mask_path = tmp_path / "id_mask.png"
    synthetic_mask().save(mask_path)
    id_map = {"schema": ID_MAP_SCHEMA, "encoding": ENCODING, "objects": SYNTHETIC_ID_MAP_OBJECTS}
    scene = validate_scene(synthetic_scene_document()).require_valid()
    light_map = synthetic_light_map()
    # "far" is not in view: nothing to report.
    assert check_frame(_gray(tmp_path), mask_path, id_map, scene, light_map) == []
    # In view but painted nowhere: hidden behind something.
    light_map["in_view"].append("far")
    [issue] = check_frame(_gray(tmp_path), mask_path, id_map, scene, light_map)
    assert (issue.code, issue.pointer) == ("frame.not_visible", "/objects/3")


def _room(openings: bool) -> dict[str, Any]:
    document = synthetic_scene_document()
    wall: dict[str, Any] = {"op": "wall", "size": [3, 0.15, 2.4]}
    if openings:
        wall["openings"] = [{"x": 0.0, "sill": 0.9, "width": 1.0, "height": 1.0}]
    document["objects"].append({"id": "wall", "geometry": wall, "location": [0, 3, 0]})
    document["world"] = {"environment": {"family": "daylight"}}
    return validate_scene(document).require_valid()


def test_open_to_sky(tmp_path: Path) -> None:
    open_up = synthetic_light_map()
    closed = {**open_up, "openness": {"open_fraction": 0.1, "open_upward_fraction": 0.0}}
    [issue] = check_frame(_gray(tmp_path), None, None, _room(True), open_up)
    assert (issue.code, issue.severity) == ("light.open_to_sky", "warning")
    assert check_frame(_gray(tmp_path), None, None, _room(True), closed) == []
    assert check_frame(_gray(tmp_path), None, None, _room(False), open_up) == []


# ----------------------------------------------------------- warning control


def _warnings(scene: dict[str, Any]) -> list[tuple[str, str]]:
    result = validate_scene(scene)
    assert result.valid, result.errors
    return [(w.code, w.pointer) for w in result.warnings]


def _sunk_orange() -> dict[str, Any]:
    scene = still_life()
    orange = scene["objects"][3]
    del orange["rest_on"]
    del orange["parent"]
    orange["location"] = [0.4, 0.3, 0.03]
    return scene


def test_attached_to_marks_an_intentional_join() -> None:
    scene = _sunk_orange()
    assert _warnings(scene) == [("physics.intersection", "/objects/3/location")]
    scene["objects"][3]["attached_to"] = ["top"]
    assert _warnings(scene) == []
    scene["objects"][3]["location"] = [0.4, 0.3, 0.5]  # attached things may hang in the air
    assert _warnings(scene) == []


def test_attached_to_must_name_another_object() -> None:
    scene = still_life()
    scene["objects"][3]["attached_to"] = ["nothing"]
    errors = {(i.code, i.pointer) for i in validate_scene(scene).errors}
    assert ("spec.unknown_reference", "/objects/3/attached_to/0") in errors


def test_accept_warnings_silences_a_code() -> None:
    scene = _sunk_orange()
    scene["accept_warnings"] = ["physics.intersection"]
    assert _warnings(scene) == []
    scene["accept_warnings"] = ["no.such_warning"]
    assert not validate_scene(scene).valid


def test_errors_cannot_be_accepted() -> None:
    assert all(code.split(".")[0] != "json" for code in ACCEPTABLE_WARNINGS)
    scene = {"accept_warnings": list(ACCEPTABLE_WARNINGS)}
    result = validate_scene(still_life() | {"objects": []})
    assert drop_accepted(list(result.errors), scene) == list(result.errors)


# -------------------------------------------------------------------- tools


@pytest.fixture
def ctx(workspace: Workspace) -> ToolContext:
    return ToolContext(workspace)


def test_describe_light(ctx: ToolContext, candidate: Candidate) -> None:
    data = REGISTRY.invoke("describe_light", {"candidate_id": candidate.id}, ctx).data
    assert [light["id"] for light in data["lights"]] == ["environment.sun", "key"]
    assert data["shadows"][0]["caster"] == "vase"
    assert data["openness"]["open_upward_fraction"] == 1.0


def test_locate_reports_colour_and_light(ctx: ToolContext, candidate: Candidate) -> None:
    point = {"candidate_id": candidate.id, "point": {"x": 35, "y": 30}}
    result = REGISTRY.invoke("locate_in_candidate", point, ctx).data["result"]
    hit = result["hit"]
    assert hit["object"] == "vase"
    assert hit["material_color"]["hex"] == "#6f93b3"
    assert {"rendered_color", "color_shift"} <= set(hit)
    assert result["light"]["lit_by"] == ["environment.sun", "key"]


def test_layout_reports_colour_and_lit_fraction(ctx: ToolContext, candidate: Candidate) -> None:
    data = REGISTRY.invoke("describe_layout", {"candidate_id": candidate.id}, ctx).data
    text = json.dumps(data)
    assert "rendered_color" in text and "lit_fraction" in text


def test_cli_names_view_images_after_what_they_show(
    capsys: pytest.CaptureFixture[str], workspace: Workspace, candidate: Candidate
) -> None:
    def call(arguments: dict[str, Any]) -> str:
        argv = ["--workspace", str(workspace.root), "call", "view_candidate"]
        assert main([*argv, "--args", json.dumps(arguments)]) == 0
        (path,) = json.loads(capsys.readouterr().out)["images"]
        return Path(path).name

    base = {"candidate_id": candidate.id}
    names = {
        call({**base, "view": "full"}),
        call({**base, "view": "light", "light": "key"}),
        call({**base, "view": "full", "region": {"x0": 0, "y0": 0, "x1": 20, "y1": 20}}),
    }
    assert len(names) == 3
    assert all(name.startswith(candidate.id) for name in names)


def test_an_object_can_accept_warnings_about_itself() -> None:
    scene = _sunk_orange()
    scene["objects"][2]["accept_warnings"] = ["physics.intersection"]  # the bowl: not it
    assert _warnings(scene) == [("physics.intersection", "/objects/3/location")]
    scene["objects"][3]["accept_warnings"] = ["physics.intersection"]
    assert _warnings(scene) == []
