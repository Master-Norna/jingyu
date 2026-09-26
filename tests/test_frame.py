"""Checks on a finished frame: exposure and objects missing from it."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from helpers import SYNTHETIC_ID_MAP_OBJECTS, synthetic_mask, synthetic_scene_document
from PIL import Image

from jingyu.frame import check_frame, luma_stats
from jingyu.idmask import ENCODING, ID_MAP_SCHEMA
from jingyu.scene import validate_scene


def _image(tmp_path: Path, value: int, name: str = "image.png") -> Path:
    path = tmp_path / name
    Image.new("RGB", (64, 48), (value, value, value)).save(path)
    return path


def _scene() -> dict[str, Any]:
    return validate_scene(synthetic_scene_document()).require_valid()


def test_luma_stats(tmp_path: Path) -> None:
    stats = luma_stats(_image(tmp_path, 255))
    assert stats["mean_luma"] == pytest.approx(1.0)
    assert (stats["clipped_fraction"], stats["crushed_fraction"]) == (1.0, 0.0)
    assert luma_stats(_image(tmp_path, 0))["crushed_fraction"] == 1.0


def test_exposure_warnings(tmp_path: Path) -> None:
    scene = _scene()
    assert check_frame(_image(tmp_path, 120), None, None, scene) == []
    [dark] = check_frame(_image(tmp_path, 30), None, None, scene)
    assert (dark.code, dark.pointer, dark.severity) == (
        "frame.underexposed",
        "/render/exposure",
        "warning",
    )
    assert dark.hint is not None and "exposure +" in dark.hint
    [bright] = check_frame(_image(tmp_path, 255), None, None, scene)
    assert bright.code == "frame.overexposed"


def test_objects_missing_from_the_frame(tmp_path: Path) -> None:
    mask_path = tmp_path / "id_mask.png"
    synthetic_mask().save(mask_path)
    id_map = {"schema": ID_MAP_SCHEMA, "encoding": ENCODING, "objects": SYNTHETIC_ID_MAP_OBJECTS}
    issues = check_frame(_image(tmp_path, 120), mask_path, id_map, _scene())
    # "far" is visible but painted nowhere; "ghost" is hidden and not expected.
    assert [(i.code, i.pointer) for i in issues] == [("frame.not_visible", "/objects/3")]


def test_supports_hidden_behind_others_are_not_reported(tmp_path: Path) -> None:
    mask_path = tmp_path / "id_mask.png"
    synthetic_mask().save(mask_path)
    id_map = {"schema": ID_MAP_SCHEMA, "encoding": ENCODING, "objects": SYNTHETIC_ID_MAP_OBJECTS}
    document = synthetic_scene_document()
    document["objects"].append(
        {
            "id": "cup",
            "geometry": {"op": "cylinder", "radius": 0.02, "height": 0.05},
            "location": [0, 50, 0],
            "rest_on": "far",
        }
    )
    scene = validate_scene(document).require_valid()
    issues = check_frame(_image(tmp_path, 120), mask_path, id_map, scene)
    # "far" now holds up the cup: structure, not subject.  The cup itself is reported.
    assert [(i.code, i.pointer) for i in issues] == [("frame.not_visible", "/objects/5")]


def test_objects_hidden_from_the_camera_are_not_expected_in_the_frame(tmp_path: Path) -> None:
    mask_path = tmp_path / "id_mask.png"
    synthetic_mask().save(mask_path)
    id_map = {"schema": ID_MAP_SCHEMA, "encoding": ENCODING, "objects": SYNTHETIC_ID_MAP_OBJECTS}
    document = synthetic_scene_document()
    document["objects"][3]["camera_visible"] = False  # "far" becomes a reflector card
    scene = validate_scene(document).require_valid()
    assert check_frame(_image(tmp_path, 120), mask_path, id_map, scene) == []
