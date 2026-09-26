"""Test helpers: synthetic scenes, id masks and candidates that need no Blender."""

from __future__ import annotations

import itertools
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from jingyu import __version__
from jingyu.bridge import BlenderRuntime
from jingyu.candidate import (
    ID_MAP_NAME,
    ID_MASK_NAME,
    IMAGE_NAME,
    SCENE_NAME,
    Candidate,
    CandidateStore,
    scene_identity,
)
from jingyu.canonical_json import pretty_bytes
from jingyu.idmask import ENCODING, ID_MAP_SCHEMA, encode_index
from jingyu.scene import validate_scene
from jingyu.workspace import Workspace

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "jingyu"

#: Default synthetic image size.
WIDTH, HEIGHT = 80, 60

#: Mask layout of the default synthetic candidate, as half-open pixel boxes
#: (x0, y0, x1, y1), painted in this order (later boxes cover earlier ones).
FLOOR_BOX = (0, 40, 80, 60)  # index 1, touches the bottom and both sides
VASE_BOX = (30, 15, 50, 50)  # index 2, covers part of the floor, touches no edge
CRATE_BOX = (0, 20, 10, 30)  # index 3, touches the left edge

#: Pixel counts that follow from the boxes above (80 x 60 = 4800 pixels).
VASE_PIXELS = 20 * 35
CRATE_PIXELS = 10 * 10
FLOOR_PIXELS = 20 * 80 - 10 * 20
BACKGROUND_PIXELS = WIDTH * HEIGHT - VASE_PIXELS - CRATE_PIXELS - FLOOR_PIXELS


def synthetic_scene_document() -> dict[str, Any]:
    """A valid scene whose objects match the synthetic id mask.

    ``far`` is visible but never drawn (not in frame); ``ghost`` is hidden.
    """

    return {
        "schema": "jingyu.scene.v1",
        "id": "synthetic",
        "materials": [
            {"id": "glaze", "family": "ceramic", "color": "#6f93b3"},
            {"id": "floor", "family": "plastic"},
        ],
        "objects": [
            {"id": "floor", "geometry": {"op": "plane", "size": [6, 6]}, "material": "floor"},
            {"id": "vase", "geometry": {"op": "vessel"}, "material": "glaze"},
            {
                "id": "crate",
                "geometry": {"op": "box", "size": [0.2, 0.2, 0.2]},
                "location": [0.4, 0, 0],
            },
            {
                "id": "far",
                "geometry": {"op": "box", "size": [0.1, 0.1, 0.1]},
                "location": [0, 50, 0],
            },
            {"id": "ghost", "geometry": {"op": "sphere", "radius": 0.1}, "visible": False},
        ],
        "lights": [
            {"id": "key", "kind": "area", "location": [-1, -1, 2], "look_at": [0, 0, 0]},
        ],
        "cameras": [{"id": "main", "location": [0, -2, 0.5], "look_at": [0, 0, 0.2]}],
        "render": {"camera": "main", "resolution": [WIDTH, HEIGHT], "samples": 1},
    }


#: id map entries for the visible objects, as the Blender pass would write them.
SYNTHETIC_ID_MAP_OBJECTS = {
    "1": {"object": "floor", "pointer": "/objects/0", "material": "floor"},
    "2": {"object": "vase", "pointer": "/objects/1", "material": "glaze"},
    "3": {"object": "crate", "pointer": "/objects/2", "material": None},
    "4": {"object": "far", "pointer": "/objects/3", "material": None},
}


def _paint(mask: Image.Image, box: tuple[int, int, int, int], index: int) -> None:
    mask.paste((*encode_index(index), 255), box)


def synthetic_mask(width: int = WIDTH, height: int = HEIGHT) -> Image.Image:
    """The default mask layout, scaled to *width* x *height*."""

    mask = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    for box, index in ((FLOOR_BOX, 1), (VASE_BOX, 2), (CRATE_BOX, 3)):
        x0, y0, x1, y1 = box
        scaled = (
            x0 * width // WIDTH,
            y0 * height // HEIGHT,
            x1 * width // WIDTH,
            y1 * height // HEIGHT,
        )
        _paint(mask, scaled, index)
    return mask


def synthetic_image(width: int, height: int) -> Image.Image:
    """A deterministic, asymmetric RGB gradient."""

    image = Image.new("RGB", (width, height))
    image.putdata(
        [
            (x * 255 // max(1, width - 1), y * 255 // max(1, height - 1), (x * y) % 256)
            for y in range(height)
            for x in range(width)
        ]
    )
    return image


@dataclass
class CandidateFactory:
    workspace: Workspace
    _counter: itertools.count[int]

    def __call__(
        self,
        *,
        width: int = WIDTH,
        height: int = HEIGHT,
        with_mask: bool = True,
        extra_pixels: Mapping[tuple[int, int], tuple[int, int, int, int]] | None = None,
        id_map: dict[str, Any] | None = None,
    ) -> Candidate:
        """Commit a synthetic candidate through the real :class:`CandidateStore`."""

        n = next(self._counter)
        normalized = validate_scene(synthetic_scene_document()).require_valid()
        normalized["render"]["resolution"] = [width, height]
        identity = scene_identity(normalized)
        # Deterministic ids that sort in creation order.
        candidate_id = f"c_20260926T12{n // 60:02d}{n % 60:02d}Z_{identity['sha256'][:8]}_{n:04x}"

        staging = self.workspace.staging_dir / f"synthetic-{n}"
        staging.mkdir(parents=True)
        synthetic_image(width, height).save(staging / IMAGE_NAME)
        (staging / SCENE_NAME).write_bytes(pretty_bytes(normalized))
        (staging / "blender.log").write_text("synthetic\n", encoding="utf-8")
        if with_mask:
            mask = synthetic_mask(width, height)
            for (x, y), rgba in (extra_pixels or {}).items():
                mask.putpixel((x, y), rgba)
            mask.save(staging / ID_MASK_NAME)
            document = id_map or {
                "schema": ID_MAP_SCHEMA,
                "encoding": ENCODING,
                "objects": SYNTHETIC_ID_MAP_OBJECTS,
            }
            (staging / ID_MAP_NAME).write_bytes(pretty_bytes(document))

        receipt = {
            "candidate_id": candidate_id,
            "created_at": f"2026-09-26T12:{n // 60:02d}:{n % 60:02d}Z",
            "jingyu_version": __version__,
            "scene": identity,
            "quality": "preview",
            "render": {
                "engine": "cycles",
                "resolution": [width, height],
                "samples": 1,
                "device": "CPU",
            },
            "blender": {"version": "synthetic"},
            "timings_ms": {},
            "warnings": [],
        }
        return CandidateStore(self.workspace).commit(staging, receipt)


@dataclass(frozen=True)
class ScriptRuntime(BlenderRuntime):
    """Runs a Python script (``sys.argv[1:]`` = request, response) in place of Blender."""

    script: str = ""

    def worker_command(self, request: Path, response: Path) -> list[str]:
        return [sys.executable, "-c", self.script, str(request), str(response)]


def script_runtime(script: str) -> ScriptRuntime:
    return ScriptRuntime("executable", sys.executable, "test-script", script)
