"""Render a scene description into a new immutable candidate."""

from __future__ import annotations

import shutil
import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from PIL import Image

from . import __version__
from .bridge import DEFAULT_TIMEOUT_S, BlenderRuntime, discover_runtime, run_worker
from .bridge.resident import WorkerPool
from .bridge.runner import LOG_NAME
from .candidate import (
    ID_MAP_NAME,
    ID_MASK_NAME,
    IMAGE_NAME,
    LIGHT_MAP_NAME,
    LIGHT_MASK_NAME,
    RENDER_NAME,
    SCENE_NAME,
    Candidate,
    CandidateStore,
    new_candidate_id,
    scene_identity,
    utc_now_iso,
)
from .canonical_json import pretty_bytes
from .errors import Issue, JingyuError
from .frame import check_frame
from .scene import validate_scene
from .scene.warnings import drop_accepted
from .style import apply_style
from .workspace import Workspace

Quality = Literal["preview", "final"]

#: Preview renders trade fidelity for speed: half resolution, few samples.
PREVIEW_SCALE = 0.5
PREVIEW_MAX_SAMPLES = 16


@dataclass(frozen=True)
class RenderOutcome:
    candidate: Candidate
    warnings: tuple[Issue, ...]


def quality_overrides(render: Mapping[str, Any], quality: Quality) -> dict[str, Any]:
    if quality == "final":
        return {}
    width, height = render["resolution"]
    return {
        "resolution": [
            max(16, round(width * PREVIEW_SCALE)),
            max(16, round(height * PREVIEW_SCALE)),
        ],
        "samples": min(int(render["samples"]), PREVIEW_MAX_SAMPLES),
    }


def render_scene(
    workspace: Workspace,
    scene: Any,
    *,
    quality: Quality = "preview",
    runtime: BlenderRuntime | None = None,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    workers: WorkerPool | None = None,
) -> RenderOutcome:
    """Validate, render in an isolated worker and commit a candidate."""

    result = validate_scene(scene)
    normalized = result.require_valid()
    runtime = runtime or discover_runtime()
    store = CandidateStore(workspace)
    staging = workspace.staging_dir / uuid.uuid4().hex
    staging.mkdir(parents=True)
    started = time.perf_counter()
    try:
        run = (workers.run if workers is not None else run_worker)(
            runtime,
            {
                "action": "render",
                "scene": normalized,
                "output_dir": str(staging),
                "overrides": quality_overrides(normalized["render"], quality),
                "passes": {"id_mask": True, "light": True},
            },
            staging,
            timeout_s=timeout_s,
        )
        response = run.response
        _check_outputs(staging, response)
        for scratch in ("request.json", "response.json"):
            (staging / scratch).unlink(missing_ok=True)
        (staging / SCENE_NAME).write_bytes(pretty_bytes(normalized))
        if response.get("id_map") is not None:
            (staging / ID_MAP_NAME).write_bytes(pretty_bytes(response["id_map"]))
        light_map = response.get("light_map")
        if light_map is not None:
            (staging / LIGHT_MAP_NAME).write_bytes(pretty_bytes(light_map))

        id_map = response.get("id_map")
        frame_warnings = check_frame(
            staging / IMAGE_NAME,
            staging / ID_MASK_NAME if id_map is not None else None,
            id_map,
            normalized,
            light_map,
        )
        warnings = [*result.warnings, *drop_accepted(frame_warnings, normalized)]
        _paint(staging, normalized["render"]["style"], id_map is not None)

        identity = scene_identity(normalized)
        receipt = {
            "candidate_id": _unused_candidate_id(store, identity["sha256"]),
            "created_at": utc_now_iso(),
            "jingyu_version": __version__,
            "scene": identity,
            "quality": quality,
            "render": response["render"],
            "blender": {**response["blender"], "runtime": runtime.to_dict()},
            "timings_ms": {
                **{k.removesuffix("_ms"): v for k, v in response.get("timings", {}).items()},
                "host_total": round((time.perf_counter() - started) * 1000),
            },
            "warnings": [w.to_dict() for w in warnings] + list(response.get("warnings", [])),
        }
        candidate = store.commit(staging, receipt)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return RenderOutcome(candidate, tuple(warnings))


def _paint(staging: Path, style: Mapping[str, Any], has_mask: bool) -> None:
    """Apply the scene's style; the photographic render is kept as render.png."""

    if style["preset"] == "none":
        return
    image_path = staging / IMAGE_NAME
    photo = staging / RENDER_NAME
    image_path.replace(photo)
    with Image.open(photo) as image:
        image.load()
        mask = None
        if has_mask:
            with Image.open(staging / ID_MASK_NAME) as opened:
                mask = opened.convert("RGBA")
        apply_style(image, mask, style).save(image_path)


def _unused_candidate_id(store: CandidateStore, scene_sha256: str) -> str:
    """A fresh id that no candidate uses yet: ids share a second and a scene hash, and
    differ only in a short random suffix, so repeated renders must not collide."""

    for _ in range(64):
        candidate_id = new_candidate_id(scene_sha256)
        if not (store.root / candidate_id).exists():
            return candidate_id
    raise JingyuError("internal.unexpected", "could not allocate an unused candidate id")


def _check_outputs(staging: Path, response: Mapping[str, Any]) -> None:
    outputs = response.get("outputs")
    if not isinstance(outputs, Mapping) or outputs.get("image") != IMAGE_NAME:
        raise JingyuError("blender.bad_response", "the worker reported no image output")
    expected = [IMAGE_NAME, LOG_NAME]
    if "id_mask" in outputs:
        if outputs["id_mask"] != ID_MASK_NAME or not isinstance(response.get("id_map"), Mapping):
            raise JingyuError("blender.bad_response", "id mask output without an id map")
        expected.append(ID_MASK_NAME)
    if "light_mask" in outputs:
        if outputs["light_mask"] != LIGHT_MASK_NAME or not isinstance(
            response.get("light_map"), Mapping
        ):
            raise JingyuError("blender.bad_response", "light mask output without a light map")
        expected.append(LIGHT_MASK_NAME)
    for name in expected:
        if not (staging / name).is_file():
            raise JingyuError("blender.bad_response", f"the worker did not write {name}")
    for key in ("render", "blender"):
        if not isinstance(response.get(key), Mapping):
            raise JingyuError("blender.bad_response", f"the response has no {key!r} section")


__all__ = [
    "PREVIEW_MAX_SAMPLES",
    "PREVIEW_SCALE",
    "Quality",
    "RenderOutcome",
    "quality_overrides",
    "render_scene",
]
