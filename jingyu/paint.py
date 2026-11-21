"""Deterministic 2D paint actions over immutable rendered candidates.

Paint never starts Blender or changes a source candidate. Object and part targets
are derived from the exact ID pass; strokes are composed only inside that mask.
"""

from __future__ import annotations

import copy
import math
import shutil
import time
import uuid
from collections.abc import Mapping, Sequence
from typing import Any

from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageOps

from . import __version__
from .candidate import (
    ID_MAP_NAME,
    ID_MASK_NAME,
    IMAGE_NAME,
    RECEIPT_NAME,
    RENDER_NAME,
    Candidate,
    CandidateStore,
    new_candidate_id,
    utc_now_iso,
)
from .canonical_json import load_file_strict
from .errors import JingyuError
from .idmask import encode_index
from .workspace import Workspace


def paint_candidate(
    workspace: Workspace,
    candidate_id: str,
    operations: Sequence[Mapping[str, Any]],
    *,
    base: str = "image",
) -> Candidate:
    """Paint an existing candidate into a new one without invoking Blender."""

    if base not in ("image", "render") or not operations:
        raise JingyuError("tool.invalid_arguments", "paint needs a base and operations")
    store = CandidateStore(workspace)
    source = store.open(candidate_id)
    problems = store.verify(source)
    if problems:
        raise JingyuError(
            "candidate.integrity_mismatch",
            f"{candidate_id}: {', '.join(problems)}",
        )

    source_path = (
        source.file(RENDER_NAME)
        if base == "render" and source.file(RENDER_NAME).is_file()
        else source.image_path
    )
    with Image.open(source_path) as original:
        original.load()
        image = original.copy()

    has_object_targets = any(
        "object" in op["target"] or "part" in op["target"] for op in operations
    )
    id_mask: Image.Image | None = None
    id_map: Mapping[str, Any] | None = None
    if has_object_targets:
        if not source.has_id_mask:
            raise JingyuError(
                "candidate.missing_pass",
                f"{candidate_id} has no id mask for object-based painting",
            )
        with Image.open(source.file(ID_MASK_NAME)) as opened:
            id_mask = opened.convert("RGBA")
        id_map = load_file_strict(source.file(ID_MAP_NAME))
        if id_mask.size != image.size or not isinstance(id_map, dict):
            raise JingyuError("candidate.integrity_mismatch", "invalid ID pass for painting")

    start = time.perf_counter()
    painted = apply_paint(image, operations, id_mask=id_mask, id_map=id_map)

    staging = workspace.staging_dir / uuid.uuid4().hex
    staging.mkdir(parents=True)
    try:
        for item in source.path.iterdir():
            if item.name in (RECEIPT_NAME, IMAGE_NAME):
                continue
            shutil.copy2(item, staging / item.name)
        # Keep the original photographic view even when the input had no style.
        if not (staging / RENDER_NAME).exists():
            shutil.copy2(source.image_path, staging / RENDER_NAME)
        painted.save(staging / IMAGE_NAME, format="PNG")
        scene_hash = source.receipt["scene"]["sha256"]
        receipt = {
            "candidate_id": _new_id(store, scene_hash),
            "created_at": utc_now_iso(),
            "jingyu_version": __version__,
            "scene": copy.deepcopy(source.receipt["scene"]),
            "quality": source.receipt["quality"],
            "render": copy.deepcopy(source.receipt["render"]),
            "blender": copy.deepcopy(source.receipt["blender"]),
            "warnings": copy.deepcopy(source.receipt.get("warnings", [])),
            "timings_ms": {"paint": round((time.perf_counter() - start) * 1000)},
            "derived_from": {"candidate_id": source.id, "kind": "paint"},
            "paint": {"base": base, "operations": copy.deepcopy(list(operations))},
        }
        return store.commit(staging, receipt)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _new_id(store: CandidateStore, scene_hash: str) -> str:
    for _ in range(64):
        candidate_id = new_candidate_id(scene_hash)
        if not (store.root / candidate_id).exists():
            return candidate_id
    raise JingyuError("internal.unexpected", "could not allocate a paint candidate id")


def apply_paint(
    image: Image.Image,
    operations: Sequence[Mapping[str, Any]],
    *,
    id_mask: Image.Image | None = None,
    id_map: Mapping[str, Any] | None = None,
) -> Image.Image:
    """Apply a sequence of local paint actions, preserving source alpha."""

    alpha = image.getchannel("A") if "A" in image.getbands() else None
    result = image.convert("RGB")
    for operation in operations:
        mask = _target_mask(result.size, operation["target"], id_mask, id_map)
        kind = operation["op"]
        if kind == "wash":
            result = _wash(result, mask, operation)
        elif kind == "hatch":
            result = _hatch(result, mask, operation)
        elif kind == "edge_ink":
            result = _edge_ink(result, mask, operation)
        else:
            raise JingyuError("tool.invalid_arguments", f"unknown paint action {kind!r}")
    if alpha is not None:
        result.putalpha(alpha)
    return result


def _target_mask(
    size: tuple[int, int],
    target: Mapping[str, Any],
    id_mask: Image.Image | None,
    id_map: Mapping[str, Any] | None,
) -> Image.Image:
    if "region" in target:
        box = target["region"]
        width, height = size
        if "x0" in box:
            x0, y0, x1, y1 = (box[key] for key in ("x0", "y0", "x1", "y1"))
        else:
            x0 = math.floor(box["u0"] * width)
            y0 = math.floor(box["v0"] * height)
            x1 = round(box["u1"] * width)
            y1 = round(box["v1"] * height)
        if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
            raise JingyuError("locate.out_of_bounds", "paint region lies outside the image")
        mask = Image.new("L", size, 0)
        ImageDraw.Draw(mask).rectangle((x0, y0, x1 - 1, y1 - 1), fill=255)
        return mask

    if id_mask is None or id_map is None:
        raise JingyuError("candidate.missing_pass", "object painting requires an ID mask")
    if id_mask.size != size:
        raise JingyuError("candidate.integrity_mismatch", "ID mask and image sizes differ")

    if "object" in target:
        object_id = target["object"]
        part_id = None
    else:
        object_id = target["part"]["object"]
        part_id = target["part"]["part"]
    indexes = [
        int(index)
        for index, entry in id_map["objects"].items()
        if entry["object"] == object_id and (part_id is None or entry.get("part") == part_id)
    ]
    if not indexes:
        raise JingyuError(
            "spec.unknown_reference", f"no mask entry for {object_id!r} / {part_id!r}"
        )
    colors = {bytes(encode_index(index)) for index in indexes}
    rgba = id_mask.convert("RGBA").tobytes()
    selected = bytearray(size[0] * size[1])
    for offset in range(0, len(rgba), 4):
        if rgba[offset + 3] and rgba[offset : offset + 3] in colors:
            selected[offset // 4] = 255
    mask = Image.frombytes("L", size, bytes(selected))
    if mask.getbbox() is None:
        raise JingyuError(
            "tool.invalid_arguments", f"{object_id!r} has no visible pixels in this candidate"
        )
    return mask


def _opacity(mask: Image.Image, strength: float) -> Image.Image:
    return mask.point(lambda value: round(value * strength))


def _composite(image: Image.Image, mask: Image.Image, color: str, strength: float) -> Image.Image:
    if strength <= 0 or mask.getbbox() is None:
        return image
    tint = Image.new("RGB", image.size, color)
    return Image.composite(tint, image, _opacity(mask, strength))


def _wash(image: Image.Image, target: Image.Image, operation: Mapping[str, Any]) -> Image.Image:
    return _composite(
        image,
        target,
        str(operation.get("color", "#6d5948")),
        float(operation.get("strength", 0.35)),
    )


def _hatch(image: Image.Image, target: Image.Image, operation: Mapping[str, Any]) -> Image.Image:
    width = float(operation.get("width", 1.5))
    strength = float(operation.get("strength", 0.65))
    if width <= 0 or strength <= 0:
        return image
    spacing = float(operation.get("spacing", 8))
    angle = math.radians(float(operation.get("angle", 45)))
    dx, dy = math.cos(angle), math.sin(angle)
    nx, ny = -dy, dx
    center_x, center_y = image.width / 2, image.height / 2
    radius = math.hypot(*image.size)
    lines = Image.new("L", image.size, 0)
    draw = ImageDraw.Draw(lines)
    for step in range(-math.ceil(radius / spacing), math.ceil(radius / spacing) + 1):
        x = center_x + nx * step * spacing
        y = center_y + ny * step * spacing
        draw.line(
            ((x - dx * radius, y - dy * radius), (x + dx * radius, y + dy * radius)),
            fill=255,
            width=max(1, round(width)),
        )
    darkness = ImageOps.invert(ImageOps.grayscale(image))
    stroke = ImageChops.multiply(ImageChops.multiply(lines, darkness), target)
    return _composite(image, stroke, str(operation.get("color", "#27211e")), strength)


def _edge_ink(image: Image.Image, target: Image.Image, operation: Mapping[str, Any]) -> Image.Image:
    width = float(operation.get("width", 1.5))
    if width <= 0:
        return image
    kernel = max(3, math.ceil(width) * 2 + 1)
    boundary = ImageChops.subtract(target, target.filter(ImageFilter.MinFilter(kernel)))
    return _composite(
        image,
        boundary,
        str(operation.get("color", "#27211e")),
        float(operation.get("strength", 0.9)),
    )


__all__ = ["apply_paint", "paint_candidate"]
