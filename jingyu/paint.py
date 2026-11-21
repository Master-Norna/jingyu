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
from itertools import pairwise
from typing import Any, cast

from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageOps

from . import __version__
from .candidate import (
    ID_MAP_NAME,
    ID_MASK_NAME,
    IMAGE_NAME,
    NORMAL_MASK_NAME,
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

    normal_mask: Image.Image | None = None
    if any(op["op"] == "hatch" and op.get("flow") == "surface" for op in operations):
        if not source.has_normal_mask:
            raise JingyuError(
                "candidate.missing_pass",
                f"{candidate_id} has no camera-space normal mask for surface hatching",
            )
        with Image.open(source.file(NORMAL_MASK_NAME)) as opened:
            if opened.mode != "RGBA" or opened.size != image.size:
                raise JingyuError(
                    "candidate.integrity_mismatch", "invalid normal pass for painting"
                )
            normal_mask = opened.copy()

    start = time.perf_counter()
    painted = apply_paint(
        image, operations, id_mask=id_mask, id_map=id_map, normal_mask=normal_mask
    )

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
    normal_mask: Image.Image | None = None,
) -> Image.Image:
    """Apply a sequence of local paint actions, preserving source alpha."""

    alpha = image.getchannel("A") if "A" in image.getbands() else None
    result = image.convert("RGB")
    if normal_mask is not None and (normal_mask.mode != "RGBA" or normal_mask.size != result.size):
        raise JingyuError("candidate.integrity_mismatch", "invalid normal pass for painting")
    for operation in operations:
        mask = _target_mask(result.size, operation["target"], id_mask, id_map)
        kind = operation["op"]
        if kind != "hatch" and "flow" in operation:
            raise JingyuError("tool.invalid_arguments", "flow is only supported for hatch")
        if kind != "hatch" and "taper" in operation:
            raise JingyuError("tool.invalid_arguments", "taper is only supported for hatch")
        if kind != "hatch" and "dryness" in operation:
            raise JingyuError("tool.invalid_arguments", "dryness is only supported for hatch")
        if kind == "wash":
            result = _wash(result, mask, operation)
        elif kind == "hatch":
            result = _hatch(result, mask, operation, normal_mask=normal_mask)
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


def _hatch(
    image: Image.Image,
    target: Image.Image,
    operation: Mapping[str, Any],
    *,
    normal_mask: Image.Image | None = None,
) -> Image.Image:
    width = float(operation.get("width", 1.5))
    strength = float(operation.get("strength", 0.65))
    flow = operation.get("flow", "straight")
    if flow in ("contour", "surface") and "region" in operation["target"]:
        raise JingyuError(
            "tool.invalid_arguments",
            f"{flow} hatching requires an object or assembly part",
            hint="Select an object or part with an exact ID pass; use tone for regions.",
        )
    if flow not in ("straight", "tone", "contour", "surface"):
        raise JingyuError("tool.invalid_arguments", f"unknown hatch flow {flow!r}")
    taper = float(operation.get("taper", 0))
    dryness = float(operation.get("dryness", 0))
    for name, value in (("taper", taper), ("dryness", dryness)):
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise JingyuError("tool.invalid_arguments", f"{name} must be between 0 and 1")
        if name in operation and flow == "straight":
            raise JingyuError(
                "tool.invalid_arguments",
                f"{name} requires a curved hatch flow: tone, contour or surface",
            )
    if flow == "surface" and normal_mask is None:
        raise JingyuError(
            "candidate.missing_pass", "surface hatching requires a camera-space normal pass"
        )
    if width <= 0 or strength <= 0:
        return image
    spacing = float(operation.get("spacing", 8))
    angle = math.radians(float(operation.get("angle", 45)))
    if flow == "straight":
        lines = _straight_hatch_lines(image.size, spacing, width, angle)
    elif flow == "tone":
        lines = _tone_hatch_lines(
            image, target, spacing, width, angle, taper=taper, dryness=dryness
        )
    elif flow == "contour":
        lines = _tone_hatch_lines(
            image, target, spacing, width, angle, contour=True, taper=taper, dryness=dryness
        )
    else:
        lines = _tone_hatch_lines(
            image,
            target,
            spacing,
            width,
            angle,
            normal_mask=normal_mask,
            taper=taper,
            dryness=dryness,
        )
    darkness = ImageOps.invert(ImageOps.grayscale(image))
    stroke = ImageChops.multiply(ImageChops.multiply(lines, darkness), target)
    return _composite(image, stroke, str(operation.get("color", "#27211e")), strength)


def _straight_hatch_lines(
    size: tuple[int, int], spacing: float, width: float, angle: float
) -> Image.Image:
    """Parallel strokes: preserve the original hatch layout exactly."""

    dx, dy = math.cos(angle), math.sin(angle)
    nx, ny = -dy, dx
    center_x, center_y = size[0] / 2, size[1] / 2
    radius = math.hypot(*size)
    lines = Image.new("L", size, 0)
    draw = ImageDraw.Draw(lines)
    for step in range(-math.ceil(radius / spacing), math.ceil(radius / spacing) + 1):
        x = center_x + nx * step * spacing
        y = center_y + ny * step * spacing
        draw.line(
            ((x - dx * radius, y - dy * radius), (x + dx * radius, y + dy * radius)),
            fill=255,
            width=max(1, round(width)),
        )
    return lines


def _tone_tangent(
    gray: Image.Image, x: float, y: float, previous: tuple[float, float]
) -> tuple[float, float]:
    """A tangent to local luminance contours; keep orientation continuous."""

    px = max(0, min(gray.width - 1, round(x)))
    py = max(0, min(gray.height - 1, round(y)))
    gx = cast(int, gray.getpixel((min(gray.width - 1, px + 2), py))) - cast(
        int, gray.getpixel((max(0, px - 2), py))
    )
    gy = cast(int, gray.getpixel((px, min(gray.height - 1, py + 2)))) - cast(
        int, gray.getpixel((px, max(0, py - 2)))
    )
    magnitude = math.hypot(gx, gy)
    if magnitude < 3:
        return previous
    tx, ty = -gy / magnitude, gx / magnitude
    if tx * previous[0] + ty * previous[1] < 0:
        return -tx, -ty
    return tx, ty


def _surface_tangent(
    normals: Image.Image, x: float, y: float, previous: tuple[float, float]
) -> tuple[float, float]:
    """Projected isodepth tangent from camera-space 3D surface normals.

    Camera Y points up, while image Y points down. Front-facing surfaces
    have no stable projected depth direction, so retain the previous angle.
    """

    px = max(0, min(normals.width - 1, round(x)))
    py = max(0, min(normals.height - 1, round(y)))
    r, g, _, alpha = cast(tuple[int, int, int, int], normals.getpixel((px, py)))
    if alpha < 128:
        return previous
    nx = r / 127.5 - 1.0
    ny = 1.0 - g / 127.5
    magnitude = math.hypot(nx, ny)
    if magnitude < 0.08:
        return previous
    tx, ty = -ny / magnitude, nx / magnitude
    if tx * previous[0] + ty * previous[1] < 0:
        return -tx, -ty
    return tx, ty


def _trace_tone(
    gray: Image.Image,
    target: Image.Image,
    origin: tuple[int, int],
    direction: tuple[float, float],
    spacing: float,
    *,
    normal_mask: Image.Image | None = None,
) -> list[tuple[float, float]]:
    """Walk a short curved stroke without leaving the requested target."""

    step = max(1.0, min(3.0, spacing / 3))
    steps = max(3, round(2 * spacing / step))
    x, y = float(origin[0]), float(origin[1])
    points: list[tuple[float, float]] = []
    for _ in range(steps):
        wanted = (
            _surface_tangent(normal_mask, x, y, direction)
            if normal_mask is not None
            else _tone_tangent(gray, x, y, direction)
        )
        vx = 0.35 * direction[0] + 0.65 * wanted[0]
        vy = 0.35 * direction[1] + 0.65 * wanted[1]
        magnitude = math.hypot(vx, vy)
        dx, dy = vx / magnitude, vy / magnitude
        next_x, next_y = x + dx * step, y + dy * step
        if not (0 <= next_x < gray.width and 0 <= next_y < gray.height):
            break
        if target.getpixel((int(next_x), int(next_y))) == 0:
            break
        x, y = next_x, next_y
        points.append((x, y))
        direction = (dx, dy)
    return points


def _draw_tapered_stroke(
    draw: ImageDraw.ImageDraw,
    points: Sequence[tuple[float, float]],
    width: float,
    taper: float,
) -> None:
    """Rasterize a short stroke with a deterministic start/end pressure envelope.

    A stroke's middle is fully inked; its endpoints shrink and lighten. No
    random state or whole-image per-stroke buffers are used.
    """

    if len(points) == 1:
        draw.point(points[0], fill=255)
        return
    last = len(points) - 1
    for index, (x, y) in enumerate(points):
        pressure = math.sin(math.pi * index / last)
        attenuation = taper * (1 - pressure)
        diameter = max(1.0, width * (1 - 0.8 * attenuation))
        opacity = max(1, round(255 * (1 - 0.8 * attenuation)))
        if index:
            draw.line(
                (points[index - 1], (x, y)),
                fill=opacity,
                width=max(1, round(diameter)),
            )
        radius = diameter / 2
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=opacity)


def _draw_dry_stroke(
    draw: ImageDraw.ImageDraw,
    points: Sequence[tuple[float, float]],
    width: float,
    taper: float,
    dryness: float,
) -> None:
    """Deplete ink along arc length and omit coherent, deterministic dry segments.

    Gaps are evaluated per short distance cell, not as independent pixel noise.
    The phase depends only on the stroke origin and cell index; no RNG state,
    per-stroke image buffer or external data is needed.
    """

    if len(points) == 1:
        draw.point(points[0], fill=255)
        return

    distances = [0.0]
    for start, end in pairwise(points):
        distances.append(distances[-1] + math.dist(start, end))
    total = max(1.0, distances[-1])
    grain_length = max(3.0, width)
    seed = (round(points[0][0]) * 41 + round(points[0][1]) * 67) & 0xFFFF
    previous_wet = False
    last = len(points) - 1

    for index, (x, y) in enumerate(points):
        progress = distances[index] / total
        cell = int(distances[index] / grain_length)
        signature = (seed + 131 * cell) ^ (17 * cell * cell)
        skipped = (signature % 101) / 101 < 0.55 * dryness * progress
        if skipped:
            previous_wet = False
            continue

        pressure = math.sin(math.pi * index / last)
        attenuation = taper * (1 - pressure)
        diameter = max(1.0, width * (1 - 0.8 * attenuation))
        ink = (1 - 0.8 * attenuation) * (1 - 0.75 * dryness * progress)
        opacity = max(1, round(255 * ink))
        if previous_wet:
            draw.line(
                (points[index - 1], (x, y)),
                fill=opacity,
                width=max(1, round(diameter)),
            )
        radius = diameter / 2
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=opacity)
        previous_wet = True


def _tone_hatch_lines(
    image: Image.Image,
    target: Image.Image,
    spacing: float,
    width: float,
    angle: float,
    *,
    contour: bool = False,
    normal_mask: Image.Image | None = None,
    taper: float = 0.0,
    dryness: float = 0.0,
) -> Image.Image:
    """Draw short strokes along tonal, silhouette or surface-normal contours."""

    lines = Image.new("L", image.size, 0)
    box = target.getbbox()
    if box is None:
        return lines

    pitch = max(2, round(spacing))
    x0, y0, x1, y1 = box
    estimated = math.ceil((x1 - x0) / pitch) * math.ceil((y1 - y0) / pitch)
    if estimated > (50_000 if taper or dryness else 250_000):
        raise JingyuError(
            "tool.invalid_arguments",
            "curved hatching exceeds the stroke budget",
            hint="Increase spacing or paint a smaller target.",
        )

    shade = ImageOps.grayscale(image).filter(ImageFilter.GaussianBlur(1.5))
    # A blurred exact-ID selection provides a screen-space silhouette field:
    # its gradients run across the boundary, not across shadows or texture.
    # At its flat interior, _tone_tangent retains the user-supplied angle.
    gray = target.filter(ImageFilter.GaussianBlur(max(1.5, spacing))) if contour else shade
    # Choose one real target pixel per tile, not a lattice centre that may miss
    # thin, diagonal or fragmented parts altogether.
    eligible = ImageChops.multiply(target, shade.point(lambda value: 255 if value < 250 else 0))
    fallback = (math.cos(angle), math.sin(angle))
    draw = ImageDraw.Draw(lines)
    on_pixel = bytes((255,))
    for y in range(y0, y1, pitch):
        for x in range(x0, x1, pitch):
            tile = eligible.crop((x, y, min(x + pitch, x1), min(y + pitch, y1)))
            offset = tile.tobytes().find(on_pixel)
            if offset < 0:
                continue
            sx, sy = x + offset % tile.width, y + offset // tile.width
            initial = (
                _surface_tangent(normal_mask, sx, sy, fallback)
                if normal_mask is not None
                else _tone_tangent(gray, sx, sy, fallback)
            )
            backward = _trace_tone(
                gray,
                target,
                (sx, sy),
                (-initial[0], -initial[1]),
                spacing,
                normal_mask=normal_mask,
            )
            forward = _trace_tone(gray, target, (sx, sy), initial, spacing, normal_mask=normal_mask)
            points = [*reversed(backward), (float(sx), float(sy)), *forward]
            if dryness:
                _draw_dry_stroke(draw, points, width, taper, dryness)
            elif taper:
                _draw_tapered_stroke(draw, points, width, taper)
            else:
                # Preserve every existing pixel when taper and dryness are zero.
                # A one-pixel-wide part may not admit a continuous segment.
                draw.point((sx, sy), fill=255)
                if len(points) >= 2:
                    draw.line(points, fill=255, width=max(1, round(width)), joint="curve")
    return lines


def _edge_ink(image: Image.Image, target: Image.Image, operation: Mapping[str, Any]) -> Image.Image:
    width = float(operation.get("width", 1.5))
    if width <= 0:
        return image
    radius = max(1, math.ceil(width))
    kernel = radius * 2 + 1
    padded = ImageOps.expand(target, border=radius, fill=0)
    eroded = padded.filter(ImageFilter.MinFilter(kernel)).crop(
        (radius, radius, radius + target.width, radius + target.height)
    )
    boundary = ImageChops.subtract(target, eroded)
    return _composite(
        image,
        boundary,
        str(operation.get("color", "#27211e")),
        float(operation.get("strength", 0.9)),
    )


__all__ = ["apply_paint", "paint_candidate"]
