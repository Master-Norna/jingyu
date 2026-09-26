"""Checks on a finished render: what the frame shows, and whether it is exposed.

The scene can be valid and still produce a picture where an object is out of
frame or the whole image is murky.  These checks turn such facts into warnings
on the receipt, so a model does not have to notice them by eye.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from PIL import Image

from .conventions import srgb_to_linear
from .errors import Issue, pointer_join
from .locate import IdMask

#: Mean display luma below which a frame reads as underexposed, and the level a
#: suggested exposure change aims for (about middle grey on screen).
DARK_MEAN_LUMA = 0.22
TARGET_MEAN_LUMA = 0.42
#: Share of pure-white pixels above which highlights are called blown.
MAX_CLIPPED_FRACTION = 0.05


def check_frame(
    image_path: Path,
    id_mask_path: Path | None,
    id_map: Mapping[str, Any] | None,
    scene: Mapping[str, Any],
) -> list[Issue]:
    issues: list[Issue] = []
    if id_mask_path is not None and id_map is not None:
        issues += _not_visible(id_mask_path, id_map, scene)
    issues += _exposure(image_path, scene)
    return issues


def _not_visible(
    id_mask_path: Path, id_map: Mapping[str, Any], scene: Mapping[str, Any]
) -> list[Issue]:
    with Image.open(id_mask_path) as image:
        mask = IdMask(image.convert("RGBA"), id_map, scene)
    missing = set(mask.layout()["not_in_frame"])
    return [
        Issue(
            "frame.not_visible",
            f"{obj['id']!r} is visible but covers no pixel of the frame",
            pointer_join("objects", index),
            severity="warning",
            hint="it is outside the camera's view or hidden behind another object",
        )
        for index, obj in enumerate(scene["objects"])
        if obj["id"] in missing
    ]


def luma_stats(image_path: Path) -> dict[str, float]:
    """Mean display luma and the shares of clipped and crushed pixels (opaque only)."""

    with Image.open(image_path) as image:
        rgba = image.convert("RGBA")
    rgba.thumbnail((256, 256))
    data = rgba.tobytes()
    total = clipped = crushed = 0
    luma_sum = 0.0
    for offset in range(0, len(data), 4):
        r, g, b, a = data[offset : offset + 4]
        if a < 128:
            continue
        total += 1
        luma = (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255.0
        luma_sum += luma
        if min(r, g, b) >= 250:
            clipped += 1
        if max(r, g, b) <= 5:
            crushed += 1
    if total == 0:
        return {"mean_luma": 0.0, "clipped_fraction": 0.0, "crushed_fraction": 0.0}
    return {
        "mean_luma": luma_sum / total,
        "clipped_fraction": clipped / total,
        "crushed_fraction": crushed / total,
    }


def _exposure(image_path: Path, scene: Mapping[str, Any]) -> list[Issue]:
    stats = luma_stats(image_path)
    current = float(scene["render"]["exposure"])
    mean = stats["mean_luma"]
    if 0.0 < mean < DARK_MEAN_LUMA and stats["clipped_fraction"] < 0.01:
        step = math.log2(srgb_to_linear(TARGET_MEAN_LUMA) / max(srgb_to_linear(mean), 1e-4))
        step = min(step, 4.0)
        return [
            Issue(
                "frame.underexposed",
                f"the frame is dark overall (mean luma {mean:.2f} of 1)",
                pointer_join("render", "exposure"),
                severity="warning",
                hint=f"if it is not meant to be a dark picture, try exposure "
                f"{current + step:+.1f} (now {current:+.1f}), or add light",
            )
        ]
    if stats["clipped_fraction"] > MAX_CLIPPED_FRACTION:
        return [
            Issue(
                "frame.overexposed",
                f"{stats['clipped_fraction']:.0%} of the frame is pure white",
                pointer_join("render", "exposure"),
                severity="warning",
                hint=f"lower exposure (now {current:+.1f}) or dim the brightest light",
            )
        ]
    return []


__all__ = ["check_frame", "luma_stats"]
