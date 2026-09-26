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


#: A room is "open" when more than this share of upward directions from the
#: camera reach the sky.
OPEN_UPWARD_FRACTION = 0.25


def check_frame(
    image_path: Path,
    id_mask_path: Path | None,
    id_map: Mapping[str, Any] | None,
    scene: Mapping[str, Any],
    light_map: Mapping[str, Any] | None = None,
) -> list[Issue]:
    issues: list[Issue] = []
    if id_mask_path is not None and id_map is not None:
        issues += _not_visible(id_mask_path, id_map, scene, light_map)
    issues += _exposure(image_path, scene)
    if light_map is not None:
        issues += _open_to_sky(scene, light_map)
    return issues


def _not_visible(
    id_mask_path: Path,
    id_map: Mapping[str, Any],
    scene: Mapping[str, Any],
    light_map: Mapping[str, Any] | None,
) -> list[Issue]:
    """Objects in the camera's view that cover no pixel: hidden behind something.

    Objects outside the view are usually deliberate (a ceiling, a bounce card) and
    are only listed by describe_layout.  Without a light pass there is no view
    test, so every missing object is reported.  Objects that hold up or carry
    others (a ``rest_on`` support, a parent) are structure, not subject: they
    are not reported when something in front hides them.
    """

    with Image.open(id_mask_path) as image:
        mask = IdMask(image.convert("RGBA"), id_map, scene)
    missing = set(mask.layout()["not_in_frame"])
    carriers = {o.get("rest_on") for o in scene["objects"]} | {
        entry.get("parent")
        for collection in ("groups", "objects", "lights", "cameras")
        for entry in scene[collection]
    }
    missing -= carriers
    in_view = None if light_map is None else set(light_map.get("in_view", []))
    return [
        Issue(
            "frame.not_visible",
            f"{obj['id']!r} is in the camera's view but covers no pixel"
            if in_view is not None
            else f"{obj['id']!r} is visible but covers no pixel of the frame",
            pointer_join("objects", index),
            severity="warning",
            hint="another object hides it completely; if it is only there to shape the light "
            '(a wall or ceiling closing the room), add "accept_warnings": '
            '["frame.not_visible"] to it, or to the group it belongs to'
            if in_view is not None
            else "it is outside the camera's view or hidden behind another object",
        )
        for index, obj in enumerate(scene["objects"])
        if obj["id"] in missing and (in_view is None or obj["id"] in in_view)
    ]


def _open_to_sky(scene: Mapping[str, Any], light_map: Mapping[str, Any]) -> list[Issue]:
    """A room with windows whose surroundings still let the sky in from above."""

    environment = scene["world"].get("environment")
    if environment is None or environment.get("family") != "daylight":
        return []
    has_openings = any(
        o["visible"] and o["geometry"]["op"] == "wall" and o["geometry"].get("openings")
        for o in scene["objects"]
    )
    openness = light_map.get("openness", {})
    upward = float(openness.get("open_upward_fraction", 0.0))
    if not has_openings or upward <= OPEN_UPWARD_FRACTION:
        return []
    return [
        Issue(
            "light.open_to_sky",
            f"the scene has walls with windows, but {upward:.0%} of the directions above "
            "the camera reach the open sky: skylight floods in everywhere, not only through "
            "the windows (cool walls, sky reflected in glossy things)",
            "/objects",
            severity="warning",
            hint="close the room with walls and a ceiling; they may stay out of frame",
        )
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
