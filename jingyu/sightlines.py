"""What the background pixels of a render look through.

The id mask says a pixel shows no object: it is the world (sky, ground, a studio
colour).  When that pixel sits inside a room, the eye reached the world through
an opening, and the useful answer is which one: "the bright strip at the left
edge is the sky through opening 0 of wall_left".  This module follows the line
of sight of such pixels on the host and reports the openings they pass.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any

from .camera import CameraModel
from .errors import pointer_join
from .placement import resolve_placement

#: At most this many background pixels are followed; counts are scaled up.
MAX_RAYS = 400


def background_openings(
    scene: Mapping[str, Any], width: int, height: int, pixels: Iterable[tuple[int, int]]
) -> list[dict[str, Any]]:
    """For background *pixels* of a *width* x *height* render of *scene*: the openings
    their lines of sight pass through first, most pixels first."""

    pixels = list(pixels)
    if not pixels:
        return []
    placement = resolve_placement(scene)
    portals = [
        (obj_id, index, portal)
        for index, obj in enumerate(scene["objects"])
        if obj["visible"] and (obj_id := obj["id"]) in placement.meshes
        for portal in placement.meshes[obj_id].portals
    ]
    if not portals:
        return []
    camera = CameraModel.from_scene(scene, placement, resolution=(width, height))
    step = max(1, len(pixels) // MAX_RAYS)
    sampled = pixels[::step]
    counts: Counter[tuple[str, int, str]] = Counter()
    for x, y in sampled:
        origin, direction = camera.pixel_ray(x, y)
        best: tuple[float, tuple[str, int, str]] | None = None
        for obj_id, index, portal in portals:
            crossing = portal.crossing(origin, direction)
            if crossing is None:
                continue
            distance, a, b = crossing
            if abs(a) <= 1.0 and abs(b) <= 1.0 and (best is None or distance < best[0]):
                best = (distance, (obj_id, index, portal.key))
        if best is not None:
            counts[best[1]] += 1
    scale = len(pixels) / len(sampled)
    return [
        {
            "object": obj_id,
            "opening": key,
            "pointer": pointer_join("objects", index, "geometry", *key.split("/")),
            "pixels": round(count * scale),
            "fraction_of_background": round(count / len(sampled), 4),
        }
        for (obj_id, index, key), count in counts.most_common()
    ]


__all__ = ["MAX_RAYS", "background_openings"]
