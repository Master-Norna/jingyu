"""Physical plausibility checks on resolved placement: sinking and hovering.

These are warnings, not errors: a nail may sink into a wall on purpose and a
lamp may hang from a ceiling that is out of frame.  They exist because a model
writing coordinates cannot see that an orange is 3 cm inside a bowl, and the
render alone rarely makes it obvious.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..errors import Issue, pointer_join
from ..geometry.spatial import WorldMesh, drop_gap, overlaps, rest_shift
from ..placement import Placement

#: Sinking deeper than this is reported.
INTERSECTION_TOLERANCE = 0.001
#: Hovering higher than this is reported.
FLOATING_TOLERANCE = 0.002


def check_physics(scene: Mapping[str, Any], placement: Placement) -> list[Issue]:
    visible = [
        (index, obj)
        for index, obj in enumerate(scene["objects"])
        if obj["visible"] and obj["id"] in placement.meshes
    ]
    meshes = {obj["id"]: placement.meshes[obj["id"]] for _, obj in visible}
    bounds = {obj_id: mesh.bounds() for obj_id, mesh in meshes.items()}

    issues: list[Issue] = []
    sunk: set[str] = set()
    for position, (a_index, a) in enumerate(visible):
        for b_index, b in visible[position + 1 :]:
            if not overlaps(bounds[a["id"]], bounds[b["id"]], INTERSECTION_TOLERANCE):
                continue
            if _attached(a, b):
                continue
            forward = _sinking(meshes[a["id"]], meshes[b["id"]], bounds[b["id"]])
            backward = _sinking(meshes[b["id"]], meshes[a["id"]], bounds[a["id"]])
            if max(forward[0], backward[0]) <= INTERSECTION_TOLERANCE:
                continue
            # An object sunk into another is held by it; do not also call it floating.
            if forward[0] > INTERSECTION_TOLERANCE:
                sunk.add(a["id"])
            if backward[0] > INTERSECTION_TOLERANCE:
                sunk.add(b["id"])
            (sinker_index, sinker), host, (depth, count) = (
                ((b_index, b), a, backward)
                if backward[0] > forward[0]
                else ((a_index, a), b, forward)
            )
            issues.append(
                Issue(
                    "physics.intersection",
                    f"{sinker['id']!r} sinks up to {depth * 1000:.0f} mm into {host['id']!r} "
                    f"({count} of its points are inside)",
                    pointer_join("objects", sinker_index, "location"),
                    severity="warning",
                    hint=_intersection_hint(sinker, host, meshes),
                )
            )

    lowest = min((box[0][2] for box in bounds.values()), default=0.0)
    for index, obj in visible:
        obj_id = obj["id"]
        # Attached things may hang; a reflector card or flag stands on a stand we do not model.
        if obj_id in sunk or obj.get("attached_to") or not obj.get("camera_visible", True):
            continue
        box = bounds[obj_id]
        gap: float | None = None
        below: str | None = None
        for other_id, mesh in meshes.items():
            other_box = bounds[other_id]
            if other_id == obj_id or other_box[0][2] > box[1][2]:
                continue
            if not _footprints_overlap(box, other_box):
                continue
            found = drop_gap(meshes[obj_id], mesh, INTERSECTION_TOLERANCE)
            if found is not None and (gap is None or found < gap):
                gap, below = found, other_id
        if gap is None:
            height = box[0][2] - lowest
            if height > FLOATING_TOLERANCE:
                issues.append(
                    Issue(
                        "physics.floating",
                        f"{obj_id!r} has nothing beneath it and hovers {height * 1000:.0f} mm "
                        "above the lowest object in the scene",
                        pointer_join("objects", index, "location"),
                        severity="warning",
                        hint="ignore this if it hangs or is fixed to something; otherwise "
                        "give it a support with rest_on",
                    )
                )
        elif gap > FLOATING_TOLERANCE:
            issues.append(
                Issue(
                    "physics.floating",
                    f"{obj_id!r} hovers {gap * 1000:.0f} mm above {below!r}",
                    pointer_join("objects", index, "location"),
                    severity="warning",
                    hint=f'set "rest_on": "{below}" to seat it, or ignore this if it hangs',
                )
            )
    return issues


def _sinking(
    a: WorldMesh, b: WorldMesh, b_bounds: tuple[tuple[float, ...], tuple[float, ...]]
) -> tuple[float, int]:
    """Deepest point of *a* inside closed mesh *b*, and how many points are inside."""

    if not b.closed:
        return 0.0, 0
    low, high = b_bounds
    deepest, count = 0.0, 0
    for vertex in a.vertices:
        if not all(low[k] <= vertex[k] <= high[k] for k in range(3)):
            continue
        depth = b.depth_inside(vertex)
        if depth > INTERSECTION_TOLERANCE:
            count += 1
            deepest = max(deepest, depth)
    return deepest, count


def _footprints_overlap(
    a: tuple[tuple[float, ...], tuple[float, ...]], b: tuple[tuple[float, ...], tuple[float, ...]]
) -> bool:
    return all(a[0][k] <= b[1][k] and b[0][k] <= a[1][k] for k in range(2))


def _attached(a: Mapping[str, Any], b: Mapping[str, Any]) -> bool:
    return b["id"] in a.get("attached_to", ()) or a["id"] in b.get("attached_to", ())


def _intersection_hint(
    a: Mapping[str, Any], b: Mapping[str, Any], meshes: Mapping[str, WorldMesh]
) -> str:
    if a.get("rest_on") == b["id"]:
        return f"{a['id']!r} rests on {b['id']!r} but also sinks into it; move it in x or y"
    joined = f'if they are joined on purpose, add "attached_to": ["{b["id"]}"] to {a["id"]!r}'
    low_a, low_b = meshes[a["id"]].bounds()[0][2], meshes[b["id"]].bounds()[0][2]
    if low_a > low_b and rest_shift(meshes[a["id"]], meshes[b["id"]]) is not None:
        return f'set "rest_on": "{b["id"]}" to seat {a["id"]!r} on it; {joined}'
    return f"move them apart; {joined}"


__all__ = ["FLOATING_TOLERANCE", "INTERSECTION_TOLERANCE", "check_physics"]
