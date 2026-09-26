"""Warnings a scene may declare as intended (``accept_warnings``).

A scene's ``accept_warnings`` covers every warning with that code; an object's
covers only the warnings about that object (their pointer is inside it).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from ..errors import Issue

#: Codes of every warning jingyu can raise about a scene or its render.
ACCEPTABLE_WARNINGS = (
    "spec.scene_unlit",
    "spec.empty_scene",
    "physics.intersection",
    "physics.floating",
    "frame.not_visible",
    "frame.underexposed",
    "frame.overexposed",
    "light.open_to_sky",
)


def drop_accepted(issues: Iterable[Issue], scene: Mapping[str, Any]) -> list[Issue]:
    """*issues* without the warnings the scene accepts."""

    accepted = set(scene.get("accept_warnings", ()))
    by_object = {
        f"/objects/{index}": set(obj.get("accept_warnings", ()))
        for index, obj in enumerate(scene.get("objects", ()))
        if isinstance(obj, Mapping) and obj.get("accept_warnings")
    }
    return [i for i in issues if not (i.severity == "warning" and _accepts(i, accepted, by_object))]


def _accepts(issue: Issue, accepted: set[str], by_object: dict[str, set[str]]) -> bool:
    if issue.code in accepted:
        return True
    parts = issue.pointer.split("/")
    if len(parts) >= 3 and parts[1] == "objects":
        return issue.code in by_object.get("/".join(parts[:3]), set())
    return False


__all__ = ["ACCEPTABLE_WARNINGS", "drop_accepted"]
