"""Warnings a scene may declare as intended (``accept_warnings``).

A scene's ``accept_warnings`` covers every warning with that code; an object's
covers only the warnings about that object (their pointer is inside it); a
group's covers the warnings about every object inside it, at any depth.
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
    nodes = {
        entry["id"]: entry
        for collection in ("groups", "objects")
        for entry in scene.get(collection, ())
        if isinstance(entry, Mapping) and isinstance(entry.get("id"), str)
    }
    by_object = {
        f"/objects/{index}": _object_accepts(obj, nodes)
        for index, obj in enumerate(scene.get("objects", ()))
        if isinstance(obj, Mapping)
    }
    return [i for i in issues if not (i.severity == "warning" and _accepts(i, accepted, by_object))]


def accepted_by_object(scene: Mapping[str, Any]) -> dict[str, set[str]]:
    """Per object id, every warning code it accepts (its own, its groups', the scene's)."""

    nodes = {
        entry["id"]: entry
        for collection in ("groups", "objects")
        for entry in scene.get(collection, ())
        if isinstance(entry, Mapping) and isinstance(entry.get("id"), str)
    }
    everywhere = set(scene.get("accept_warnings", ()))
    return {
        obj["id"]: everywhere | _object_accepts(obj, nodes)
        for obj in scene.get("objects", ())
        if isinstance(obj, Mapping) and isinstance(obj.get("id"), str)
    }


def _object_accepts(obj: Mapping[str, Any], nodes: Mapping[str, Mapping[str, Any]]) -> set[str]:
    """The object's own accepted codes plus those of every group it sits in."""

    codes = set(obj.get("accept_warnings", ()))
    groups = {g["id"] for g in nodes.values() if "geometry" not in g}
    seen: set[str] = set()
    parent = obj.get("parent")
    while isinstance(parent, str) and parent in nodes and parent not in seen:
        seen.add(parent)
        if parent in groups:
            codes |= set(nodes[parent].get("accept_warnings", ()))
        parent = nodes[parent].get("parent")
    return codes


def _accepts(issue: Issue, accepted: set[str], by_object: dict[str, set[str]]) -> bool:
    if issue.code in accepted:
        return True
    parts = issue.pointer.split("/")
    if len(parts) >= 3 and parts[1] == "objects":
        return issue.code in by_object.get("/".join(parts[:3]), set())
    return False


__all__ = ["ACCEPTABLE_WARNINGS", "accepted_by_object", "drop_accepted"]
