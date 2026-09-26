"""Checks that JSON Schema cannot express: identity, references and consistency.

These run on the normalised scene (defaults filled), so every field exists.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from ..environments import ENVIRONMENTS
from ..errors import Issue, pointer_join
from ..geometry import GEOMETRY
from ..materials import MATERIALS

#: Id namespaces.  Groups, objects, lights and cameras are all nodes of one
#: scene and share a namespace; materials have their own.
_NAMESPACES = (
    (("materials",), "Material ids are unique among materials."),
    (
        ("groups", "objects", "lights", "cameras"),
        "Ids are unique across groups, objects, lights and cameras.",
    ),
)
_NODES = ("groups", "objects", "lights", "cameras")


def check_scene(scene: Mapping[str, Any]) -> list[Issue]:
    issues: list[Issue] = []
    issues += _check_ids(scene)
    issues += _check_references(scene)
    issues += _check_hierarchy(scene)
    issues += _check_generators(scene)
    issues += _check_aims(scene)
    issues += _check_ranges(scene)
    issues += _check_warnings(scene)
    return issues


def _check_ids(scene: Mapping[str, Any]) -> list[Issue]:
    issues: list[Issue] = []
    for namespace, hint in _NAMESPACES:
        first_seen: dict[str, str] = {}
        for collection in namespace:
            for index, entry in enumerate(scene[collection]):
                pointer = pointer_join(collection, index, "id")
                ident = entry["id"]
                if ident in first_seen:
                    issues.append(
                        Issue(
                            "spec.duplicate_id",
                            f"id {ident!r} is already used at {first_seen[ident]}",
                            pointer,
                            hint=hint,
                        )
                    )
                else:
                    first_seen[ident] = pointer
    return issues


def _ids(scene: Mapping[str, Any], collection: str) -> list[str]:
    return [entry["id"] for entry in scene[collection]]


def _check_references(scene: Mapping[str, Any]) -> list[Issue]:
    issues: list[Issue] = []
    materials = set(_ids(scene, "materials"))
    for index, obj in enumerate(scene["objects"]):
        ref = obj.get("material")
        if ref is not None and ref not in materials:
            issues.append(
                Issue(
                    "spec.unknown_reference",
                    f"material {ref!r} is not defined",
                    pointer_join("objects", index, "material"),
                    hint=f"defined materials: {sorted(materials) or 'none'}",
                )
            )
    cameras = _ids(scene, "cameras")
    if scene["render"]["camera"] not in cameras:
        issues.append(
            Issue(
                "spec.unknown_reference",
                f"camera {scene['render']['camera']!r} is not defined",
                pointer_join("render", "camera"),
                hint=f"defined cameras: {cameras}",
            )
        )
    return issues


def _check_hierarchy(scene: Mapping[str, Any]) -> list[Issue]:
    """parent names a group or an object, parents form no loop, rest_on names another object."""

    issues: list[Issue] = []
    carriers = {e["id"]: e for c in ("groups", "objects") for e in scene[c]}
    for collection in _NODES:
        for index, entry in enumerate(scene[collection]):
            parent = entry.get("parent")
            if parent is None:
                continue
            if parent == entry["id"] or parent not in carriers:
                issues.append(
                    Issue(
                        "spec.unknown_reference",
                        f"parent {parent!r} is not another group or object",
                        pointer_join(collection, index, "parent"),
                        hint=f"groups: {sorted(g['id'] for g in scene['groups']) or 'none'}; "
                        "any object id also works",
                    )
                )
    for collection in ("groups", "objects"):
        for index, entry in enumerate(scene[collection]):
            seen = {entry["id"]}
            parent = entry.get("parent")
            if parent == entry["id"]:  # reported above as an unknown reference
                continue
            while parent in carriers:
                if parent in seen:
                    issues.append(
                        Issue(
                            "spec.parent_cycle",
                            f"{entry['id']!r} is (indirectly) its own parent",
                            pointer_join(collection, index, "parent"),
                        )
                    )
                    break
                seen.add(parent)
                parent = carriers[parent].get("parent")
    objects = set(_ids(scene, "objects"))
    for index, obj in enumerate(scene["objects"]):
        for position, other in enumerate(obj.get("attached_to", [])):
            if other == obj["id"] or other not in objects:
                issues.append(
                    Issue(
                        "spec.unknown_reference",
                        f"attached_to {other!r} is not another object",
                        pointer_join("objects", index, "attached_to", position),
                        hint=f"defined objects: {sorted(objects)}",
                    )
                )
        support = obj.get("rest_on")
        if support is None:
            continue
        if support == obj["id"]:
            issues.append(
                Issue(
                    "spec.unknown_reference",
                    "an object cannot rest on itself",
                    pointer_join("objects", index, "rest_on"),
                )
            )
        elif support not in objects:
            issues.append(
                Issue(
                    "spec.unknown_reference",
                    f"rest_on {support!r} is not an object",
                    pointer_join("objects", index, "rest_on"),
                    hint=f"defined objects: {sorted(objects)}",
                )
            )
    return issues


def _check_generators(scene: Mapping[str, Any]) -> list[Issue]:
    issues: list[Issue] = []
    materials = {m["id"] for m in scene["materials"]}
    for index, obj in enumerate(scene["objects"]):
        geometry = obj["geometry"]
        definition = GEOMETRY.get(geometry["op"])
        for part, material in (obj.get("part_materials") or {}).items():
            pointer = pointer_join("objects", index, "part_materials", part)
            if part not in definition.parts:
                parts = ", ".join(definition.parts) or "none: it is one piece"
                issues.append(
                    Issue(
                        "spec.unknown_reference",
                        f"{geometry['op']} has no part {part!r}",
                        pointer,
                        hint=f"parts of {geometry['op']}: {parts}",
                    )
                )
            elif material not in materials:
                issues.append(
                    Issue(
                        "spec.unknown_reference",
                        f"material {material!r} is not defined",
                        pointer,
                        hint=f"defined materials: {sorted(materials) or 'none'}",
                    )
                )
        params = {k: v for k, v in geometry.items() if k != "op"}
        for param, message in definition.check(params):
            issues.append(
                Issue(
                    "spec.invalid_parameter",
                    f"{geometry['op']}: {message}",
                    pointer_join("objects", index, "geometry", param),
                )
            )
    environment = scene["world"].get("environment")
    if environment is not None:
        env_family = ENVIRONMENTS.get(environment["family"])
        params = {k: v for k, v in environment.items() if k != "family"}
        for param, message in env_family.check(params):
            issues.append(
                Issue(
                    "spec.invalid_parameter",
                    f"{environment['family']}: {message}",
                    pointer_join("world", "environment", param),
                )
            )
    for index, material in enumerate(scene["materials"]):
        family = MATERIALS.get(material["family"])
        params = {k: v for k, v in material.items() if k not in ("family", "id")}
        for param, message in family.check(params):
            issues.append(
                Issue(
                    "spec.invalid_parameter",
                    f"{material['family']}: {message}",
                    pointer_join("materials", index, param),
                )
            )
    return issues


def _check_aims(scene: Mapping[str, Any]) -> list[Issue]:
    issues: list[Issue] = []
    for collection in ("cameras", "lights"):
        for index, entry in enumerate(scene[collection]):
            target = entry.get("look_at")
            if target is not None and math.dist(target, entry["location"]) < 1e-9:
                issues.append(
                    Issue(
                        "spec.degenerate_aim",
                        "look_at equals location, so there is no direction to aim",
                        pointer_join(collection, index, "look_at"),
                    )
                )
    return issues


def _check_ranges(scene: Mapping[str, Any]) -> list[Issue]:
    issues: list[Issue] = []
    for index, camera in enumerate(scene["cameras"]):
        if camera["clip_start"] >= camera["clip_end"]:
            issues.append(
                Issue(
                    "spec.invalid_range",
                    "clip_start must be below clip_end",
                    pointer_join("cameras", index, "clip_start"),
                )
            )
    return issues


def _check_warnings(scene: Mapping[str, Any]) -> list[Issue]:
    issues: list[Issue] = []
    visible = [o for o in scene["objects"] if o["visible"]]
    if not visible:
        issues.append(
            Issue("spec.empty_scene", "no visible objects", "/objects", severity="warning")
        )
    world = scene["world"]
    lit = (
        any(_emits(light) for light in scene["lights"])
        or (
            "environment" not in world
            and world["strength"] > 0
            and world["color"].lower() != "#000000"
        )
        or ("environment" in world and _environment_lit(world["environment"]))
        or any(_material_emits(m) for m in scene["materials"])
    )
    if not lit:
        issues.append(
            Issue(
                "spec.scene_unlit",
                "no light, emissive material or lit world; the render will be black",
                "/lights",
                severity="warning",
            )
        )
    return issues


def _environment_lit(environment: Mapping[str, Any]) -> bool:
    try:
        return ENVIRONMENTS.run(environment).lit
    except ValueError:  # an invalid parameter, reported by _check_generators
        return True


def _emits(light: Mapping[str, Any]) -> bool:
    return float(light.get("power_w", light.get("strength", 0.0))) > 0.0


def _material_emits(material: Mapping[str, Any]) -> bool:
    return material["family"] == "principled" and float(material["emission_strength"]) > 0.0


__all__ = ["check_scene"]
