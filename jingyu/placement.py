"""Where every node of a normalised scene ends up in world space.

Standard library only: the host resolves placement to report problems before
rendering, and the Blender worker resolves the same placement to build the
scene, so both sides agree to the last digit.

* A node's matrix is ``parent_world @ T @ R @ S``; groups chain the same way.
* An object with ``rest_on`` keeps its world x and y, and moves vertically until
  its lowest points touch the topmost surface of the supporting object beneath
  it.  The vertical move is converted back into the object's local frame, so a
  rotated group still works.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from .errors import Issue, pointer_join
from .geometry import GEOMETRY
from .geometry.spatial import WorldMesh, rest_shift
from .geometry.transform import IDENTITY, Mat4, Vec3, compose, multiply, solve_linear

_UNIT_SCALE = (1.0, 1.0, 1.0)


@dataclass
class Placement:
    """World matrices, resolved object locations and world meshes of one scene."""

    #: World matrix of every group and object, and of every light and camera
    #: before aiming (``look_at`` replaces the rotation part).
    world: dict[str, Mat4] = field(default_factory=dict)
    #: World matrix of each node's parent (identity at the top level).
    parent_world: dict[str, Mat4] = field(default_factory=dict)
    #: Local location of every object after ``rest_on`` is applied.
    locations: dict[str, Vec3] = field(default_factory=dict)
    #: Objects in world space, keyed by id.
    meshes: dict[str, WorldMesh] = field(default_factory=dict)
    #: Objects whose height came from ``rest_on``.
    rested: dict[str, str] = field(default_factory=dict)
    issues: list[Issue] = field(default_factory=list)


def resolve_placement(scene: Mapping[str, Any], *, build_meshes: bool = True) -> Placement:
    """Resolve groups, parents and ``rest_on`` for a scene that passed semantic checks.

    Failures (a support with nothing under the object, a ``rest_on`` cycle) are
    ``spec.placement_failed`` issues; the object then keeps its given location.
    """

    placement = Placement()
    groups = {g["id"]: g for g in scene.get("groups", [])}
    for group_id in groups:
        _group_world(group_id, groups, placement)

    objects = {o["id"]: (index, o) for index, o in enumerate(scene["objects"])}
    for obj in scene["objects"]:
        parent = _parent_matrix(obj, placement)
        placement.parent_world[obj["id"]] = parent
        location = tuple(float(v) for v in obj["location"])
        placement.locations[obj["id"]] = (location[0], location[1], location[2])
        placement.world[obj["id"]] = multiply(
            parent, compose(obj["location"], obj["rotation"], obj["scale"])
        )
        if build_meshes or obj.get("rest_on") is not None or _supports_other(obj, scene):
            mesh = GEOMETRY.run(obj["geometry"])
            placement.meshes[obj["id"]] = WorldMesh.from_mesh(mesh, placement.world[obj["id"]])

    unresolved: set[str] = set()
    for obj_id in _rest_order(scene, objects, placement):
        index, obj = objects[obj_id]
        support = obj["rest_on"]
        if support in unresolved:  # already reported for the support
            unresolved.add(obj_id)
            continue
        shift = rest_shift(placement.meshes[obj_id], placement.meshes[support])
        if shift is None:
            unresolved.add(obj_id)
            placement.issues.append(
                Issue(
                    "spec.placement_failed",
                    f"{obj_id!r} cannot rest on {support!r}: no part of {support!r} is "
                    f"directly below it",
                    pointer_join("objects", index, "rest_on"),
                    hint=f"move {obj_id!r} in x or y so it is above {support!r}",
                )
            )
            continue
        _move_vertically(obj_id, shift, placement)
        placement.rested[obj_id] = support

    for collection in ("lights", "cameras"):
        for entry in scene[collection]:
            parent = _parent_matrix(entry, placement)
            placement.parent_world[entry["id"]] = parent
            rotation = entry.get("rotation", (0.0, 0.0, 0.0))
            placement.world[entry["id"]] = multiply(
                parent, compose(entry["location"], rotation, _UNIT_SCALE)
            )
    return placement


def _group_world(group_id: str, groups: Mapping[str, Any], placement: Placement) -> Mat4:
    if group_id in placement.world:
        return placement.world[group_id]
    group = groups[group_id]
    parent_id = group.get("parent")
    parent = IDENTITY if parent_id is None else _group_world(parent_id, groups, placement)
    placement.parent_world[group_id] = parent
    scale = float(group["scale"])
    world = multiply(parent, compose(group["location"], group["rotation"], (scale, scale, scale)))
    placement.world[group_id] = world
    return world


def _parent_matrix(entry: Mapping[str, Any], placement: Placement) -> Mat4:
    parent_id = entry.get("parent")
    return IDENTITY if parent_id is None else placement.world[parent_id]


def _supports_other(obj: Mapping[str, Any], scene: Mapping[str, Any]) -> bool:
    return any(other.get("rest_on") == obj["id"] for other in scene["objects"])


def _rest_order(
    scene: Mapping[str, Any],
    objects: Mapping[str, tuple[int, Mapping[str, Any]]],
    placement: Placement,
) -> list[str]:
    """Objects with ``rest_on``, supports first; cycles become issues."""

    order: list[str] = []
    state: dict[str, str] = {}

    def visit(obj_id: str) -> bool:
        if state.get(obj_id) == "done":
            return True
        if state.get(obj_id) in ("visiting", "failed"):
            return False
        state[obj_id] = "visiting"
        support = objects[obj_id][1].get("rest_on")
        ok = support is None or visit(support)
        state[obj_id] = "done" if ok else "failed"
        if ok and support is not None:
            order.append(obj_id)
        return ok

    for obj in scene["objects"]:
        if obj.get("rest_on") is None or state.get(obj["id"]) in ("done", "failed"):
            continue
        if not visit(obj["id"]):
            index = objects[obj["id"]][0]
            placement.issues.append(
                Issue(
                    "spec.placement_failed",
                    f"rest_on of {obj['id']!r} leads into a cycle: objects cannot support "
                    "each other in a loop",
                    pointer_join("objects", index, "rest_on"),
                )
            )
    return order


def _move_vertically(obj_id: str, shift: float, placement: Placement) -> None:
    parent = placement.parent_world[obj_id]
    delta = solve_linear(parent, (0.0, 0.0, shift))
    x, y, z = placement.locations[obj_id]
    placement.locations[obj_id] = (x + delta[0], y + delta[1], z + delta[2])
    world = list(placement.world[obj_id])
    world[11] += shift
    placement.world[obj_id] = tuple(world)
    placement.meshes[obj_id] = placement.meshes[obj_id].translated((0.0, 0.0, shift))


__all__ = ["Placement", "resolve_placement"]
