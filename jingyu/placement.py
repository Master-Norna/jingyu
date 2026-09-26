"""Where every node of a normalised scene ends up in world space.

Standard library only: the host resolves placement to report problems before
rendering, and the Blender worker resolves the same placement to build the
scene, so both sides agree to the last digit.

* A node's matrix is ``parent_world @ T @ R @ S``.  A parent is a group or an
  object; nodes are resolved parents first, so a child follows its parent
  wherever ``rest_on`` put it (the oranges move with the bowl).
* An object with ``rest_on`` keeps its world x and y, and moves vertically until
  its lowest points touch the topmost surface of the supporting object beneath
  it.  The vertical move is converted back into the object's local frame, so a
  rotated group still works.
* Lights and cameras are aimed here too (``look_at`` or ``rotation``), so the
  host can project and cast rays exactly as the renderer sees the scene.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from .errors import Issue, pointer_join
from .geometry import GEOMETRY
from .geometry.spatial import WorldMesh, rest_shift
from .geometry.transform import (
    IDENTITY,
    Mat4,
    Vec3,
    apply,
    compose,
    look_rotation,
    multiply,
    rotation_part,
    rotation_xyz,
    solve_linear,
    with_translation,
)

_UNIT_SCALE = (1.0, 1.0, 1.0)


@dataclass
class Placement:
    """World matrices, resolved object locations and world meshes of one scene."""

    #: World matrix of every group and object, and of every light and camera
    #: before aiming (``look_at`` replaces the rotation part).
    world: dict[str, Mat4] = field(default_factory=dict)
    #: World matrix of each node's parent (identity at the top level).
    parent_world: dict[str, Mat4] = field(default_factory=dict)
    #: Lights and cameras after aiming: rotation only (no scale), looking along -Z.
    oriented: dict[str, Mat4] = field(default_factory=dict)
    #: Local location of every object after ``rest_on`` is applied.
    locations: dict[str, Vec3] = field(default_factory=dict)
    #: Objects in world space, keyed by id.
    meshes: dict[str, WorldMesh] = field(default_factory=dict)
    #: Objects whose height came from ``rest_on``.
    rested: dict[str, str] = field(default_factory=dict)
    issues: list[Issue] = field(default_factory=list)


def resolve_placement(scene: Mapping[str, Any], *, build_meshes: bool = True) -> Placement:
    """Resolve parents, groups and ``rest_on`` for a scene that passed semantic checks.

    Failures (a support with nothing under the object, a loop through parents and
    supports) are ``spec.placement_failed`` issues; the object then keeps its
    given location.
    """

    placement = Placement()
    nodes: dict[str, tuple[str, int, Mapping[str, Any]]] = {}
    for collection in ("groups", "objects"):
        for index, entry in enumerate(scene.get(collection, [])):
            nodes[entry["id"]] = (collection, index, entry)
    need_mesh = {o.get("rest_on") for o in scene["objects"]} - {None}

    failed: set[str] = set()
    for node_id in _order(nodes, placement):
        collection, index, entry = nodes[node_id]
        parent_id = entry.get("parent")
        if parent_id in failed:
            failed.add(node_id)  # the parent's problem is reported once, for the parent
        parent = IDENTITY if parent_id is None else placement.world[parent_id]
        placement.parent_world[node_id] = parent
        if collection == "groups":
            scale = float(entry["scale"])
            placement.world[node_id] = multiply(
                parent, compose(entry["location"], entry["rotation"], (scale, scale, scale))
            )
            continue
        x, y, z = (float(v) for v in entry["location"])
        placement.locations[node_id] = (x, y, z)
        placement.world[node_id] = multiply(
            parent, compose(entry["location"], entry["rotation"], entry["scale"])
        )
        support = entry.get("rest_on")
        if build_meshes or support is not None or node_id in need_mesh:
            mesh = GEOMETRY.run(entry["geometry"])
            placement.meshes[node_id] = WorldMesh.from_mesh(mesh, placement.world[node_id])
        if support is None or node_id in failed:
            continue
        if support in failed:
            failed.add(node_id)
            continue
        shift = rest_shift(placement.meshes[node_id], placement.meshes[support])
        if shift is None:
            failed.add(node_id)
            placement.issues.append(
                Issue(
                    "spec.placement_failed",
                    f"{node_id!r} cannot rest on {support!r}: no part of {support!r} is "
                    f"directly below it",
                    pointer_join("objects", index, "rest_on"),
                    hint=f"move {node_id!r} in x or y so it is above {support!r}",
                )
            )
            continue
        _move_vertically(node_id, shift, placement)
        placement.rested[node_id] = support

    for collection in ("lights", "cameras"):
        for entry in scene[collection]:
            parent_id = entry.get("parent")
            parent = placement.world.get(parent_id, IDENTITY) if parent_id else IDENTITY
            placement.parent_world[entry["id"]] = parent
            rotation = entry.get("rotation", (0.0, 0.0, 0.0))
            placement.world[entry["id"]] = multiply(
                parent, compose(entry["location"], rotation, _UNIT_SCALE)
            )
            placement.oriented[entry["id"]] = aim_matrix(
                entry["location"], entry.get("look_at"), entry.get("rotation"), parent
            )
    return placement


def aim_matrix(
    location: Any, look_at: Any | None, rotation_deg: Any | None, parent: Mat4 = IDENTITY
) -> Mat4:
    """World matrix of a light or camera: *location* and *rotation_deg* are relative to
    *parent*; *look_at* is a world-space point.  Parent scale does not reach it."""

    position = apply(parent, location)
    if look_at is not None:
        target = tuple(float(v) for v in look_at)
        direction = tuple(t - p for t, p in zip(target, position, strict=True))
        return with_translation(look_rotation(direction), position)
    local = rotation_xyz(rotation_deg if rotation_deg is not None else (0.0, 0.0, 0.0))
    return with_translation(multiply(rotation_part(parent), local), position)


def _order(
    nodes: Mapping[str, tuple[str, int, Mapping[str, Any]]], placement: Placement
) -> list[str]:
    """Groups and objects with every parent and support before the nodes that need it.

    A node on a loop (through parents, supports or both) is reported once and left
    out, together with everything that depends on it.
    """

    order: list[str] = []
    state: dict[str, str] = {}

    def needs(node_id: str) -> list[str]:
        entry = nodes[node_id][2]
        found = [entry.get("parent"), entry.get("rest_on")]
        return [n for n in found if isinstance(n, str) and n in nodes]

    def visit(node_id: str) -> bool:
        if state.get(node_id) == "done":
            return True
        if state.get(node_id) in ("visiting", "failed"):
            return False
        state[node_id] = "visiting"
        ok = all(visit(other) for other in needs(node_id))
        state[node_id] = "done" if ok else "failed"
        if ok:
            order.append(node_id)
        return ok

    for node_id, (collection, index, entry) in nodes.items():
        if state.get(node_id) in ("done", "failed"):
            continue
        if not visit(node_id):
            field_name = "rest_on" if entry.get("rest_on") is not None else "parent"
            placement.issues.append(
                Issue(
                    "spec.placement_failed",
                    f"{field_name} of {node_id!r} leads into a cycle: things cannot hold or "
                    "carry each other in a loop",
                    pointer_join(collection, index, field_name),
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


__all__ = ["Placement", "aim_matrix", "resolve_placement"]
