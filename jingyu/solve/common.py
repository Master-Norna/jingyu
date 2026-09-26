"""Pieces the solvers share: which objects a request means, sampling them, and
predicting where things land in a frame."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from ..camera import CameraModel
from ..errors import JingyuError
from ..geometry.spatial import WorldMesh
from ..geometry.transform import Vec3
from ..placement import Placement

#: Per object, at most this many vertices stand in for its shape when projecting.
MAX_POINTS_PER_OBJECT = 600

#: Directions whose extreme vertices are always kept, so sampling never shrinks
#: an object's outline: the axes, the face diagonals and the corners of a cube.
_DIRECTIONS: tuple[Vec3, ...] = tuple(
    (x, y, z)
    for x in (-1.0, 0.0, 1.0)
    for y in (-1.0, 0.0, 1.0)
    for z in (-1.0, 0.0, 1.0)
    if (x, y, z) != (0.0, 0.0, 0.0)
)


def expand_ids(scene: Mapping[str, Any], ids: Sequence[str]) -> list[str]:
    """Visible objects meant by *ids*: an object with everything parented to it, or
    every object inside a group, at any depth."""

    parents = {
        entry["id"]: entry.get("parent")
        for collection in ("groups", "objects")
        for entry in scene[collection]
    }
    known = set(parents)
    unknown = [i for i in ids if i not in known]
    if unknown:
        raise JingyuError(
            "spec.unknown_reference",
            f"no group or object named {', '.join(map(repr, unknown))}",
            hint="name objects or groups of the scene",
        )
    wanted = set(ids)
    chosen = []
    for obj in scene["objects"]:
        if not obj["visible"]:
            continue
        node: str | None = obj["id"]
        seen: set[str] = set()
        while node is not None and node not in seen:
            if node in wanted:
                chosen.append(obj["id"])
                break
            seen.add(node)
            node = parents.get(node)
    if not chosen:
        raise JingyuError(
            "solve.no_solution",
            f"{', '.join(map(repr, ids))} contain no visible object",
            hint="name at least one visible object, or a group that holds one",
        )
    return chosen


def sample_points(mesh: WorldMesh, limit: int = MAX_POINTS_PER_OBJECT) -> list[Vec3]:
    """A few hundred vertices that keep the mesh's outline: a regular subsample plus
    the extreme vertex in each of 26 directions."""

    vertices = mesh.vertices
    step = max(1, math.ceil(len(vertices) / limit))
    chosen = {id(v): v for v in vertices[::step]}
    for direction in _DIRECTIONS:
        best = max(
            vertices, key=lambda v: v[0] * direction[0] + v[1] * direction[1] + v[2] * direction[2]
        )
        chosen[id(best)] = best
    return list(chosen.values())


def centre_of(points: Iterable[Vec3]) -> Vec3:
    """Centre of the axis-aligned box around *points*."""

    xs, ys, zs = zip(*points, strict=True)
    return ((min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0, (min(zs) + max(zs)) / 2.0)


def frame_box(camera: CameraModel, points: Iterable[Vec3]) -> dict[str, float] | None:
    """Box in frame coordinates around the points in front of the camera."""

    us: list[float] = []
    vs: list[float] = []
    for point in points:
        projected = camera.project(point)
        if projected is not None and projected[2] >= camera.clip_start:
            us.append(projected[0])
            vs.append(projected[1])
    if not us:
        return None
    return {"u0": min(us), "v0": min(vs), "u1": max(us), "v1": max(vs)}


def predicted_layout(
    scene: Mapping[str, Any], placement: Placement, camera: CameraModel
) -> list[dict[str, Any]]:
    """Where each visible object's outline would fall in the frame (before occlusion)."""

    rows = []
    for obj in scene["objects"]:
        mesh = placement.meshes.get(obj["id"])
        if not obj["visible"] or not obj["camera_visible"] or mesh is None:
            continue
        box = frame_box(camera, sample_points(mesh, 200))
        if box is None or box["u1"] < 0 or box["u0"] > 1 or box["v1"] < 0 or box["v0"] > 1:
            continue
        rows.append(
            {
                "object": obj["id"],
                "bbox_uv": {k: round(v, 4) for k, v in box.items()},
                "cut_by_frame_edge": box["u0"] < 0
                or box["v0"] < 0
                or box["u1"] > 1
                or box["v1"] > 1,
            }
        )
    return rows


def round_vec(values: Sequence[float], digits: int = 4) -> list[float]:
    return [round(float(v), digits) + 0.0 for v in values]


__all__ = [
    "MAX_POINTS_PER_OBJECT",
    "centre_of",
    "expand_ids",
    "frame_box",
    "predicted_layout",
    "round_vec",
    "sample_points",
]
