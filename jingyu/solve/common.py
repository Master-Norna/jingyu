"""Pieces the solvers share: which objects a request means, sampling them, and
predicting where things land in a frame."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from ..camera import CameraModel
from ..errors import JingyuError
from ..geometry import GEOMETRY
from ..geometry.raycast import SceneHit, cast
from ..geometry.spatial import WorldMesh
from ..geometry.transform import Vec3
from ..materials import MATERIALS
from ..materials.assign import part_materials
from ..placement import Placement

#: A ray passes through at most this many clear faces.
_MAX_PASSES = 16

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


class Sight:
    """Rays through the visible objects of a scene that pass through clear faces.

    Glass lets sunlight and sight through, so a window pane neither shades the
    table nor hides it from the camera.  Clearness is per face: a room's panes
    are clear, its walls are not.
    """

    def __init__(
        self, scene: Mapping[str, Any], placement: Placement, skip: Iterable[str] = ()
    ) -> None:
        skipped = set(skip)
        self.meshes = {
            obj["id"]: placement.meshes[obj["id"]]
            for obj in scene["objects"]
            if obj["visible"] and obj["id"] in placement.meshes and obj["id"] not in skipped
        }
        self.clear = clear_faces(scene, self.meshes)

    def cast(
        self, origin: Sequence[float], direction: Sequence[float], t_max: float = math.inf
    ) -> SceneHit | None:
        start = (float(origin[0]), float(origin[1]), float(origin[2]))
        for _ in range(_MAX_PASSES):
            found = cast(self.meshes, start, direction, t_max)
            if found is None:
                return None
            faces = self.clear.get(found.object, set())
            if faces is not None and found.hit.face not in faces:
                return found
            step = found.hit.distance + 1e-6
            start = (
                start[0] + direction[0] * step,
                start[1] + direction[1] * step,
                start[2] + direction[2] * step,
            )
            t_max -= step
        return None


def clear_faces(
    scene: Mapping[str, Any], meshes: Mapping[str, WorldMesh]
) -> dict[str, set[int] | None]:
    """Per object with clear surfaces, the faces that let light through (None: all)."""

    refractive: dict[str, bool] = {}

    def clear(spec: Mapping[str, Any]) -> bool:
        key = repr(sorted(spec.items()))
        if key not in refractive:
            refractive[key] = MATERIALS.run(spec).refractive
        return refractive[key]

    materials = {m["id"]: {k: v for k, v in m.items() if k != "id"} for m in scene["materials"]}
    found: dict[str, set[int] | None] = {}
    for obj in scene["objects"]:
        mesh = meshes.get(obj["id"])
        if mesh is None:
            continue
        assignment = part_materials(obj, GEOMETRY.get(obj["geometry"]["op"]))
        see_through = {
            part
            for part, assigned in assignment.items()
            if (assigned.material is not None and clear(materials[assigned.material]))
            or (
                assigned.material is None
                and assigned.default is not None
                and clear(assigned.default)
            )
        }
        if not see_through:
            continue
        if None in see_through:
            found[obj["id"]] = None
            continue
        source = mesh.source
        wanted = {source.parts.index(p) for p in see_through if p in source.parts}
        found[obj["id"]] = {i for i, part in enumerate(source.face_parts) if part in wanted}
    return found


def round_vec(values: Sequence[float], digits: int = 4) -> list[float]:
    return [round(float(v), digits) + 0.0 for v in values]


__all__ = [
    "MAX_POINTS_PER_OBJECT",
    "Sight",
    "centre_of",
    "clear_faces",
    "expand_ids",
    "frame_box",
    "predicted_layout",
    "round_vec",
    "sample_points",
]
