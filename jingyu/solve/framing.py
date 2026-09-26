"""Frame a subject: from "these things, here in the frame, this big" to a camera.

The camera keeps the direction it views the subject from (or takes one given
as azimuth and elevation), then two things are solved together:

* its aim, in closed form, so the subject's box lands centred on ``at``;
* its distance (or, when the distance is fixed, its focal length; for an
  orthographic camera, its ortho scale) so the subject spans ``size`` of the
  frame.

The aim is solved for a point and then corrected until the centre of the
subject's projected box (not just its middle point) sits on ``at``; the size
follows from the projected box too, so off-centre and wide-angle framings come
out right, not only the simple case.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from ..camera import CameraModel
from ..errors import JingyuError
from ..geometry.raycast import cast, unit
from ..geometry.transform import Vec3, look_rotation, solve_linear, with_translation
from ..placement import Placement
from .common import centre_of, expand_ids, frame_box, predicted_layout, round_vec, sample_points

SizeOf = Literal["larger", "width", "height"]

_MIN_LENS_MM = 1.0
_MAX_LENS_MM = 5000.0
_ITERATIONS = 40


@dataclass(frozen=True)
class FramingRequest:
    subject: tuple[str, ...]
    at: tuple[float, float] = (0.5, 0.5)
    size: float = 0.5
    size_of: SizeOf = "larger"
    camera: str | None = None
    azimuth: float | None = None
    elevation: float | None = None
    lens_mm: float | None = None
    distance: float | None = None


def frame_subject(
    scene: Mapping[str, Any], placement: Placement, request: FramingRequest
) -> dict[str, Any]:
    """Solve the camera for *request*; returns the camera, an edit and a prediction."""

    if request.lens_mm is not None and request.distance is not None:
        raise JingyuError(
            "tool.invalid_arguments",
            "give lens_mm or distance, not both: one is kept and the other is solved",
        )
    if not 0.0 < request.size <= 4.0:
        raise JingyuError("tool.invalid_arguments", "size must be above 0 (1 is the full frame)")
    current = CameraModel.from_scene(scene, placement, request.camera)
    entry = next(c for c in scene["cameras"] if c["id"] == current.id)
    objects = expand_ids(scene, request.subject)
    points = [p for obj_id in objects for p in sample_points(placement.meshes[obj_id])]
    centre = centre_of(points)
    radius = max(math.dist(centre, p) for p in points)

    direction = _view_direction(current, centre, request)
    if current.orthographic:
        camera, lens_or_scale = _solve_orthographic(
            current, points, centre, radius, direction, request
        )
    else:
        camera, lens_or_scale = _solve_perspective(
            current, entry, points, centre, radius, direction, request
        )

    box = frame_box(camera, points)
    assert box is not None
    width, height = box["u1"] - box["u0"], box["v1"] - box["v0"]
    distance = math.dist(camera.position, centre)
    look_at = tuple(p + f * distance for p, f in zip(camera.position, camera.forward, strict=True))
    parent = placement.parent_world[current.id]
    offset = tuple(
        p - t for p, t in zip(camera.position, (parent[3], parent[7], parent[11]), strict=True)
    )
    location = solve_linear(parent, offset)

    values: dict[str, Any] = {"location": round_vec(location), "look_at": round_vec(look_at)}
    if camera.orthographic:
        values["ortho_scale"] = round(lens_or_scale, 4)
    else:
        values["lens_mm"] = round(lens_or_scale, 3)
    if "rotation" in entry:
        values["rotation"] = None  # merge patch: look_at replaces the rotation
    elevation = math.degrees(math.asin(max(-1.0, min(1.0, direction[2]))))
    azimuth = math.degrees(math.atan2(direction[1], direction[0])) % 360.0

    return {
        "camera": {
            "id": current.id,
            **{k: v for k, v in values.items() if v is not None},
            "distance": round(distance, 4),
            "azimuth": round(azimuth, 2),
            "elevation": round(elevation, 2),
        },
        "operations": [{"op": "merge", "path": f"/cameras/@{current.id}", "value": values}],
        "subject": {
            "objects": objects,
            "bbox_uv": {k: round(v, 4) for k, v in box.items()},
            "centre_uv": {
                "u": round((box["u0"] + box["u1"]) / 2, 4),
                "v": round((box["v0"] + box["v1"]) / 2, 4),
            },
            "size": {"width": round(width, 4), "height": round(height, 4)},
        },
        "frame": predicted_layout(scene, placement, camera),
        "warnings": _warnings(
            {o["id"] for o in scene["objects"] if not (o["visible"] and o["camera_visible"])},
            placement,
            camera,
            objects,
            points,
            request,
            width,
            height,
        ),
    }


def _view_direction(current: CameraModel, centre: Vec3, request: FramingRequest) -> Vec3:
    """Unit vector from the subject toward the camera."""

    offset = tuple(p - c for p, c in zip(current.position, centre, strict=True))
    if math.hypot(*offset) < 1e-9:
        if request.azimuth is None or request.elevation is None:
            raise JingyuError(
                "solve.no_solution",
                "the camera sits at the subject's centre, so it has no direction to keep",
                hint="give azimuth and elevation",
            )
        offset = (1.0, 0.0, 0.0)
    here = unit(offset)
    azimuth = request.azimuth
    elevation = request.elevation
    if azimuth is None:
        azimuth = (
            math.degrees(math.atan2(here[1], here[0]))
            if math.hypot(here[0], here[1]) > 1e-9
            else 0.0
        )
    if elevation is None:
        elevation = math.degrees(math.asin(max(-1.0, min(1.0, here[2]))))
    if not -90.0 <= elevation <= 90.0:
        raise JingyuError("tool.invalid_arguments", "elevation must lie in [-90, 90] degrees")
    e, a = math.radians(elevation), math.radians(azimuth)
    return (math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e))


def _camera(
    template: CameraModel, position: Sequence[float], forward: Sequence[float], half_width: float
) -> CameraModel:
    rotation = with_translation(look_rotation(forward), position)
    return CameraModel(
        id=template.id,
        position=(float(position[0]), float(position[1]), float(position[2])),
        right=(rotation[0], rotation[4], rotation[8]),
        up=(rotation[1], rotation[5], rotation[9]),
        forward=unit(forward),
        orthographic=template.orthographic,
        half_width=half_width,
        half_height=half_width * template.height / template.width,
        shift_x=template.shift_x,
        shift_y=template.shift_y,
        clip_start=template.clip_start,
        clip_end=template.clip_end,
        width=template.width,
        height=template.height,
    )


def _aim(target_direction: Vec3, plane_x: float, plane_y: float) -> Vec3 | None:
    """Forward direction of a level camera that sees *target_direction* at image-plane
    point (plane_x, plane_y); None when no level camera can (too close to straight up
    or down for that offset)."""

    k = math.sqrt(plane_x * plane_x + plane_y * plane_y + 1.0)
    sine = k * target_direction[2] / math.sqrt(1.0 + plane_y * plane_y)
    if abs(sine) > 1.0:
        return None
    pitch = math.asin(sine) - math.atan(plane_y)
    if abs(pitch) > math.pi / 2:
        return None
    heading = math.atan2(target_direction[1], target_direction[0]) - math.atan2(
        -plane_x, math.cos(pitch) - plane_y * math.sin(pitch)
    )
    return (
        math.cos(pitch) * math.cos(heading),
        math.cos(pitch) * math.sin(heading),
        math.sin(pitch),
    )


def _size(box: Mapping[str, float], size_of: SizeOf, aspect: float) -> float:
    width, height = box["u1"] - box["u0"], box["v1"] - box["v0"]
    if size_of == "width":
        return width
    if size_of == "height":
        return height
    return max(width, height)


def _place_perspective(
    template: CameraModel,
    points: Sequence[Vec3],
    centre: Vec3,
    direction: Vec3,
    distance: float,
    half_width: float,
    at: tuple[float, float],
) -> tuple[CameraModel, dict[str, float]]:
    """A camera at *distance* along *direction*, aimed so the subject's box centres on *at*."""

    position = tuple(c + d * distance for c, d in zip(centre, direction, strict=True))
    towards = unit(tuple(c - p for c, p in zip(centre, position, strict=True)))
    goal_u, goal_v = at
    camera = _camera(template, position, towards, half_width)
    box: dict[str, float] | None = None
    for _ in range(8):
        plane_x, plane_y = camera.uv_to_plane(goal_u, goal_v)
        forward = _aim(towards, plane_x, plane_y)
        if forward is None:
            raise JingyuError(
                "solve.no_solution",
                "no level camera can look this steeply and still put the subject there",
                hint="lower the elevation, or move 'at' toward the frame's centre",
            )
        camera = _camera(template, position, forward, half_width)
        box = frame_box(camera, points)
        if box is None:
            break
        du = at[0] - (box["u0"] + box["u1"]) / 2
        dv = at[1] - (box["v0"] + box["v1"]) / 2
        if abs(du) < 1e-5 and abs(dv) < 1e-5:
            break
        goal_u += du
        goal_v += dv
    if box is None:
        raise JingyuError("solve.no_solution", "the subject ends up behind the camera")
    return camera, box


def _solve_perspective(
    template: CameraModel,
    entry: Mapping[str, Any],
    points: Sequence[Vec3],
    centre: Vec3,
    radius: float,
    direction: Vec3,
    request: FramingRequest,
) -> tuple[CameraModel, float]:
    sensor_half = float(entry["sensor_width_mm"]) / 2.0
    current_lens = float(entry["lens_mm"])
    aspect = template.width / template.height
    if request.distance is not None:
        distance = float(request.distance)
        if distance <= radius + template.clip_start:
            raise JingyuError(
                "solve.no_solution",
                f"at {distance:g} m the camera would be inside the subject (it reaches "
                f"{radius:.3g} m from its centre)",
            )
        lens = current_lens
        for _ in range(_ITERATIONS):
            half_width = sensor_half / lens
            camera, box = _place_perspective(
                template, points, centre, direction, distance, half_width, request.at
            )
            ratio = _size(box, request.size_of, aspect) / request.size
            if abs(ratio - 1.0) < 1e-5:
                break
            lens = min(_MAX_LENS_MM, max(_MIN_LENS_MM, lens / ratio))
        return camera, lens

    lens = float(request.lens_mm if request.lens_mm is not None else current_lens)
    half_width = sensor_half / lens
    closest = (radius + template.clip_start) * 1.02
    distance = max(math.dist(template.position, centre), closest)
    camera, box = _place_perspective(
        template, points, centre, direction, distance, half_width, request.at
    )
    for _ in range(_ITERATIONS):
        ratio = _size(box, request.size_of, aspect) / request.size
        if abs(ratio - 1.0) < 1e-5:
            break
        wanted = max(closest, distance * ratio)
        if wanted == distance:  # already as close as the subject allows
            break
        distance = wanted
        camera, box = _place_perspective(
            template, points, centre, direction, distance, half_width, request.at
        )
    return camera, lens


def _solve_orthographic(
    template: CameraModel,
    points: Sequence[Vec3],
    centre: Vec3,
    radius: float,
    direction: Vec3,
    request: FramingRequest,
) -> tuple[CameraModel, float]:
    if request.lens_mm is not None:
        raise JingyuError("tool.invalid_arguments", "an orthographic camera has no lens_mm")
    distance = (
        float(request.distance)
        if request.distance is not None
        else max(math.dist(template.position, centre), (radius + template.clip_start) * 1.02)
    )
    forward = tuple(-d for d in direction)
    half_width = template.half_width
    aspect = template.width / template.height
    camera = template
    for _ in range(_ITERATIONS):
        goal_u, goal_v = request.at
        for _ in range(4):
            probe = _camera(template, centre, forward, half_width)
            plane_x, plane_y = probe.uv_to_plane(goal_u, goal_v)
            position = tuple(
                c + d * distance - plane_x * r - plane_y * q
                for c, d, r, q in zip(centre, direction, probe.right, probe.up, strict=True)
            )
            camera = _camera(template, position, forward, half_width)
            box = frame_box(camera, points)
            assert box is not None
            goal_u += request.at[0] - (box["u0"] + box["u1"]) / 2
            goal_v += request.at[1] - (box["v0"] + box["v1"]) / 2
        assert box is not None
        ratio = _size(box, request.size_of, aspect) / request.size
        if abs(ratio - 1.0) < 1e-5:
            break
        half_width *= ratio
    return camera, half_width * 2.0


def _warnings(
    unseen: set[str],
    placement: Placement,
    camera: CameraModel,
    subject: Sequence[str],
    points: Sequence[Vec3],
    request: FramingRequest,
    width: float,
    height: float,
) -> list[dict[str, str]]:
    warnings: list[dict[str, str]] = []
    achieved = {"larger": max(width, height), "width": width, "height": height}[request.size_of]
    if abs(achieved - request.size) > 0.01:
        warnings.append(
            {
                "code": "solve.size_unreached",
                "message": f"the subject spans {achieved:.2f} of the frame, not "
                f"{request.size:.2f}: the camera cannot come closer without entering it (or "
                "the lens hit its limit)",
            }
        )
    inside = [
        obj_id
        for obj_id, mesh in placement.meshes.items()
        if mesh.closed and mesh.depth_inside(camera.position) > 0.0
    ]
    for obj_id in inside:
        warnings.append(
            {
                "code": "solve.camera_inside",
                "message": f"the camera would stand inside {obj_id!r}",
                "hint": "choose another azimuth or elevation, or a longer lens from farther away",
            }
        )
    blockers: dict[str, int] = {}
    checked = points[:: max(1, len(points) // 96)]
    others = {
        k: v
        for k, v in placement.meshes.items()
        if k not in subject and k not in inside and k not in unseen
    }
    for point in checked:
        offset = tuple(p - c for p, c in zip(point, camera.position, strict=True))
        length = math.hypot(*offset)
        if length < 1e-9:
            continue
        found = cast(others, camera.position, unit(offset), length - 1e-4)
        if found is not None:
            blockers[found.object] = blockers.get(found.object, 0) + 1
    for obj_id, count in sorted(blockers.items(), key=lambda item: -item[1]):
        warnings.append(
            {
                "code": "solve.subject_hidden",
                "message": f"{obj_id!r} stands between the camera and about "
                f"{count / len(checked):.0%} of the subject",
                "hint": f"hide {obj_id!r} in a subtraction render, move it, or change the view",
            }
        )
    return warnings


__all__ = ["FramingRequest", "frame_subject"]
