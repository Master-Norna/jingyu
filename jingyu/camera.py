"""The render camera on the host: project world points into the frame, cast rays out.

Standard library only.  The camera's pose comes from :mod:`jingyu.placement`
(the same matrix the worker gives Blender) and its lens model follows Blender's:
the sensor width spans the frame's width (horizontal fit), the height follows
the resolution's aspect, and ``shift`` moves the frame by a fraction of its width.

Frame coordinates are the ones every tool uses: ``u`` from the left edge and
``v`` from the top edge, both in [0, 1] across the image.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .errors import JingyuError
from .geometry.transform import Mat4, Vec3, column, translation
from .placement import Placement


@dataclass(frozen=True)
class CameraModel:
    id: str
    position: Vec3
    right: Vec3
    up: Vec3
    #: The direction the camera looks (its local -Z).
    forward: Vec3
    orthographic: bool
    #: Half the frame's width and height: on the image plane at distance 1 for a
    #: perspective camera, in metres for an orthographic one.
    half_width: float
    half_height: float
    shift_x: float
    shift_y: float
    clip_start: float
    clip_end: float
    width: int
    height: int

    @classmethod
    def from_scene(
        cls,
        scene: Mapping[str, Any],
        placement: Placement,
        camera_id: str | None = None,
        resolution: Sequence[int] | None = None,
    ) -> CameraModel:
        """The camera *camera_id* (default: the render camera) of a normalised scene."""

        camera_id = camera_id or scene["render"]["camera"]
        entry = next((c for c in scene["cameras"] if c["id"] == camera_id), None)
        if entry is None:
            raise JingyuError(
                "spec.unknown_reference",
                f"camera {camera_id!r} is not defined",
                hint=f"cameras: {', '.join(c['id'] for c in scene['cameras'])}",
            )
        width, height = (int(v) for v in (resolution or scene["render"]["resolution"]))
        return cls.from_matrix(camera_id, placement.oriented[camera_id], entry, width, height)

    @classmethod
    def from_matrix(
        cls, camera_id: str, matrix: Mat4, entry: Mapping[str, Any], width: int, height: int
    ) -> CameraModel:
        ortho = entry["projection"] == "orthographic"
        if ortho:
            half_width = float(entry["ortho_scale"]) / 2.0
        else:
            half_width = float(entry["sensor_width_mm"]) / 2.0 / float(entry["lens_mm"])
        back = column(matrix, 2)
        return cls(
            id=camera_id,
            position=translation(matrix),
            right=column(matrix, 0),
            up=column(matrix, 1),
            forward=(-back[0], -back[1], -back[2]),
            orthographic=ortho,
            half_width=half_width,
            half_height=half_width * height / width,
            shift_x=float(entry["shift"][0]),
            shift_y=float(entry["shift"][1]),
            clip_start=float(entry["clip_start"]),
            clip_end=float(entry["clip_end"]),
            width=width,
            height=height,
        )

    # ------------------------------------------------------------ mapping

    def to_camera(self, point: Sequence[float]) -> Vec3:
        """*point* in camera coordinates: (right, up, distance in front)."""

        d = (point[0] - self.position[0], point[1] - self.position[1], point[2] - self.position[2])
        return (_dot(d, self.right), _dot(d, self.up), _dot(d, self.forward))

    def plane_to_uv(self, x: float, y: float) -> tuple[float, float]:
        """Image-plane coordinates (depth 1 or metres for orthographic) to frame u, v."""

        u = 0.5 + x / (2.0 * self.half_width) - self.shift_x
        v = 0.5 - y / (2.0 * self.half_height) + self.shift_y * self.half_width / self.half_height
        return u, v

    def uv_to_plane(self, u: float, v: float) -> tuple[float, float]:
        x = (u - 0.5 + self.shift_x) * 2.0 * self.half_width
        y = (0.5 - v + self.shift_y * self.half_width / self.half_height) * 2.0 * self.half_height
        return x, y

    def project(self, point: Sequence[float]) -> tuple[float, float, float] | None:
        """(u, v, depth) of a world point; None when it is not in front of the camera."""

        x, y, depth = self.to_camera(point)
        if depth <= 1e-12:
            return None
        if self.orthographic:
            u, v = self.plane_to_uv(x, y)
        else:
            u, v = self.plane_to_uv(x / depth, y / depth)
        return u, v, depth

    def ray(self, u: float, v: float) -> tuple[Vec3, Vec3]:
        """Origin and unit direction of the line of sight through frame point (u, v)."""

        x, y = self.uv_to_plane(u, v)
        if self.orthographic:
            origin = tuple(
                p + x * r + y * q
                for p, r, q in zip(self.position, self.right, self.up, strict=True)
            )
            return (origin[0], origin[1], origin[2]), self.forward
        direction = tuple(
            f + x * r + y * q for f, r, q in zip(self.forward, self.right, self.up, strict=True)
        )
        length = math.sqrt(_dot(direction, direction))
        return self.position, (
            direction[0] / length,
            direction[1] / length,
            direction[2] / length,
        )

    def pixel_ray(self, x: int, y: int) -> tuple[Vec3, Vec3]:
        """The line of sight through the centre of pixel (x, y)."""

        return self.ray((x + 0.5) / self.width, (y + 0.5) / self.height)


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return float(a[0] * b[0] + a[1] * b[1] + a[2] * b[2])


__all__ = ["CameraModel"]
