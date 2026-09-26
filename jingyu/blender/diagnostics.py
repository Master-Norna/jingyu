"""The light pass: which lights reach each visible point, and what shades it.

A model looking at a render can see that the fruit is dark, but not why: it may
face away from the sun, sit in the shadow of the window wall, or be lit only by
the sky.  This pass answers with geometry instead of guesswork.  It casts one
ray per sample from the camera, then one shadow ray toward each light, using
Blender's own scene ray casting, so it agrees with what the renderer sees.

Results:

* ``light_mask.png``: per sample, a 24-bit mask of the lights that reach the
  surface directly (bit *i* is light *i* of the light map); alpha 0 where the
  camera sees no surface.
* the light map (``jingyu.light-map.v1``): the lights, per object the lit
  fraction for each light, the biggest shadows (caster, receiver, light), the
  objects inside the camera's view, and how open the camera's surroundings are.

Area lights are treated as their centre, so penumbrae are not measured; sky
light is not a light here (it comes from everywhere).
"""

from __future__ import annotations

import math
import struct
import zlib
from collections import Counter, defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import bpy
from mathutils import Vector

from .build import BuiltObject

LIGHT_MAP_SCHEMA = "jingyu.light-map.v1"
#: Longest side of the sample grid; enough to locate light and shadow shapes.
MAX_SAMPLES_SIDE = 240
MAX_LIGHTS = 24
_EPSILON = 1e-4
_MAX_PASS_THROUGH = 8
_OPENNESS_RAYS = 256


def light_pass(
    scene: Any,
    objects: Sequence[BuiltObject],
    lights: Sequence[tuple[str, str, Any]],
    refractive: set[str],
    filepath: Path,
) -> dict[str, Any]:
    """Run the light pass; *lights* are ``(id, kind, blender_object)``."""

    for obj in scene.objects:
        if obj.type == "MESH":
            obj.hide_viewport = obj.hide_render
    view_layer = scene.view_layers[0]
    view_layer.update()
    depsgraph = bpy.context.evaluated_depsgraph_get()

    by_name = {o.blender_object.name: o.id for o in objects}
    lights = list(lights)[:MAX_LIGHTS]
    width, height = _grid(scene)
    camera = scene.camera
    rays = _camera_rays(scene, camera, width, height)

    mask = bytearray(width * height * 4)
    samples: Counter[str] = Counter()
    lit: dict[str, Counter[str]] = defaultdict(Counter)
    facing_away: dict[str, Counter[str]] = defaultdict(Counter)
    shadows: Counter[tuple[str, str, str]] = Counter()
    for index, (origin, direction) in enumerate(rays):
        hit, location, normal, _, obj, _ = scene.ray_cast(depsgraph, origin, direction)
        if not hit or obj is None or obj.name not in by_name:
            continue
        receiver = by_name[obj.name]
        samples[receiver] += 1
        if normal.dot(direction) > 0:
            normal = -normal
        bits = 0
        for bit, (light_id, kind, lamp) in enumerate(lights):
            caster = _occluder(scene, depsgraph, location, normal, kind, lamp, refractive, by_name)
            if caster is None:
                bits |= 1 << bit
                lit[receiver][light_id] += 1
            elif caster == "":
                facing_away[receiver][light_id] += 1
            else:
                shadows[(light_id, caster, receiver)] += 1
        offset = index * 4
        mask[offset : offset + 4] = bytes(((bits >> 16) & 255, (bits >> 8) & 255, bits & 255, 255))

    _write_png(filepath, width, height, bytes(mask))
    total = {name: count for name, count in samples.items()}
    return {
        "schema": LIGHT_MAP_SCHEMA,
        "width": width,
        "height": height,
        "lights": [
            {"bit": bit, "id": light_id, "kind": kind}
            for bit, (light_id, kind, _) in enumerate(lights)
        ],
        "objects": {
            name: {
                "samples": count,
                "lit_fraction": {
                    light_id: round(lit[name][light_id] / count, 4) for light_id, _, _ in lights
                },
                "facing_away_fraction": {
                    light_id: round(facing_away[name][light_id] / count, 4)
                    for light_id, _, _ in lights
                },
            }
            for name, count in sorted(total.items())
        },
        "shadows": [
            {
                "light": light_id,
                "caster": caster,
                "receiver": receiver,
                "fraction_of_receiver": round(count / total[receiver], 4),
            }
            for (light_id, caster, receiver), count in shadows.most_common(24)
        ],
        "in_view": sorted(o.id for o in objects if _in_view(scene, camera, o.blender_object)),
        "openness": _openness(scene, depsgraph, camera.matrix_world.translation),
    }


def _grid(scene: Any) -> tuple[int, int]:
    x, y = scene.render.resolution_x, scene.render.resolution_y
    scale = min(1.0, MAX_SAMPLES_SIDE / max(x, y))
    return max(1, round(x * scale)), max(1, round(y * scale))


def _camera_rays(scene: Any, camera: Any, width: int, height: int) -> list[tuple[Vector, Vector]]:
    """One ray per sample, row by row from the top-left, through sample centres."""

    matrix = camera.matrix_world
    frame = [Vector(corner) for corner in camera.data.view_frame(scene=scene)]
    top_right, bottom_right, bottom_left, top_left = frame
    ortho = camera.data.type == "ORTHO"
    forward = (matrix.to_3x3() @ Vector((0.0, 0.0, -1.0))).normalized()
    rays = []
    for row in range(height):
        v = (row + 0.5) / height
        left = top_left.lerp(bottom_left, v)
        right = top_right.lerp(bottom_right, v)
        for column in range(width):
            local = left.lerp(right, (column + 0.5) / width)
            if ortho:
                origin = matrix @ Vector((local.x, local.y, 0.0))
                rays.append((origin, forward))
            else:
                point = matrix @ local
                origin = matrix.translation
                rays.append((origin.copy(), (point - origin).normalized()))
    return rays


def _occluder(
    scene: Any,
    depsgraph: Any,
    point: Vector,
    normal: Vector,
    kind: str,
    lamp: Any,
    refractive: set[str],
    by_name: dict[str, str],
) -> str | None:
    """None when *lamp* reaches *point*; "" when it faces away; else the caster id."""

    axis = (lamp.matrix_world.to_3x3() @ Vector((0.0, 0.0, -1.0))).normalized()
    if kind == "sun":
        direction = -axis
        distance = 1.0e7
    else:
        offset = lamp.matrix_world.translation - point
        distance = offset.length
        if distance < _EPSILON:
            return None
        direction = offset / distance
        if kind == "spot" and direction.dot(-axis) < math.cos(lamp.data.spot_size / 2.0):
            return ""
        if kind == "area" and direction.dot(axis) > 0:
            return ""
    if normal.dot(direction) <= 0:
        return ""
    origin = point + normal * _EPSILON
    for _ in range(_MAX_PASS_THROUGH):
        hit, location, _, _, obj, _ = scene.ray_cast(
            depsgraph, origin, direction, distance=distance
        )
        if not hit or obj is None:
            return None
        caster = by_name.get(obj.name)
        if caster is None or caster not in refractive:
            return caster or obj.name
        travelled = (location - origin).length + _EPSILON
        origin = location + direction * _EPSILON
        distance -= travelled
        if distance <= 0:
            return None
    return None


def _in_view(scene: Any, camera: Any, obj: Any) -> bool:
    """Whether *obj*'s bounding box reaches into the camera's view.

    The box edges are clipped to the part in front of the camera, projected onto
    the frame plane, and the projection's bounds compared with the frame.  That is
    conservative at the frame's corners, never wrong the other way.
    """

    frame = [Vector(corner) for corner in camera.data.view_frame(scene=scene)]
    x0, x1 = min(c.x for c in frame), max(c.x for c in frame)
    y0, y1 = min(c.y for c in frame), max(c.y for c in frame)
    to_camera = camera.matrix_world.inverted()
    corners = [to_camera @ (obj.matrix_world @ Vector(c)) for c in obj.bound_box]
    if camera.data.type == "ORTHO":
        projected = [(p.x, p.y) for p in corners]
    else:
        near = -max(camera.data.clip_start, _EPSILON)
        front = [p for p in corners if p.z <= near]
        for i, a in enumerate(corners):
            for b in corners[i + 1 :]:
                if (a.z - near) * (b.z - near) < 0:  # the edge crosses the near plane
                    front.append(a.lerp(b, (near - a.z) / (b.z - a.z)))
        if not front:
            return False  # wholly behind the camera
        depth = frame[0].z
        projected = [(p.x * depth / p.z, p.y * depth / p.z) for p in front]
    xs = [p[0] for p in projected]
    ys = [p[1] for p in projected]
    return max(xs) >= x0 and min(xs) <= x1 and max(ys) >= y0 and min(ys) <= y1


def _openness(scene: Any, depsgraph: Any, origin: Vector) -> dict[str, float]:
    """Share of directions from the camera that escape to the sky, all round and upward."""

    escaped = escaped_up = up = 0
    golden = math.pi * (3.0 - math.sqrt(5.0))
    for i in range(_OPENNESS_RAYS):
        z = 1.0 - 2.0 * (i + 0.5) / _OPENNESS_RAYS
        radius = math.sqrt(max(0.0, 1.0 - z * z))
        direction = Vector((math.cos(golden * i) * radius, math.sin(golden * i) * radius, z))
        hit = scene.ray_cast(depsgraph, origin, direction)[0]
        if z > 0:
            up += 1
        if not hit:
            escaped += 1
            if z > 0:
                escaped_up += 1
    return {
        "open_fraction": round(escaped / _OPENNESS_RAYS, 4),
        "open_upward_fraction": round(escaped_up / max(up, 1), 4),
    }


def _write_png(path: Path, width: int, height: int, rgba: bytes) -> None:
    """A minimal RGBA PNG writer, so the pass needs no image library."""

    rows = b"".join(
        b"\x00" + rgba[row * width * 4 : (row + 1) * width * 4] for row in range(height)
    )

    def chunk(tag: bytes, data: bytes) -> bytes:
        body = tag + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))

    header = struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(rows, 6))
        + chunk(b"IEND", b"")
    )


__all__ = ["LIGHT_MAP_SCHEMA", "light_pass"]
