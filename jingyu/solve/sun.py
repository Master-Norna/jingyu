"""Aim the sun: from "sunlight lands here" to a sun elevation and azimuth.

Indoors the sun reaches a spot only through an opening, so the answer is the
direction from the spot through the window; with the elevation (the time of
day) or the azimuth fixed, the other angle is searched along the opening.
Every candidate direction is checked with rays: a reveal, a shelf or the
frame of the window may still cast the spot into shadow, and then the next
direction through the opening is tried.  Outdoors, with no opening, the
missing angle is chosen so nothing stands between the spot and the sun.

A target object stands for the part of it the camera sees: the chosen direction
is the one that puts the largest share of that in sunlight.

The result also traces sunbeams through the opening into the room, so it can
say where the patch of sunlight falls: on which objects, and where in the frame.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ..camera import CameraModel
from ..conventions import sun_rotation_deg
from ..errors import JingyuError, pointer_join
from ..geometry.mesh import Portal
from ..geometry.raycast import cast, unit
from ..geometry.transform import Vec3
from ..materials import MATERIALS
from ..placement import Placement
from .common import centre_of, frame_box, round_vec, sample_points

ENVIRONMENT_SUN = "environment.sun"
#: Below this elevation the daylight sun is too faint to give direct light.
LOW_SUN = 3.0
_ELEVATION_RANGE = (-10.0, 90.0)
_STEP = 0.25
_PATCH_GRID = 16
#: Lines of sight per side of the grid that finds what the camera sees of an object.
_SEEN_GRID = 8
#: For an object target, this many directions are compared before settling for the
#: best partly lit one; a single point takes the first direction that reaches it.
_OBJECT_BUDGET = 60
_MAX_TRIES = 400


@dataclass(frozen=True)
class SunRequest:
    point: Vec3 | None = None
    object: str | None = None
    #: A spot in the render camera's frame (u from the left, v from the top).
    uv: tuple[float, float] | None = None
    #: The object with the opening and the opening's key (e.g. "openings/0").
    through: tuple[str, str] | None = None
    elevation: float | None = None
    azimuth: float | None = None
    light: str | None = None


@dataclass(frozen=True)
class _Opening:
    object: str
    index: int
    portal: Portal


def aim_sun(scene: Mapping[str, Any], placement: Placement, request: SunRequest) -> dict[str, Any]:
    light, current = _light(scene, placement, request.light)
    meshes = {
        o["id"]: placement.meshes[o["id"]]
        for o in scene["objects"]
        if o["visible"] and o["id"] in placement.meshes
    }
    clear = _transmissive(scene)
    opaque = {k: v for k, v in meshes.items() if k not in clear}
    target, samples, target_object = _target(scene, placement, meshes, request)
    openings = _openings(scene, meshes, request.through)
    budget = 1 if len(samples) == 1 else _OBJECT_BUDGET

    best: tuple[float, Vec3, _Opening | None, Counter[str]] | None = None
    for tried, (direction, opening) in enumerate(
        _candidates(target, openings, request, current), start=1
    ):
        shade = _shade(opaque, samples, direction)
        lit = 1.0 - sum(shade.values()) / len(samples)
        if best is None or lit > best[0] + 1e-9:
            best = (lit, direction, opening, shade)
        if lit >= 1.0 or (lit > 0.0 and tried >= budget) or tried >= _MAX_TRIES:
            break
    warnings: list[dict[str, str]] = []
    if best is None:
        where = "through that opening" if request.through else "through any opening"
        raise JingyuError(
            "solve.no_solution",
            f"no sun direction reaches the target {where} with the angles given",
            hint="the target may be higher than the window, or the fixed elevation or "
            "azimuth may miss the opening; free one of them",
        )
    lit, direction, opening, shade = best
    if lit < (1.0 if len(samples) == 1 else 0.5):
        caster = shade.most_common(1)[0][0]
        if caster == target_object:
            warnings.append(
                {
                    "code": "solve.sun_blocked",
                    "message": f"at best {lit:.0%} of what the camera sees of {caster!r} is "
                    "in sunlight: the camera faces its shadow side (backlight)",
                    "hint": "move the camera toward the sun's side for a lit subject, or keep "
                    "the backlight for rim light",
                }
            )
        else:
            warnings.append(
                {
                    "code": "solve.sun_blocked",
                    "message": f"at best {lit:.0%} of the target is in sunlight; the rest is "
                    f"shaded by {caster!r}",
                    "hint": f"move {caster!r}, or free the elevation or azimuth",
                }
            )
    elevation = math.degrees(math.asin(max(-1.0, min(1.0, direction[2]))))
    azimuth = math.degrees(math.atan2(direction[1], direction[0])) % 360.0
    if elevation < LOW_SUN:
        warnings.append(
            {
                "code": "solve.low_sun",
                "message": f"the sun would stand {elevation:.1f} degrees high: at the horizon "
                "daylight has almost no direct sun",
                "hint": "choose a target lower than the opening, or a higher opening",
            }
        )

    result: dict[str, Any] = {
        "light": light,
        "sun": {"elevation": round(elevation, 2), "azimuth": round(azimuth, 2)},
        "target": {
            "point": round_vec(target),
            "object": target_object,
            "samples": len(samples),
            "lit_fraction": round(lit, 4),
        },
        "operations": _operations(light, elevation, azimuth),
        "warnings": warnings,
    }
    if opening is not None:
        result["through"] = {
            "object": opening.object,
            "opening": opening.portal.key,
            "pointer": pointer_join(
                "objects", opening.index, "geometry", *opening.portal.key.split("/")
            ),
            "elevation_range": _range(target, opening.portal, azimuth, "elevation"),
            "azimuth_range": _range(target, opening.portal, elevation, "azimuth"),
        }
        result["patch"] = _patch(scene, placement, opaque, opening.portal, direction)
    return result


def _light(scene: Mapping[str, Any], placement: Placement, wanted: str | None) -> tuple[str, Vec3]:
    """The light to aim and the current direction toward it."""

    environment = scene["world"].get("environment")
    suns = [light for light in scene["lights"] if light["kind"] == "sun"]
    if wanted is None:
        if environment is not None and environment["family"] == "daylight":
            wanted = ENVIRONMENT_SUN
        elif len(suns) == 1:
            wanted = suns[0]["id"]
        else:
            raise JingyuError(
                "solve.no_solution",
                "there is no single sun to aim",
                hint="use a daylight environment, or name a light of kind sun",
            )
    if wanted == ENVIRONMENT_SUN:
        if environment is None or environment["family"] != "daylight":
            raise JingyuError(
                "solve.no_solution",
                "the environment has no sun",
                hint='set world.environment to {"family": "daylight"}',
            )
        e, a = math.radians(environment["sun_elevation"]), math.radians(environment["sun_azimuth"])
        return wanted, (math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e))
    light = next((entry for entry in suns if entry["id"] == wanted), None)
    if light is None:
        raise JingyuError(
            "spec.unknown_reference",
            f"{wanted!r} is not a light of kind sun",
            hint=f"suns: {', '.join([ENVIRONMENT_SUN] + [s['id'] for s in suns])}",
        )
    matrix = placement.oriented[wanted]
    # A sun lamp shines along its local -Z, so the sun itself lies along +Z.
    return wanted, unit((matrix[2], matrix[6], matrix[10]))


def _transmissive(scene: Mapping[str, Any]) -> set[str]:
    clear_materials = {
        m["id"]
        for m in scene["materials"]
        if MATERIALS.run({k: v for k, v in m.items() if k != "id"}).refractive
    }
    return {o["id"] for o in scene["objects"] if o.get("material") in clear_materials}


def _target(
    scene: Mapping[str, Any],
    placement: Placement,
    meshes: Mapping[str, Any],
    request: SunRequest,
) -> tuple[Vec3, list[Vec3], str | None]:
    """The target's anchor point, the surface points that must be lit, and its object.

    An object stands for the part of it the camera sees (or, when it is out of
    view, its whole surface): lighting "the bowl" means lighting what a viewer
    sees of it, not one point that may lie in its own shadow.
    """

    given = [request.point is not None, request.object is not None, request.uv is not None]
    if sum(given) != 1:
        raise JingyuError("tool.invalid_arguments", "give exactly one of point, object or uv")
    if request.point is not None:
        point = (float(request.point[0]), float(request.point[1]), float(request.point[2]))
        return point, [point], None
    camera = CameraModel.from_scene(scene, placement)
    if request.object is not None:
        mesh = meshes.get(request.object)
        if mesh is None:
            raise JingyuError(
                "spec.unknown_reference", f"{request.object!r} is not a visible object"
            )
        samples = _seen_surface(camera, meshes, request.object) or sample_points(mesh, 48)
        return centre_of(samples), samples, request.object
    assert request.uv is not None
    origin, direction = camera.ray(*request.uv)
    found = cast(meshes, origin, direction)
    if found is None:
        raise JingyuError(
            "solve.no_solution",
            "that spot of the frame shows no object, only the world behind",
            hint="point at a surface where the sunlight should land",
        )
    return found.hit.point, [found.hit.point], found.object


def _seen_surface(camera: CameraModel, meshes: Mapping[str, Any], obj_id: str) -> list[Vec3]:
    """Points of *obj_id* the camera sees: lines of sight on a grid over its frame box."""

    box = frame_box(camera, sample_points(meshes[obj_id], 200))
    if box is None:
        return []
    u0, u1 = max(0.0, box["u0"]), min(1.0, box["u1"])
    v0, v1 = max(0.0, box["v0"]), min(1.0, box["v1"])
    if u0 >= u1 or v0 >= v1:
        return []
    points = []
    for i in range(_SEEN_GRID):
        for j in range(_SEEN_GRID):
            u = u0 + (u1 - u0) * (i + 0.5) / _SEEN_GRID
            v = v0 + (v1 - v0) * (j + 0.5) / _SEEN_GRID
            found = cast(meshes, *camera.ray(u, v))
            if found is not None and found.object == obj_id:
                # Lift the point off the surface toward the camera so its own facet
                # does not shade it.
                points.append(
                    tuple(
                        p - d * 1e-4
                        for p, d in zip(found.hit.point, camera.ray(u, v)[1], strict=True)
                    )
                )
    return points  # type: ignore[return-value]


def _shade(opaque: Mapping[str, Any], samples: Sequence[Vec3], toward_sun: Vec3) -> Counter[str]:
    """For each object, how many of *samples* it keeps out of the sun."""

    shade: Counter[str] = Counter()
    for point in samples:
        origin = tuple(p + d * 1e-4 for p, d in zip(point, toward_sun, strict=True))
        found = cast(opaque, origin, toward_sun)
        if found is not None:
            shade[found.object] += 1
    return shade


def _openings(
    scene: Mapping[str, Any], meshes: Mapping[str, Any], through: tuple[str, str] | None
) -> list[_Opening]:
    index = {o["id"]: i for i, o in enumerate(scene["objects"])}
    found = [
        _Opening(obj_id, index[obj_id], portal)
        for obj_id, mesh in meshes.items()
        for portal in mesh.portals
    ]
    if through is None:
        return found
    obj_id, key = through
    chosen = [o for o in found if o.object == obj_id and o.portal.key == key]
    if not chosen:
        available = [f"{o.object} {o.portal.key}" for o in found] or ["none"]
        raise JingyuError(
            "spec.unknown_reference",
            f"{obj_id!r} has no visible opening {key!r}",
            hint=f"openings: {', '.join(available)}",
        )
    return chosen


def _direction(elevation: float, azimuth: float) -> Vec3:
    e, a = math.radians(elevation), math.radians(azimuth)
    return (math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e))


def _through(target: Vec3, portal: Portal, direction: Vec3) -> float | None:
    """How central the beam crosses the opening (0 centre, 1 edge); None if it misses."""

    crossing = portal.crossing(target, direction)
    if crossing is None:
        return None
    _, a, b = crossing
    centrality = max(abs(a), abs(b))
    return centrality if centrality <= 1.0 else None


def _sweep(fixed: float, sweep: str) -> Iterable[tuple[float, float]]:
    """(elevation, azimuth) pairs with one angle fixed and the other stepped."""

    if sweep == "azimuth":
        steps = int(360.0 / _STEP)
        return ((fixed, i * _STEP) for i in range(steps))
    low, high = _ELEVATION_RANGE
    steps = int((high - low) / _STEP) + 1
    return ((low + i * _STEP, fixed) for i in range(steps))


def _candidates(
    target: Vec3,
    openings: Sequence[_Opening],
    request: SunRequest,
    current: Vec3,
) -> Iterable[tuple[Vec3, _Opening | None]]:
    """Directions toward the sun to try, best first."""

    ranked: list[tuple[float, Vec3, _Opening | None]] = []
    e_fixed, a_fixed = request.elevation, request.azimuth

    def preference(direction: Vec3, centrality: float) -> float:
        # Central through the opening first; among equals, closest to the sun now.
        cosine = sum(d * c for d, c in zip(direction, current, strict=True))
        return centrality + 0.05 * math.acos(max(-1.0, min(1.0, cosine)))

    if not openings:
        if e_fixed is not None and a_fixed is not None:
            return [(_direction(e_fixed, a_fixed), None)]
        here_e = math.degrees(math.asin(max(-1.0, min(1.0, current[2]))))
        if a_fixed is None:
            pairs = _sweep(e_fixed if e_fixed is not None else here_e, "azimuth")
        else:
            pairs = _sweep(a_fixed, "elevation")
        for elevation, azimuth in pairs:
            if elevation <= 0.0:
                continue
            direction = _direction(elevation, azimuth)
            ranked.append((preference(direction, 0.0), direction, None))
        ranked.sort(key=lambda item: item[0])
        return [(d, o) for _, d, o in ranked]

    for opening in openings:
        portal = opening.portal
        if e_fixed is not None and a_fixed is not None:
            direction = _direction(e_fixed, a_fixed)
            centrality = _through(target, portal, direction)
            if centrality is not None:
                ranked.append((preference(direction, centrality), direction, opening))
            continue
        if e_fixed is None and a_fixed is None:
            grid = [i / 4.0 - 1.0 for i in range(1, 8)]  # -0.75 .. 0.75
            for a in grid:
                for b in grid:
                    direction = unit(
                        tuple(p - t for p, t in zip(portal.point(a, b), target, strict=True))
                    )
                    if direction[2] <= 0.0:
                        continue
                    ranked.append((preference(direction, max(abs(a), abs(b))), direction, opening))
            continue
        if e_fixed is not None:
            pairs = _sweep(e_fixed, "azimuth")
        else:
            assert a_fixed is not None
            pairs = _sweep(a_fixed, "elevation")
        for elevation, azimuth in pairs:
            direction = _direction(elevation, azimuth)
            centrality = _through(target, portal, direction)
            if centrality is not None:
                ranked.append((preference(direction, centrality), direction, opening))
    ranked.sort(key=lambda item: item[0])
    return [(d, o) for _, d, o in ranked]


def _range(target: Vec3, portal: Portal, fixed: float, sweep: str) -> list[float] | None:
    """The span of the swept angle over which the beam passes the opening."""

    inside = [
        (elevation if sweep == "elevation" else azimuth)
        for elevation, azimuth in _sweep(fixed, sweep)
        if _through(target, portal, _direction(elevation, azimuth)) is not None
    ]
    if not inside:
        return None
    if sweep == "azimuth" and inside[0] == 0.0 and inside[-1] >= 360.0 - _STEP:
        # The span wraps through 0: report it from its start before 360.
        gap = max(range(1, len(inside)), key=lambda i: inside[i] - inside[i - 1], default=None)
        if gap is not None and inside[gap] - inside[gap - 1] > _STEP * 1.5:
            return [round(inside[gap], 2), round(inside[gap - 1] + 360.0, 2)]
    return [round(inside[0], 2), round(inside[-1], 2)]


def _patch(
    scene: Mapping[str, Any],
    placement: Placement,
    opaque: Mapping[str, Any],
    portal: Portal,
    toward_sun: Vec3,
) -> list[dict[str, Any]]:
    """Where sunbeams through the opening land: per object, share and frame box."""

    beam = (-toward_sun[0], -toward_sun[1], -toward_sun[2])
    camera = CameraModel.from_scene(scene, placement)
    hits: dict[str, list[Vec3]] = {}
    total = 0
    for i in range(_PATCH_GRID):
        for j in range(_PATCH_GRID):
            a = -1.0 + (2 * i + 1) / _PATCH_GRID
            b = -1.0 + (2 * j + 1) / _PATCH_GRID
            origin = portal.point(a, b)
            # Start outside, so the far side of a thick wall's reveal can shade the beam.
            start = tuple(o + s * 0.5 for o, s in zip(origin, toward_sun, strict=True))
            total += 1
            found = cast(opaque, start, beam)
            if found is not None:
                hits.setdefault(found.object, []).append(found.hit.point)
    rows = []
    for obj_id, points in sorted(hits.items(), key=lambda item: -len(item[1])):
        projected = [p for p in (camera.project(q) for q in points) if p is not None]
        in_frame = [(u, v) for u, v, _ in projected if 0.0 <= u <= 1.0 and 0.0 <= v <= 1.0]
        row: dict[str, Any] = {
            "object": obj_id,
            "fraction_of_beam": round(len(points) / total, 4),
            "world_box": [
                round_vec([min(p[k] for p in points) for k in range(3)]),
                round_vec([max(p[k] for p in points) for k in range(3)]),
            ],
        }
        if in_frame:
            row["bbox_uv"] = {
                "u0": round(min(u for u, _ in in_frame), 4),
                "v0": round(min(v for _, v in in_frame), 4),
                "u1": round(max(u for u, _ in in_frame), 4),
                "v1": round(max(v for _, v in in_frame), 4),
            }
        row["in_frame_fraction"] = round(len(in_frame) / len(points), 4)
        rows.append(row)
    return rows


def _operations(light: str, elevation: float, azimuth: float) -> list[dict[str, Any]]:
    e, a = round(elevation, 2), round(azimuth, 2)
    if light == ENVIRONMENT_SUN:
        return [
            {
                "op": "merge",
                "path": "/world/environment",
                "value": {"sun_elevation": e, "sun_azimuth": a},
            }
        ]
    rotation = [round(v, 3) for v in sun_rotation_deg(e, a)]
    return [
        {
            "op": "merge",
            "path": f"/lights/@{light}",
            "value": {"rotation": rotation, "look_at": None},
        }
    ]


__all__ = ["ENVIRONMENT_SUN", "SunRequest", "aim_sun"]
