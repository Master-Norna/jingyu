"""Arbitrary rays against world meshes, and aiming lights and cameras."""

from __future__ import annotations

import math
import random

import pytest

from jingyu.geometry import GEOMETRY
from jingyu.geometry.raycast import _intersect, cast, cast_mesh, unit
from jingyu.geometry.spatial import WorldMesh
from jingyu.geometry.transform import apply_direction, compose, look_rotation
from jingyu.placement import aim_matrix

WALL = {
    "op": "wall",
    "size": [4.0, 0.15, 2.6],
    "openings": [{"x": -0.6, "sill": 0.9, "width": 1.2, "height": 1.3}],
}


def _world(
    spec: dict[str, object], location=(0.0, 0.0, 0.0), rotation=(0.0, 0.0, 0.0)
) -> WorldMesh:
    normalized = {**GEOMETRY.get(str(spec["op"])).catalog_entry()["examples"][0], **spec}
    for name, schema in GEOMETRY.get(str(spec["op"])).params.items():
        if name not in normalized and "default" in schema:
            normalized[name] = schema["default"]
    return WorldMesh.from_mesh(GEOMETRY.run(normalized), compose(location, rotation, (1, 1, 1)))


def test_the_first_surface_along_a_ray_is_found_even_when_grazing_an_edge() -> None:
    wall = _world(WALL, rotation=(0, 0, 90))
    # y = 0 runs exactly along the edge between two triangles of the front face.
    hit = cast_mesh(wall, (-2, 0, 1.5), (1, 0, 0))
    assert hit is not None
    assert hit.distance == pytest.approx(2 - 0.075)
    assert hit.normal == pytest.approx((-1, 0, 0), abs=1e-12)
    # Through the opening there is nothing to hit.
    assert cast_mesh(wall, (-2, -0.6, 1.5), (1, 0, 0)) is None
    # A limit shorter than the distance to the wall finds nothing either.
    assert cast_mesh(wall, (-2, 0, 1.5), (1, 0, 0), t_max=1.0) is None


def test_the_hierarchy_agrees_with_testing_every_triangle() -> None:
    vessel = _world({"op": "vessel"})
    rng = random.Random(7)
    for _ in range(60):
        origin = (rng.uniform(-0.3, 0.3), -1.0, rng.uniform(-0.05, 0.3))
        direction = unit((rng.uniform(-0.2, 0.2), 1.0, rng.uniform(-0.2, 0.2)))
        hit = cast_mesh(vessel, origin, direction)
        distances = [
            t
            for t in (
                _intersect(vessel.vertices, tri, *origin, *direction) for tri in vessel.triangles
            )
            if t is not None
        ]
        if not distances:
            assert hit is None
        else:
            assert hit is not None and hit.distance == pytest.approx(min(distances), abs=1e-12)


def test_casting_into_a_scene_returns_the_nearest_object() -> None:
    near = _world({"op": "box", "size": [0.2, 0.2, 0.2]}, location=(0, 1, 0))
    far = _world({"op": "box", "size": [0.2, 0.2, 0.2]}, location=(0, 3, 0))
    meshes = {"far": far, "near": near}
    found = cast(meshes, (0, 0, 0.1), (0, 1, 0))
    assert found is not None and found.object == "near"
    assert found.hit.distance == pytest.approx(0.9)
    skipped = cast(meshes, (0, 0, 0.1), (0, 1, 0), skip=["near"])
    assert skipped is not None and skipped.object == "far"
    assert cast(meshes, (0, 0, 0.1), (0, -1, 0)) is None


@pytest.mark.parametrize(
    "direction", [(0, 1, 0), (1, -1, 0.3), (0.2, 0.1, -3), (0, 0, -1), (0, 0, 1)]
)
def test_look_rotation_points_minus_z_along_the_direction_without_roll(
    direction: tuple[float, float, float],
) -> None:
    rotation = look_rotation(direction)
    forward = apply_direction(rotation, (0, 0, -1))
    assert forward == pytest.approx(unit(direction), abs=1e-12)
    right = apply_direction(rotation, (1, 0, 0))
    up = apply_direction(rotation, (0, 1, 0))
    assert right[2] == pytest.approx(0.0, abs=1e-12)  # the horizon stays level
    assert up[2] >= -1e-12  # the frame is not upside down
    assert math.isclose(sum(r * u for r, u in zip(right, up, strict=True)), 0.0, abs_tol=1e-12)


def test_aim_matrix_uses_the_parent_for_position_and_rotation_but_not_scale() -> None:
    parent = compose((1, 2, 0), (0, 0, 90), (2, 2, 2))
    aimed = aim_matrix((1, 0, 0), None, (0, 0, 0), parent)
    assert (aimed[3], aimed[7], aimed[11]) == pytest.approx((1, 4, 0))
    assert apply_direction(aimed, (1, 0, 0)) == pytest.approx((0, 1, 0), abs=1e-12)
    looking = aim_matrix((0, 0, 0), (1, 5, 0), None, parent)
    assert apply_direction(looking, (0, 0, -1)) == pytest.approx((0, 1, 0), abs=1e-12)
