from __future__ import annotations

import math
from collections.abc import Mapping
from itertools import pairwise
from typing import Any

import pytest

from jingyu.errors import JingyuError
from jingyu.generator import GeneratorDef
from jingyu.geometry import GEOMETRY, MeshData
from jingyu.geometry.curves import linspace, pchip, smooth_monotone
from jingyu.geometry.lathe import close_solid, revolve, shell
from jingyu.geometry.ops import vessel_profile
from jingyu.scene import scene_schema
from jingyu.scene.normalize import apply_defaults

#: Flat shapes are single-sided surfaces, not solids.
FLAT_OPS = {"plane"}
#: A room stands on its floor surface: the floor slab lies below the origin.
ORIGIN_ON_FLOOR = {"room"}
#: A sweep keeps the path it is given (a handle's ends on its mug's side).
KEEPS_COORDINATES = {"sweep"}


def _defaults(definition: GeneratorDef[Any]) -> dict[str, Any]:
    return _params(definition, {})


def _params(definition: GeneratorDef[Any], example: Mapping[str, Any]) -> dict[str, Any]:
    """Parameters as a scene hands them to a generator: every default filled, nested too."""

    schema = definition.branch_schema("op")
    filled = apply_defaults({"op": definition.name, **example}, schema, scene_schema())
    return {k: v for k, v in filled.items() if k != "op"}


def _cases() -> list[Any]:
    cases = []
    for definition in GEOMETRY:
        if not definition.required:
            cases.append(pytest.param(definition.name, _defaults(definition), id=definition.name))
        for index, example in enumerate(definition.examples):
            params = _params(definition, example)
            cases.append(pytest.param(definition.name, params, id=f"{definition.name}-ex{index}"))
    return cases


def _vessel(**overrides: Any) -> dict[str, Any]:
    return {**_defaults(GEOMETRY.get("vessel")), "segments": 24, **overrides}


def _assert_well_formed(mesh: MeshData) -> None:
    count = len(mesh.vertices)
    assert count > 0 and mesh.faces
    for face in mesh.faces:
        assert len(face) >= 3
        assert len(set(face)) == len(face), f"face repeats a vertex: {face}"
        assert all(0 <= i < count for i in face), f"face indexes a missing vertex: {face}"
    for a, b in mesh.sharp_edges:
        assert 0 <= a < count and 0 <= b < count
    assert all(math.isfinite(c) for v in mesh.vertices for c in v)


def test_every_op_has_an_example() -> None:
    assert all(definition.examples for definition in GEOMETRY)


@pytest.mark.parametrize(("name", "params"), _cases())
def test_ops_produce_valid_meshes_from_defaults_and_examples(
    name: str, params: dict[str, Any]
) -> None:
    definition = GEOMETRY.get(name)
    assert definition.check(params) == []
    mesh = definition.run(params)
    _assert_well_formed(mesh)
    if name in FLAT_OPS:
        return
    assert mesh.is_closed_manifold()
    assert mesh.signed_volume() > 0
    # Solids stand on their origin.
    if name in KEEPS_COORDINATES:
        pass
    elif name in ORIGIN_ON_FLOOR:
        assert mesh.bounds()[0][2] == pytest.approx(-params["slab_thickness"] * params["floor"])
    else:
        assert mesh.bounds()[0][2] == pytest.approx(0.0, abs=1e-12)
    if mesh.parts:
        assert set(mesh.parts) == set(definition.parts)
        assert len(mesh.face_parts) == len(mesh.faces)


def test_registry_run_dispatches_on_op() -> None:
    mesh = GEOMETRY.run({"op": "box", "size": [2, 4, 3]})
    assert mesh.bounds() == ((-1.0, -2.0, 0.0), (1.0, 2.0, 3.0))
    assert mesh.signed_volume() == pytest.approx(24.0)


def test_plane_is_a_centred_upward_facing_quad() -> None:
    mesh = GEOMETRY.run({"op": "plane", "size": [2, 6]})
    assert mesh.bounds() == ((-1.0, -3.0, 0.0), (1.0, 3.0, 0.0))
    assert len(mesh.faces) == 1
    a, b, c = (mesh.vertices[i] for i in mesh.faces[0][:3])
    normal_z = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    assert normal_z > 0


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        ({"op": "cylinder", "radius": 0.5, "height": 2.0, "segments": 256}, math.pi * 0.25 * 2),
        (
            {"op": "cone", "radius": 0.5, "height": 3.0, "top_radius": 0.0, "segments": 256},
            math.pi * 0.25 * 3 / 3,
        ),
        (
            {"op": "sphere", "radius": 0.5, "segments": 256, "rings": 128},
            4 / 3 * math.pi * 0.125,
        ),
    ],
)
def test_lathe_primitives_approach_their_analytic_volume(
    spec: dict[str, Any], expected: float
) -> None:
    definition = GEOMETRY.get(spec["op"])
    mesh = definition.run(_params(definition, spec))
    assert mesh.signed_volume() == pytest.approx(expected, rel=2e-3)


def test_truncated_cone_is_closed() -> None:
    mesh = GEOMETRY.get("cone").run(
        {"radius": 0.2, "height": 0.3, "top_radius": 0.1, "segments": 12}
    )
    assert mesh.is_closed_manifold()
    assert mesh.bounds()[1][2] == pytest.approx(0.3)


# ---------------------------------------------------------------- vessels


@pytest.mark.parametrize("open_", [True, False])
def test_vessel_rim_is_flat_at_the_full_height(open_: bool) -> None:
    params = _vessel(open=open_)
    mesh = GEOMETRY.get("vessel").run(params)
    top = max(v[2] for v in mesh.vertices)
    assert abs(top - params["height"]) <= 1e-9
    rim = [v for v in mesh.vertices if abs(v[2] - params["height"]) <= 1e-9]
    # The rim is whole rings, not a single stray vertex.
    assert len(rim) >= params["segments"]


def test_open_vessel_is_a_hollow_shell_and_closed_vessel_is_solid() -> None:
    definition = GEOMETRY.get("vessel")
    hollow = definition.run(_vessel(open=True))
    solid = definition.run(_vessel(open=False))
    for mesh in (hollow, solid):
        assert mesh.is_closed_manifold()
        assert mesh.signed_volume() > 0
    assert hollow.signed_volume() < solid.signed_volume() / 2

    def on_axis(mesh: MeshData) -> list[float]:
        return sorted(v[2] for v in mesh.vertices if v[0] == v[1] == 0.0)

    # A solid is capped on the axis at the top; a shell's inner floor sits one thickness up.
    assert on_axis(solid) == pytest.approx([0.0, _vessel()["height"]])
    assert on_axis(hollow) == pytest.approx([0.0, _vessel()["thickness"]])


def test_vessel_profile_passes_through_its_control_rings() -> None:
    params = _vessel(profile_samples=201, belly_at=0.5, neck_at=0.75)
    profile = dict((round(z, 9), r) for r, z in vessel_profile(params))
    h = params["height"]
    assert profile[0.0] == pytest.approx(params["base_radius"])
    assert profile[round(0.5 * h, 9)] == pytest.approx(params["belly_radius"])
    assert profile[round(0.75 * h, 9)] == pytest.approx(params["neck_radius"])
    assert profile[round(h, 9)] == pytest.approx(params["lip_radius"])
    radii = list(profile.values())
    assert min(radii) >= min(params[k] for k in ("base_radius", "neck_radius")) - 1e-12
    assert max(radii) <= params["belly_radius"] + 1e-12


@pytest.mark.parametrize(("belly_at", "neck_at"), [(0.8, 0.8), (0.9, 0.5), (0.5, 1.0)])
def test_vessel_check_requires_belly_below_neck(belly_at: float, neck_at: float) -> None:
    problems = GEOMETRY.get("vessel").check(_vessel(belly_at=belly_at, neck_at=neck_at))
    assert [param for param, _ in problems] == ["neck_at"]


def test_vessel_check_rejects_walls_thicker_than_the_neck() -> None:
    params = _vessel(thickness=0.05)
    problems = GEOMETRY.get("vessel").check(params)
    assert [param for param, _ in problems] == ["thickness"]
    with pytest.raises(JingyuError) as info:
        GEOMETRY.get("vessel").run(params)
    assert info.value.code == "geometry.invalid_profile"


def test_closed_vessel_ignores_wall_thickness() -> None:
    assert GEOMETRY.get("vessel").check(_vessel(open=False, thickness=0.05)) == []


# ------------------------------------------------------------------ lathe


@pytest.mark.parametrize(
    "profile",
    [
        [[0.05, 0.0]],  # a single point
        [[0.05, 0.0], [0.0, 0.05], [0.05, 0.1]],  # an interior point on the axis
        [[0.05, 0.0], [0.05, 0.0], [0.04, 0.1]],  # coincident points
        [[0.0, 0.0], [0.0, 0.1]],  # nothing off the axis
    ],
)
def test_lathe_rejects_bad_solid_profiles(profile: list[list[float]]) -> None:
    definition = GEOMETRY.get("lathe")
    params = _params(definition, {"profile": profile})
    assert [param for param, _ in definition.check(params)] == ["profile"]
    with pytest.raises(JingyuError) as info:
        definition.run(params)
    assert info.value.code == "geometry.invalid_profile"


@pytest.mark.parametrize(
    ("profile", "thickness"),
    [
        ([[0.0, 0.0], [0.05, 0.1]], 0.003),  # a shell wall touching the axis
        ([[0.01, 0.0], [0.01, 0.1]], 0.02),  # thicker than the radius
        ([[0.05, 0.0], [0.05, 0.01]], 0.02),  # floor reaches the rim
    ],
)
def test_lathe_rejects_bad_shell_profiles(profile: list[list[float]], thickness: float) -> None:
    with pytest.raises(JingyuError) as info:
        shell([tuple(p) for p in profile], thickness)  # type: ignore[misc]
    assert info.value.code == "geometry.invalid_profile"


@pytest.mark.parametrize(
    ("polyline", "segments"),
    [
        ([(0.0, 0.0), (0.1, 0.0), (0.0, 0.1)], 2),  # too few segments
        ([(0.1, 0.0), (0.1, 0.1), (0.0, 0.1)], 8),  # does not start on the axis
        ([(0.0, 0.0), (0.1, 0.1)], 8),  # too few points
    ],
)
def test_revolve_rejects_malformed_polylines(
    polyline: list[tuple[float, float]], segments: int
) -> None:
    with pytest.raises(JingyuError) as info:
        revolve(polyline, segments)
    assert info.value.code == "geometry.invalid_profile"


def test_close_solid_adds_axis_points_only_where_missing() -> None:
    assert close_solid([(0.1, 0.0), (0.1, 0.2)]) == [(0.0, 0.0), (0.1, 0.0), (0.1, 0.2), (0.0, 0.2)]
    assert close_solid([(0.0, 0.0), (0.1, 0.1), (1e-12, 0.2)]) == [
        (0.0, 0.0),
        (0.1, 0.1),
        (0.0, 0.2),
    ]


def test_sharp_edges_follow_profile_corners() -> None:
    cylinder = revolve([(0.0, 0.0), (0.1, 0.0), (0.1, 0.2), (0.0, 0.2)], 8, sharp_angle=30.0)
    assert len(cylinder.sharp_edges) == 16  # both rims, 8 edges each
    smooth = revolve([(0.0, 0.0), (0.1, 0.0), (0.1, 0.2), (0.0, 0.2)], 8, sharp_angle=180.0)
    assert smooth.sharp_edges == ()


# ------------------------------------------------------------------ curves


def test_pchip_preserves_monotone_data() -> None:
    xs = [0.0, 1.0, 1.5, 4.0, 5.0]
    ys = [0.0, 0.1, 2.0, 2.05, 7.0]
    samples = linspace(0.0, 5.0, 501)
    values = pchip(xs, ys, samples)
    assert all(b >= a - 1e-12 for a, b in pairwise(values))
    assert min(values) >= 0.0 and max(values) <= 7.0
    falling = pchip(xs, [-y for y in ys], samples)
    assert all(b <= a + 1e-12 for a, b in pairwise(falling))


def test_pchip_interpolates_control_points_and_clamps_outside() -> None:
    xs, ys = [0.0, 1.0, 3.0], [1.0, 3.0, 2.0]
    assert pchip(xs, ys, xs) == pytest.approx(ys)
    assert pchip(xs, ys, [-1.0, 4.0]) == [1.0, 2.0]
    # No overshoot between a local maximum and its neighbours.
    assert max(pchip(xs, ys, linspace(0.0, 3.0, 301))) <= 3.0 + 1e-12


def _second_differences(f: Any, x: float, step: float = 1e-5) -> tuple[float, float]:
    """Second derivative just left and just right of *x*."""

    def d2(c: float) -> float:
        a, b, e = f([c - step, c, c + step])
        return float((a - 2 * b + e) / step**2)

    return d2(x - 2 * step), d2(x + 2 * step)


def test_smooth_monotone_keeps_curvature_continuous_at_control_points() -> None:
    # A vessel profile: base, belly (maximum), neck (minimum), lip.
    xs, ys = [0.0, 0.072, 0.1408, 0.16], [0.06, 0.092, 0.062, 0.066]

    def smooth(samples: list[float]) -> list[float]:
        return smooth_monotone(xs, ys, samples)

    def cubic(samples: list[float]) -> list[float]:
        return pchip(xs, ys, samples)

    before, after = _second_differences(cubic, xs[1])
    assert abs(after - before) > 0.5 * abs(before)  # the cubic's curvature jumps here
    before, after = _second_differences(smooth, xs[1])
    assert after == pytest.approx(before, rel=0.05)


@pytest.mark.parametrize(
    "ys",
    [
        [0.0, 0.1, 2.0, 2.05, 7.0],
        [1.0, 3.0, 2.0, 2.0, 5.0],
        [0.05, 0.09, 0.02, 0.03, 0.0],
    ],
)
def test_smooth_monotone_stays_within_its_control_points(ys: list[float]) -> None:
    xs = [0.0, 1.0, 1.5, 4.0, 5.0]
    assert smooth_monotone(xs, ys, xs) == pytest.approx(ys)
    samples = linspace(0.0, 5.0, 2001)
    values = smooth_monotone(xs, ys, samples)
    for (x0, y0), (x1, y1) in pairwise(zip(xs, ys, strict=True)):
        segment = [v for x, v in zip(samples, values, strict=True) if x0 <= x <= x1]
        assert min(y0, y1) - 1e-9 <= min(segment) and max(segment) <= max(y0, y1) + 1e-9
        rising = y1 >= y0
        assert all((b >= a - 1e-9) if rising else (b <= a + 1e-9) for a, b in pairwise(segment))


def test_pchip_with_two_points_is_linear() -> None:
    assert pchip([0.0, 2.0], [1.0, 5.0], [0.5, 1.0, 1.5]) == pytest.approx([2.0, 3.0, 4.0])


@pytest.mark.parametrize(
    ("xs", "ys"), [([0.0], [1.0]), ([0.0, 1.0], [1.0]), ([0.0, 0.0], [1.0, 2.0])]
)
def test_pchip_rejects_bad_control_points(xs: list[float], ys: list[float]) -> None:
    with pytest.raises(ValueError):
        pchip(xs, ys, [0.5])


def test_linspace_hits_both_ends_exactly() -> None:
    values = linspace(0.0, 0.3, 7)
    assert len(values) == 7
    assert values[0] == 0.0 and values[-1] == 0.3
    with pytest.raises(ValueError):
        linspace(0.0, 1.0, 1)


def test_wall_with_openings_is_a_closed_solid_with_the_right_volume() -> None:
    params = {
        "op": "wall",
        "size": [4.0, 0.15, 2.6],
        "openings": [
            {"x": -0.6, "sill": 0.9, "width": 1.2, "height": 1.3},
            {"x": 1.2, "sill": 0.0, "width": 0.9, "height": 2.1},
        ],
    }
    mesh = GEOMETRY.run(params)
    assert mesh.is_closed_manifold()
    solid = 4.0 * 2.6 - 1.2 * 1.3 - 0.9 * 2.1
    assert math.isclose(mesh.signed_volume(), solid * 0.15, rel_tol=1e-9)
    assert mesh.bounds() == ((-2.0, -0.075, 0.0), (2.0, 0.075, 2.6))


@pytest.mark.parametrize(
    ("openings", "message"),
    [
        ([{"x": 1.9, "sill": 0.5, "width": 0.5, "height": 1.0}], "extends beyond"),
        ([{"x": 0.0, "sill": 2.0, "width": 0.5, "height": 1.0}], "extends beyond"),
        ([{"x": 0.0, "sill": 0.0, "width": 4.0, "height": 2.6}], "whole wall"),
        (
            [
                {"x": -1.0, "sill": 0.0, "width": 2.0, "height": 2.6},
                {"x": 0.9, "sill": 0.0, "width": 2.2, "height": 2.6},
            ],
            "whole wall",
        ),
    ],
)
def test_wall_check_rejects_bad_openings(openings: list[dict[str, float]], message: str) -> None:
    wall = GEOMETRY.get("wall")
    problems = wall.check(wall.with_defaults({"size": [4.0, 0.15, 2.6], "openings": openings}))
    assert [p[0] for p in problems] == ["openings"]
    assert message in problems[0][1]


# ----------------------------------------------------------------- assemblies


def _faces_of(mesh: MeshData, part: str) -> list[int]:
    return [i for i, p in enumerate(mesh.face_parts) if mesh.parts[p] == part]


def test_framed_glazed_openings_have_parts_and_a_clear_portal() -> None:
    mesh = GEOMETRY.run(
        {
            "op": "wall",
            "size": [3.0, 0.2, 2.6],
            "openings": [
                {"x": 0.5, "sill": 0.9, "width": 1.0, "height": 1.2, "frame": 0.06, "glass": True},
                {"x": -0.8, "sill": 0.0, "width": 0.9, "height": 2.1, "frame": 0.05},
            ],
        }
    )
    assert mesh.is_closed_manifold()
    assert mesh.parts == ("walls", "frames", "glass")
    assert _faces_of(mesh, "frames") and len(_faces_of(mesh, "glass")) == 6
    window, door = mesh.portals
    # The clear opening lies inside the frame; a door has no frame across its threshold.
    assert window.half_width[0] == pytest.approx(0.5 - 0.06)
    assert window.half_height[2] == pytest.approx(0.6 - 0.06)
    assert door.centre[2] - door.half_height[2] == pytest.approx(0.0)
    assert door.half_height[2] == pytest.approx((2.1 - 0.05) / 2)


def test_a_frame_wider_than_its_opening_is_rejected() -> None:
    wall = GEOMETRY.get("wall")
    params = wall.with_defaults(
        {
            "size": [3, 0.2, 2.6],
            "openings": [{"x": 0, "sill": 1, "width": 0.1, "height": 1, "frame": 0.06}],
        }
    )
    assert "fills the whole opening" in wall.check(params)[0][1]


def test_a_room_has_named_walls_and_openings_keyed_in_order() -> None:
    mesh = GEOMETRY.run(
        {
            "op": "room",
            "size": [4.0, 5.0, 2.7],
            "openings": [
                {"wall": "right", "x": 0.5, "sill": 0.9, "width": 1.0, "height": 1.2},
                {
                    "wall": "left",
                    "x": -1.0,
                    "sill": 0.9,
                    "width": 1.2,
                    "height": 1.3,
                    "glass": True,
                },
                {"wall": "front", "x": 1.0, "sill": 0.0, "width": 0.9, "height": 2.1},
            ],
        }
    )
    assert mesh.is_closed_manifold()
    assert [p.key for p in mesh.portals] == ["openings/0", "openings/1", "openings/2"]
    right, left, front = mesh.portals
    assert right.centre[0] == pytest.approx(2.0 + 0.075)
    # Seen from inside, x runs to the right: on the right (+x) wall that is -y.
    assert right.centre[1] == pytest.approx(-0.5)
    assert left.centre[0] == pytest.approx(-2.075) and left.centre[1] == pytest.approx(-1.0)
    assert front.centre[1] == pytest.approx(-2.575) and front.centre[0] == pytest.approx(-1.0)
    (_, _, z0), (_, _, z1) = mesh.bounds()
    assert (z0, z1) == pytest.approx((-0.12, 2.7 + 0.12))
    assert _faces_of(mesh, "glass")


def test_a_room_can_leave_out_its_ceiling_and_floor() -> None:
    mesh = GEOMETRY.run({"op": "room", "size": [3, 3, 2.5], "ceiling": False, "floor": False})
    assert not _faces_of(mesh, "ceiling") and not _faces_of(mesh, "floor")
    assert mesh.bounds()[1][2] == pytest.approx(2.5)


@pytest.mark.parametrize("legs", ["square", "round", "tapered"])
def test_table_top_sits_at_the_given_height(legs: str) -> None:
    mesh = GEOMETRY.run({"op": "table", "size": [1.2, 0.7, 0.74], "legs": legs})
    (x0, y0, z0), (x1, y1, z1) = mesh.bounds()
    assert (x1 - x0, y1 - y0, z0, z1) == pytest.approx((1.2, 0.7, 0.0, 0.74))
    top = _faces_of(mesh, "top")
    assert min(mesh.vertices[i][2] for f in top for i in mesh.faces[f]) == pytest.approx(
        0.74 - 0.035
    )


def test_chair_back_rises_above_the_seat_on_the_plus_y_side() -> None:
    mesh = GEOMETRY.run({"op": "chair", "seat_height": 0.46, "back_height": 0.4})
    back = [mesh.vertices[i] for f in _faces_of(mesh, "back") for i in mesh.faces[f]]
    assert max(v[2] for v in back) == pytest.approx(0.86)
    assert min(v[1] for v in back) > 0.1
    stool = GEOMETRY.run({"op": "chair", "back_height": 0.0})
    assert not _faces_of(stool, "back")
    assert stool.bounds()[1][2] == pytest.approx(0.45)


def test_shelf_boards_are_evenly_spaced() -> None:
    spec = {"op": "shelf", "size": [0.8, 0.3, 1.0], "shelves": 3, "plinth": 0.0, "bevel": 0.0}
    mesh = GEOMETRY.run(spec)
    # Without rounding, the shelf faces lie at each board's bottom and top.
    heights = sorted(
        {round(mesh.vertices[i][2], 9) for f in _faces_of(mesh, "shelves") for i in mesh.faces[f]}
    )
    assert len(heights) == 6
    bottoms = [*heights[::2], 1.0 - 0.02]  # ... and the carcass top board
    tops = [0.02, *heights[1::2]]  # the carcass bottom board first
    gaps = [b - t for t, b in zip(tops, bottoms, strict=True)]
    assert gaps == pytest.approx([gaps[0]] * 4)


def test_rounded_boxes_keep_their_size_and_round_their_edges() -> None:
    plain = GEOMETRY.run({"op": "box", "size": [0.4, 0.2, 0.1]})
    round_ = GEOMETRY.run({"op": "box", "size": [0.4, 0.2, 0.1], "bevel": 0.01})
    assert plain.bounds() == round_.bounds()
    assert not plain.smooth and round_.smooth
    # A rounded box is the inner box grown by a ball of the radius; the facets of the
    # rounding lie inside the true curve, so a little more is lost than exactly.
    r, (a, b, c) = 0.01, (0.4 - 0.02, 0.2 - 0.02, 0.1 - 0.02)
    exact = a * b * c + 2 * r * (a * b + b * c + a * c) + math.pi * r * r * (a + b + c)
    exact += 4 / 3 * math.pi * r**3
    assert round_.signed_volume() == pytest.approx(exact, rel=0.01)
    assert round_.signed_volume() < exact
    box = GEOMETRY.get("box")
    assert box.check(box.with_defaults({"size": [0.4, 0.2, 0.1], "bevel": 0.06}))


# ------------------------------------------------------------ free-form shapes


def test_extrude_handles_concave_outlines_either_way_round() -> None:
    l_shape = [[0, 0], [1.2, 0], [1.2, 0.6], [0.6, 0.6], [0.6, 1.4], [0, 1.4]]
    for profile in (l_shape, list(reversed(l_shape))):
        mesh = GEOMETRY.run({"op": "extrude", "profile": profile, "height": 0.9})
        assert mesh.is_closed_manifold()
        area = 1.2 * 0.6 + 0.6 * 0.8
        assert mesh.signed_volume() == pytest.approx(area * 0.9)
    tapered = GEOMETRY.run({"op": "extrude", "profile": l_shape, "height": 0.9, "taper": 0.5})
    assert tapered.is_closed_manifold() and tapered.signed_volume() < area * 0.9
    bow_tie = {"op": "extrude", "profile": [[0, 0], [1, 1], [1, 0], [0, 1]], "height": 1}
    extrude = GEOMETRY.get("extrude")
    assert extrude.check(extrude.with_defaults({k: v for k, v in bow_tie.items() if k != "op"}))


def test_sweep_makes_a_closed_tube_that_narrows() -> None:
    mesh = GEOMETRY.run(
        {
            "op": "sweep",
            "path": [[0, 0, 0], [0, 0, 0.5], [0.3, 0, 0.8]],
            "radius": 0.02,
            "radius_end": 0.01,
        }
    )
    assert mesh.is_closed_manifold() and mesh.signed_volume() > 0
    top = [v for v in mesh.vertices if v[2] > 0.79]
    assert max(abs(v[1]) for v in top) < 0.0101
    straight = GEOMETRY.run({"op": "sweep", "path": [[0, 0, 0], [0, 0, 1]], "radius": 0.1})
    assert straight.signed_volume() == pytest.approx(math.pi * 0.01, rel=0.02)
    sweep = GEOMETRY.get("sweep")
    doubled = sweep.with_defaults({"path": [[0, 0, 0], [0, 0, 0], [1, 0, 0]], "radius": 0.1})
    assert sweep.check(doubled)


def test_scatter_keeps_its_spacing_and_checks_its_item() -> None:
    spec = {
        "op": "scatter",
        "item": {"op": "box", "size": [0.02, 0.02, 0.02]},
        "area": [0.5, 0.5],
        "count": 40,
        "spacing": 0.06,
        "scale_jitter": 0.0,
    }
    mesh = GEOMETRY.run(spec)
    assert mesh.is_closed_manifold()
    centres = [
        tuple(sum(mesh.vertices[k + i][a] for i in range(8)) / 8 for a in range(2))
        for k in range(0, len(mesh.vertices), 8)
    ]
    assert 10 < len(centres) <= 40
    assert (
        min(math.dist(a, b) for i, a in enumerate(centres) for b in centres[i + 1 :]) >= 0.06 - 1e-9
    )
    assert GEOMETRY.run({**spec, "seed": 1}).vertices != mesh.vertices
    scatter = GEOMETRY.get("scatter")
    bad = scatter.with_defaults(
        {
            **{k: v for k, v in spec.items() if k != "op"},
            "item": {"op": "box", "size": [1, 1, 1], "bevel": 0.9},
        }
    )
    assert "box" in scatter.check(bad)[0][1]


def test_things_rest_on_terrain_and_rocks_sit_flat() -> None:
    from jingyu.scene import validate_scene

    scene = {
        "schema": "jingyu.scene.v1",
        "id": "hill",
        "objects": [
            {
                "id": "land",
                "geometry": {"op": "terrain", "size": [20, 20], "relief": 3, "resolution": 48},
            },
            {
                "id": "stone",
                "geometry": {"op": "rock", "seed": 3},
                "location": [2, -1, 0],
                "rest_on": "land",
            },
        ],
        "cameras": [{"id": "c", "location": [0, -30, 10], "look_at": [0, 0, 1]}],
        "render": {"camera": "c"},
    }
    result = validate_scene(scene)
    assert result.valid, result.issues
    stone = result.placement.meshes["stone"]
    assert stone.bounds()[0][2] > 0.2  # it sits on the land, above its base
    rock = GEOMETRY.run({"op": "rock", "sink": 0.3})
    flat = [v for v in rock.vertices if v[2] < 1e-12]
    assert len(flat) > 20  # a flat face where it is cut into the ground


def test_trees_have_trunk_and_foliage_with_their_own_materials() -> None:
    for kind in ("broadleaf", "conifer"):
        mesh = GEOMETRY.run({"op": "tree", "kind": kind, "height": 5})
        assert mesh.parts == ("trunk", "foliage")
        assert {mesh.parts[p] for p in mesh.face_parts} == {"trunk", "foliage"}
        assert mesh.bounds()[0][2] == pytest.approx(0.0)
        assert mesh.bounds()[1][2] == pytest.approx(5.0, rel=0.25)
    tree = GEOMETRY.get("tree")
    assert tree.part_defaults["trunk"]["family"] == "wood"


def test_grass_density_sets_the_number_of_blades() -> None:
    sparse = GEOMETRY.run({"op": "grass", "size": [1, 1], "density": 100})
    dense = GEOMETRY.run({"op": "grass", "size": [1, 1], "density": 400})
    assert len(sparse.faces) == 4 * 100 and len(dense.faces) == 4 * 400
    assert dense.is_closed_manifold()
    patchy = GEOMETRY.run({"op": "grass", "size": [1, 1], "density": 400, "patchiness": 0.8})
    assert len(patchy.faces) < len(dense.faces)
