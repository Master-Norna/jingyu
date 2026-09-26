from __future__ import annotations

import math
from collections.abc import Mapping
from itertools import pairwise
from typing import Any

import pytest

from jingyu.errors import JingyuError
from jingyu.generator import GeneratorDef
from jingyu.geometry import GEOMETRY, MeshData
from jingyu.geometry.curves import linspace, pchip
from jingyu.geometry.lathe import close_solid, revolve, shell
from jingyu.geometry.ops import vessel_profile

#: Flat shapes are single-sided surfaces, not solids.
FLAT_OPS = {"plane"}


def _defaults(definition: GeneratorDef[Any]) -> dict[str, Any]:
    return {k: s["default"] for k, s in definition.params.items() if "default" in s}


def _params(definition: GeneratorDef[Any], example: Mapping[str, Any]) -> dict[str, Any]:
    return {**_defaults(definition), **{k: v for k, v in example.items() if k != "op"}}


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
    assert mesh.bounds()[0][2] == pytest.approx(0.0, abs=1e-12)


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
    params = {"size": [4.0, 0.15, 2.6], "openings": openings}
    problems = GEOMETRY.get("wall").check(params)
    assert [p[0] for p in problems] == ["openings"]
    assert message in problems[0][1]
