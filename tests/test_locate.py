from __future__ import annotations

from typing import Any

import pytest
from helpers import (
    BACKGROUND_PIXELS,
    CRATE_PIXELS,
    FLOOR_PIXELS,
    HEIGHT,
    VASE_PIXELS,
    WIDTH,
    CandidateFactory,
)

from jingyu.candidate import Candidate
from jingyu.errors import JingyuError
from jingyu.idmask import ENCODING, ID_MAP_SCHEMA, encode_index
from jingyu.locate import Box, IdMask

VASE_HIT = {
    "object": "vase",
    "pointer": "/objects/1",
    "material": "glaze",
    "material_pointer": "/materials/0",
}
CRATE_HIT = {
    "object": "crate",
    "pointer": "/objects/2",
    "material": None,
    "material_pointer": None,
}


@pytest.fixture
def mask(candidate: Candidate) -> IdMask:
    return IdMask.load(candidate)


def _code(call: Any, *args: Any, **kwargs: Any) -> str:
    with pytest.raises(JingyuError) as info:
        call(*args, **kwargs)
    return info.value.code


def test_mask_has_the_image_size(mask: IdMask) -> None:
    assert (mask.width, mask.height) == (WIDTH, HEIGHT)


def test_locate_point_hits_object_pointer_and_material(mask: IdMask) -> None:
    result = mask.locate_point({"x": 40, "y": 20})
    assert result["hit"] == VASE_HIT
    assert result["background"] is False
    assert result["point"] == {"x": 40, "y": 20, "u": 40.5 / WIDTH, "v": 20.5 / HEIGHT}
    # The whole 7 x 7 neighbourhood lies inside the vase.
    assert [(r["object"], r["pixels"]) for r in result["neighborhood"]] == [("vase", 49)]


def test_locate_point_reports_objects_without_material(mask: IdMask) -> None:
    assert mask.locate_point({"x": 5, "y": 25})["hit"] == CRATE_HIT


def test_locate_point_accepts_normalised_coordinates(mask: IdMask) -> None:
    result = mask.locate_point({"u": 0.5, "v": 0.4})
    assert (result["point"]["x"], result["point"]["y"]) == (40, 24)
    assert result["hit"] == VASE_HIT
    # u = v = 1 is the last pixel, not one past it.
    corner = mask.locate_point({"u": 1.0, "v": 1.0})
    assert (corner["point"]["x"], corner["point"]["y"]) == (WIDTH - 1, HEIGHT - 1)
    assert corner["hit"]["object"] == "floor"


def test_locate_point_on_background(mask: IdMask) -> None:
    result = mask.locate_point({"x": 70, "y": 5})
    assert result["hit"] is None
    assert result["background"] is True
    assert result["neighborhood"] == []


def test_radius_search_finds_nearby_objects(mask: IdMask) -> None:
    # Two pixels left of the vase: the 7 x 7 box reaches two of its columns.
    result = mask.locate_point({"x": 28, "y": 20}, radius=3)
    assert result["hit"] is None
    assert result["neighborhood"] == [{**VASE_HIT, "pixels": 14, "fraction_of_region": 14 / 49}]
    assert mask.locate_point({"x": 28, "y": 20}, radius=1)["neighborhood"] == []
    assert mask.locate_point({"x": 28, "y": 20}, radius=0)["neighborhood"] == []


def test_radius_search_is_clipped_at_the_frame(mask: IdMask) -> None:
    result = mask.locate_point({"x": 0, "y": 59}, radius=2)
    # Only the 3 x 3 corner inside the frame counts.
    assert result["neighborhood"] == [
        {
            "object": "floor",
            "pointer": "/objects/0",
            "material": "floor",
            "material_pointer": "/materials/1",
            "pixels": 9,
            "fraction_of_region": 1.0,
        }
    ]


@pytest.mark.parametrize(
    "point",
    [
        {"x": WIDTH, "y": 0},
        {"x": 0, "y": HEIGHT},
        {"x": -1, "y": 10},
        {"u": -0.1, "v": 0.5},
        {"u": 0.5, "v": 1.5},
        # Just outside [0, 1] is outside too, not the nearest edge pixel.
        {"u": -0.001, "v": 0.5},
        {"u": 0.5, "v": 1.001},
    ],
)
def test_points_outside_the_image_are_rejected(mask: IdMask, point: dict[str, float]) -> None:
    assert _code(mask.locate_point, point) == "locate.out_of_bounds"


def test_points_need_a_coordinate_pair(mask: IdMask) -> None:
    assert _code(mask.locate_point, {"x": 1}) == "tool.invalid_arguments"
    assert _code(mask.locate_point, {"x": 1, "v": 0.5}) == "tool.invalid_arguments"


def test_region_over_one_object(mask: IdMask) -> None:
    result = mask.locate_region({"x0": 30, "y0": 15, "x1": 50, "y1": 50})
    assert result["region"] == {"x0": 30, "y0": 15, "x1": 50, "y1": 50}
    assert result["pixels"] == VASE_PIXELS
    assert result["background_fraction"] == 0.0
    assert result["objects"] == [
        {**VASE_HIT, "pixels": VASE_PIXELS, "fraction_of_region": 1.0, "fraction_of_object": 1.0}
    ]


def test_region_statistics_split_objects_and_background(mask: IdMask) -> None:
    result = mask.locate_region({"x0": 0, "y0": 35, "x1": 40, "y1": 45})
    assert result["pixels"] == 400
    assert result["background_fraction"] == 150 / 400
    rows = {r["object"]: r for r in result["objects"]}
    assert [r["object"] for r in result["objects"]] == ["floor", "vase"]
    assert rows["floor"]["pixels"] == 150
    assert rows["floor"]["fraction_of_object"] == 150 / FLOOR_PIXELS
    assert rows["vase"]["pixels"] == 100
    assert rows["vase"]["fraction_of_region"] == 100 / 400


def test_full_frame_region_by_uv(mask: IdMask) -> None:
    result = mask.locate_region({"u0": 0, "v0": 0, "u1": 1, "v1": 1})
    assert result["pixels"] == WIDTH * HEIGHT
    assert result["background_fraction"] == BACKGROUND_PIXELS / (WIDTH * HEIGHT)
    assert [(r["object"], r["pixels"]) for r in result["objects"]] == [
        ("floor", FLOOR_PIXELS),
        ("vase", VASE_PIXELS),
        ("crate", CRATE_PIXELS),
    ]


@pytest.mark.parametrize(
    "region",
    [
        {"x0": 10, "y0": 10, "x1": 10, "y1": 20},
        {"x0": 20, "y0": 10, "x1": 10, "y1": 20},
        {"x0": 0, "y0": 0, "x1": WIDTH + 1, "y1": 10},
        {"x0": -5, "y0": 0, "x1": 10, "y1": 10},
        {"u0": 0.5, "v0": 0.5, "u1": 0.5, "v1": 0.9},
        {"u0": 0.0, "v0": 0.0, "u1": 1.2, "v1": 1.0},
        {"u0": -0.001, "v0": 0.0, "u1": 0.5, "v1": 0.5},
    ],
)
def test_empty_or_outside_regions_are_rejected(mask: IdMask, region: dict[str, float]) -> None:
    assert _code(mask.locate_region, region) == "locate.out_of_bounds"


def test_regions_need_four_coordinates(mask: IdMask) -> None:
    assert _code(mask.locate_region, {"x0": 0, "y0": 0, "x1": 5}) == "tool.invalid_arguments"


def test_layout_fractions_sum_to_one(mask: IdMask) -> None:
    layout = mask.layout()
    assert (layout["width"], layout["height"]) == (WIDTH, HEIGHT)
    assert layout["unassigned_pixels"] == 0
    total = layout["background_fraction"] + sum(o["area_fraction"] for o in layout["objects"])
    assert total == pytest.approx(1.0)
    assert [o["object"] for o in layout["objects"]] == ["floor", "vase", "crate"]
    assert [o["pixels"] for o in layout["objects"]] == [FLOOR_PIXELS, VASE_PIXELS, CRATE_PIXELS]


def test_layout_boxes_centroids_and_frame_edges(mask: IdMask) -> None:
    objects = {o["object"]: o for o in mask.layout()["objects"]}
    vase = objects["vase"]
    assert vase["bbox"] == {"x0": 30, "y0": 15, "x1": 50, "y1": 50}
    assert vase["bbox_uv"] == {"u0": 30 / 80, "v0": 15 / 60, "u1": 50 / 80, "v1": 50 / 60}
    assert vase["centroid_uv"]["u"] == pytest.approx(0.5)
    assert vase["centroid_uv"]["v"] == pytest.approx((32 + 0.5) / 60)
    assert vase["material_pointer"] == "/materials/0"
    assert vase["touches_frame_edge"] is False
    assert objects["floor"]["touches_frame_edge"] is True
    assert objects["crate"]["touches_frame_edge"] is True


def test_layout_lists_visible_objects_that_are_out_of_frame(mask: IdMask) -> None:
    # "far" is visible but drew no pixels; "ghost" is hidden and is not expected at all.
    assert mask.layout()["not_in_frame"] == ["far"]


def test_corrupt_and_unmapped_pixels(make_candidate: CandidateFactory) -> None:
    candidate = make_candidate(
        extra_pixels={(70, 5): (0, 0, 1, 128), (71, 5): (*encode_index(99), 255)}
    )
    mask = IdMask.load(candidate)
    layout = mask.layout()
    assert layout["unassigned_pixels"] == 2
    total = layout["background_fraction"] + sum(o["area_fraction"] for o in layout["objects"])
    assert total == pytest.approx(1 - 2 / (WIDTH * HEIGHT))
    corrupt = mask.locate_point({"x": 70, "y": 5})
    assert (corrupt["hit"], corrupt["background"]) == (None, False)
    assert _code(mask.locate_point, {"x": 71, "y": 5}) == "locate.ambiguous_pixels"
    # Region statistics skip pixels they cannot attribute.
    region = mask.locate_region({"x0": 68, "y0": 3, "x1": 74, "y1": 8})
    assert region["objects"] == []
    assert region["background_fraction"] == 28 / 30


def test_candidates_without_a_mask_cannot_be_located(make_candidate: CandidateFactory) -> None:
    candidate = make_candidate(with_mask=False)
    assert not candidate.has_id_mask
    assert _code(IdMask.load, candidate) == "candidate.missing_pass"


def test_an_id_map_with_an_unknown_schema_is_rejected(make_candidate: CandidateFactory) -> None:
    candidate = make_candidate(
        id_map={"schema": "jingyu.id-map.v0", "encoding": ENCODING, "objects": {}}
    )
    assert ID_MAP_SCHEMA != "jingyu.id-map.v0"
    assert _code(IdMask.load, candidate) == "candidate.integrity_mismatch"


def test_box_area_is_never_negative() -> None:
    assert Box(0, 0, 4, 5).area == 20
    assert Box(5, 5, 4, 9).area == 0
