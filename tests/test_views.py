from __future__ import annotations

import io

import pytest
from helpers import HEIGHT, WIDTH, CandidateFactory
from PIL import Image

from jingyu.candidate import Candidate
from jingyu.errors import JingyuError
from jingyu.views import GLANCE_SIZE, VIEWS, RenderedView, render_view


def _decode(view: RenderedView) -> Image.Image:
    image = Image.open(io.BytesIO(view.png))
    assert image.format == "PNG"
    image.load()
    return image


@pytest.fixture
def other(make_candidate: CandidateFactory, candidate: Candidate) -> Candidate:
    # Created after `candidate`, at twice its size.
    return make_candidate(width=2 * WIDTH, height=2 * HEIGHT)


EXPECTED_SIZES = {
    "full": (WIDTH, HEIGHT),
    "render": (WIDTH, HEIGHT),  # without a style, the image itself
    "glance": (WIDTH, HEIGHT),
    "flip": (WIDTH, HEIGHT),
    "squint": (WIDTH, HEIGHT),
    "grayscale": (WIDTH, HEIGHT),
    "values": (WIDTH, HEIGHT),
    "id_mask": (WIDTH, HEIGHT),
    "saturation": (WIDTH, HEIGHT),
    "light": (WIDTH, HEIGHT),
    # Both sides at the smaller height plus a 4 px gap.
    "compare": (WIDTH + 4 + WIDTH, HEIGHT),
}


def test_every_view_has_an_expected_size() -> None:
    assert set(EXPECTED_SIZES) == set(VIEWS)


@pytest.mark.parametrize("view", list(VIEWS))
def test_every_view_renders_a_png_of_the_expected_size(
    candidate: Candidate, other: Candidate, view: str
) -> None:
    rendered = render_view(candidate, view, other=other)
    assert (rendered.width, rendered.height) == EXPECTED_SIZES[view]
    assert _decode(rendered).size == EXPECTED_SIZES[view]
    assert rendered.meta["view"] == view


def test_large_images_are_fitted(make_candidate: CandidateFactory) -> None:
    big = make_candidate(width=400, height=300)
    glance = render_view(big, "glance")
    assert (glance.width, glance.height) == (GLANCE_SIZE, 144)
    full = render_view(big, "full", max_size=100)
    assert _decode(full).size == (100, 75)
    assert render_view(big, "full").width == 400


def test_full_view_is_the_render_itself(candidate: Candidate) -> None:
    with Image.open(candidate.image_path) as original:
        expected = original.convert("RGB").tobytes()
    assert _decode(render_view(candidate, "full")).convert("RGB").tobytes() == expected


def test_flip_mirrors_left_to_right(candidate: Candidate) -> None:
    with Image.open(candidate.image_path) as original:
        source = original.convert("RGB")
    flipped = _decode(render_view(candidate, "flip")).convert("RGB")
    for y in (0, HEIGHT // 2, HEIGHT - 1):
        assert flipped.getpixel((0, y)) == source.getpixel((WIDTH - 1, y))
        assert flipped.getpixel((WIDTH - 1, y)) == source.getpixel((0, y))


def test_grayscale_has_one_channel(candidate: Candidate) -> None:
    assert _decode(render_view(candidate, "grayscale")).mode == "L"


def test_squint_blurs_in_proportion_to_the_image(candidate: Candidate) -> None:
    rendered = render_view(candidate, "squint")
    assert rendered.meta["blur_radius_px"] == pytest.approx(0.02 * WIDTH)


@pytest.mark.parametrize("levels", [2, 5, 16])
def test_values_view_quantises_luminance(candidate: Candidate, levels: int) -> None:
    rendered = render_view(candidate, "values", levels=levels)
    histogram = _decode(rendered).convert("L").histogram()
    distinct = {value for value, count in enumerate(histogram) if count}
    assert len(distinct) <= levels
    assert {0, 255} <= distinct
    distribution = rendered.meta["value_distribution"]
    assert rendered.meta["levels"] == levels
    assert len(distribution) == levels
    assert sum(distribution) == pytest.approx(1.0, abs=1e-3)


@pytest.mark.parametrize("levels", [1, 17])
def test_values_view_rejects_bad_levels(candidate: Candidate, levels: int) -> None:
    with pytest.raises(JingyuError) as info:
        render_view(candidate, "values", levels=levels)
    assert info.value.code == "tool.invalid_arguments"


def test_id_mask_view_has_a_legend_of_drawn_objects(candidate: Candidate) -> None:
    rendered = render_view(candidate, "id_mask")
    legend = rendered.meta["legend"]
    assert [entry["object"] for entry in legend] == ["floor", "vase", "crate"]
    colours = {entry["color"] for entry in legend}
    assert len(colours) == 3
    image = _decode(rendered).convert("RGB")
    assert image.getpixel((70, 5)) == (40, 40, 40)  # background
    assert "#{:02x}{:02x}{:02x}".format(*image.getpixel((40, 20))) == legend[1]["color"]


def test_id_mask_view_needs_a_mask(make_candidate: CandidateFactory) -> None:
    with pytest.raises(JingyuError) as info:
        render_view(make_candidate(with_mask=False), "id_mask")
    assert info.value.code == "candidate.missing_pass"


def test_compare_names_both_sides(candidate: Candidate, other: Candidate) -> None:
    rendered = render_view(candidate, "compare", other=other)
    assert rendered.meta["left"] == candidate.id
    assert rendered.meta["right"] == other.id


def test_compare_needs_another_candidate(candidate: Candidate) -> None:
    with pytest.raises(JingyuError) as info:
        render_view(candidate, "compare")
    assert info.value.code == "tool.invalid_arguments"


@pytest.mark.parametrize("view", ["thumbnail", "", "FULL"])
def test_unknown_views_are_rejected(candidate: Candidate, view: str) -> None:
    with pytest.raises(JingyuError) as info:
        render_view(candidate, view)
    assert info.value.code == "view.unknown_view"
    assert info.value.hint and "glance" in info.value.hint
