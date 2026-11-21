"""Style presets are deterministic image operations that follow the scene's edges."""

from __future__ import annotations

import random
from itertools import pairwise

import pytest
from helpers import CRATE_BOX, VASE_BOX, synthetic_mask
from PIL import Image, ImageDraw

from jingyu.style import PRESETS, apply_style, flatten, kuwahara, outlines


def _render() -> Image.Image:
    image = Image.new("RGB", (80, 60), (200, 190, 170))
    draw = ImageDraw.Draw(image)
    draw.rectangle((VASE_BOX[0], VASE_BOX[1], VASE_BOX[2] - 1, VASE_BOX[3] - 1), fill=(60, 90, 140))
    draw.rectangle(
        (CRATE_BOX[0], CRATE_BOX[1], CRATE_BOX[2] - 1, CRATE_BOX[3] - 1), fill=(150, 90, 40)
    )
    return image


@pytest.mark.parametrize("preset", [p for p in PRESETS if p != "none"])
def test_every_preset_changes_the_image_the_same_way_twice(preset: str) -> None:
    render, mask = _render(), synthetic_mask()
    first = apply_style(render, mask, {"preset": preset})
    second = apply_style(render, mask, {"preset": preset})
    assert first.size == render.size
    assert first.tobytes() == second.tobytes()
    assert first.tobytes() != render.tobytes()
    half = apply_style(render, mask, {"preset": preset, "strength": 0.5})
    assert half.tobytes() not in (first.tobytes(), render.tobytes())


def test_none_leaves_the_render_alone() -> None:
    render = _render()
    assert apply_style(render, None, {"preset": "none"}) is render
    with pytest.raises(ValueError, match="preset"):
        apply_style(render, None, {"preset": "pastel"})


def test_outlines_follow_the_id_mask() -> None:
    lines = outlines(synthetic_mask(), (80, 60), 1.0)
    x0, y0, x1, y1 = VASE_BOX
    assert lines.getpixel((x0, (y0 + y1) // 2)) > 128  # the vase's left edge
    assert lines.getpixel(((x0 + x1) // 2, (y0 + y1) // 2)) < 32  # inside the vase
    assert outlines(None, (80, 60), 1.0).getbbox() is None


def test_outlines_do_not_wrap_across_opposite_image_borders() -> None:
    mask = Image.new("RGBA", (24, 16), (0, 0, 0, 0))
    draw = ImageDraw.Draw(mask)
    draw.rectangle((23, 5, 23, 11), fill=(0, 0, 1, 255))
    lines = outlines(mask, mask.size, 1.0)
    assert lines.getpixel((23, 8)) > 128
    assert lines.getpixel((0, 8)) == 0

    mask = Image.new("RGBA", (24, 16), (0, 0, 0, 0))
    draw = ImageDraw.Draw(mask)
    draw.rectangle((5, 15, 12, 15), fill=(0, 0, 1, 255))
    lines = outlines(mask, mask.size, 1.0)
    assert lines.getpixel((8, 15)) > 128
    assert lines.getpixel((8, 0)) == 0


@pytest.mark.parametrize("preset", [p for p in PRESETS if p != "none"])
@pytest.mark.parametrize("strength", [0.5, 1.0])
def test_style_preserves_transparent_background(preset: str, strength: float) -> None:
    source = _render().convert("RGBA")
    source.putalpha(Image.new("L", source.size, 0))
    source.putpixel((10, 10), (30, 80, 150, 128))
    source.putpixel((30, 20), (90, 40, 20, 255))
    painted = apply_style(source, synthetic_mask(), {"preset": preset, "strength": strength})
    assert painted.mode == "RGBA"
    assert painted.getchannel("A").tobytes() == source.getchannel("A").tobytes()


@pytest.mark.parametrize("preset", ["ink", "cel"])
def test_zero_line_width_removes_all_outlines(preset: str) -> None:
    render, mask = _render(), synthetic_mask()
    assert outlines(mask, render.size, 0).getbbox() is None
    without_lines = apply_style(render, mask, {"preset": preset, "line_width": 0})
    without_mask = apply_style(render, None, {"preset": preset, "line_width": 0})
    with_lines = apply_style(render, mask, {"preset": preset, "line_width": 40})
    assert without_lines.tobytes() == without_mask.tobytes()
    assert without_lines.tobytes() != with_lines.tobytes()


def test_kuwahara_does_not_wrap_pixels_from_right_border_to_left() -> None:
    rng = random.Random(0)
    image = Image.new("L", (48, 36))
    image.putdata([rng.randrange(256) for _ in range(48 * 36)])
    changed = image.copy()
    for y in range(36):
        for x in range(43, 48):
            changed.putpixel((x, y), 255 - image.getpixel((x, y)))
    original = kuwahara(image.convert("RGB"), 5)
    altered = kuwahara(changed.convert("RGB"), 5)
    # The far right is outside the left edge's filter neighbourhood.
    for y in range(5, 31):
        for x in range(2):
            assert original.getpixel((x, y)) == altered.getpixel((x, y))


def test_ink_draws_dark_lines_on_edges() -> None:
    styled = apply_style(_render(), synthetic_mask(), {"preset": "ink", "paper": 0.0})
    x0, y0, _, y1 = VASE_BOX
    edge = sum(styled.getpixel((x0, (y0 + y1) // 2)))
    inside = sum(styled.getpixel((x0 + 8, (y0 + y1) // 2)))
    assert edge < inside


def test_flatten_quantises_value_and_keeps_hue() -> None:
    ramp = Image.new("RGB", (256, 1))
    ramp.putdata([(v, v // 2, 0) for v in range(256)])
    bands = flatten(ramp, 3, 0.0)
    values = {max(p) for p in bands.get_flattened_data()}
    assert len(values) <= 3
    assert all(p[2] == 0 for p in bands.get_flattened_data())


def test_kuwahara_keeps_edges_and_smooths_texture() -> None:
    noisy = Image.new("RGB", (40, 20))
    noisy.putdata(
        [
            ((x * 37 + y * 11) % 60 + (0 if x < 20 else 150),) * 3
            for y in range(20)
            for x in range(40)
        ]
    )
    smooth = kuwahara(noisy, 3)
    left = [smooth.getpixel((x, 10))[0] for x in range(4, 16)]
    assert max(left) - min(left) < 40  # texture flattened
    assert smooth.getpixel((25, 10))[0] - smooth.getpixel((14, 10))[0] > 100  # the edge survives


@pytest.mark.parametrize("preset", ["charcoal", "woodcut"])
def test_new_styles_are_neutral_and_preserve_light_dark_structure(preset: str) -> None:
    image = Image.new("RGB", (96, 48), (245, 245, 245))
    draw = ImageDraw.Draw(image)
    draw.rectangle((48, 0, 95, 47), fill=(20, 20, 20))
    painted = apply_style(image, None, {"preset": preset, "paper": 0, "line_width": 0})
    bright = painted.getpixel((24, 20))
    dark = painted.getpixel((72, 20))
    assert sum(dark) < sum(bright) * 0.7
    assert all(abs(a - b) < 25 for pixel in (bright, dark) for a, b in pairwise(pixel))


@pytest.mark.parametrize("preset", ["charcoal", "woodcut"])
def test_new_styles_respect_zero_outline_width(preset: str) -> None:
    render = _render()
    with_mask = apply_style(render, synthetic_mask(), {"preset": preset, "line_width": 0})
    without_mask = apply_style(render, None, {"preset": preset, "line_width": 0})
    assert with_mask.tobytes() == without_mask.tobytes()


@pytest.mark.parametrize("preset", ["charcoal", "woodcut"])
def test_new_styles_seed_controls_repeatable_paper(preset: str) -> None:
    image = _render()
    options = {"preset": preset, "paper": 1.0}
    first = apply_style(image, None, {**options, "seed": 9})
    again = apply_style(image, None, {**options, "seed": 9})
    changed = apply_style(image, None, {**options, "seed": 10})
    assert first.tobytes() == again.tobytes()
    assert first.tobytes() != changed.tobytes()


def test_new_painting_presets_are_accepted_by_scene_validation() -> None:
    from jingyu.scene import minimal_scene, validate_scene

    for preset in ("charcoal", "woodcut"):
        scene = minimal_scene()
        scene["render"]["style"] = {"preset": preset}
        assert validate_scene(scene).require_valid()["render"]["style"]["preset"] == preset


@pytest.mark.parametrize("preset", ["pencil", "stippling", "gouache"])
def test_broader_painting_presets_validate_as_scenes(preset: str) -> None:
    from jingyu.scene import minimal_scene, validate_scene

    scene = minimal_scene()
    scene["render"]["style"] = {"preset": preset, "seed": 17}
    normalized = validate_scene(scene).require_valid()
    assert normalized["render"]["style"]["preset"] == preset


def test_new_style_families_are_visually_distinct() -> None:
    source = _render()
    images = [
        apply_style(source, synthetic_mask(), {"preset": preset, "paper": 0})
        for preset in ("pencil", "stippling", "gouache")
    ]
    assert len({image.tobytes() for image in images}) == 3
    assert all(image.tobytes() != source.tobytes() for image in images)
    pencil, stippling, gouache = images
    assert all(max(pixel) - min(pixel) < 26 for pixel in pencil.getdata())
    assert all(max(pixel) - min(pixel) < 26 for pixel in stippling.getdata())
    assert any(max(pixel) - min(pixel) > 35 for pixel in gouache.getdata())


def test_pencil_crosshatches_dark_regions_more_than_light_regions() -> None:
    image = Image.new("RGB", (100, 60), (245, 245, 245))
    ImageDraw.Draw(image).rectangle((50, 0, 99, 59), fill=(24, 24, 24))
    result = apply_style(image, None, {"preset": "pencil", "paper": 0, "line_width": 0})
    left = sum(sum(result.getpixel((x, y))) for y in range(8, 52) for x in range(8, 42))
    right = sum(sum(result.getpixel((x, y))) for y in range(8, 52) for x in range(58, 92))
    assert right < left * 0.75


def test_stippling_has_spatial_dots_and_seeded_locations() -> None:
    image = Image.new("RGB", (100, 60), (247, 247, 247))
    ImageDraw.Draw(image).rectangle((50, 0, 99, 59), fill=(8, 8, 8))
    options = {"preset": "stippling", "paper": 0}
    first = apply_style(image, None, {**options, "seed": 11})
    again = apply_style(image, None, {**options, "seed": 11})
    different = apply_style(image, None, {**options, "seed": 12})
    assert first.tobytes() == again.tobytes()
    assert first.tobytes() != different.tobytes()
    left = [first.getpixel((x, y))[0] for y in range(8, 52) for x in range(8, 42)]
    right = [first.getpixel((x, y))[0] for y in range(8, 52) for x in range(58, 92)]
    assert sum(v < 80 for v in right) > sum(v < 80 for v in left) + 50
    assert any(v > 225 for v in right)  # exposed paper between discrete dots


def test_gouache_keeps_colour_without_hard_cel_outlines() -> None:
    source, ids = _render(), synthetic_mask()
    gau = apply_style(source, ids, {"preset": "gouache", "paper": 0})
    with_zero_lines = apply_style(
        source, None, {"preset": "gouache", "paper": 0, "line_width": 0}
    )
    assert gau.tobytes() == with_zero_lines.tobytes()
    assert gau.getpixel((VASE_BOX[0] + 5, VASE_BOX[1] + 5))[2] > 100
    assert gau.tobytes() != apply_style(source, ids, {"preset": "oil", "paper": 0}).tobytes()
