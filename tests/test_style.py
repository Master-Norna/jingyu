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
        apply_style(render, None, {"preset": "unsupported_style"})


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
    with_zero_lines = apply_style(source, None, {"preset": "gouache", "paper": 0, "line_width": 0})
    assert gau.tobytes() == with_zero_lines.tobytes()
    assert gau.getpixel((VASE_BOX[0] + 5, VASE_BOX[1] + 5))[2] > 100
    assert gau.tobytes() != apply_style(source, ids, {"preset": "oil", "paper": 0}).tobytes()


@pytest.mark.parametrize("preset", ["pastel", "etching", "mosaic"])
def test_new_artistic_styles_validate_as_scenes(preset: str) -> None:
    from jingyu.scene import minimal_scene, validate_scene

    scene = minimal_scene()
    scene["render"]["style"] = {"preset": preset, "seed": 23}
    assert validate_scene(scene).require_valid()["render"]["style"]["preset"] == preset


def test_new_styles_use_three_distinct_drawing_mechanisms() -> None:
    scene = _render()
    results = [
        apply_style(scene, synthetic_mask(), {"preset": preset, "paper": 0})
        for preset in ("pastel", "etching", "mosaic")
    ]
    assert len({im.tobytes() for im in results}) == 3
    pastel, etching, mosaic = results
    assert any(max(pixel) - min(pixel) > 35 for pixel in pastel.getdata())
    assert all(max(pixel) - min(pixel) < 28 for pixel in etching.getdata())
    assert any(max(pixel) - min(pixel) > 35 for pixel in mosaic.getdata())


def test_etching_uses_shadow_crosshatching_with_exposed_paper() -> None:
    source = Image.new("RGB", (100, 60), (248, 248, 248))
    ImageDraw.Draw(source).rectangle((50, 0, 99, 59), fill=(17, 17, 17))
    result = apply_style(source, None, {"preset": "etching", "paper": 0, "line_width": 0})
    lights = [result.getpixel((x, y))[0] for y in range(8, 52) for x in range(8, 42)]
    shadows = [result.getpixel((x, y))[0] for y in range(8, 52) for x in range(58, 92)]
    assert sum(shadows) < sum(lights) * 0.8
    assert any(v > 210 for v in shadows)  # open spaces between ink strokes


def test_mosaic_has_grout_and_preserves_object_colour() -> None:
    source = Image.new("RGB", (120, 84), (30, 110, 190))
    first = apply_style(source, None, {"preset": "mosaic", "paper": 0, "seed": 17})
    second = apply_style(source, None, {"preset": "mosaic", "paper": 0, "seed": 18})
    assert first.tobytes() != second.tobytes()
    # Every tile has an opaque coloured centre surrounded by contrasting mortar.
    grout = first.getpixel((0, 0))
    tile = first.getpixel((3, 3))
    assert sum(tile) > sum(grout)
    assert tile[2] > tile[1] > tile[0]
    assert first.getpixel((6, 3)) == grout


def test_pastel_is_coloured_soft_and_seeded() -> None:
    source = Image.new("RGB", (96, 64), (55, 100, 190))
    settings = {"preset": "pastel", "paper": 1}
    first = apply_style(source, None, {**settings, "seed": 7})
    repeat = apply_style(source, None, {**settings, "seed": 7})
    other = apply_style(source, None, {**settings, "seed": 8})
    assert first.tobytes() == repeat.tobytes()
    assert first.tobytes() != other.tobytes()
    colours = list(first.getdata())
    assert any(pixel[2] > pixel[1] > pixel[0] for pixel in colours)
    assert len(set(colours)) > 20


@pytest.mark.parametrize("preset", ["pastel", "etching", "mosaic"])
def test_new_styles_keep_legacy_alpha_and_strength(preset: str) -> None:
    source = _render().convert("RGBA")
    alpha = Image.new("L", source.size, 0)
    ImageDraw.Draw(alpha).rectangle((12, 9, 51, 38), fill=128)
    source.putalpha(alpha)
    full = apply_style(source, synthetic_mask(), {"preset": preset, "strength": 1})
    half = apply_style(source, synthetic_mask(), {"preset": preset, "strength": 0.5})
    assert full.getchannel("A").tobytes() == alpha.tobytes()
    assert half.getchannel("A").tobytes() == alpha.tobytes()
    assert full.tobytes() != half.tobytes()


@pytest.mark.parametrize("preset", ["crayon", "screenprint", "papercut"])
def test_more_style_families_are_valid_scenes(preset: str) -> None:
    from jingyu.scene import minimal_scene, validate_scene

    scene = minimal_scene()
    scene["render"]["style"] = {"preset": preset, "seed": 9}
    assert validate_scene(scene).require_valid()["render"]["style"]["preset"] == preset


def test_crayon_has_colour_marks_and_exposed_paper() -> None:
    source = Image.new("RGB", (100, 70), (27, 94, 185))
    spec = {"preset": "crayon", "paper": 0, "seed": 4}
    drawn = apply_style(source, None, spec)
    repeat = apply_style(source, None, spec)
    other = apply_style(source, None, {**spec, "seed": 5})
    assert drawn.tobytes() == repeat.tobytes()
    assert drawn.tobytes() != other.tobytes()
    pixels = list(drawn.getdata())
    assert any(v[2] > v[1] > v[0] for v in pixels)
    assert any(v[0] > 220 and v[1] > 220 for v in pixels)


def test_screenprint_uses_limited_inks_halftone_and_seeded_registration() -> None:
    source = Image.new("RGB", (100, 70), (245, 212, 180))
    ImageDraw.Draw(source).rectangle((48, 0, 99, 69), fill=(30, 45, 82))
    spec = {"preset": "screenprint", "paper": 0, "seed": 7}
    printed = apply_style(source, None, spec)
    again = apply_style(source, None, spec)
    shifted = apply_style(source, None, {**spec, "seed": 8})
    assert printed.tobytes() == again.tobytes()
    assert printed.tobytes() != shifted.tobytes()
    assert len(set(printed.getdata())) < 30
    dark = [printed.getpixel((x, y)) for y in range(10, 60) for x in range(58, 95)]
    assert len(set(dark)) >= 2  # a distinct halftone ink plate over the flat colour


def test_papercut_uses_colour_regions_and_drop_shadows() -> None:
    source = Image.new("RGB", (108, 72), (223, 185, 110))
    ImageDraw.Draw(source).rectangle((26, 16, 81, 54), fill=(43, 89, 162))
    spec = {"preset": "papercut", "paper": 0, "seed": 12}
    collage = apply_style(source, None, spec)
    assert collage.tobytes() == apply_style(source, None, spec).tobytes()
    assert collage.tobytes() != source.tobytes()
    center = collage.getpixel((50, 35))
    outer = collage.getpixel((8, 8))
    assert center[2] > center[0]
    assert outer[0] > outer[2]
    assert len(set(collage.getdata())) >= 3  # paper, layer colours, cast shadows
    assert len(set(collage.getdata())) < 30


def test_new_styles_are_distinct_and_keep_legacy_alpha_strength() -> None:
    source = _render().convert("RGBA")
    alpha = Image.new("L", source.size, 0)
    ImageDraw.Draw(alpha).rectangle((12, 9, 51, 38), fill=128)
    source.putalpha(alpha)
    results = []
    for preset in ("crayon", "screenprint", "papercut"):
        original = {"preset": preset, "strength": 1}
        full = apply_style(source, synthetic_mask(), original)
        half = apply_style(source, synthetic_mask(), {**original, "strength": 0.5})
        assert full.getchannel("A").tobytes() == alpha.tobytes()
        assert half.getchannel("A").tobytes() == alpha.tobytes()
        assert full.tobytes() != half.tobytes()
        results.append(full.tobytes())
    assert len(set(results)) == 3


@pytest.mark.parametrize("preset", ["sumi", "stainedglass", "cyanotype"])
def test_ink_glass_blueprint_styles_are_valid_scene_presets(preset: str) -> None:
    from jingyu.scene import minimal_scene, validate_scene

    scene = minimal_scene()
    scene["render"]["style"] = {"preset": preset, "seed": 21}
    assert validate_scene(scene).require_valid()["render"]["style"]["preset"] == preset


def test_sumi_has_monochrome_washes_and_darker_brushwork() -> None:
    source = Image.new("RGB", (120, 72), (240, 180, 110))
    ImageDraw.Draw(source).rectangle((60, 0, 119, 71), fill=(24, 39, 57))
    base = {"preset": "sumi", "line_width": 0, "paper": 0, "seed": 17}
    first = apply_style(source, None, base)
    assert first.tobytes() == apply_style(source, None, base).tobytes()
    assert first.tobytes() != apply_style(source, None, {**base, "seed": 18}).tobytes()
    assert all(max(pixel) - min(pixel) < 16 for pixel in first.getdata())
    light = [first.getpixel((x, y))[0] for y in range(9, 61) for x in range(8, 50)]
    dark = [first.getpixel((x, y))[0] for y in range(9, 61) for x in range(70, 112)]
    assert sum(dark) < sum(light) * 0.60
    assert len(set(dark)) > 5  # layered wash tones, not one binary black area


def test_stainedglass_has_dark_leading_and_coloured_panes() -> None:
    source = Image.new("RGB", (100, 68), (26, 80, 194))
    ImageDraw.Draw(source).rectangle((50, 0, 99, 67), fill=(214, 75, 34))
    glass = apply_style(source, None, {"preset": "stainedglass", "paper": 0})
    left = glass.getpixel((20, 30))
    right = glass.getpixel((80, 30))
    seam = glass.getpixel((50, 30))
    assert left[2] > left[0]
    assert right[0] > right[2]
    assert sum(seam) < min(sum(left), sum(right)) * 0.6
    assert glass.getpixel((22, 30)) == left


def test_cyanotype_is_blue_monochrome_and_follows_luminance() -> None:
    source = Image.new("RGB", (104, 68), (243, 240, 240))
    ImageDraw.Draw(source).rectangle((52, 0, 103, 67), fill=(17, 27, 34))
    blue = apply_style(source, None, {"preset": "cyanotype", "paper": 0, "line_width": 0})
    dark = blue.getpixel((80, 34))
    light = blue.getpixel((23, 34))
    assert dark[2] > dark[0] * 2
    assert sum(dark) < sum(light) * 0.60
    assert light[0] > 200
    assert len(set(blue.getdata())) > 3  # softened transitions, not flat posterization


@pytest.mark.parametrize("preset", ["sumi", "stainedglass", "cyanotype"])
def test_new_media_keep_alpha_strength_seed_and_existing_render(preset: str) -> None:
    image = _render().convert("RGBA")
    original = image.tobytes()
    alpha = Image.new("L", image.size, 0)
    ImageDraw.Draw(alpha).rectangle((12, 9, 51, 38), fill=128)
    image.putalpha(alpha)
    opts = {"preset": preset, "paper": 1, "seed": 7}
    first = apply_style(image, synthetic_mask(), opts)
    again = apply_style(image, synthetic_mask(), opts)
    second_seed = apply_style(image, synthetic_mask(), {**opts, "seed": 8})
    half = apply_style(image, synthetic_mask(), {**opts, "strength": 0.5})
    assert first.tobytes() == again.tobytes()
    assert first.tobytes() != second_seed.tobytes()
    assert first.tobytes() != half.tobytes()
    assert first.getchannel("A").tobytes() == alpha.tobytes()
    assert half.getchannel("A").tobytes() == alpha.tobytes()
    assert image.convert("RGB").tobytes() == _render().tobytes()
    assert original != image.tobytes()


@pytest.mark.parametrize("preset", ["frottage", "embroidery", "glaze"])
def test_material_media_validate_with_scene_schema(preset: str) -> None:
    from jingyu.scene import minimal_scene, validate_scene

    scene = minimal_scene()
    scene["render"]["style"] = {"preset": preset, "seed": 13}
    assert validate_scene(scene).require_valid()["render"]["style"]["preset"] == preset


def test_frottage_has_relief_and_more_ink_on_dark_surfaces() -> None:
    source = Image.new("RGB", (120, 70), (240, 240, 240))
    ImageDraw.Draw(source).rectangle((60, 0, 119, 69), fill=(25, 25, 25))
    spec = {"preset": "frottage", "paper": 0, "seed": 13}
    first = apply_style(source, None, spec)
    second = apply_style(source, None, {**spec, "seed": 14})
    assert first.tobytes() != second.tobytes()
    assert all(max(pixel) - min(pixel) < 25 for pixel in first.getdata())
    left = sum(first.getpixel((x, y))[0] for y in range(10, 60) for x in range(10, 50))
    right = sum(first.getpixel((x, y))[0] for y in range(10, 60) for x in range(70, 110))
    assert right < left * 0.70
    dark_values = {first.getpixel((x, y)) for y in range(10, 60) for x in range(70, 110)}
    assert len(dark_values) >= 3  # rubbing plus separate directional wax/graphite marks


def test_embroidery_preserves_colour_regions_with_visible_stitches_and_cloth() -> None:
    source = Image.new("RGB", (100, 70), (35, 90, 170))
    ImageDraw.Draw(source).rectangle((50, 0, 99, 69), fill=(195, 61, 30))
    first = apply_style(source, None, {"preset": "embroidery", "paper": 0})
    left = first.getpixel((15, 15))
    right = first.getpixel((65, 15))
    assert left[2] > left[0] or first.getpixel((12, 12))[2] > 100
    assert right[0] > right[2] or first.getpixel((62, 12))[0] > 100
    assert len(set(first.getdata())) >= 5  # dark thread, highlight and cloth
    assert any(pixel == (231, 223, 204) for pixel in first.getdata())


def test_glaze_has_colour_pools_and_seeded_reflections() -> None:
    source = Image.new("RGB", (120, 72), (35, 100, 178))
    ImageDraw.Draw(source).rectangle((55, 10, 110, 63), fill=(184, 72, 54))
    base = {"preset": "glaze", "paper": 0, "seed": 11}
    first = apply_style(source, None, base)
    repeat = apply_style(source, None, base)
    second = apply_style(source, None, {**base, "seed": 12})
    assert first.tobytes() == repeat.tobytes()
    assert first.tobytes() != second.tobytes()
    assert first.getpixel((20, 30))[2] > first.getpixel((20, 30))[0]
    assert first.getpixel((77, 40))[0] > first.getpixel((77, 40))[2]
    assert len(set(first.getdata())) > 10


@pytest.mark.parametrize("preset", ["frottage", "embroidery", "glaze"])
def test_material_media_preserve_alpha_strength_and_source(preset: str) -> None:
    source = _render().convert("RGBA")
    alpha = Image.new("L", source.size, 0)
    ImageDraw.Draw(alpha).rectangle((12, 9, 51, 38), fill=128)
    source.putalpha(alpha)
    before = source.tobytes()
    opts = {"preset": preset, "seed": 31, "paper": 0.75}
    full = apply_style(source, synthetic_mask(), opts)
    repeated = apply_style(source, synthetic_mask(), opts)
    half = apply_style(source, synthetic_mask(), {**opts, "strength": 0.5})
    assert full.tobytes() == repeated.tobytes()
    assert full.tobytes() != half.tobytes()
    assert full.getchannel("A").tobytes() == alpha.tobytes()
    assert half.getchannel("A").tobytes() == alpha.tobytes()
    assert source.tobytes() == before
