"""Style presets are deterministic image operations that follow the scene's edges."""

from __future__ import annotations

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
