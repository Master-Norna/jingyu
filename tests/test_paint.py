"""Painting a rendered candidate is local, repeatable and never changes its parent."""

from __future__ import annotations

import copy
import math

import pytest
from helpers import SYNTHETIC_ID_MAP_OBJECTS, CandidateFactory
from PIL import Image, ImageDraw

from jingyu.candidate import IMAGE_NAME, RENDER_NAME, CandidateStore
from jingyu.errors import JingyuError
from jingyu.paint import _tone_tangent, apply_paint, paint_candidate
from jingyu.tools.builtin import REGISTRY
from jingyu.tools.registry import ToolContext


def test_paint_object_is_local_and_keeps_the_parent(make_candidate: CandidateFactory) -> None:
    source = make_candidate()
    store = CandidateStore(make_candidate.workspace)
    before = source.image_path.read_bytes()
    painted = paint_candidate(
        make_candidate.workspace,
        source.id,
        [{"op": "wash", "target": {"object": "vase"}, "color": "#ffffff", "strength": 1}],
    )
    assert painted.id != source.id
    assert source.image_path.read_bytes() == before
    assert store.verify(source) == []
    assert store.verify(painted) == []
    assert painted.receipt["derived_from"] == {"candidate_id": source.id, "kind": "paint"}
    assert painted.receipt["paint"]["base"] == "image"
    assert painted.file(RENDER_NAME).read_bytes() == before
    assert painted.file("id_mask.png").read_bytes() == source.file("id_mask.png").read_bytes()
    with Image.open(source.image_path) as original, Image.open(painted.image_path) as changed:
        assert changed.getpixel((35, 20)) == (255, 255, 255)
        assert changed.getpixel((15, 20)) == original.getpixel((15, 20))
        assert changed.getpixel((20, 50)) == original.getpixel((20, 50))


def test_paint_region_without_an_id_pass(make_candidate: CandidateFactory) -> None:
    source = make_candidate(with_mask=False)
    painted = paint_candidate(
        make_candidate.workspace,
        source.id,
        [
            {
                "op": "wash",
                "target": {"region": {"u0": 0, "v0": 0, "u1": 0.25, "v1": 0.5}},
                "color": "#ff0000",
                "strength": 1,
            }
        ],
    )
    assert not painted.has_id_mask
    with Image.open(source.image_path) as before, Image.open(painted.image_path) as after:
        assert after.getpixel((10, 10)) == (255, 0, 0)
        assert after.getpixel((60, 40)) == before.getpixel((60, 40))


def test_paint_assembly_part_selection(make_candidate: CandidateFactory) -> None:
    entries = copy.deepcopy(SYNTHETIC_ID_MAP_OBJECTS)
    entries["2"]["part"] = "body"
    source = make_candidate(
        id_map={"schema": "jingyu.id-map.v1", "encoding": "rgb24-index", "objects": entries}
    )
    child = paint_candidate(
        make_candidate.workspace,
        source.id,
        [
            {
                "op": "wash",
                "target": {"part": {"object": "vase", "part": "body"}},
                "color": "#221100",
                "strength": 1,
            }
        ],
    )
    with Image.open(child.image_path) as image:
        assert image.getpixel((35, 20)) == (34, 17, 0)
    with pytest.raises(JingyuError, match="no mask entry") as exc:
        paint_candidate(
            make_candidate.workspace,
            source.id,
            [
                {
                    "op": "wash",
                    "target": {"part": {"object": "vase", "part": "lip"}},
                    "color": "#ffffff",
                }
            ],
        )
    assert exc.value.code == "spec.unknown_reference"


def test_unknown_hidden_and_missing_mask_fail_without_a_candidate(
    make_candidate: CandidateFactory,
) -> None:
    source = make_candidate()
    without_mask = make_candidate(with_mask=False)
    store = CandidateStore(make_candidate.workspace)
    initial = len(store.list())
    for source_id, target, code in [
        (source.id, {"object": "nope"}, "spec.unknown_reference"),
        (source.id, {"object": "far"}, "tool.invalid_arguments"),
        (without_mask.id, {"object": "vase"}, "candidate.missing_pass"),
    ]:
        with pytest.raises(JingyuError) as exc:
            paint_candidate(
                make_candidate.workspace,
                source_id,
                [{"op": "edge_ink", "target": target, "color": "#000000"}],
            )
        assert exc.value.code == code
    assert len(store.list()) == initial
    assert list(make_candidate.workspace.staging_dir.iterdir()) == []


def test_hatching_dark_values_but_not_whites() -> None:
    source = Image.new("RGB", (80, 60), "white")
    ImageDraw.Draw(source).rectangle((0, 0, 39, 59), fill=(30, 30, 30))
    changed = apply_paint(
        source,
        [
            {
                "op": "hatch",
                "target": {"region": {"x0": 0, "y0": 0, "x1": 80, "y1": 60}},
                "color": "#ffffff",
                "angle": 90,
                "spacing": 5,
                "width": 2,
                "strength": 1,
            }
        ],
    )
    assert changed.getpixel((50, 20)) == source.getpixel((50, 20))
    assert any(changed.getpixel((x, 20)) != source.getpixel((x, 20)) for x in range(40))


def test_ink_stays_inside_target_edges() -> None:
    source = Image.new("RGB", (40, 30), "white")
    changed = apply_paint(
        source,
        [
            {
                "op": "edge_ink",
                "target": {"region": {"x0": 8, "y0": 6, "x1": 28, "y1": 24}},
                "color": "#000000",
                "width": 2,
                "strength": 1,
            }
        ],
    )
    assert changed.getpixel((8, 12)) == (0, 0, 0)
    assert changed.getpixel((18, 12)) == (255, 255, 255)
    assert changed.getpixel((7, 12)) == (255, 255, 255)


def test_full_frame_edge_ink_strokes_all_four_borders() -> None:
    source = Image.new("RGB", (40, 30), "white")
    painted = apply_paint(
        source,
        [
            {
                "op": "edge_ink",
                "target": {"region": {"u0": 0, "v0": 0, "u1": 1, "v1": 1}},
                "color": "#000000",
                "width": 2,
                "strength": 1,
            }
        ],
    )
    assert painted.getpixel((0, 15)) == (0, 0, 0)
    assert painted.getpixel((39, 15)) == (0, 0, 0)
    assert painted.getpixel((20, 0)) == (0, 0, 0)
    assert painted.getpixel((20, 29)) == (0, 0, 0)
    assert painted.getpixel((20, 15)) == (255, 255, 255)


def test_paint_preserves_rgba_and_is_repeatable() -> None:
    source = Image.new("RGBA", (32, 24), (120, 170, 210, 128))
    source.putpixel((3, 3), (120, 170, 210, 0))
    actions = [
        {
            "op": "hatch",
            "target": {"region": {"x0": 0, "y0": 0, "x1": 32, "y1": 24}},
            "color": "#222222",
            "angle": 25,
            "spacing": 4,
        },
        {
            "op": "wash",
            "target": {"region": {"x0": 0, "y0": 0, "x1": 32, "y1": 24}},
            "color": "#663300",
            "strength": 0.2,
        },
    ]
    first = apply_paint(source, actions)
    second = apply_paint(source, actions)
    assert first.mode == "RGBA"
    assert first.getchannel("A").tobytes() == source.getchannel("A").tobytes()
    assert first.tobytes() == second.tobytes()


def test_paint_chain_can_return_to_original_render(make_candidate: CandidateFactory) -> None:
    source = make_candidate()
    action = {"op": "wash", "target": {"object": "vase"}, "color": "#ffffff", "strength": 1}
    first = paint_candidate(make_candidate.workspace, source.id, [action])
    second = paint_candidate(make_candidate.workspace, first.id, [action], base="render")
    assert first.receipt["derived_from"]["candidate_id"] == source.id
    assert second.receipt["derived_from"]["candidate_id"] == first.id
    assert second.file(RENDER_NAME).read_bytes() == source.file(IMAGE_NAME).read_bytes()
    assert CandidateStore(make_candidate.workspace).verify(second) == []


def test_tool_contract_returns_image_and_rejects_bad_actions(
    make_candidate: CandidateFactory,
) -> None:
    source = make_candidate()
    ctx = ToolContext(make_candidate.workspace)
    arguments = {
        "candidate_id": source.id,
        "operations": [{"op": "wash", "target": {"object": "vase"}, "color": "#abcdef"}],
    }
    result = REGISTRY.invoke("paint_candidate", arguments, ctx)
    assert result.data["derived_from"]["candidate_id"] == source.id
    assert len(result.images) == 1
    assert (
        CandidateStore(make_candidate.workspace).verify(
            CandidateStore(make_candidate.workspace).open(result.data["candidate_id"])
        )
        == []
    )
    for invalid in [
        {"op": "spray", "target": {"object": "vase"}},
        {"op": "wash", "target": {"region": {"x0": 1}}, "color": "#123456"},
        {"op": "wash", "target": {"object": "vase"}, "color": "red"},
    ]:
        with pytest.raises(JingyuError) as exc:
            REGISTRY.invoke("paint_candidate", {**arguments, "operations": [invalid]}, ctx)
        assert exc.value.code == "tool.invalid_arguments"


def test_tone_tangent_follows_isoluminance_and_preserves_flat_fallback() -> None:
    x_gradient = Image.new("L", (40, 40))
    x_gradient.putdata([x * 6 for _ in range(40) for x in range(40)])
    y_gradient = Image.new("L", (40, 40))
    y_gradient.putdata([y * 6 for y in range(40) for _ in range(40)])

    tangent_x = _tone_tangent(x_gradient, 20, 20, (1.0, 0.0))
    tangent_y = _tone_tangent(y_gradient, 20, 20, (1.0, 0.0))
    assert abs(tangent_x[0]) < 0.01 and abs(tangent_x[1]) > 0.99
    assert tangent_y[0] > 0.99 and abs(tangent_y[1]) < 0.01

    flat = Image.new("L", (40, 40), 120)
    assert _tone_tangent(flat, 20, 20, (0.6, 0.8)) == (0.6, 0.8)


def test_tone_hatching_is_local_curved_and_repeatable() -> None:
    image = Image.new("RGB", (96, 96))
    pixels = []
    for y in range(96):
        for x in range(96):
            value = min(245, round(8 + 3 * math.hypot(x - 48, y - 48)))
            pixels.append((value, value, value))
    image.putdata(pixels)
    base = {
        "op": "hatch",
        "target": {"region": {"x0": 12, "y0": 12, "x1": 84, "y1": 84}},
        "color": "#141414",
        "spacing": 7,
        "width": 1.5,
        "strength": 1,
    }

    painted = apply_paint(image, [{**base, "flow": "tone"}])
    repeated = apply_paint(image, [{**base, "flow": "tone"}])
    straight = apply_paint(image, [{**base, "flow": "straight"}])
    legacy = apply_paint(image, [base])
    assert painted.tobytes() == repeated.tobytes()
    assert painted.tobytes() != image.tobytes()
    assert painted.tobytes() != straight.tobytes()
    assert straight.tobytes() == legacy.tobytes()
    assert painted.getpixel((4, 4)) == image.getpixel((4, 4))
    assert painted.getpixel((91, 91)) == image.getpixel((91, 91))


def test_tone_flow_is_recorded_and_rejects_other_paint_actions(
    make_candidate: CandidateFactory,
) -> None:
    source = make_candidate()
    ctx = ToolContext(make_candidate.workspace)
    hatch = {
        "op": "hatch",
        "target": {"object": "vase"},
        "flow": "tone",
        "spacing": 5,
        "color": "#222222",
    }
    result = REGISTRY.invoke(
        "paint_candidate",
        {"candidate_id": source.id, "operations": [hatch], "return_image": False},
        ctx,
    )
    child = CandidateStore(make_candidate.workspace).open(result.data["candidate_id"])
    assert child.receipt["paint"]["operations"][0]["flow"] == "tone"
    assert CandidateStore(make_candidate.workspace).verify(child) == []
    assert CandidateStore(make_candidate.workspace).verify(source) == []

    for invalid in [
        {**hatch, "flow": "unknown"},
        {"op": "wash", "target": {"object": "vase"}, "flow": "tone"},
    ]:
        with pytest.raises(JingyuError) as exc:
            REGISTRY.invoke(
                "paint_candidate",
                {"candidate_id": source.id, "operations": [invalid]},
                ctx,
            )
        assert exc.value.code == "tool.invalid_arguments"


def test_tone_hatching_bounds_the_number_of_strokes() -> None:
    image = Image.new("RGB", (1200, 1200), (60, 60, 60))
    with pytest.raises(JingyuError) as exc:
        apply_paint(
            image,
            [
                {
                    "op": "hatch",
                    "flow": "tone",
                    "target": {"region": {"u0": 0, "v0": 0, "u1": 1, "v1": 1}},
                    "spacing": 2,
                }
            ],
        )
    assert exc.value.code == "tool.invalid_arguments"
    assert "spacing" in (exc.value.hint or "")


def test_tone_hatching_paints_a_one_pixel_diagonal_object() -> None:
    image = Image.new("RGB", (96, 96), (24, 24, 24))
    mask = Image.new("RGBA", image.size, (0, 0, 0, 0))
    for x in range(96):
        mask.putpixel((x, 95 - x), (0, 0, 1, 255))
    operation = {
        "op": "hatch",
        "target": {"object": "sliver"},
        "flow": "tone",
        "spacing": 8,
        "angle": 45,
        "color": "#ffffff",
        "strength": 1,
    }
    painted = apply_paint(
        image,
        [operation],
        id_mask=mask,
        id_map={"objects": {"1": {"object": "sliver"}}},
    )
    assert sum(painted.getpixel((x, 95 - x)) != (24, 24, 24) for x in range(96)) >= 8
    assert painted.getpixel((0, 0)) == (24, 24, 24)
    assert painted.getpixel((48, 48)) == (24, 24, 24)
    assert painted.tobytes() == apply_paint(
        image, [operation], id_mask=mask, id_map={"objects": {"1": {"object": "sliver"}}}
    ).tobytes()
