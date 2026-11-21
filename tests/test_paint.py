"""Painting a rendered candidate is local, repeatable and never changes its parent."""

from __future__ import annotations

import copy

import pytest
from helpers import SYNTHETIC_ID_MAP_OBJECTS, VASE_BOX, CandidateFactory
from PIL import Image, ImageDraw

from jingyu.candidate import IMAGE_NAME, RENDER_NAME, CandidateStore
from jingyu.errors import JingyuError
from jingyu.paint import apply_paint, paint_candidate
from jingyu.tools.builtin import REGISTRY
from jingyu.tools.registry import ToolContext
from jingyu.workspace import Workspace


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
        [{"op": "wash", "target": {"part": {"object": "vase", "part": "body"}},
          "color": "#221100", "strength": 1}],
    )
    with Image.open(child.image_path) as image:
        assert image.getpixel((35, 20)) == (34, 17, 0)
    with pytest.raises(JingyuError, match="no mask entry") as exc:
        paint_candidate(
            make_candidate.workspace,
            source.id,
            [{"op": "wash", "target": {"part": {"object": "vase", "part": "lip"}},
              "color": "#ffffff"}],
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
                make_candidate.workspace, source_id,
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
        [{
            "op": "hatch", "target": {"region": {"x0": 0, "y0": 0, "x1": 80, "y1": 60}},
            "color": "#ffffff", "angle": 90, "spacing": 5, "width": 2, "strength": 1,
        }],
    )
    assert changed.getpixel((50, 20)) == source.getpixel((50, 20))
    assert any(changed.getpixel((x, 20)) != source.getpixel((x, 20)) for x in range(40))


def test_ink_stays_inside_target_edges() -> None:
    source = Image.new("RGB", (40, 30), "white")
    changed = apply_paint(
        source,
        [{
            "op": "edge_ink", "target": {
                "region": {"x0": 8, "y0": 6, "x1": 28, "y1": 24}
            }, "color": "#000000", "width": 2, "strength": 1,
        }],
    )
    assert changed.getpixel((8, 12)) == (0, 0, 0)
    assert changed.getpixel((18, 12)) == (255, 255, 255)
    assert changed.getpixel((7, 12)) == (255, 255, 255)


def test_paint_preserves_rgba_and_is_repeatable() -> None:
    source = Image.new("RGBA", (32, 24), (120, 170, 210, 128))
    source.putpixel((3, 3), (120, 170, 210, 0))
    actions = [
        {
            "op": "hatch", "target": {
                "region": {"x0": 0, "y0": 0, "x1": 32, "y1": 24}
            }, "color": "#222222", "angle": 25, "spacing": 4,
        },
        {
            "op": "wash", "target": {
                "region": {"x0": 0, "y0": 0, "x1": 32, "y1": 24}
            }, "color": "#663300", "strength": 0.2,
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
        "operations": [
            {"op": "wash", "target": {"object": "vase"}, "color": "#abcdef"}
        ],
    }
    result = REGISTRY.invoke("paint_candidate", arguments, ctx)
    assert result.data["derived_from"]["candidate_id"] == source.id
    assert len(result.images) == 1
    assert CandidateStore(make_candidate.workspace).verify(
        CandidateStore(make_candidate.workspace).open(result.data["candidate_id"])
    ) == []
    for invalid in [
        {"op": "spray", "target": {"object": "vase"}},
        {"op": "wash", "target": {"region": {"x0": 1}}, "color": "#123456"},
        {"op": "wash", "target": {"object": "vase"}, "color": "red"},
    ]:
        with pytest.raises(JingyuError) as exc:
            REGISTRY.invoke(
                "paint_candidate", {**arguments, "operations": [invalid]}, ctx
            )
        assert exc.value.code == "tool.invalid_arguments"
