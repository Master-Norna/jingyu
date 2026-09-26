"""The picture charter: measurements of a render beside a series' identity, as questions."""

from __future__ import annotations

import json
from typing import Any

import pytest
from PIL import Image

from jingyu.candidate import Candidate
from jingyu.charter import check_charter, dominant_colours, validate_charter
from jingyu.errors import JingyuError
from jingyu.tools import REGISTRY, ToolContext
from jingyu.workspace import Workspace


def _charter(**extra: Any) -> dict[str, Any]:
    return {"schema": "jingyu.charter.v1", "id": "series", "subject": "a vase", **extra}


def _layout(u0: float, v0: float, u1: float, v1: float) -> dict[str, Any]:
    return {"objects": [{"object": "vase", "bbox_uv": {"u0": u0, "v0": v0, "u1": u1, "v1": v1}}]}


def _scene(preset: str = "none") -> dict[str, Any]:
    return {"render": {"style": {"preset": preset}}}


def test_charters_are_validated() -> None:
    assert validate_charter(_charter(value_key="low"))["value_key"] == "low"
    with pytest.raises(JingyuError) as info:
        validate_charter(_charter(value_key="dim"))
    assert info.value.code == "charter.invalid"


def test_a_frame_that_matches_raises_no_question_but_the_mood() -> None:
    image = Image.new("RGB", (64, 48), (40, 40, 44))
    image.paste((200, 60, 40), (20, 10, 44, 38))
    charter = _charter(value_key="low", palette=["#c83c28"], style="none")
    report = check_charter(charter, image, _scene(), None)
    assert report["facts"]["value"]["key"] == "low"
    assert report["facts"]["palette"][0]["present"]
    assert report["questions"] == []
    moody = check_charter({**charter, "light_mood": "a quiet evening"}, image, _scene(), None)
    assert len(moody["questions"]) == 1 and "quiet evening" in moody["questions"][0]


def test_differences_become_questions_not_scores() -> None:
    image = Image.new("RGB", (64, 48), (230, 230, 225))
    charter = _charter(value_key="low", contrast="hard", palette=["#1f3b73"], style="ink")
    report = check_charter(charter, image, _scene(), None)
    text = " ".join(report["questions"])
    assert "low key" in text and "hard contrast" in text and "#1f3b73" in text and "ink" in text
    assert "score" not in json.dumps(report)


def test_the_subject_is_held_against_the_composition_skeleton() -> None:
    image = Image.new("RGB", (64, 48), (120, 120, 120))
    charter = _charter(
        subject_objects=["vase"], composition={"skeleton": "thirds", "subject_size": 0.3}
    )
    centred = check_charter(charter, image, _scene(), _layout(0.4, 0.4, 0.6, 0.6))
    thirds = [[round(u, 3), round(v, 3)] for u in (1 / 3, 2 / 3) for v in (1 / 3, 2 / 3)]
    assert centred["facts"]["subject"]["skeleton_point"] in thirds
    assert any("thirds" in q for q in centred["questions"])
    placed = check_charter(charter, image, _scene(), _layout(0.23, 0.18, 0.43, 0.48))
    assert not any("thirds" in q or "frame_subject" in q for q in placed["questions"])
    missing = check_charter(charter, image, _scene(), {"objects": []})
    assert any("not in the frame" in q for q in missing["questions"])


def test_dominant_colours_find_the_main_colours() -> None:
    image = Image.new("RGB", (40, 40), (10, 20, 200))
    image.paste((240, 240, 0), (0, 0, 40, 10))
    colours = dominant_colours(image, 4)
    assert colours[0]["share"] == pytest.approx(0.75, abs=0.02)


def test_the_tool_reads_the_charter_from_the_workspace(
    workspace: Workspace, candidate: Candidate
) -> None:
    (workspace.root / "series.json").write_text(
        json.dumps(_charter(value_key="high")), encoding="utf-8"
    )
    data = REGISTRY.invoke(
        "check_charter",
        {"candidate_id": candidate.id, "charter_path": "series.json"},
        ToolContext(workspace),
    ).data
    assert data["charter"] == "series"
    assert "value" in data["facts"] and isinstance(data["questions"], list)
