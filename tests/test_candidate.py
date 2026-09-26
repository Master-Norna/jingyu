from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest
from helpers import CandidateFactory

from jingyu.candidate import (
    CANDIDATE_ID_RE,
    ID_MAP_NAME,
    ID_MASK_NAME,
    IMAGE_NAME,
    LIGHT_MAP_NAME,
    LIGHT_MASK_NAME,
    RECEIPT_NAME,
    RECEIPT_SCHEMA,
    SCENE_NAME,
    Candidate,
    CandidateStore,
    new_candidate_id,
    scene_identity,
)
from jingyu.canonical_json import CANONICALIZATION, canonical_sha256, file_sha256, pretty_bytes
from jingyu.errors import JingyuError
from jingyu.workspace import Workspace


def _code(store: CandidateStore, candidate_id: object) -> str:
    with pytest.raises(JingyuError) as info:
        store.open(candidate_id)  # type: ignore[arg-type]
    return info.value.code


def test_new_candidate_ids_are_well_formed() -> None:
    moment = datetime(2026, 9, 26, 12, 0, 5, tzinfo=UTC)
    candidate_id = new_candidate_id("1a2b3c4d" + "0" * 56, now=moment)
    assert CANDIDATE_ID_RE.match(candidate_id)
    assert candidate_id.startswith("c_20260926T120005Z_1a2b3c4d_")
    assert CANDIDATE_ID_RE.match(new_candidate_id("f" * 64))


def test_scene_identity_hashes_the_canonical_form() -> None:
    scene = {"id": "s", "b": 1, "a": 2}
    assert scene_identity(scene) == {
        "id": "s",
        "sha256": canonical_sha256(scene),
        "canonicalization": CANONICALIZATION,
    }


def test_commit_publishes_the_staging_directory(workspace: Workspace, candidate: Candidate) -> None:
    assert candidate.path == workspace.candidates_dir / candidate.id
    assert candidate.path.is_dir()
    # The staging directory was renamed into place, not copied.
    assert list(workspace.staging_dir.iterdir()) == []
    receipt = json.loads((candidate.path / RECEIPT_NAME).read_text(encoding="utf-8"))
    assert receipt == candidate.receipt
    assert (candidate.path / RECEIPT_NAME).read_bytes() == pretty_bytes(receipt)
    assert receipt["schema"] == RECEIPT_SCHEMA
    names = {IMAGE_NAME, ID_MASK_NAME, ID_MAP_NAME, SCENE_NAME, "blender.log"}
    names |= {LIGHT_MASK_NAME, LIGHT_MAP_NAME}
    assert set(receipt["files"]) == names
    for name, meta in receipt["files"].items():
        path = candidate.path / name
        assert meta == {"sha256": file_sha256(path), "bytes": path.stat().st_size}


def test_commit_refuses_to_overwrite_an_existing_candidate(
    workspace: Workspace, candidate: Candidate
) -> None:
    staging = workspace.staging_dir / "again"
    staging.mkdir(parents=True)
    (staging / IMAGE_NAME).write_bytes(b"png")
    with pytest.raises(JingyuError) as info:
        CandidateStore(workspace).commit(staging, dict(candidate.receipt))
    assert info.value.code == "candidate.integrity_mismatch"
    assert (candidate.path / IMAGE_NAME).read_bytes() != b"png"


def test_commit_rejects_directories_in_staging(workspace: Workspace) -> None:
    staging = workspace.staging_dir / "nested"
    (staging / "sub").mkdir(parents=True)
    receipt = {"candidate_id": "c_20260926T120000Z_00000000_0000"}
    with pytest.raises(JingyuError) as info:
        CandidateStore(workspace).commit(staging, receipt)
    assert info.value.code == "candidate.integrity_mismatch"
    assert not (workspace.candidates_dir / receipt["candidate_id"]).exists()


def test_open_returns_the_committed_candidate(workspace: Workspace, candidate: Candidate) -> None:
    opened = CandidateStore(workspace).open(candidate.id)
    assert opened == candidate
    assert opened.has_id_mask
    assert opened.has_light_map
    assert opened.image_path == candidate.path / IMAGE_NAME
    assert opened.scene()["id"] == "synthetic"


def test_summary_reports_the_receipt(candidate: Candidate) -> None:
    assert candidate.summary() == {
        "candidate_id": candidate.id,
        "created_at": candidate.receipt["created_at"],
        "scene_id": "synthetic",
        "scene_sha256": candidate.receipt["scene"]["sha256"],
        "quality": "preview",
        "engine": "cycles",
        "resolution": [80, 60],
    }


def test_list_is_newest_first_and_limited(
    workspace: Workspace, make_candidate: CandidateFactory
) -> None:
    store = CandidateStore(workspace)
    assert store.list() == []
    made = [make_candidate() for _ in range(3)]
    assert [c.id for c in store.list()] == [c.id for c in reversed(made)]
    assert [c.id for c in store.list(limit=2)] == [made[2].id, made[1].id]


def test_list_skips_foreign_and_broken_entries(
    workspace: Workspace, make_candidate: CandidateFactory
) -> None:
    good = make_candidate()
    (workspace.candidates_dir / "notes.txt").write_text("x", encoding="utf-8")
    (workspace.candidates_dir / "c_20990101T000000Z_00000000_0000").mkdir()
    assert [c.id for c in CandidateStore(workspace).list()] == [good.id]


def test_list_without_a_candidates_directory_is_empty(tmp_path) -> None:
    assert CandidateStore(Workspace.at(tmp_path / "fresh")).list() == []


@pytest.mark.parametrize(
    "candidate_id",
    [
        "",
        "c_bad",
        "../candidates",
        "c_20260926T120000Z_1a2b3c4d_5e6f/../x",
        "C_20260926T120000Z_1A2B3C4D_5E6F",
        "c_20260926T120000Z_1a2b3c4d",
        "c_20260926T120000Z_1a2b3c4d_5e6f\n",
        None,
        42,
    ],
)
def test_open_rejects_malformed_ids(workspace: Workspace, candidate_id: object) -> None:
    assert _code(CandidateStore(workspace), candidate_id) == "candidate.invalid_id"


def test_open_of_an_unknown_candidate_fails(workspace: Workspace) -> None:
    assert _code(CandidateStore(workspace), "c_20260926T120000Z_1a2b3c4d_5e6f") == (
        "candidate.not_found"
    )


def test_open_rejects_a_receipt_with_the_wrong_schema(
    workspace: Workspace, candidate: Candidate
) -> None:
    receipt = dict(candidate.receipt, schema="jingyu.receipt.v0")
    (candidate.path / RECEIPT_NAME).write_bytes(pretty_bytes(receipt))
    assert _code(CandidateStore(workspace), candidate.id) == "candidate.integrity_mismatch"


def test_verify_passes_on_a_fresh_candidate(workspace: Workspace, candidate: Candidate) -> None:
    assert CandidateStore(workspace).verify(candidate) == []


def test_verify_detects_tampered_files(workspace: Workspace, candidate: Candidate) -> None:
    store = CandidateStore(workspace)
    with (candidate.path / IMAGE_NAME).open("ab") as handle:
        handle.write(b"\0")
    assert store.verify(candidate) == [f"{IMAGE_NAME} does not match its sha256"]


def test_verify_detects_missing_and_extra_files(workspace: Workspace, candidate: Candidate) -> None:
    (candidate.path / ID_MAP_NAME).unlink()
    (candidate.path / "notes.txt").write_text("x", encoding="utf-8")
    assert CandidateStore(workspace).verify(candidate) == [
        "unexpected file notes.txt",
        f"missing file {ID_MAP_NAME}",
    ]


def test_verify_detects_a_receipt_scene_identity_mismatch(
    workspace: Workspace, candidate: Candidate
) -> None:
    receipt = json.loads(json.dumps(candidate.receipt))
    receipt["scene"]["sha256"] = "0" * 64
    forged = Candidate(candidate.id, candidate.path, receipt)
    assert CandidateStore(workspace).verify(forged) == [
        "scene.json does not match the receipt's scene identity"
    ]


def test_get_candidate_tool_reports_tampering(workspace: Workspace, candidate: Candidate) -> None:
    from jingyu.tools import REGISTRY, ToolContext

    (candidate.path / SCENE_NAME).write_text("{}", encoding="utf-8")
    data = REGISTRY.invoke(
        "get_candidate", {"candidate_id": candidate.id}, ToolContext(workspace)
    ).data
    assert data["integrity"] == {
        "ok": False,
        "problems": [f"{SCENE_NAME} does not match its sha256"],
    }
