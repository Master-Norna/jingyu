"""Immutable render candidates and their receipts.

Every render produces a new candidate directory; nothing is ever overwritten.
A candidate holds the exact normalised scene that produced it, the image, the
id mask and its map, the Blender log and a receipt binding them together by
SHA-256.  ``verify`` recomputes every hash and rejects missing or extra files.
"""

from __future__ import annotations

import builtins
import re
import secrets
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .canonical_json import (
    CANONICALIZATION,
    canonical_sha256,
    file_sha256,
    load_file_strict,
    pretty_bytes,
)
from .errors import JingyuError
from .workspace import Workspace

RECEIPT_SCHEMA = "jingyu.receipt.v1"
RECEIPT_NAME = "receipt.json"
SCENE_NAME = "scene.json"
IMAGE_NAME = "image.png"
#: The photographic render, kept when a style painted image.png.
RENDER_NAME = "render.png"
ID_MASK_NAME = "id_mask.png"
ID_MAP_NAME = "id_map.json"
LIGHT_MASK_NAME = "light_mask.png"
LIGHT_MAP_NAME = "light_map.json"

CANDIDATE_ID_RE = re.compile(r"^c_[0-9]{8}T[0-9]{6}Z_[0-9a-f]{8}_[0-9a-f]{4}$")


def new_candidate_id(scene_sha256: str, now: datetime | None = None) -> str:
    moment = (now or datetime.now(UTC)).astimezone(UTC)
    return f"c_{moment:%Y%m%dT%H%M%SZ}_{scene_sha256[:8]}_{secrets.token_hex(2)}"


def utc_now_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass(frozen=True)
class Candidate:
    id: str
    path: Path
    receipt: dict[str, Any]

    def file(self, name: str) -> Path:
        return self.path / name

    @property
    def image_path(self) -> Path:
        return self.path / IMAGE_NAME

    @property
    def has_id_mask(self) -> bool:
        return ID_MASK_NAME in self.receipt.get("files", {})

    @property
    def has_light_map(self) -> bool:
        return LIGHT_MAP_NAME in self.receipt.get("files", {})

    def scene(self) -> dict[str, Any]:
        document = load_file_strict(self.path / SCENE_NAME)
        if not isinstance(document, dict):
            raise JingyuError("candidate.integrity_mismatch", "scene.json is not an object")
        return document

    def summary(self) -> dict[str, Any]:
        receipt = self.receipt
        return {
            "candidate_id": self.id,
            "created_at": receipt["created_at"],
            "scene_id": receipt["scene"]["id"],
            "scene_sha256": receipt["scene"]["sha256"],
            "quality": receipt["quality"],
            "engine": receipt["render"]["engine"],
            "resolution": receipt["render"]["resolution"],
        }


class CandidateStore:
    def __init__(self, workspace: Workspace) -> None:
        self.workspace = workspace

    @property
    def root(self) -> Path:
        return self.workspace.candidates_dir

    def open(self, candidate_id: str) -> Candidate:
        if not isinstance(candidate_id, str) or not CANDIDATE_ID_RE.fullmatch(candidate_id):
            raise JingyuError(
                "candidate.invalid_id",
                f"{candidate_id!r} is not a candidate id",
                hint="Candidate ids look like c_20260926T120000Z_1a2b3c4d_5e6f.",
            )
        path = self.root / candidate_id
        receipt_path = path / RECEIPT_NAME
        if not receipt_path.is_file():
            raise JingyuError("candidate.not_found", f"no candidate {candidate_id}")
        receipt = load_file_strict(receipt_path)
        if not isinstance(receipt, dict) or receipt.get("schema") != RECEIPT_SCHEMA:
            raise JingyuError(
                "candidate.integrity_mismatch", f"{candidate_id} has an invalid receipt"
            )
        return Candidate(candidate_id, path, receipt)

    def list(self, limit: int = 20) -> list[Candidate]:
        if not self.root.is_dir():
            return []
        ids = sorted(
            (p.name for p in self.root.iterdir() if CANDIDATE_ID_RE.fullmatch(p.name)),
            reverse=True,
        )
        found = []
        for candidate_id in ids[:limit]:
            try:
                found.append(self.open(candidate_id))
            except JingyuError:
                continue
        return found

    def commit(self, staging: Path, receipt: dict[str, Any]) -> Candidate:
        """Hash every staged file, write the receipt and publish atomically."""

        files: dict[str, dict[str, Any]] = {}
        for item in sorted(staging.iterdir()):
            if item.name == RECEIPT_NAME:
                continue
            if not item.is_file() or item.is_symlink():
                raise JingyuError(
                    "candidate.integrity_mismatch", f"unexpected entry {item.name} in staging"
                )
            files[item.name] = {"sha256": file_sha256(item), "bytes": item.stat().st_size}
        document = {**receipt, "schema": RECEIPT_SCHEMA, "files": files}
        (staging / RECEIPT_NAME).write_bytes(pretty_bytes(document))
        final = self.root / str(receipt["candidate_id"])
        if final.exists():
            raise JingyuError(
                "candidate.integrity_mismatch", f"candidate {final.name} already exists"
            )
        staging.replace(final)
        return Candidate(final.name, final, document)

    def verify(self, candidate: Candidate) -> builtins.list[str]:
        """Return a list of problems; empty means the candidate is intact."""

        problems: builtins.list[str] = []
        expected = candidate.receipt.get("files", {})
        present = {p.name for p in candidate.path.iterdir()} - {RECEIPT_NAME}
        for name in sorted(present - set(expected)):
            problems.append(f"unexpected file {name}")
        for name, meta in sorted(expected.items()):
            path = candidate.path / name
            if not path.is_file():
                problems.append(f"missing file {name}")
                continue
            if path.is_symlink():
                problems.append(f"{name} is a symlink")
                continue
            if file_sha256(path) != meta.get("sha256"):
                problems.append(f"{name} does not match its sha256")
        if not problems:
            scene_hash = canonical_sha256(candidate.scene())
            if scene_hash != candidate.receipt["scene"]["sha256"]:
                problems.append("scene.json does not match the receipt's scene identity")
        return problems


def scene_identity(scene: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": scene["id"],
        "sha256": canonical_sha256(scene),
        "canonicalization": CANONICALIZATION,
    }


__all__ = [
    "CANDIDATE_ID_RE",
    "ID_MAP_NAME",
    "ID_MASK_NAME",
    "IMAGE_NAME",
    "LIGHT_MAP_NAME",
    "LIGHT_MASK_NAME",
    "RECEIPT_NAME",
    "RECEIPT_SCHEMA",
    "SCENE_NAME",
    "Candidate",
    "CandidateStore",
    "new_candidate_id",
    "scene_identity",
    "utc_now_iso",
]
