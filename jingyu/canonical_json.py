"""Strict JSON parsing and canonical JSON identity.

Structured documents (scenes, receipts, id maps) are identified by the SHA-256
of their canonical form, so indentation, line endings and key order never change
an identity.  Opaque files (images, logs) are hashed by their bytes instead.

Standard library only: the Blender worker imports this module.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

from .errors import JingyuError

HASH_ALGORITHM = "SHA-256"
CANONICALIZATION = "jingyu-json-v1"


def _reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    document: dict[str, Any] = {}
    for key, value in pairs:
        if key in document:
            raise JingyuError(
                "json.duplicate_key",
                f"duplicate JSON object key {key!r}",
                hint="Each key may appear once per object.",
            )
        document[key] = value
    return document


def _reject_constant(name: str) -> Any:
    raise JingyuError(
        "json.non_finite_number",
        f"{name} is not a valid JSON number",
        hint="Use finite numbers only.",
    )


def _finite_float(literal: str) -> float:
    # A literal such as 1e999 parses to infinity without passing through parse_constant.
    value = float(literal)
    if not math.isfinite(value):
        raise JingyuError(
            "json.non_finite_number",
            f"{literal} overflows to infinity",
            hint="Use finite numbers only.",
        )
    return value


def _finite_int(literal: str) -> int:
    try:
        value = int(literal)
        float(value)
    except (ValueError, OverflowError):
        raise JingyuError(
            "json.non_finite_number",
            f"an integer with {len(literal.lstrip('-'))} digits is out of the finite number range",
            hint="Use finite numbers only.",
        ) from None
    return value


def loads_strict(text: str) -> Any:
    """Parse JSON, rejecting duplicate keys, NaN/Infinity and numbers that overflow."""

    try:
        return json.loads(
            text,
            object_pairs_hook=_reject_duplicates,
            parse_constant=_reject_constant,
            parse_float=_finite_float,
            parse_int=_finite_int,
        )
    except json.JSONDecodeError as exc:
        raise JingyuError(
            "json.invalid",
            f"invalid JSON at line {exc.lineno} column {exc.colno}: {exc.msg}",
        ) from exc


def load_file_strict(path: str | Path) -> Any:
    """Read a UTF-8 JSON file (a leading BOM is tolerated) strictly."""

    try:
        text = Path(path).read_text(encoding="utf-8-sig")
    except UnicodeDecodeError as exc:
        raise JingyuError(
            "json.invalid", f"the file is not UTF-8 text: {exc.reason} at byte {exc.start}"
        ) from exc
    return loads_strict(text)


def canonical_bytes(document: Any) -> bytes:
    """Return the canonical UTF-8 form of a JSON-compatible value."""

    return json.dumps(
        document,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(document: Any) -> str:
    """Return the SHA-256 hex digest of the canonical form of *document*."""

    return hashlib.sha256(canonical_bytes(document)).hexdigest()


def pretty_bytes(document: Any) -> bytes:
    """Human-friendly, deterministic serialisation used for files on disk."""

    text = json.dumps(document, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
    return (text + "\n").encode("utf-8")


def file_sha256(path: str | Path) -> str:
    """Return the SHA-256 hex digest of a file's bytes."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "CANONICALIZATION",
    "HASH_ALGORITHM",
    "canonical_bytes",
    "canonical_sha256",
    "file_sha256",
    "load_file_strict",
    "loads_strict",
    "pretty_bytes",
]
