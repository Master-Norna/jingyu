"""Bounded lookup into the Jingyu visual constitution.

The constitution is a source of questions, not a checklist: models read a few
relevant clauses when a review needs another angle, never the whole text by
default.  Lookups are limited to :data:`MAX_IDS` ids per call.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from typing import Any

from .errors import JingyuError

RESOURCE = "景语视觉宪法-v0.1.md"
VERSION = "v0.1"
MAX_IDS = 12

_HEADING = re.compile(r"^(#{1,6})\s+(J\d+(?:\.\d+)?)｜(.+?)\s*$")
_CLAUSE = re.compile(r"^\*\s+\*\*(J\d+(?:\.\d+)+)｜(.+?)\*\*：(.+)$")
_ANY_HEADING = re.compile(r"^#{1,6}\s")


@dataclass(frozen=True)
class Entry:
    id: str
    kind: str  # "section" or "clause"
    title: str
    text: str


@dataclass(frozen=True)
class Constitution:
    version: str
    sha256: str
    entries: dict[str, Entry]

    def index(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "sha256": self.sha256,
            "sections": [
                {"id": e.id, "title": e.title} for e in self.entries.values() if e.kind == "section"
            ],
            "clauses": [
                {"id": e.id, "title": e.title} for e in self.entries.values() if e.kind == "clause"
            ],
        }

    def lookup(self, ids: list[str]) -> list[dict[str, str]]:
        if len(ids) > MAX_IDS:
            raise JingyuError(
                "tool.invalid_arguments",
                f"at most {MAX_IDS} ids per lookup",
                hint="Pick the few clauses that matter for this review.",
            )
        found = []
        for ident in ids:
            entry = self.entries.get(ident)
            if entry is None:
                raise JingyuError(
                    "constitution.unknown_clause",
                    f"no clause or section {ident!r}",
                    hint="Call without ids to get the index.",
                )
            found.append(
                {"id": entry.id, "kind": entry.kind, "title": entry.title, "text": entry.text}
            )
        return found


def parse(text: str, version: str = VERSION) -> Constitution:
    entries: dict[str, Entry] = {}
    lines = text.splitlines()
    current: tuple[str, str] | None = None
    buffer: list[str] = []

    def flush() -> None:
        if current is not None:
            ident, title = current
            entries[ident] = Entry(ident, "section", title, "\n".join(buffer).strip())

    for line in lines:
        heading = _HEADING.match(line)
        if heading or _ANY_HEADING.match(line):
            flush()
            buffer = []
            current = (heading.group(2), heading.group(3)) if heading else None
            if heading:
                buffer.append(line)
            continue
        if current is not None:
            buffer.append(line)
        clause = _CLAUSE.match(line)
        if clause:
            ident, title, body = clause.groups()
            entries[ident] = Entry(ident, "clause", title, body.strip())
    flush()
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return Constitution(
        version, digest, dict(sorted(entries.items(), key=lambda kv: _sort_key(kv[0])))
    )


def _sort_key(ident: str) -> tuple[int, ...]:
    return tuple(int(part) for part in ident[1:].split("."))


def resource_text() -> str:
    return (
        resources.files("jingyu")
        .joinpath("resources", "constitution", RESOURCE)
        .read_text(encoding="utf-8")
    )


@lru_cache(maxsize=1)
def load() -> Constitution:
    return parse(resource_text())


__all__ = ["MAX_IDS", "RESOURCE", "VERSION", "Constitution", "Entry", "load", "parse"]
