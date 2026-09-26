from __future__ import annotations

import hashlib
import re

import pytest
from helpers import PACKAGE_ROOT, REPO_ROOT

from jingyu import constitution
from jingyu.errors import JingyuError

PACKAGED = PACKAGE_ROOT / "resources" / "constitution" / constitution.RESOURCE
DOCS_COPY = REPO_ROOT / "docs" / "视觉宪法" / constitution.RESOURCE


def test_packaged_copy_is_byte_identical_to_the_docs_copy() -> None:
    assert PACKAGED.read_bytes() == DOCS_COPY.read_bytes(), (
        f"copy {DOCS_COPY.relative_to(REPO_ROOT)} over {PACKAGED.relative_to(REPO_ROOT)}"
    )


def test_resource_is_read_from_the_package() -> None:
    text = constitution.resource_text()
    assert text == PACKAGED.read_text(encoding="utf-8")
    assert constitution.load().sha256 == hashlib.sha256(text.encode("utf-8")).hexdigest()
    assert constitution.load().version == constitution.VERSION


def test_load_finds_sections_and_clauses() -> None:
    entries = constitution.load().entries
    sections = [e for e in entries.values() if e.kind == "section"]
    clauses = [e for e in entries.values() if e.kind == "clause"]
    assert {"J0", "J1", "J7", "J7.1"} <= {e.id for e in sections}
    assert len(clauses) > 50
    for entry in entries.values():
        assert entry.title.strip() and entry.text.strip()
    j7_1 = entries["J7.1"]
    assert j7_1.kind == "section"
    assert j7_1.text.startswith("## J7.1｜")


def test_every_clause_id_in_the_text_is_defined_once() -> None:
    text = constitution.resource_text()
    clause_ids = re.findall(r"^\*\s+\*\*(J\d+(?:\.\d+)+)｜", text, flags=re.MULTILINE)
    heading_ids = re.findall(r"^#{1,6}\s+(J\d+(?:\.\d+)?)｜", text, flags=re.MULTILINE)
    ids = clause_ids + heading_ids
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    assert duplicates == []
    assert set(ids) == set(constitution.load().entries)


def test_entries_are_in_numeric_order() -> None:
    ids = list(constitution.load().entries)
    keys = [tuple(int(p) for p in i[1:].split(".")) for i in ids]
    assert keys == sorted(keys)


def test_index_lists_ids_and_titles_only() -> None:
    document = constitution.load()
    index = document.index()
    assert index["version"] == document.version
    assert index["sha256"] == document.sha256
    listed = [e["id"] for e in index["sections"] + index["clauses"]]
    assert sorted(listed) == sorted(document.entries)
    assert all(set(e) == {"id", "title"} for e in index["sections"] + index["clauses"])


def test_lookup_returns_known_entries_in_request_order() -> None:
    found = constitution.load().lookup(["J7.1", "J1.01", "J0"])
    assert [e["id"] for e in found] == ["J7.1", "J1.01", "J0"]
    assert found[1] == {
        "id": "J1.01",
        "kind": "clause",
        "title": "整体先读",
        "text": constitution.load().entries["J1.01"].text,
    }


def test_lookup_accepts_up_to_the_limit() -> None:
    clauses = [i for i, e in constitution.load().entries.items() if e.kind == "clause"]
    assert len(constitution.load().lookup(clauses[: constitution.MAX_IDS])) == 12


def test_lookup_rejects_more_than_twelve_ids() -> None:
    clauses = [i for i, e in constitution.load().entries.items() if e.kind == "clause"]
    with pytest.raises(JingyuError) as info:
        constitution.load().lookup(clauses[: constitution.MAX_IDS + 1])
    assert info.value.code == "tool.invalid_arguments"


@pytest.mark.parametrize("ident", ["J99", "J1.999", "j1.01", ""])
def test_lookup_of_an_unknown_id_fails(ident: str) -> None:
    with pytest.raises(JingyuError) as info:
        constitution.load().lookup(["J1.01", ident])
    assert info.value.code == "constitution.unknown_clause"


def test_parse_handles_a_small_document() -> None:
    text = "\n".join(
        [
            "# Title",
            "intro",
            "# J1｜One",
            "body",
            "* **J1.01｜First**：the first clause.",
            "## Not a section",
            "* **J1.02｜Second**：outside any section.",
        ]
    )
    document = constitution.parse(text, version="test")
    assert list(document.entries) == ["J1", "J1.01", "J1.02"]
    assert document.entries["J1"].text == "# J1｜One\nbody\n* **J1.01｜First**：the first clause."
    assert document.entries["J1.02"].text == "outside any section."
