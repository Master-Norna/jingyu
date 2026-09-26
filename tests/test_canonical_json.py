from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from jingyu.canonical_json import (
    canonical_bytes,
    canonical_sha256,
    file_sha256,
    load_file_strict,
    loads_strict,
    pretty_bytes,
)
from jingyu.errors import JingyuError


def _code(text: str) -> str:
    with pytest.raises(JingyuError) as info:
        loads_strict(text)
    return info.value.code


def test_strict_parse_accepts_plain_json() -> None:
    assert loads_strict('{"a": [1, 2.5, "x", null, true]}') == {"a": [1, 2.5, "x", None, True]}


@pytest.mark.parametrize(
    "text",
    ['{"a": 1, "a": 2}', '{"outer": {"k": 1, "k": 1}}', '[{"x": 0}, {"y": 1, "y": 2}]'],
)
def test_duplicate_keys_are_rejected_at_any_depth(text: str) -> None:
    assert _code(text) == "json.duplicate_key"


@pytest.mark.parametrize(
    "text",
    [
        '{"a": NaN}',
        "[Infinity]",
        "-Infinity",
        # Literals that overflow a double would otherwise parse to infinity.
        "[1e999]",
        "-1e400",
        "1" + "0" * 400,
        "9" * 5000,
    ],
)
def test_non_finite_numbers_are_rejected(text: str) -> None:
    assert _code(text) == "json.non_finite_number"


@pytest.mark.parametrize("text", ["", "{", '{"a": }', "[1,]", "{'a': 1}", '{"a": 1} x'])
def test_malformed_json_reports_json_invalid(text: str) -> None:
    assert _code(text) == "json.invalid"


def test_large_but_finite_numbers_survive() -> None:
    assert loads_strict("[1e308, -1e308, 123456789012345678901234567890]") == [
        1e308,
        -1e308,
        123456789012345678901234567890,
    ]


def test_canonical_bytes_ignore_key_order_and_whitespace() -> None:
    a = loads_strict('{"b": 1, "a": {"y": [1, 2], "x": "é"}}')
    b = loads_strict('{\n  "a": {"x": "é", "y": [1, 2]},\n  "b": 1\n}')
    assert canonical_bytes(a) == canonical_bytes(b)
    assert canonical_bytes(a) == '{"a":{"x":"é","y":[1,2]},"b":1}'.encode()
    assert canonical_sha256(a) == canonical_sha256(b)


def test_canonical_sha_is_stable_and_content_sensitive() -> None:
    document = {"id": "x", "values": [1, 2, 3]}
    assert canonical_sha256(document) == hashlib.sha256(canonical_bytes(document)).hexdigest()
    assert canonical_sha256(document) == (
        hashlib.sha256(b'{"id":"x","values":[1,2,3]}').hexdigest()
    )
    assert canonical_sha256({**document, "values": [1, 2, 4]}) != canonical_sha256(document)


def test_canonical_form_refuses_non_finite_values() -> None:
    with pytest.raises(ValueError):
        canonical_bytes({"a": float("nan")})


def test_pretty_bytes_are_sorted_indented_and_newline_terminated() -> None:
    text = pretty_bytes({"b": 1, "a": "é"}).decode("utf-8")
    assert text == '{\n  "a": "é",\n  "b": 1\n}\n'


def test_load_file_tolerates_a_utf8_bom(tmp_path: Path) -> None:
    path = tmp_path / "bom.json"
    path.write_bytes(b"\xef\xbb\xbf" + '{"名": 1}'.encode())
    assert load_file_strict(path) == {"名": 1}


def test_load_file_applies_strict_rules(tmp_path: Path) -> None:
    path = tmp_path / "dup.json"
    path.write_text('{"a": 1, "a": 1}', encoding="utf-8")
    with pytest.raises(JingyuError) as info:
        load_file_strict(path)
    assert info.value.code == "json.duplicate_key"


def test_load_file_rejects_non_utf8_as_invalid_json(tmp_path: Path) -> None:
    path = tmp_path / "latin1.json"
    path.write_bytes(b'{"a": "\xff"}')
    with pytest.raises(JingyuError) as info:
        load_file_strict(path)
    assert info.value.code == "json.invalid"


def test_load_file_propagates_missing_files(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_file_strict(tmp_path / "missing.json")


def test_file_sha256_hashes_raw_bytes(tmp_path: Path) -> None:
    path = tmp_path / "blob.bin"
    payload = bytes(range(256)) * 5000
    path.write_bytes(payload)
    assert file_sha256(path) == hashlib.sha256(payload).hexdigest()
