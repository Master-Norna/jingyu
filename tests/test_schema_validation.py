"""JSON Schema patterns use ECMA-262 anchors, not Python's."""

from __future__ import annotations

import re

import pytest

from jingyu.conventions import COLOR_PATTERN, ID_PATTERN
from jingyu.schema_validation import Validator, ecma_to_python, pattern_matches


@pytest.mark.parametrize(
    ("pattern", "translated"),
    [
        (r"^[a-z]+$", r"^[a-z]+\Z"),
        (r"^a$|^b$", r"^a\Z|^b\Z"),
        (r"^[$]+$", r"^[$]+\Z"),
        (r"^\$[0-9]+$", r"^\$[0-9]+\Z"),
        (r"^[\]$]$", r"^[\]$]\Z"),
        (r"abc", r"abc"),
    ],
)
def test_dollar_outside_classes_becomes_end_of_string(pattern: str, translated: str) -> None:
    assert ecma_to_python(pattern) == translated
    re.compile(translated)


@pytest.mark.parametrize(
    ("pattern", "text", "expected"),
    [
        (ID_PATTERN, "vase", True),
        (ID_PATTERN, "vase\n", False),
        (COLOR_PATTERN, "#ffffff", True),
        (COLOR_PATTERN, "#ffffff\n", False),
        (r"^\$[0-9]+$", "$12", True),
        (r"^[$]+$", "$$", True),
        (r"b", "abc", True),
    ],
)
def test_pattern_matches(pattern: str, text: str, expected: bool) -> None:
    assert pattern_matches(pattern, text) is expected


def test_the_validator_rejects_a_trailing_newline() -> None:
    validator = Validator({"type": "string", "pattern": ID_PATTERN})
    assert validator.is_valid("vase")
    errors = list(validator.iter_errors("vase\n"))
    assert [e.validator for e in errors] == ["pattern"]
