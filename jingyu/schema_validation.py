"""JSON Schema validation with the regular-expression semantics the spec requires.

JSON Schema patterns follow ECMA-262, where ``$`` (without the multiline flag)
matches only at the very end of the input.  Python's ``re`` also lets ``$``
match before a final newline, so with the stock validator an id such as
``"vase\\n"`` or a colour ``"#ffffff\\n"`` would pass ``^...$``.  The validator
here translates ``$`` into Python's end-of-string anchor before matching.
Every schema in Jingyu is validated through it.
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Mapping
from functools import lru_cache
from typing import Any

from jsonschema import Draft202012Validator, validators
from jsonschema.exceptions import ValidationError


def ecma_to_python(pattern: str) -> str:
    """Translate ``$`` outside character classes into Python's end-of-string ``\\Z``."""

    out: list[str] = []
    index, in_class = 0, False
    while index < len(pattern):
        char = pattern[index]
        if char == "\\":
            out.append(pattern[index : index + 2])
            index += 2
            continue
        if in_class:
            in_class = char != "]"
        elif char == "[":
            in_class = True
        elif char == "$":
            char = "\\Z"
        out.append(char)
        index += 1
    return "".join(out)


@lru_cache(maxsize=256)
def _compile(pattern: str) -> re.Pattern[str]:
    return re.compile(ecma_to_python(pattern))


def pattern_matches(pattern: str, text: str) -> bool:
    """Whether *text* matches the JSON Schema *pattern* (unanchored, ECMA-262 ``$``)."""

    return _compile(pattern).search(text) is not None


def _pattern(
    validator: Any, pattern: str, instance: Any, schema: Mapping[str, Any]
) -> Iterator[ValidationError]:
    if validator.is_type(instance, "string") and not pattern_matches(pattern, instance):
        yield ValidationError(f"{instance!r} does not match {pattern!r}")


Validator = validators.extend(Draft202012Validator, {"pattern": _pattern})


def check_schema(schema: Mapping[str, Any]) -> None:
    Draft202012Validator.check_schema(schema)


__all__ = ["Validator", "check_schema", "ecma_to_python", "pattern_matches"]
