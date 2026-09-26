"""Scene validation: strict parse, version gate, JSON Schema, defaults, semantics."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

from jsonschema.exceptions import ValidationError

from ..canonical_json import canonical_sha256, load_file_strict, loads_strict
from ..errors import Issue, JingyuError, pointer_join
from ..schema_validation import Validator, check_schema
from .normalize import apply_defaults
from .schema import SCENE_SCHEMA, scene_schema
from .semantic import check_scene

MAX_SCHEMA_ISSUES = 100
_AIM_ONE_OF = [{"required": ["look_at"]}, {"required": ["rotation"]}]


@dataclass(frozen=True)
class ValidationResult:
    """Outcome of validating one scene document."""

    issues: tuple[Issue, ...] = ()
    normalized: dict[str, Any] | None = None
    scene_sha256: str | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    @property
    def valid(self) -> bool:
        return self.normalized is not None and not any(i.severity == "error" for i in self.issues)

    @property
    def errors(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == "error"]

    @property
    def warnings(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == "warning"]

    def require_valid(self) -> dict[str, Any]:
        """Return the normalised scene or raise ``spec.invalid`` with every issue."""

        if not self.valid or self.normalized is None:
            raise JingyuError(
                "spec.invalid",
                f"the scene has {len(self.errors)} error(s); first: "
                + (self.errors[0].message if self.errors else "unknown"),
                hint="Fix the issues listed in details.issues; pointers are JSON Pointers.",
                details={"issues": [i.to_dict() for i in self.issues]},
            )
        return self.normalized


@lru_cache(maxsize=1)
def _validator() -> Any:
    schema = scene_schema()
    check_schema(schema)
    return Validator(schema)


def validate_scene(document: Any) -> ValidationResult:
    """Validate an already-parsed scene document."""

    if not isinstance(document, Mapping):
        return ValidationResult(
            (Issue("spec.schema_violation", "the scene must be a JSON object", ""),)
        )
    marker = document.get("schema")
    if marker != SCENE_SCHEMA:
        return ValidationResult(
            (
                Issue(
                    "spec.unsupported_schema",
                    f"expected schema {SCENE_SCHEMA!r}, got {marker!r}",
                    "/schema",
                    hint=f'Set "schema": "{SCENE_SCHEMA}".',
                ),
            )
        )

    schema_issues = _non_finite_issues(document) or _schema_issues(document)
    if schema_issues:
        return ValidationResult(tuple(schema_issues))

    validator = _validator()
    normalized = apply_defaults(dict(document), validator.schema, validator.schema)
    issues = check_scene(normalized)
    has_errors = any(i.severity == "error" for i in issues)
    return ValidationResult(
        tuple(issues),
        normalized,
        None if has_errors else canonical_sha256(normalized),
    )


def validate_scene_text(text: str) -> ValidationResult:
    try:
        document = loads_strict(text)
    except JingyuError as exc:
        return ValidationResult((Issue(exc.code, exc.message, ""),))
    return validate_scene(document)


def validate_scene_file(path: str | Path) -> ValidationResult:
    try:
        document = load_file_strict(path)
    except JingyuError as exc:
        return ValidationResult((Issue(exc.code, exc.message, ""),))
    return validate_scene(document)


def _non_finite_issues(value: Any, path: tuple[str | int, ...] = ()) -> list[Issue]:
    """NaN, infinities and integers beyond the float range in an already-parsed document.

    Strict parsing rejects them in text, but documents from other sources (Python
    callers, MCP clients) may still carry them, and JSON Schema accepts them as numbers.
    """

    if isinstance(value, bool):
        return []
    if isinstance(value, float | int):
        try:
            finite = math.isfinite(value)
        except OverflowError:
            finite = False
        if finite:
            return []
        return [
            Issue(
                "json.non_finite_number",
                "the value is not a finite JSON number",
                pointer_join(*path),
                hint="Use finite numbers only.",
            )
        ]
    if isinstance(value, Mapping):
        return [i for k, v in value.items() for i in _non_finite_issues(v, (*path, str(k)))]
    if isinstance(value, list | tuple):
        return [i for n, v in enumerate(value) for i in _non_finite_issues(v, (*path, n))]
    return []


def _schema_issues(document: Any) -> list[Issue]:
    errors = sorted(_validator().iter_errors(document), key=_error_sort_key)
    issues: list[Issue] = []
    seen: set[tuple[str, str]] = set()
    for error in errors:
        pointer = pointer_join(*error.absolute_path)
        key = (pointer, error.message)
        if key in seen:
            continue
        seen.add(key)
        issues.append(Issue("spec.schema_violation", error.message, pointer, hint=_hint(error)))
        if len(issues) >= MAX_SCHEMA_ISSUES:
            break
    return issues


def _error_sort_key(error: ValidationError) -> tuple[str, str]:
    return (pointer_join(*error.absolute_path), error.message)


def _hint(error: ValidationError) -> str | None:
    schema = error.schema if isinstance(error.schema, Mapping) else {}
    if error.validator == "additionalProperties":
        allowed = sorted(schema.get("properties", {}))
        return f"allowed fields here: {', '.join(allowed)}"
    if error.validator == "enum":
        return f"allowed values: {', '.join(map(str, schema.get('enum', [])))}"
    if error.validator == "oneOf" and error.validator_value == _AIM_ONE_OF:
        return "give exactly one of look_at or rotation"
    if error.validator == "pattern":
        return f"must match {schema.get('pattern')}"
    return None


__all__ = [
    "ValidationResult",
    "validate_scene",
    "validate_scene_file",
    "validate_scene_text",
]
