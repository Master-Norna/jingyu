from __future__ import annotations

import ast
import re

import pytest
from helpers import PACKAGE_ROOT

from jingyu.errors import ERROR_CODES, Issue, JingyuError, pointer_join

CODE_RE = re.compile(r"^[a-z][a-z_]*(\.[a-z][a-z_]*)+$")


@pytest.mark.parametrize("code", sorted(ERROR_CODES))
def test_every_code_is_dotted_ascii_with_a_meaning(code: str) -> None:
    assert code.isascii()
    assert CODE_RE.match(code), code
    assert ERROR_CODES[code].strip()


def test_unregistered_code_is_rejected() -> None:
    with pytest.raises(ValueError, match="unregistered error code"):
        JingyuError("no.such_code", "boom")
    with pytest.raises(ValueError, match="unregistered error code"):
        Issue("no.such_code", "boom")


def test_error_message_carries_the_code() -> None:
    error = JingyuError("tool.unknown", "no tool named 'x'")
    assert str(error) == "[tool.unknown] no tool named 'x'"
    assert isinstance(error, Exception)


def test_to_dict_omits_empty_optional_fields() -> None:
    assert JingyuError("tool.unknown", "m").to_dict() == {"code": "tool.unknown", "message": "m"}
    assert Issue("spec.invalid", "m").to_dict() == {
        "code": "spec.invalid",
        "severity": "error",
        "pointer": "",
        "message": "m",
    }


def test_error_round_trips_through_dict() -> None:
    error = JingyuError(
        "blender.timeout", "too slow", hint="lower samples", details={"log_tail": ["a", "b"]}
    )
    data = error.to_dict()
    assert data == {
        "code": "blender.timeout",
        "message": "too slow",
        "hint": "lower samples",
        "details": {"log_tail": ["a", "b"]},
    }
    again = JingyuError.from_dict(data)
    assert (again.code, again.message, again.hint, dict(again.details)) == (
        error.code,
        error.message,
        error.hint,
        dict(error.details),
    )


@pytest.mark.parametrize(
    "payload",
    [{}, {"code": "no.such_code", "message": "x"}, {"code": 7}, {"message": "only"}],
)
def test_from_dict_maps_unknown_payloads_to_bad_response(payload: dict[str, object]) -> None:
    assert JingyuError.from_dict(payload).code == "blender.bad_response"


def test_from_dict_ignores_malformed_optional_fields() -> None:
    error = JingyuError.from_dict({"code": "tool.unknown", "hint": 3, "details": ["x"]})
    assert error.message == ""
    assert error.hint is None
    assert error.details == {}


def test_issue_keeps_hint_and_severity() -> None:
    issue = Issue("spec.scene_unlit", "dark", "/lights", severity="warning", hint="add a light")
    assert issue.to_dict() == {
        "code": "spec.scene_unlit",
        "severity": "warning",
        "pointer": "/lights",
        "message": "dark",
        "hint": "add a light",
    }


@pytest.mark.parametrize(
    ("parts", "expected"),
    [
        ((), ""),
        (("objects", 0, "id"), "/objects/0/id"),
        (("a/b",), "/a~1b"),
        (("m~n",), "/m~0n"),
        (("~/",), "/~0~1"),
        (("",), "/"),
    ],
)
def test_pointer_join_escapes_per_rfc_6901(parts: tuple[str | int, ...], expected: str) -> None:
    assert pointer_join(*parts) == expected


# --------------------------------------------------------- static code scan


def _literal_codes() -> dict[str, list[str]]:
    """Every string literal used as an error code anywhere in the package.

    Covers the first argument (or ``code=``) of ``JingyuError(...)`` and ``Issue(...)``,
    ``unknown_code="..."`` keywords and ``{"code": "..."}`` dict entries.
    """

    found: dict[str, list[str]] = {}

    def add(node: ast.AST, where: str) -> None:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            found.setdefault(node.value, []).append(where)

    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            where = f"{path.relative_to(PACKAGE_ROOT.parent)}:{getattr(node, 'lineno', '?')}"
            if isinstance(node, ast.Call):
                func = node.func
                name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
                if name in ("JingyuError", "Issue"):
                    if node.args:
                        add(node.args[0], where)
                    for keyword in node.keywords:
                        if keyword.arg == "code":
                            add(keyword.value, where)
                for keyword in node.keywords:
                    if keyword.arg == "unknown_code":
                        add(keyword.value, where)
            elif isinstance(node, ast.Dict):
                for key, value in zip(node.keys, node.values, strict=True):
                    if isinstance(key, ast.Constant) and key.value == "code":
                        add(value, where)
    return found


def test_every_code_literal_in_the_source_is_registered() -> None:
    found = _literal_codes()
    # Sanity: the scan sees codes from several layers, including the Blender worker.
    assert {"json.duplicate_key", "geometry.invalid_profile", "internal.unexpected"} <= set(found)
    unregistered = {code: where for code, where in found.items() if code not in ERROR_CODES}
    assert not unregistered, f"add these codes to ERROR_CODES: {unregistered}"
