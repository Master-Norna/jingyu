"""Edit a scene by operations instead of rewriting it: JSON Patch with id selectors.

Operations follow RFC 6902 (``add``, ``remove``, ``replace``, ``move``, ``copy``,
``test``) plus ``merge``, which applies an RFC 7386 merge patch to the value at
``path``.  A path segment ``@<id>`` selects the element of an array whose ``id``
is ``<id>``, so ``/objects/@bowl/location/2`` survives reordering and does not
require counting array positions.

Patching is all or nothing: the first failing operation raises ``patch.failed``
naming the operation and the path, and the document is left unchanged.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from typing import Any

from ..errors import JingyuError, pointer_join

OPS = ("add", "remove", "replace", "move", "copy", "test", "merge")


def apply_patch(document: Any, operations: Sequence[Mapping[str, Any]]) -> Any:
    """Return a patched deep copy of *document*."""

    result = copy.deepcopy(document)
    for number, operation in enumerate(operations):
        try:
            result = _apply_one(result, operation)
        except _PatchError as exc:
            raise JingyuError(
                "patch.failed",
                f"operation {number} ({operation.get('op')} {operation.get('path')}): {exc}",
                hint="Paths are JSON Pointers; use @<id> to select an array element by id.",
                details={"operation": number},
            ) from None
    return result


def resolve_pointer(document: Any, path: str) -> str:
    """Rewrite ``@id`` segments of *path* to array indices in *document*."""

    try:
        tokens = _tokens(path)
        resolved: list[str | int] = []
        node = document
        for token in tokens:
            key = _key(node, token, allow_end=False)
            resolved.append(key)
            node = node[key]
    except (_PatchError, KeyError) as exc:
        raise JingyuError("patch.failed", f"{path}: {exc}") from None
    return pointer_join(*resolved)


class _PatchError(Exception):
    pass


def _tokens(path: Any) -> list[str]:
    if not isinstance(path, str):
        raise _PatchError("path must be a string")
    if path == "":
        return []
    if not path.startswith("/"):
        raise _PatchError("a JSON Pointer starts with '/'")
    return [t.replace("~1", "/").replace("~0", "~") for t in path[1:].split("/")]


def _key(node: Any, token: str, *, allow_end: bool) -> str | int:
    if isinstance(node, dict):
        return token
    if not isinstance(node, list):
        raise _PatchError(f"cannot descend into a {type(node).__name__} at {token!r}")
    if token.startswith("@"):
        ident = token[1:]
        for index, item in enumerate(node):
            if isinstance(item, dict) and item.get("id") == ident:
                return index
        known = [item.get("id") for item in node if isinstance(item, dict) and "id" in item]
        raise _PatchError(f"no element with id {ident!r}; ids here: {known}")
    if token == "-" and allow_end:
        return len(node)
    if not token.isdigit() or (token != "0" and token.startswith("0")):
        raise _PatchError(f"{token!r} is not an array index (use a number or @<id>)")
    index = int(token)
    if index > len(node) or (index == len(node) and not allow_end):
        raise _PatchError(f"index {index} is out of range (length {len(node)})")
    return index


def _parent(document: Any, path: str, *, allow_end: bool) -> tuple[Any, str | int]:
    tokens = _tokens(path)
    if not tokens:
        raise _PatchError("this operation needs a path below the root")
    node = document
    for token in tokens[:-1]:
        key = _key(node, token, allow_end=False)
        try:
            node = node[key]
        except KeyError:
            raise _PatchError(f"{token!r} does not exist") from None
    return node, _key(node, tokens[-1], allow_end=allow_end)


def _get(document: Any, path: str) -> Any:
    node = document
    for token in _tokens(path):
        key = _key(node, token, allow_end=False)
        try:
            node = node[key]
        except KeyError:
            raise _PatchError(f"{token!r} does not exist") from None
    return node


def _add(document: Any, path: str, value: Any) -> Any:
    if _tokens(path) == []:
        return value
    parent, key = _parent(document, path, allow_end=True)
    if isinstance(parent, list):
        assert isinstance(key, int)
        parent.insert(key, value)
    else:
        parent[key] = value
    return document


def _remove(document: Any, path: str) -> tuple[Any, Any]:
    parent, key = _parent(document, path, allow_end=False)
    try:
        return document, parent.pop(key)
    except KeyError:
        raise _PatchError(f"{key!r} does not exist") from None


def _merge(target: Any, patch: Any) -> Any:
    if not isinstance(patch, dict):
        return copy.deepcopy(patch)
    result = dict(target) if isinstance(target, dict) else {}
    for key, value in patch.items():
        if value is None:
            result.pop(key, None)
        else:
            result[key] = _merge(result.get(key), value)
    return result


def _apply_one(document: Any, operation: Mapping[str, Any]) -> Any:
    op = operation.get("op")
    path = operation.get("path")
    if op not in OPS:
        raise _PatchError(f"unknown op; use one of {', '.join(OPS)}")
    if op in ("add", "replace", "test", "merge") and "value" not in operation:
        raise _PatchError("this operation needs a value")
    value = copy.deepcopy(operation.get("value"))
    if op == "add":
        return _add(document, str(path), value)
    if op == "remove":
        return _remove(document, str(path))[0]
    if op == "replace":
        _get(document, str(path))
        if _tokens(path) == []:
            return value
        parent, key = _parent(document, str(path), allow_end=False)
        parent[key] = value
        return document
    if op == "merge":
        merged = _merge(_get(document, str(path)), value)
        if _tokens(path) == []:
            return merged
        parent, key = _parent(document, str(path), allow_end=False)
        parent[key] = merged
        return document
    if op == "test":
        actual = _get(document, str(path))
        same_type = type(actual) is type(value) or _both_numbers(actual, value)
        if actual != value or not same_type:
            raise _PatchError(f"test failed: found {actual!r}")
        return document
    source = operation.get("from")
    if not isinstance(source, str):
        raise _PatchError(f"{op} needs a 'from' pointer")
    if op == "move":
        if str(path).startswith(source + "/"):
            raise _PatchError("cannot move a value into itself")
        document, moved = _remove(document, source)
        return _add(document, str(path), moved)
    return _add(document, str(path), copy.deepcopy(_get(document, source)))


def _both_numbers(a: Any, b: Any) -> bool:
    return all(isinstance(v, int | float) and not isinstance(v, bool) for v in (a, b))


__all__ = ["OPS", "apply_patch", "resolve_pointer"]
