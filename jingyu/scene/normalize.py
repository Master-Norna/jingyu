"""Fill schema defaults so that every consumer sees one explicit, complete scene.

The filler walks the same JSON Schema that validated the document, so defaults
live in exactly one place.  It understands the subset of JSON Schema the scene
schema uses: ``$ref`` into ``$defs``, ``properties`` with ``default``,
``items``, ``allOf`` and ``if``/``then`` branches selected by ``const``
discriminators.  It must only be run on documents that already validated.

Integral floats in ``integer`` positions (``1280.0``) become ``int`` so that the
renderer and the canonical hash never see two spellings of one value.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any


def apply_defaults(instance: Any, schema: Mapping[str, Any], root: Mapping[str, Any]) -> Any:
    """Return a deep copy of *instance* with defaults filled in."""

    return _fill(copy.deepcopy(instance), schema, root)


def _by_id(root: Mapping[str, Any], ref: str) -> Mapping[str, Any]:
    """The schema that declares ``$id`` *ref*: the root itself or one of its ``$defs``."""

    if root.get("$id") == ref:
        return root
    for candidate in root.get("$defs", {}).values():
        if isinstance(candidate, Mapping) and candidate.get("$id") == ref:
            return candidate
    raise ValueError(f"unsupported $ref {ref!r}")


def _resolve(schema: Mapping[str, Any], root: Mapping[str, Any]) -> Mapping[str, Any]:
    seen = 0
    while "$ref" in schema:
        ref = schema["$ref"]
        if not isinstance(ref, str):
            raise ValueError(f"unsupported $ref {ref!r}")
        if ref.startswith("#/$defs/"):
            target = root["$defs"][ref.removeprefix("#/$defs/")]
        else:
            target = _by_id(root, ref)
        merged = {k: v for k, v in schema.items() if k != "$ref"}
        schema = {**target, **merged}
        seen += 1
        if seen > 32:
            raise ValueError("$ref chain too deep")
    return schema


def _matches_consts(condition: Mapping[str, Any], instance: Any) -> bool:
    if not isinstance(instance, dict):
        return False
    props = condition.get("properties", {})
    for key in condition.get("required", []):
        if key not in instance:
            return False
    for key, sub in props.items():
        if "const" in sub and instance.get(key) != sub["const"]:
            return False
    return True


def _fill(instance: Any, schema: Mapping[str, Any], root: Mapping[str, Any]) -> Any:
    schema = _resolve(schema, root)

    if schema.get("type") == "integer" and isinstance(instance, float) and instance.is_integer():
        instance = int(instance)

    if isinstance(instance, dict):
        for key, sub in schema.get("properties", {}).items():
            if key not in instance and "default" in sub:
                instance[key] = copy.deepcopy(sub["default"])
            if key in instance:
                instance[key] = _fill(instance[key], sub, root)
        for sub in schema.get("allOf", []):
            instance = _fill(instance, sub, root)
        condition = schema.get("if")
        if condition is not None and _matches_consts(condition, instance):
            then = schema.get("then")
            if then is not None:
                instance = _fill(instance, then, root)
    elif isinstance(instance, list):
        items = schema.get("items")
        if isinstance(items, Mapping):
            instance = [_fill(item, items, root) for item in instance]
        prefix = schema.get("prefixItems")
        if isinstance(prefix, list):
            for index, sub in enumerate(prefix[: len(instance)]):
                instance[index] = _fill(instance[index], sub, root)
    return instance


__all__ = ["apply_defaults"]
