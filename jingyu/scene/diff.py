"""What changed between two scenes, with array elements matched by id.

Paths use the same ``@<id>`` segments as :mod:`jingyu.scene.patch`, so a change
reads as ``/objects/@bowl/location`` and can be fed straight back as an edit.
Arrays of plain values (vectors, colours) are compared as a whole.
"""

from __future__ import annotations

from typing import Any


def _escape(token: str) -> str:
    return token.replace("~", "~0").replace("/", "~1")


def _has_ids(value: Any) -> bool:
    return (
        isinstance(value, list)
        and bool(value)
        and all(isinstance(item, dict) and isinstance(item.get("id"), str) for item in value)
    )


def diff_scenes(before: Any, after: Any) -> list[dict[str, Any]]:
    """Every difference as ``{path, change, before?, after?}``; change is added,
    removed or changed."""

    changes: list[dict[str, Any]] = []
    _diff(before, after, "", changes)
    return changes


def _diff(a: Any, b: Any, path: str, out: list[dict[str, Any]]) -> None:
    if isinstance(a, dict) and isinstance(b, dict):
        for key in list(a) + [k for k in b if k not in a]:
            child = f"{path}/{_escape(str(key))}"
            if key not in b:
                out.append({"path": child, "change": "removed", "before": a[key]})
            elif key not in a:
                out.append({"path": child, "change": "added", "after": b[key]})
            else:
                _diff(a[key], b[key], child, out)
        return
    if (_has_ids(a) or a == []) and (_has_ids(b) or b == []) and (a or b):
        left = {item["id"]: item for item in a}
        right = {item["id"]: item for item in b}
        for ident, item in left.items():
            child = f"{path}/@{_escape(ident)}"
            if ident not in right:
                out.append({"path": child, "change": "removed", "before": item})
            else:
                _diff(item, right[ident], child, out)
        for ident, item in right.items():
            if ident not in left:
                out.append({"path": f"{path}/@{_escape(ident)}", "change": "added", "after": item})
        order_a = [i for i in left if i in right]
        order_b = [i for i in right if i in left]
        if order_a != order_b:
            out.append({"path": path, "change": "reordered", "before": order_a, "after": order_b})
        return
    if a != b or (type(a) is not type(b) and not _numbers(a, b)):
        out.append({"path": path, "change": "changed", "before": a, "after": b})


def _numbers(a: Any, b: Any) -> bool:
    return all(isinstance(v, int | float) and not isinstance(v, bool) for v in (a, b))


__all__ = ["diff_scenes"]
