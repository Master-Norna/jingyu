"""Which material each part of an object wears.

An object's ``material`` dresses the whole of it; an assembled geometry (a table,
a room) has named parts, and ``part_materials`` can dress each one differently.
Some parts come with a default of their own (window panes are clear glass), so a
plaster room does not get plaster windows.  Standard library only: the Blender
worker resolves materials the same way the host reports them.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ..generator import GeneratorDef


@dataclass(frozen=True)
class PartMaterial:
    """A scene material id, or else a generator's default family call, or neither
    (neutral grey)."""

    material: str | None
    default: Mapping[str, Any] | None = None


def part_materials(
    obj: Mapping[str, Any], definition: GeneratorDef[Any]
) -> dict[str | None, PartMaterial]:
    """Per part (None for an object without parts), the material it wears."""

    own = obj.get("material")
    if not definition.parts:
        return {None: PartMaterial(own)}
    chosen = obj.get("part_materials") or {}
    result: dict[str | None, PartMaterial] = {}
    for part in definition.parts:
        if part in chosen:
            result[part] = PartMaterial(chosen[part])
        elif part in definition.part_defaults:
            result[part] = PartMaterial(None, definition.part_defaults[part])
        else:
            result[part] = PartMaterial(own)
    return result


__all__ = ["PartMaterial", "part_materials"]
