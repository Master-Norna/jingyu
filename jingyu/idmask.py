"""The object id mask encoding, shared by the Blender pass and the host decoder.

Each visible object gets a 1-based index encoded as a 24-bit RGB value
(``index = r << 16 | g << 8 | b``) and rendered as flat, unlit colour with one
centred sample per pixel.  Background pixels have alpha 0.  Decoding is exact:
there is no palette matching and no tolerance.

Standard library only: the Blender worker imports this module.
"""

from __future__ import annotations

ENCODING = "rgb24-index"
ID_MAP_SCHEMA = "jingyu.id-map.v1"
MAX_INDEX = (1 << 24) - 1
BACKGROUND = 0


def encode_index(index: int) -> tuple[int, int, int]:
    if not 1 <= index <= MAX_INDEX:
        raise ValueError(f"id mask index must be in [1, {MAX_INDEX}], got {index}")
    return (index >> 16) & 0xFF, (index >> 8) & 0xFF, index & 0xFF


def decode_rgba(r: int, g: int, b: int, a: int) -> int | None:
    """Return the object index, ``BACKGROUND`` (0), or ``None`` for a corrupt pixel."""

    if a == 0:
        return BACKGROUND
    if a != 255:
        return None
    index = (r << 16) | (g << 8) | b
    return index if index else None


__all__ = [
    "BACKGROUND",
    "ENCODING",
    "ID_MAP_SCHEMA",
    "MAX_INDEX",
    "decode_rgba",
    "encode_index",
]
