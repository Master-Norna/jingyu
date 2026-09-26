from __future__ import annotations

import pytest

from jingyu.idmask import BACKGROUND, MAX_INDEX, decode_rgba, encode_index


@pytest.mark.parametrize("index", [1, 2, 255, 256, 257, 65535, 65536, 1 << 23, MAX_INDEX])
def test_encode_decode_round_trip(index: int) -> None:
    r, g, b = encode_index(index)
    assert all(0 <= c <= 255 for c in (r, g, b))
    assert decode_rgba(r, g, b, 255) == index


def test_encoding_is_big_endian_rgb() -> None:
    assert encode_index(1) == (0, 0, 1)
    assert encode_index(256) == (0, 1, 0)
    assert encode_index(1 << 16) == (1, 0, 0)
    assert encode_index(MAX_INDEX) == (255, 255, 255)
    assert MAX_INDEX == 2**24 - 1


@pytest.mark.parametrize("index", [0, -1, MAX_INDEX + 1])
def test_indices_outside_the_24_bit_range_cannot_be_encoded(index: int) -> None:
    # Index 0 is reserved for the background, which is transparent rather than black.
    with pytest.raises(ValueError):
        encode_index(index)


@pytest.mark.parametrize("rgb", [(0, 0, 0), (0, 0, 1), (255, 255, 255)])
def test_transparent_pixels_are_background(rgb: tuple[int, int, int]) -> None:
    assert decode_rgba(*rgb, 0) == BACKGROUND == 0


def test_opaque_black_is_not_a_valid_object() -> None:
    assert decode_rgba(0, 0, 0, 255) is None


@pytest.mark.parametrize("alpha", [1, 128, 254])
def test_partially_transparent_pixels_are_corrupt(alpha: int) -> None:
    assert decode_rgba(0, 0, 1, alpha) is None
