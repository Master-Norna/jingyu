from __future__ import annotations

import re
from itertools import pairwise

import pytest

from jingyu import conventions
from jingyu.conventions import (
    ID_PATTERN,
    linear_to_srgb,
    parse_hex,
    srgb_hex_to_linear,
    srgb_to_linear,
)

GRID = [i / 100 for i in range(101)]


@pytest.mark.parametrize("value", GRID)
def test_srgb_linear_round_trip(value: float) -> None:
    assert linear_to_srgb(srgb_to_linear(value)) == pytest.approx(value, abs=1e-12)
    assert srgb_to_linear(linear_to_srgb(value)) == pytest.approx(value, abs=1e-12)


def test_transfer_functions_fix_the_end_points_and_are_monotone() -> None:
    assert srgb_to_linear(0.0) == 0.0
    assert srgb_to_linear(1.0) == pytest.approx(1.0)
    assert linear_to_srgb(1.0) == pytest.approx(1.0)
    encoded = [srgb_to_linear(v) for v in GRID]
    assert all(b > a for a, b in pairwise(encoded))
    # An sRGB value of 0.5 is about 21.4 % linear light.
    assert srgb_to_linear(0.5) == pytest.approx(0.214, abs=1e-3)


@pytest.mark.parametrize(
    ("color", "expected"),
    [
        ("#000000", (0.0, 0.0, 0.0)),
        ("#ffffff", (1.0, 1.0, 1.0)),
        ("#FFFFFF", (1.0, 1.0, 1.0)),
        ("#ff8000", (1.0, 128 / 255, 0.0)),
        ("#0a0B0c", (10 / 255, 11 / 255, 12 / 255)),
    ],
)
def test_parse_hex_accepts_six_digit_colours(
    color: str, expected: tuple[float, float, float]
) -> None:
    assert parse_hex(color) == pytest.approx(expected)


@pytest.mark.parametrize(
    "color",
    [
        "",
        "fff",
        "#fff",
        "ffffff",
        "#fffffff",
        "#12345g",
        "# 12345",
        "#ff00ff00",
        "red",
        "#ffffff\n",
    ],
)
def test_parse_hex_rejects_other_spellings(color: str) -> None:
    with pytest.raises(ValueError):
        parse_hex(color)


def test_hex_to_linear_decodes_srgb() -> None:
    assert srgb_hex_to_linear("#ffffff") == (1.0, 1.0, 1.0)
    assert srgb_hex_to_linear("#000000") == (0.0, 0.0, 0.0)
    r, g, b = srgb_hex_to_linear("#808080")
    assert r == g == b == pytest.approx(srgb_to_linear(128 / 255))


@pytest.mark.parametrize(
    "ident", ["a", "vase", "vase-01", "a_b", "x9", "a" * 63, "k" + "-" * 62, "floor_2-b"]
)
def test_id_pattern_accepts_slugs(ident: str) -> None:
    assert re.fullmatch(ID_PATTERN, ident)


@pytest.mark.parametrize(
    "ident",
    ["", "A", "Vase", "1a", "-a", "_a", "a b", "a.b", "a/b", "é", "vase!", "a" * 64],
)
def test_id_pattern_rejects_everything_else(ident: str) -> None:
    assert not re.fullmatch(ID_PATTERN, ident)


def test_summary_states_the_conventions() -> None:
    summary = conventions.summary()
    assert summary["length_unit"] == "m"
    assert summary["angle_unit"] == "deg"
    assert summary["up_axis"] == "Z"
    assert summary["id_pattern"] == ID_PATTERN
    assert all(isinstance(v, str) and v for v in summary.values())
