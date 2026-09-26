from __future__ import annotations

import json

from helpers import REPO_ROOT
from jsonschema import Draft202012Validator

from jingyu.reference import stale_files
from jingyu.scene import scene_schema
from jingyu.scene.schema import DIALECT

SCHEMA_FILE = REPO_ROOT / "schemas" / "scene.v1.schema.json"
REGENERATE = (
    "regenerate from the repository root with: python -m jingyu.reference --write "
    "(the schema alone: python -m jingyu.cli schema --write schemas/scene.v1.schema.json)"
)


def test_scene_schema_is_valid_draft_2020_12() -> None:
    schema = scene_schema()
    assert schema["$schema"] == DIALECT
    Draft202012Validator.check_schema(schema)


def test_scene_schema_is_a_fresh_copy() -> None:
    scene_schema()["properties"].clear()
    assert scene_schema()["properties"]


def test_exported_scene_schema_matches_the_code() -> None:
    expected = json.dumps(scene_schema(), ensure_ascii=False, indent=2) + "\n"
    assert SCHEMA_FILE.is_file(), f"{SCHEMA_FILE} is missing; {REGENERATE}"
    actual = SCHEMA_FILE.read_text(encoding="utf-8")
    assert actual == expected, f"{SCHEMA_FILE.name} is stale; {REGENERATE}"


def test_generated_reference_files_are_fresh() -> None:
    stale = stale_files(REPO_ROOT)
    assert stale == [], f"stale generated files {stale}; {REGENERATE}"
