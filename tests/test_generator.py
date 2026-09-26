from __future__ import annotations

from typing import Any

import pytest
from jsonschema import Draft202012Validator

from jingyu.environments import ENVIRONMENTS
from jingyu.errors import JingyuError
from jingyu.generator import GeneratorDef, GeneratorRegistry, discriminated_union, number
from jingyu.geometry import GEOMETRY
from jingyu.materials import MATERIALS

REGISTRIES = [
    pytest.param(GEOMETRY, id="geometry"),
    pytest.param(MATERIALS, id="materials"),
    pytest.param(ENVIRONMENTS, id="environments"),
]


def _definitions() -> list[Any]:
    return [
        pytest.param(registry, definition, id=f"{registry.discriminator}:{definition.name}")
        for registry in (GEOMETRY, MATERIALS, ENVIRONMENTS)
        for definition in registry
    ]


@pytest.mark.parametrize("registry", REGISTRIES)
def test_union_schema_is_valid_draft_2020_12(registry: GeneratorRegistry[Any]) -> None:
    Draft202012Validator.check_schema(registry.union_schema())
    Draft202012Validator.check_schema(
        registry.union_schema(extra_properties={"id": {"type": "string"}}, extra_required=("id",))
    )


@pytest.mark.parametrize(
    ("registry", "code"),
    [
        (GEOMETRY, "geometry.unknown_op"),
        (MATERIALS, "material.unknown_family"),
        (ENVIRONMENTS, "environment.unknown_family"),
    ],
)
def test_unknown_discriminator_raises_the_registry_code(
    registry: GeneratorRegistry[Any], code: str
) -> None:
    assert registry.unknown_code == code
    with pytest.raises(JingyuError) as info:
        registry.get("no_such_generator")
    assert info.value.code == code
    assert info.value.hint and registry.names()[0] in info.value.hint
    with pytest.raises(JingyuError) as info:
        registry.run({registry.discriminator: "no_such_generator"})
    assert info.value.code == code


@pytest.mark.parametrize("registry", REGISTRIES)
def test_catalog_lists_every_generator(registry: GeneratorRegistry[Any]) -> None:
    catalog = registry.catalog()
    assert [entry["name"] for entry in catalog] == registry.names()
    assert len(set(registry.names())) == len(registry.names())
    for entry in catalog:
        assert entry["summary"].strip()
        assert isinstance(entry["params"], dict)
        assert set(entry) == {"name", "summary", "params", "required", "examples"}


def test_catalog_entries_are_copies() -> None:
    entry = GEOMETRY.catalog()[0]
    entry["params"].clear()
    assert GEOMETRY.catalog()[0]["params"]


@pytest.mark.parametrize(("registry", "definition"), _definitions())
def test_params_are_described_and_required_iff_without_default(
    registry: GeneratorRegistry[Any], definition: GeneratorDef[Any]
) -> None:
    without_default = [k for k, s in definition.params.items() if "default" not in s]
    assert list(definition.required) == without_default
    branch = definition.branch_schema(registry.discriminator)
    assert branch["required"] == [registry.discriminator, *without_default]
    assert branch["additionalProperties"] is False
    assert definition.catalog_entry()["required"] == without_default
    for name, schema in definition.params.items():
        assert schema.get("description", "").strip(), f"{definition.name}.{name}"


@pytest.mark.parametrize(("registry", "definition"), _definitions())
def test_defaults_and_examples_satisfy_their_own_schemas(
    registry: GeneratorRegistry[Any], definition: GeneratorDef[Any]
) -> None:
    for name, schema in definition.params.items():
        if "default" in schema:
            errors = list(Draft202012Validator(dict(schema)).iter_errors(schema["default"]))
            assert not errors, f"{definition.name}.{name} default: {errors[0].message}"
    branch = Draft202012Validator(definition.branch_schema(registry.discriminator))
    for example in definition.examples:
        assert example[registry.discriminator] == definition.name
        errors = list(branch.iter_errors(dict(example)))
        assert not errors, errors[0].message


@pytest.mark.parametrize("registry", REGISTRIES)
def test_union_reports_errors_from_the_selected_branch(registry: GeneratorRegistry[Any]) -> None:
    validator = Draft202012Validator(registry.union_schema())
    for definition in registry:
        example = dict(definition.examples[0])
        assert list(validator.iter_errors(example)) == []
        errors = list(validator.iter_errors({**example, "bogus": 1}))
        assert [e.validator for e in errors] == ["additionalProperties"]
    errors = list(validator.iter_errors({registry.discriminator: "no_such_generator"}))
    assert [e.validator for e in errors] == ["enum"]


def _toy(name: str = "toy", **params: dict[str, Any]) -> GeneratorDef[int]:
    return GeneratorDef[int](name=name, summary="A toy.", params=params, run=lambda p: 1)


def test_register_enforces_names_descriptions_and_uniqueness() -> None:
    registry: GeneratorRegistry[int] = GeneratorRegistry("toy", "kind", "geometry.unknown_op")
    registry.register(_toy(size=number("Size.", default=1.0)))
    with pytest.raises(ValueError, match="registered twice"):
        registry.register(_toy())
    with pytest.raises(ValueError, match="invalid generator name"):
        registry.register(_toy("Bad-Name"))
    with pytest.raises(ValueError, match="needs a description"):
        registry.register(_toy("other", size={"type": "number"}))
    with pytest.raises(ValueError, match="invalid parameter name"):
        registry.register(_toy("other", Size=number("Size.")))
    with pytest.raises(ValueError, match="declares a parameter named"):
        registry.register(_toy("other", kind=number("Kind.")))
    assert registry.names() == ["toy"]


def test_discriminated_union_requires_the_discriminator() -> None:
    schema = discriminated_union("op", {"a": {"type": "object"}}, "Which.")
    Draft202012Validator.check_schema(schema)
    assert list(Draft202012Validator(schema).iter_errors({}))
