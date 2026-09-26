"""The generator abstraction shared by geometry operators and material families.

A generator is a pure function from a small set of named, typed parameters to a
result (a mesh, a material recipe, ...).  Collecting generators instead of asset
files means that a new object differs from an old one by a parameter value, not
by a new file.

Each parameter is declared as a JSON Schema fragment.  The rule is simple and
enforced at registration: a parameter is either required (no ``default``) or
optional with a ``default``, and every parameter has a ``description``.  The
scene JSON Schema, the defaults applied during normalisation and the catalogue
shown to models are all derived from these declarations, so they cannot drift.

Standard library only: the Blender worker imports this module.
"""

from __future__ import annotations

import copy
import re
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

from .errors import JingyuError

T = TypeVar("T")

#: A problem found by a generator's cross-parameter check: (param, message).
ParamProblem = tuple[str, str]

_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{0,31}$")


def _no_problems(params: Mapping[str, Any]) -> list[ParamProblem]:
    return []


@dataclass(frozen=True)
class GeneratorDef(Generic[T]):
    """Declaration of one generator."""

    name: str
    summary: str
    params: Mapping[str, Mapping[str, Any]]
    run: Callable[[Mapping[str, Any]], T]
    check: Callable[[Mapping[str, Any]], list[ParamProblem]] = _no_problems
    examples: tuple[Mapping[str, Any], ...] = ()

    @property
    def required(self) -> tuple[str, ...]:
        return tuple(k for k, s in self.params.items() if "default" not in s)

    def branch_schema(self, discriminator: str) -> dict[str, Any]:
        """JSON Schema for ``{discriminator: name, **params}``."""

        properties: dict[str, Any] = {
            discriminator: {"const": self.name, "description": self.summary}
        }
        properties.update({k: copy.deepcopy(dict(v)) for k, v in self.params.items()})
        return {
            "type": "object",
            "title": self.name,
            "description": self.summary,
            "additionalProperties": False,
            "required": [discriminator, *self.required],
            "properties": properties,
        }

    def catalog_entry(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "summary": self.summary,
            "params": {k: copy.deepcopy(dict(v)) for k, v in self.params.items()},
            "required": list(self.required),
            "examples": [copy.deepcopy(dict(e)) for e in self.examples],
        }


@dataclass
class GeneratorRegistry(Generic[T]):
    """An ordered, name-unique set of generators of one kind."""

    kind: str
    discriminator: str
    unknown_code: str
    _defs: dict[str, GeneratorDef[T]] = field(default_factory=dict)

    def register(self, definition: GeneratorDef[T]) -> GeneratorDef[T]:
        if not _NAME_RE.fullmatch(definition.name):
            raise ValueError(f"invalid generator name {definition.name!r}")
        if definition.name in self._defs:
            raise ValueError(f"{self.kind} generator {definition.name!r} registered twice")
        if self.discriminator in definition.params:
            raise ValueError(
                f"{definition.name!r} declares a parameter named {self.discriminator!r}"
            )
        for pname, pschema in definition.params.items():
            if not _NAME_RE.fullmatch(pname):
                raise ValueError(f"{definition.name}: invalid parameter name {pname!r}")
            if not pschema.get("description"):
                raise ValueError(f"{definition.name}.{pname} needs a description")
        self._defs[definition.name] = definition
        return definition

    def get(self, name: str) -> GeneratorDef[T]:
        try:
            return self._defs[name]
        except KeyError:
            raise JingyuError(
                self.unknown_code,
                f"unknown {self.kind} {name!r}",
                hint=f"available: {', '.join(self._defs)}",
            ) from None

    def names(self) -> list[str]:
        return list(self._defs)

    def __iter__(self) -> Iterator[GeneratorDef[T]]:
        return iter(self._defs.values())

    def union_schema(
        self,
        extra_properties: Mapping[str, Mapping[str, Any]] | None = None,
        extra_required: tuple[str, ...] = (),
    ) -> dict[str, Any]:
        """A discriminated union over all registered generators.

        ``if``/``then`` branches are used instead of ``oneOf`` so that a
        validation error names the one branch the discriminator selected,
        rather than reporting that no branch matched.
        """

        branches: dict[str, dict[str, Any]] = {}
        for definition in self._defs.values():
            branch = definition.branch_schema(self.discriminator)
            for key, value in (extra_properties or {}).items():
                branch["properties"][key] = copy.deepcopy(dict(value))
            branch["required"] = [*extra_required, *branch["required"]]
            branches[definition.name] = branch
        return discriminated_union(self.discriminator, branches, f"Which {self.kind} to use.")

    def catalog(self) -> list[dict[str, Any]]:
        return [d.catalog_entry() for d in self._defs.values()]

    def run(self, spec: Mapping[str, Any]) -> T:
        """Run the generator named by ``spec[discriminator]`` on normalised params."""

        definition = self.get(str(spec[self.discriminator]))
        params = {k: v for k, v in spec.items() if k != self.discriminator}
        return definition.run(params)


def discriminated_union(
    discriminator: str, branches: Mapping[str, Mapping[str, Any]], description: str
) -> dict[str, Any]:
    """JSON Schema selecting one branch by the constant value of *discriminator*.

    ``if``/``then`` is used instead of ``oneOf`` so that a validation error
    names the branch the discriminator selected rather than reporting that no
    branch matched.  Each branch must itself declare the discriminator.
    """

    return {
        "type": "object",
        "required": [discriminator],
        "properties": {discriminator: {"enum": list(branches), "description": description}},
        "allOf": [
            {
                "if": {
                    "properties": {discriminator: {"const": name}},
                    "required": [discriminator],
                },
                "then": copy.deepcopy(dict(branch)),
            }
            for name, branch in branches.items()
        ],
    }


def number(
    description: str,
    *,
    default: float | None = None,
    minimum: float | None = None,
    maximum: float | None = None,
    exclusive_minimum: float | None = None,
) -> dict[str, Any]:
    """Helper for a numeric parameter schema."""

    schema: dict[str, Any] = {"type": "number", "description": description}
    if minimum is not None:
        schema["minimum"] = minimum
    if maximum is not None:
        schema["maximum"] = maximum
    if exclusive_minimum is not None:
        schema["exclusiveMinimum"] = exclusive_minimum
    if default is not None:
        schema["default"] = default
    return schema


def integer(
    description: str,
    *,
    default: int | None = None,
    minimum: int | None = None,
    maximum: int | None = None,
) -> dict[str, Any]:
    """Helper for an integer parameter schema."""

    schema: dict[str, Any] = {"type": "integer", "description": description}
    if minimum is not None:
        schema["minimum"] = minimum
    if maximum is not None:
        schema["maximum"] = maximum
    if default is not None:
        schema["default"] = default
    return schema


__all__ = [
    "GeneratorDef",
    "GeneratorRegistry",
    "ParamProblem",
    "discriminated_union",
    "integer",
    "number",
]
