"""A transport-independent tool registry.

Every capability exposed to people and models is declared once as a
:class:`Tool`: a name, a description, JSON Schemas for input and output and a
handler.  The CLI and the MCP server are thin adapters over this registry, so
both always offer the same tools with the same contracts.

Arguments are validated against the input schema before the handler runs and
the structured result is validated against the output schema after, so a
contract violation is caught at the boundary with ``tool.invalid_arguments`` or
``tool.invalid_output`` instead of surfacing as a confusing downstream error.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any

from ..errors import JingyuError, pointer_join
from ..schema_validation import Validator, check_schema
from ..workspace import Workspace

_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{1,63}$")


@dataclass(frozen=True)
class ImagePayload:
    """An image returned alongside a tool's structured result."""

    png: bytes
    caption: str
    mime_type: str = "image/png"


@dataclass(frozen=True)
class ToolResult:
    data: dict[str, Any]
    images: tuple[ImagePayload, ...] = ()


@dataclass(frozen=True)
class ToolContext:
    """What a handler may touch: the workspace and a render timeout."""

    workspace: Workspace
    render_timeout_s: float = 900.0


Handler = Callable[[ToolContext, dict[str, Any]], ToolResult]


@dataclass(frozen=True)
class Tool:
    name: str
    title: str
    description: str
    input_schema: dict[str, Any]
    output_schema: dict[str, Any]
    handler: Handler
    read_only: bool = True
    destructive: bool = False
    idempotent: bool = True
    returns_images: bool = False

    @cached_property
    def _input_validator(self) -> Any:
        return Validator(self.input_schema)

    @cached_property
    def _output_validator(self) -> Any:
        return Validator(self.output_schema)

    def validate_arguments(self, arguments: Mapping[str, Any]) -> None:
        errors = sorted(
            self._input_validator.iter_errors(dict(arguments)),
            key=lambda e: list(map(str, e.absolute_path)),
        )
        if errors:
            raise JingyuError(
                "tool.invalid_arguments",
                f"{self.name}: {errors[0].message}",
                hint="See the tool's input schema.",
                details={
                    "errors": [
                        {"pointer": pointer_join(*e.absolute_path), "message": e.message}
                        for e in errors[:20]
                    ]
                },
            )

    def validate_output(self, data: Mapping[str, Any]) -> None:
        error = next(iter(self._output_validator.iter_errors(dict(data))), None)
        if error is not None:
            raise JingyuError(
                "tool.invalid_output",
                f"{self.name} produced invalid output at "
                f"{pointer_join(*error.absolute_path) or '/'}: {error.message}",
            )


@dataclass
class ToolRegistry:
    _tools: dict[str, Tool] = field(default_factory=dict)

    def add(self, tool: Tool) -> Tool:
        if not _NAME_RE.fullmatch(tool.name):
            raise ValueError(f"invalid tool name {tool.name!r}")
        if tool.name in self._tools:
            raise ValueError(f"tool {tool.name!r} registered twice")
        for label, schema in (("input", tool.input_schema), ("output", tool.output_schema)):
            check_schema(schema)
            if schema.get("type") != "object":
                raise ValueError(f"{tool.name}: {label} schema must describe an object")
        if tool.input_schema.get("additionalProperties") is not False:
            raise ValueError(f"{tool.name}: input schema must set additionalProperties false")
        self._tools[tool.name] = tool
        return tool

    def get(self, name: str) -> Tool:
        try:
            return self._tools[name]
        except KeyError:
            raise JingyuError(
                "tool.unknown", f"no tool named {name!r}", hint=f"tools: {', '.join(self._tools)}"
            ) from None

    def __iter__(self) -> Iterator[Tool]:
        return iter(self._tools.values())

    def names(self) -> list[str]:
        return list(self._tools)

    def invoke(
        self, name: str, arguments: Mapping[str, Any] | None, context: ToolContext
    ) -> ToolResult:
        tool = self.get(name)
        args = dict(arguments or {})
        tool.validate_arguments(args)
        result = tool.handler(context, args)
        tool.validate_output(result.data)
        return result


__all__ = ["Handler", "ImagePayload", "Tool", "ToolContext", "ToolRegistry", "ToolResult"]
