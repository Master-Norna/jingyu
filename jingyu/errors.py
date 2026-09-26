"""Stable, machine-readable errors and issues.

Every failure that crosses a process or tool boundary carries a stable dotted
ASCII ``code`` from :data:`ERROR_CODES`.  Callers (people, CLIs and models)
branch on the code; the message is for humans and may change between versions.

This module is standard-library only so that the Blender worker can raise the
same codes the host understands.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

Severity = Literal["error", "warning"]

#: code -> one-line meaning.  ``docs/错误码.md`` is generated from this table and
#: a test fails if a raised code is missing here.
ERROR_CODES: dict[str, str] = {
    # JSON documents
    "json.invalid": "The document is not valid JSON.",
    "json.duplicate_key": "A JSON object repeats a key; the document is ambiguous.",
    "json.non_finite_number": "NaN or Infinity appeared where JSON requires a finite number.",
    # Scene description
    "spec.schema_violation": "The scene does not match the scene JSON Schema.",
    "spec.unsupported_schema": "The scene declares a schema version this build does not support.",
    "spec.duplicate_id": "Two scene entities share the same id.",
    "spec.unknown_reference": "A field refers to an id that does not exist or has the wrong kind.",
    "spec.invalid_parameter": "A generator parameter is out of its valid range or inconsistent.",
    "spec.degenerate_aim": "A camera or light looks at its own location.",
    "spec.invalid_range": "A lower bound is not below its upper bound.",
    "spec.invalid": "The scene has validation errors; see details.issues.",
    "spec.scene_unlit": "Warning: nothing lights the scene, the image will be black.",
    "spec.empty_scene": "Warning: the scene has no visible objects.",
    # Generators
    "geometry.unknown_op": "The geometry operator is not registered.",
    "geometry.invalid_profile": "A lathe profile cannot be revolved into a valid solid.",
    "material.unknown_family": "The material family is not registered.",
    # Blender runtime
    "blender.not_found": "No Blender runtime (executable or bpy module) could be found.",
    "blender.unsupported_version": "The Blender runtime is older than the supported minimum.",
    "blender.timeout": "The Blender worker did not finish within the time limit.",
    "blender.crashed": "The Blender worker exited without writing a response.",
    "blender.bad_response": "The Blender worker wrote a response that violates the protocol.",
    "blender.build_failed": "Building the Blender scene from the description failed.",
    "blender.render_failed": "Blender failed while rendering.",
    "blender.engine_unavailable": "The requested render engine is not available in this runtime.",
    "blender.gpu_unavailable": "A GPU device was required but none is usable.",
    "worker.bad_request": "The worker request violates the worker protocol.",
    # Candidates
    "candidate.not_found": "No candidate with this id exists in the workspace.",
    "candidate.invalid_id": "The candidate id is malformed.",
    "candidate.integrity_mismatch": "Candidate files do not match their receipt.",
    "candidate.missing_pass": "The candidate has no pass required for this operation.",
    # Locating and viewing
    "locate.out_of_bounds": "The requested point or region lies outside the image.",
    "locate.ambiguous_pixels": "The id mask contains pixels that decode to no object.",
    "view.unknown_view": "The requested view is not supported.",
    # Workspace and tools
    "workspace.path_escape": "A path points outside the workspace root.",
    "workspace.not_found": "A path inside the workspace does not exist.",
    "io.unreadable": "A file cannot be read: it is missing, a directory, or not permitted.",
    "tool.unknown": "No tool with this name is registered.",
    "tool.invalid_arguments": "Tool arguments do not match the tool's input schema.",
    "tool.invalid_output": "A tool produced output that violates its output schema.",
    "constitution.unknown_clause": "The constitution has no clause or section with this id.",
    "internal.unexpected": "An unexpected internal error occurred.",
}


@dataclass(frozen=True)
class Issue:
    """One validation finding, located by a JSON Pointer into the document."""

    code: str
    message: str
    pointer: str = ""
    severity: Severity = "error"
    hint: str | None = None

    def __post_init__(self) -> None:
        _require_known(self.code)

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "code": self.code,
            "severity": self.severity,
            "pointer": self.pointer,
            "message": self.message,
        }
        if self.hint:
            data["hint"] = self.hint
        return data


@dataclass
class JingyuError(Exception):
    """An expected failure with a stable code, safe to show to people and models."""

    code: str
    message: str
    hint: str | None = None
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_known(self.code)
        super().__init__(f"[{self.code}] {self.message}")

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.hint:
            data["hint"] = self.hint
        if self.details:
            data["details"] = dict(self.details)
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> JingyuError:
        code = data.get("code")
        if not isinstance(code, str) or code not in ERROR_CODES:
            return cls(
                "blender.bad_response",
                f"unrecognised error payload: {dict(data)!r}",
            )
        details = data.get("details")
        return cls(
            code,
            str(data.get("message", "")),
            hint=data.get("hint") if isinstance(data.get("hint"), str) else None,
            details=details if isinstance(details, Mapping) else {},
        )


def pointer_join(*parts: str | int) -> str:
    """Build an RFC 6901 JSON Pointer from path parts."""

    escaped = (str(p).replace("~", "~0").replace("/", "~1") for p in parts)
    return "".join("/" + p for p in escaped)


def _require_known(code: str) -> None:
    if code not in ERROR_CODES:
        raise ValueError(f"unregistered error code {code!r}; add it to ERROR_CODES")


__all__ = [
    "ERROR_CODES",
    "Issue",
    "JingyuError",
    "Severity",
    "pointer_join",
]
