"""The built-in tools, shared verbatim by the CLI and the MCP server."""

from __future__ import annotations

import copy
from typing import Any

from .. import __version__, conventions
from .. import constitution as constitution_mod
from ..candidate import CandidateStore
from ..canonical_json import load_file_strict
from ..doctor import diagnose
from ..errors import ERROR_CODES, JingyuError
from ..geometry import GEOMETRY
from ..locate import IdMask
from ..materials import MATERIALS
from ..render import render_scene
from ..scene import SCENE_SCHEMA, minimal_scene, scene_schema, validate_scene
from ..views import VIEWS, render_view
from .registry import ImagePayload, Tool, ToolContext, ToolRegistry, ToolResult

# ------------------------------------------------------------------ schemas

_ANY_OBJECT: dict[str, Any] = {"type": "object"}
_STRING = {"type": "string"}
_CANDIDATE_ID = {
    "type": "string",
    "description": "A candidate id such as c_20260926T120000Z_1a2b3c4d_5e6f.",
}
_ISSUE = {
    "type": "object",
    "required": ["code", "severity", "pointer", "message"],
    "additionalProperties": False,
    "properties": {
        "code": _STRING,
        "severity": {"enum": ["error", "warning"]},
        "pointer": {"type": "string", "description": "JSON Pointer into the scene."},
        "message": _STRING,
        "hint": _STRING,
    },
}
_SCENE_SOURCE_PROPS: dict[str, Any] = {
    "scene": {"type": "object", "description": "The scene description (jingyu.scene.v1)."},
    "scene_path": {
        "type": "string",
        "description": "Path of a scene JSON file inside the workspace.",
    },
}
_SCENE_SOURCE_ONE_OF = [{"required": ["scene"]}, {"required": ["scene_path"]}]
_CANDIDATE_SUMMARY = {
    "type": "object",
    "required": ["candidate_id", "created_at", "scene_id", "scene_sha256", "quality", "engine"],
    "properties": {
        "candidate_id": _STRING,
        "created_at": _STRING,
        "scene_id": _STRING,
        "scene_sha256": _STRING,
        "quality": {"enum": ["preview", "final"]},
        "engine": _STRING,
        "resolution": {"type": "array", "items": {"type": "integer"}},
    },
}


def _input(
    properties: dict[str, Any], required: list[str] | None = None, **extra: Any
) -> dict[str, Any]:
    schema: dict[str, Any] = {
        "type": "object",
        "additionalProperties": False,
        "properties": properties,
    }
    if required:
        schema["required"] = required
    schema.update(extra)
    return schema


def _output(properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {"type": "object", "properties": properties, "required": required}


# ------------------------------------------------------------------ helpers


def _load_scene(ctx: ToolContext, args: dict[str, Any]) -> Any:
    if "scene" in args:
        return copy.deepcopy(args["scene"])
    path = ctx.workspace.resolve(args["scene_path"])
    try:
        return load_file_strict(path)
    except OSError as exc:
        reason = exc.strerror or str(exc)
        raise JingyuError("io.unreadable", f"cannot read {args['scene_path']}: {reason}") from exc


def _hide(scene: Any, hidden: list[str]) -> Any:
    if not hidden or not isinstance(scene, dict):
        return scene
    objects = scene.get("objects", [])
    known = {o.get("id") for o in objects if isinstance(o, dict)}
    missing = sorted(set(hidden) - known)
    if missing:
        raise JingyuError(
            "spec.unknown_reference",
            f"cannot hide unknown object(s): {', '.join(missing)}",
            hint=f"objects: {', '.join(sorted(str(k) for k in known))}",
        )
    for obj in objects:
        if obj.get("id") in hidden:
            obj["visible"] = False
    return scene


def _store(ctx: ToolContext) -> CandidateStore:
    return CandidateStore(ctx.workspace)


# ------------------------------------------------------------------ handlers

GUIDE_WORKFLOW = [
    "Read the conventions below and the primitive catalogue (list_generators). Shapes are "
    "geometry operators with parameters, surfaces are material families with parameters; "
    "a new object is a new parameter line, never a new file.",
    "Write a scene description (schema jingyu.scene.v1, see get_scene_schema) and run "
    "validate_scene until it is valid. Issues carry JSON Pointers and hints.",
    "render_scene with quality 'preview' first. Every render is a new immutable candidate; "
    "never edit candidate files, change the scene and render again.",
    "Review the whole before the parts (constitution J1): view_candidate with glance, "
    "squint, grayscale or values, and flip; compare with the previous candidate; "
    "describe_layout shows how objects are distributed in the frame.",
    "If something looks wrong but you cannot name it, point at it: locate_in_candidate "
    "returns the object, material and the JSON Pointers to edit.",
    "To test whether an element earns its place, render again with hide=[its id] and "
    "compare (subtraction review, J7.1 four).",
    "get_constitution returns the index or a few clauses; use clauses to ask questions, "
    "not as reasons. Render 'final' once the preview holds.",
]


def _get_guide(ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
    return ToolResult(
        {
            "jingyu_version": __version__,
            "summary": (
                "Jingyu builds images from declarative 3D scenes rendered by Blender. "
                "Geometry and materials come from parametric generators, every render "
                "is an immutable candidate, and each pixel can be traced back to the "
                "scene entry that produced it."
            ),
            "workflow": GUIDE_WORKFLOW,
            "conventions": conventions.summary(),
            "scene_schema": SCENE_SCHEMA,
            "tools": [{"name": t.name, "title": t.title} for t in REGISTRY],
            "workspace": str(ctx.workspace.root),
        }
    )


def _get_scene_schema(ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
    return ToolResult(
        {
            "schema_id": SCENE_SCHEMA,
            "json_schema": scene_schema(),
            "example": minimal_scene(),
            "conventions": conventions.summary(),
        }
    )


def _list_generators(ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
    kind = args.get("kind", "all")
    data: dict[str, Any] = {}
    if kind in ("all", "geometry"):
        data["geometry"] = GEOMETRY.catalog()
    if kind in ("all", "materials"):
        data["materials"] = MATERIALS.catalog()
    return ToolResult(data)


def _validate_scene(ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
    result = validate_scene(_load_scene(ctx, args))
    data: dict[str, Any] = {
        "valid": result.valid,
        "issues": [i.to_dict() for i in result.issues],
        "scene_sha256": result.scene_sha256,
    }
    if args.get("include_normalized") and result.normalized is not None:
        data["normalized"] = result.normalized
    return ToolResult(data)


def _render_scene(ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
    scene = _hide(_load_scene(ctx, args), args.get("hide", []))
    outcome = render_scene(
        ctx.workspace,
        scene,
        quality=args.get("quality", "preview"),
        timeout_s=ctx.render_timeout_s,
    )
    candidate = outcome.candidate
    receipt = candidate.receipt
    images: tuple[ImagePayload, ...] = ()
    if args.get("return_image", True):
        view = render_view(candidate, "full", max_size=args.get("max_image_size", 1024))
        images = (ImagePayload(view.png, f"{candidate.id} ({receipt['quality']})"),)
    return ToolResult(
        {
            "candidate_id": candidate.id,
            "path": ctx.workspace.relative(candidate.path),
            "scene_sha256": receipt["scene"]["sha256"],
            "quality": receipt["quality"],
            "render": receipt["render"],
            "timings_ms": receipt["timings_ms"],
            "warnings": receipt["warnings"],
        },
        images,
    )


def _list_candidates(ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
    candidates = _store(ctx).list(limit=args.get("limit", 20))
    return ToolResult({"candidates": [c.summary() for c in candidates]})


def _get_candidate(ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
    store = _store(ctx)
    candidate = store.open(args["candidate_id"])
    problems = store.verify(candidate)
    return ToolResult(
        {
            "candidate_id": candidate.id,
            "path": ctx.workspace.relative(candidate.path),
            "receipt": candidate.receipt,
            "integrity": {"ok": not problems, "problems": problems},
        }
    )


def _view_candidate(ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
    store = _store(ctx)
    candidate = store.open(args["candidate_id"])
    other = store.open(args["other_candidate_id"]) if "other_candidate_id" in args else None
    view = render_view(
        candidate,
        args["view"],
        max_size=args.get("max_size", 1024),
        levels=args.get("levels", 5),
        other=other,
    )
    return ToolResult(
        {
            "candidate_id": candidate.id,
            "view": args["view"],
            "description": VIEWS[args["view"]],
            "width": view.width,
            "height": view.height,
            "meta": view.meta,
        },
        (ImagePayload(view.png, f"{candidate.id} [{args['view']}]"),),
    )


def _locate(ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
    candidate = _store(ctx).open(args["candidate_id"])
    mask = IdMask.load(candidate)
    if "point" in args:
        result = mask.locate_point(args["point"], radius=args.get("radius", 3))
        mode = "point"
    else:
        result = mask.locate_region(args["region"])
        mode = "region"
    return ToolResult({"candidate_id": candidate.id, "mode": mode, "result": result})


def _describe_layout(ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
    candidate = _store(ctx).open(args["candidate_id"])
    return ToolResult({"candidate_id": candidate.id, "layout": IdMask.load(candidate).layout()})


def _get_constitution(ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
    document = constitution_mod.load()
    data: dict[str, Any] = {"version": document.version, "sha256": document.sha256}
    ids = args.get("ids")
    if ids:
        data["entries"] = document.lookup(list(ids))
    else:
        data["index"] = document.index()
    return ToolResult(data)


def _diagnose_environment(ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
    return ToolResult(diagnose(deep=bool(args.get("deep", False))))


def _list_error_codes(ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
    return ToolResult({"codes": [{"code": k, "meaning": v} for k, v in ERROR_CODES.items()]})


# ----------------------------------------------------------------- registry

_POINT = {
    "type": "object",
    "additionalProperties": False,
    "description": "Pixel x/y from the top-left, or normalised u/v in [0, 1].",
    "properties": {
        "x": {"type": "integer", "minimum": 0},
        "y": {"type": "integer", "minimum": 0},
        "u": {"type": "number", "minimum": 0, "maximum": 1},
        "v": {"type": "number", "minimum": 0, "maximum": 1},
    },
    "oneOf": [{"required": ["x", "y"]}, {"required": ["u", "v"]}],
}
_REGION = {
    "type": "object",
    "additionalProperties": False,
    "description": "Pixel box x0/y0/x1/y1 (x1, y1 exclusive) or normalised u0/v0/u1/v1.",
    "properties": {
        **{k: {"type": "integer", "minimum": 0} for k in ("x0", "y0", "x1", "y1")},
        **{k: {"type": "number", "minimum": 0, "maximum": 1} for k in ("u0", "v0", "u1", "v1")},
    },
    "oneOf": [{"required": ["x0", "y0", "x1", "y1"]}, {"required": ["u0", "v0", "u1", "v1"]}],
}

REGISTRY = ToolRegistry()

REGISTRY.add(
    Tool(
        name="get_guide",
        title="Get the Jingyu guide",
        description=(
            "Start here: what Jingyu is, the recommended workflow, conventions and the tool list."
        ),
        input_schema=_input({}),
        output_schema=_output(
            {"workflow": {"type": "array", "items": _STRING}, "conventions": _ANY_OBJECT},
            ["summary", "workflow", "conventions", "tools"],
        ),
        handler=_get_guide,
    )
)
REGISTRY.add(
    Tool(
        name="get_scene_schema",
        title="Get the scene schema",
        description=(
            "The JSON Schema of a scene description, a minimal valid example and the conventions."
        ),
        input_schema=_input({}),
        output_schema=_output(
            {"schema_id": _STRING, "json_schema": _ANY_OBJECT, "example": _ANY_OBJECT},
            ["schema_id", "json_schema", "example", "conventions"],
        ),
        handler=_get_scene_schema,
    )
)
REGISTRY.add(
    Tool(
        name="list_generators",
        title="List geometry operators and material families",
        description=(
            "The primitive catalogue: every geometry operator and material family with its "
            "parameters, units, ranges, defaults and examples."
        ),
        input_schema=_input({"kind": {"enum": ["all", "geometry", "materials"], "default": "all"}}),
        output_schema=_output({"geometry": {"type": "array"}, "materials": {"type": "array"}}, []),
        handler=_list_generators,
    )
)
REGISTRY.add(
    Tool(
        name="validate_scene",
        title="Validate a scene",
        description=(
            "Check a scene description without rendering. Returns every issue with a JSON "
            "Pointer and a hint, and the scene's content hash when valid."
        ),
        input_schema=_input(
            {
                **_SCENE_SOURCE_PROPS,
                "include_normalized": {
                    "type": "boolean",
                    "default": False,
                    "description": "Also return the scene with every default filled in.",
                },
            },
            oneOf=_SCENE_SOURCE_ONE_OF,
        ),
        output_schema=_output(
            {
                "valid": {"type": "boolean"},
                "issues": {"type": "array", "items": _ISSUE},
                "scene_sha256": {"type": ["string", "null"]},
                "normalized": _ANY_OBJECT,
            },
            ["valid", "issues", "scene_sha256"],
        ),
        handler=_validate_scene,
    )
)
REGISTRY.add(
    Tool(
        name="render_scene",
        title="Render a scene",
        description=(
            "Validate and render a scene in an isolated Blender process, creating a new "
            "immutable candidate. Returns the candidate id and, by default, the image."
        ),
        input_schema=_input(
            {
                **_SCENE_SOURCE_PROPS,
                "quality": {
                    "enum": ["preview", "final"],
                    "default": "preview",
                    "description": "preview: half resolution and at most 16 samples.",
                },
                "hide": {
                    "type": "array",
                    "items": _STRING,
                    "uniqueItems": True,
                    "default": [],
                    "description": "Object ids to leave out of this render (subtraction review).",
                },
                "return_image": {"type": "boolean", "default": True},
                "max_image_size": {
                    "type": "integer",
                    "minimum": 64,
                    "maximum": 4096,
                    "default": 1024,
                },
            },
            oneOf=_SCENE_SOURCE_ONE_OF,
        ),
        output_schema=_output(
            {
                "candidate_id": _STRING,
                "path": _STRING,
                "scene_sha256": _STRING,
                "quality": {"enum": ["preview", "final"]},
                "render": _ANY_OBJECT,
                "timings_ms": _ANY_OBJECT,
                "warnings": {"type": "array"},
            },
            ["candidate_id", "path", "scene_sha256", "quality", "render", "timings_ms", "warnings"],
        ),
        handler=_render_scene,
        read_only=False,
        idempotent=False,
        returns_images=True,
    )
)
REGISTRY.add(
    Tool(
        name="list_candidates",
        title="List candidates",
        description="Recent render candidates in the workspace, newest first.",
        input_schema=_input(
            {"limit": {"type": "integer", "minimum": 1, "maximum": 500, "default": 20}}
        ),
        output_schema=_output(
            {"candidates": {"type": "array", "items": _CANDIDATE_SUMMARY}}, ["candidates"]
        ),
        handler=_list_candidates,
    )
)
REGISTRY.add(
    Tool(
        name="get_candidate",
        title="Get a candidate",
        description="A candidate's receipt and a fresh integrity check of all its files.",
        input_schema=_input({"candidate_id": _CANDIDATE_ID}, ["candidate_id"]),
        output_schema=_output(
            {
                "candidate_id": _STRING,
                "path": _STRING,
                "receipt": _ANY_OBJECT,
                "integrity": {
                    "type": "object",
                    "required": ["ok", "problems"],
                    "properties": {
                        "ok": {"type": "boolean"},
                        "problems": {"type": "array", "items": _STRING},
                    },
                },
            },
            ["candidate_id", "path", "receipt", "integrity"],
        ),
        handler=_get_candidate,
    )
)
REGISTRY.add(
    Tool(
        name="view_candidate",
        title="View a candidate",
        description=(
            "Look at a candidate one of several ways: full, glance, flip, squint, grayscale, "
            "values, id_mask, or compare (beside other_candidate_id)."
        ),
        input_schema=_input(
            {
                "candidate_id": _CANDIDATE_ID,
                "view": {"enum": list(VIEWS), "description": "Which view."},
                "max_size": {"type": "integer", "minimum": 64, "maximum": 4096, "default": 1024},
                "levels": {
                    "type": "integer",
                    "minimum": 2,
                    "maximum": 16,
                    "default": 5,
                    "description": "Value steps for the values view.",
                },
                "other_candidate_id": _CANDIDATE_ID,
            },
            ["candidate_id", "view"],
        ),
        output_schema=_output(
            {
                "candidate_id": _STRING,
                "view": _STRING,
                "description": _STRING,
                "width": {"type": "integer"},
                "height": {"type": "integer"},
                "meta": _ANY_OBJECT,
            },
            ["candidate_id", "view", "width", "height", "meta"],
        ),
        handler=_view_candidate,
        returns_images=True,
    )
)
REGISTRY.add(
    Tool(
        name="locate_in_candidate",
        title="Locate what is at a point or in a region",
        description=(
            "Point at the image and get the scene back: the object, its material and the "
            "JSON Pointers to edit, from the candidate's exact id mask."
        ),
        input_schema=_input(
            {
                "candidate_id": _CANDIDATE_ID,
                "point": _POINT,
                "region": _REGION,
                "radius": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 64,
                    "default": 3,
                    "description": "Neighbourhood radius in pixels for point queries.",
                },
            },
            ["candidate_id"],
            oneOf=[{"required": ["point"]}, {"required": ["region"]}],
        ),
        output_schema=_output(
            {"candidate_id": _STRING, "mode": {"enum": ["point", "region"]}, "result": _ANY_OBJECT},
            ["candidate_id", "mode", "result"],
        ),
        handler=_locate,
    )
)
REGISTRY.add(
    Tool(
        name="describe_layout",
        title="Describe the layout of a candidate",
        description=(
            "How the frame is distributed: each visible object's area, bounding box and "
            "centroid, the background share, and which objects are out of frame."
        ),
        input_schema=_input({"candidate_id": _CANDIDATE_ID}, ["candidate_id"]),
        output_schema=_output(
            {"candidate_id": _STRING, "layout": _ANY_OBJECT}, ["candidate_id", "layout"]
        ),
        handler=_describe_layout,
    )
)
REGISTRY.add(
    Tool(
        name="get_constitution",
        title="Read the visual constitution",
        description=(
            "Without ids: the index of sections and clauses. With up to 12 ids (such as "
            "J1.03 or J7.1): their text. Use clauses to raise questions, not as reasons."
        ),
        input_schema=_input(
            {
                "ids": {
                    "type": "array",
                    "items": {"type": "string", "pattern": r"^J\d+(\.\d+)*$"},
                    "maxItems": constitution_mod.MAX_IDS,
                    "uniqueItems": True,
                }
            }
        ),
        output_schema=_output(
            {
                "version": _STRING,
                "sha256": _STRING,
                "index": _ANY_OBJECT,
                "entries": {"type": "array"},
            },
            ["version", "sha256"],
        ),
        handler=_get_constitution,
    )
)
REGISTRY.add(
    Tool(
        name="diagnose_environment",
        title="Diagnose the environment",
        description=(
            "Versions, the Blender runtime in use and its GPUs. With deep=true also "
            "renders a tiny scene with each engine and reports which work."
        ),
        input_schema=_input({"deep": {"type": "boolean", "default": False}}),
        output_schema=_output(
            {"ok": {"type": "boolean"}, "problems": {"type": "array"}}, ["ok", "problems"]
        ),
        handler=_diagnose_environment,
    )
)
REGISTRY.add(
    Tool(
        name="list_error_codes",
        title="List error codes",
        description="Every stable error code with its meaning.",
        input_schema=_input({}),
        output_schema=_output({"codes": {"type": "array"}}, ["codes"]),
        handler=_list_error_codes,
    )
)

__all__ = ["GUIDE_WORKFLOW", "REGISTRY"]
