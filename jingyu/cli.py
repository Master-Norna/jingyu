"""Command line interface.

Every command is a thin wrapper over a registered tool, so the CLI and the MCP
server always behave the same.  Results are printed as JSON on stdout; images a
tool returns are written to files whose paths are added under ``"images"``.

Exit codes: 0 success, 1 a reported failure (JSON ``{"error": ...}`` on stdout,
or an invalid scene for ``validate``), 2 usage errors.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

from . import __version__
from .bridge import DEFAULT_TIMEOUT_S
from .canonical_json import load_file_strict, loads_strict
from .errors import JingyuError
from .scene import scene_schema
from .tools import REGISTRY, ToolContext, ToolResult
from .workspace import Workspace

_SAFE = re.compile(r"[^A-Za-z0-9_.-]+")


def _configure_streams() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8")


def _print(document: Any) -> None:
    sys.stdout.write(json.dumps(document, ensure_ascii=False, indent=2) + "\n")


def _save_images(result: ToolResult, directory: Path, stem: str) -> list[str]:
    if not result.images:
        return []
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for index, image in enumerate(result.images):
        suffix = f"-{index}" if len(result.images) > 1 else ""
        path = directory / f"{_SAFE.sub('_', stem)}{suffix}.png"
        path.write_bytes(image.png)
        paths.append(str(path))
    return paths


def _invoke(
    context: ToolContext, name: str, arguments: dict[str, Any], image_dir: Path, stem: str
) -> dict[str, Any]:
    result = REGISTRY.invoke(name, arguments, context)
    data = dict(result.data)
    saved = _save_images(result, image_dir, stem)
    if saved:
        data["images"] = saved
    return data


def _parse_args_json(raw: str | None) -> dict[str, Any]:
    if raw is None:
        return {}
    document = _read_local_json(raw[1:]) if raw.startswith("@") else loads_strict(raw)
    if not isinstance(document, dict):
        raise JingyuError("tool.invalid_arguments", "--args must be a JSON object")
    return document


def _read_local_json(path: str) -> Any:
    """Files named on the command line (scenes, ``--args @file``) are ordinary local paths
    (relative to the current directory), not workspace paths; their content is passed to
    the tool inline."""

    try:
        return load_file_strict(Path(path))
    except OSError as exc:
        raise JingyuError("io.unreadable", f"cannot read {path}: {exc.strerror or exc}") from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="jingyu", description="Jingyu image toolkit")
    parser.add_argument("--version", action="version", version=f"jingyu {__version__}")
    parser.add_argument("--workspace", help="workspace root (default: $JINGYU_WORKSPACE or cwd)")
    parser.add_argument(
        "--timeout", type=float, default=DEFAULT_TIMEOUT_S, help="render timeout in seconds"
    )
    parser.add_argument(
        "--image-dir", help="where returned images are written (default: <workspace>/views)"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("tools", help="list the available tools")

    call = sub.add_parser("call", help="call any tool with JSON arguments")
    call.add_argument("tool")
    call.add_argument("--args", help="JSON object, or @file.json")

    validate = sub.add_parser("validate", help="validate a scene file")
    validate.add_argument("scene")
    validate.add_argument(
        "--normalized", action="store_true", help="also print the normalised scene"
    )

    render = sub.add_parser("render", help="render a scene file into a new candidate")
    render.add_argument("scene")
    render.add_argument("--final", action="store_true", help="full resolution and samples")
    render.add_argument(
        "--hide", nargs="*", default=[], metavar="ID", help="object ids to leave out"
    )

    view = sub.add_parser("view", help="write one view of a candidate to a PNG")
    view.add_argument("candidate")
    view.add_argument("view")
    view.add_argument("--other", help="second candidate for the compare view")
    view.add_argument("--levels", type=int, default=5)
    view.add_argument("--max-size", type=int, default=1024)

    locate = sub.add_parser("locate", help="what is at a pixel or in a region")
    locate.add_argument("candidate")
    where = locate.add_mutually_exclusive_group(required=True)
    where.add_argument("--xy", nargs=2, type=int, metavar=("X", "Y"))
    where.add_argument("--uv", nargs=2, type=float, metavar=("U", "V"))
    where.add_argument("--box", nargs=4, type=int, metavar=("X0", "Y0", "X1", "Y1"))

    layout = sub.add_parser("layout", help="how objects are distributed in the frame")
    layout.add_argument("candidate")

    doctor = sub.add_parser("doctor", help="diagnose the environment")
    doctor.add_argument("--deep", action="store_true", help="test-render with every engine")

    schema = sub.add_parser("schema", help="print the scene JSON Schema")
    schema.add_argument("--write", metavar="PATH", help="write it to a file instead")
    return parser


def _command_arguments(args: argparse.Namespace) -> tuple[str, dict[str, Any], str]:
    command = args.command
    if command == "call":
        return args.tool, _parse_args_json(args.args), args.tool
    if command == "validate":
        return (
            "validate_scene",
            {"scene": _read_local_json(args.scene), "include_normalized": args.normalized},
            "validate",
        )
    if command == "render":
        return (
            "render_scene",
            {
                "scene": _read_local_json(args.scene),
                "quality": "final" if args.final else "preview",
                "hide": args.hide,
                "return_image": False,
            },
            "render",
        )
    if command == "view":
        arguments: dict[str, Any] = {
            "candidate_id": args.candidate,
            "view": args.view,
            "levels": args.levels,
            "max_size": args.max_size,
        }
        if args.other:
            arguments["other_candidate_id"] = args.other
        return "view_candidate", arguments, f"{args.candidate}-{args.view}"
    if command == "locate":
        arguments = {"candidate_id": args.candidate}
        if args.xy:
            arguments["point"] = {"x": args.xy[0], "y": args.xy[1]}
        elif args.uv:
            arguments["point"] = {"u": args.uv[0], "v": args.uv[1]}
        else:
            x0, y0, x1, y1 = args.box
            arguments["region"] = {"x0": x0, "y0": y0, "x1": x1, "y1": y1}
        return "locate_in_candidate", arguments, "locate"
    if command == "layout":
        return "describe_layout", {"candidate_id": args.candidate}, "layout"
    if command == "doctor":
        return "diagnose_environment", {"deep": args.deep}, "doctor"
    raise AssertionError(command)


def main(argv: list[str] | None = None) -> int:
    _configure_streams()
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "tools":
        _print([{"name": t.name, "title": t.title, "description": t.description} for t in REGISTRY])
        return 0
    if args.command == "schema":
        text = json.dumps(scene_schema(), ensure_ascii=False, indent=2) + "\n"
        if args.write:
            Path(args.write).write_text(text, encoding="utf-8")
        else:
            sys.stdout.write(text)
        return 0

    try:
        workspace = Workspace.at(args.workspace)
        context = ToolContext(workspace, render_timeout_s=args.timeout)
        image_dir = Path(args.image_dir) if args.image_dir else workspace.root / "views"
        name, arguments, stem = _command_arguments(args)
        data = _invoke(context, name, arguments, image_dir, stem)
    except JingyuError as exc:
        _print({"error": exc.to_dict()})
        return 1
    _print(data)
    if args.command == "validate" and not data.get("valid"):
        return 1
    if args.command == "doctor" and not data.get("ok"):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
