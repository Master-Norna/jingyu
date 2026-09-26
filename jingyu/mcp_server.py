"""MCP server over stdio, exposing the tool registry.

This is the only module that imports the optional ``mcp`` package.  It uses the
low-level server so that each tool's hand-written JSON Schemas are advertised
exactly as declared; arguments are validated by the registry, not by
signature introspection.
"""

from __future__ import annotations

import argparse
import base64
import json
import traceback
from typing import Any

import anyio
import anyio.to_thread
import mcp_types as types
from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server

from . import __version__
from .bridge import DEFAULT_TIMEOUT_S
from .bridge.resident import WorkerPool
from .errors import JingyuError
from .tools import REGISTRY, Tool, ToolContext, ToolResult
from .workspace import Workspace

INSTRUCTIONS = (
    "Jingyu renders images from declarative 3D scenes. Call get_guide first. "
    "Shapes and materials are parametric generators (list_generators). Validate, render "
    "a preview, review it (view_candidate: glance, squint, values, flip, compare), point "
    "at problems with locate_in_candidate, edit the scene and render again."
)


def mcp_tool(tool: Tool) -> types.Tool:
    return types.Tool(
        name=tool.name,
        title=tool.title,
        description=tool.description,
        input_schema=tool.input_schema,
        output_schema=tool.output_schema,
        annotations=types.ToolAnnotations(
            title=tool.title,
            read_only_hint=tool.read_only,
            destructive_hint=tool.destructive,
            idempotent_hint=tool.idempotent,
            open_world_hint=False,
        ),
    )


def success_result(result: ToolResult) -> types.CallToolResult:
    content: list[Any] = [
        types.TextContent(type="text", text=json.dumps(result.data, ensure_ascii=False))
    ]
    for image in result.images:
        content.append(types.TextContent(type="text", text=image.caption))
        content.append(
            types.ImageContent(
                type="image",
                data=base64.b64encode(image.png).decode("ascii"),
                mime_type=image.mime_type,
            )
        )
    return types.CallToolResult(content=content, structured_content=result.data, is_error=False)


def error_result(error: JingyuError) -> types.CallToolResult:
    payload = {"error": error.to_dict()}
    return types.CallToolResult(
        content=[types.TextContent(type="text", text=json.dumps(payload, ensure_ascii=False))],
        is_error=True,
    )


def build_server(context: ToolContext) -> Server[Any]:
    async def list_tools(ctx: Any, params: Any) -> types.ListToolsResult:
        return types.ListToolsResult(tools=[mcp_tool(t) for t in REGISTRY])

    async def call_tool(ctx: Any, params: types.CallToolRequestParams) -> types.CallToolResult:
        arguments = dict(params.arguments or {})
        try:
            result = await anyio.to_thread.run_sync(
                REGISTRY.invoke, params.name, arguments, context
            )
        except JingyuError as exc:
            return error_result(exc)
        except Exception as exc:  # never let one tool crash the server
            return error_result(
                JingyuError(
                    "internal.unexpected",
                    f"{type(exc).__name__}: {exc}",
                    details={"traceback": traceback.format_exc()},
                )
            )
        return success_result(result)

    return Server(
        "jingyu",
        version=__version__,
        title="Jingyu",
        instructions=INSTRUCTIONS,
        on_list_tools=list_tools,
        on_call_tool=call_tool,
    )


async def _serve(server: Server[Any]) -> None:
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="jingyu-mcp", description="Jingyu MCP server (stdio)")
    parser.add_argument("--workspace", help="workspace root (default: $JINGYU_WORKSPACE or cwd)")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S)
    args = parser.parse_args(argv)
    workspace = Workspace.at(args.workspace)
    # The server lives for a whole conversation: keep a Blender worker warm.
    workers = WorkerPool(workspace.root / ".jingyu" / "workers")
    context = ToolContext(workspace, render_timeout_s=args.timeout, workers=workers)
    try:
        anyio.run(_serve, build_server(context))
    finally:
        workers.close()


__all__ = ["INSTRUCTIONS", "build_server", "error_result", "main", "mcp_tool", "success_result"]
