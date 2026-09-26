from __future__ import annotations

import base64
import json
from collections.abc import Awaitable, Callable
from typing import Any

import pytest
from helpers import CandidateFactory

from jingyu.scene import minimal_scene
from jingyu.tools import REGISTRY, ToolContext
from jingyu.workspace import Workspace

pytestmark = pytest.mark.mcp


def _with_client(workspace: Workspace, body: Callable[[Any], Awaitable[None]]) -> None:
    pytest.importorskip("mcp")
    anyio = pytest.importorskip("anyio")
    from mcp.client import Client

    from jingyu.mcp_server import build_server

    async def main() -> None:
        async with Client(build_server(ToolContext(workspace)), raise_exceptions=True) as client:
            await body(client)

    anyio.run(main)


def test_list_tools_advertises_every_registry_tool(workspace: Workspace) -> None:
    async def body(client: Any) -> None:
        listed = {tool.name: tool for tool in (await client.list_tools()).tools}
        assert list(listed) == REGISTRY.names()
        for tool in REGISTRY:
            advertised = listed[tool.name]
            assert advertised.input_schema == tool.input_schema
            assert advertised.output_schema == tool.output_schema
            assert advertised.description == tool.description
            assert advertised.annotations.read_only_hint is tool.read_only

    _with_client(workspace, body)


def test_validate_scene_returns_structured_content(workspace: Workspace) -> None:
    async def body(client: Any) -> None:
        result = await client.call_tool("validate_scene", {"scene": minimal_scene()})
        assert result.is_error is False
        assert result.structured_content["valid"] is True
        assert len(result.structured_content["scene_sha256"]) == 64
        assert json.loads(result.content[0].text) == result.structured_content

    _with_client(workspace, body)


def test_unknown_tool_is_an_error_result(workspace: Workspace) -> None:
    async def body(client: Any) -> None:
        result = await client.call_tool("draw_a_cat", {})
        assert result.is_error is True
        payload = json.loads(result.content[0].text)
        assert payload["error"]["code"] == "tool.unknown"

    _with_client(workspace, body)


def test_invalid_arguments_are_an_error_result(workspace: Workspace) -> None:
    async def body(client: Any) -> None:
        result = await client.call_tool("list_generators", {"kind": "lights"})
        assert result.is_error is True
        assert json.loads(result.content[0].text)["error"]["code"] == "tool.invalid_arguments"

    _with_client(workspace, body)


def test_unexpected_exceptions_do_not_crash_the_server(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = REGISTRY.invoke

    def invoke(name: str, arguments: Any, context: ToolContext) -> Any:
        if name == "list_error_codes":
            raise RuntimeError("boom")
        return original(name, arguments, context)

    monkeypatch.setattr(REGISTRY, "invoke", invoke)

    async def body(client: Any) -> None:
        result = await client.call_tool("list_error_codes", {})
        assert result.is_error is True
        error = json.loads(result.content[0].text)["error"]
        assert error["code"] == "internal.unexpected"
        assert "RuntimeError: boom" in error["message"]
        # The server keeps serving.
        assert (await client.call_tool("get_guide", {})).is_error is False

    _with_client(workspace, body)


def test_images_are_returned_as_image_content(
    workspace: Workspace, make_candidate: CandidateFactory
) -> None:
    candidate = make_candidate()

    async def body(client: Any) -> None:
        result = await client.call_tool(
            "view_candidate", {"candidate_id": candidate.id, "view": "glance"}
        )
        assert result.is_error is False
        kinds = [item.type for item in result.content]
        assert kinds == ["text", "text", "image"]
        image = result.content[2]
        assert image.mime_type == "image/png"
        assert base64.b64decode(image.data).startswith(b"\x89PNG")

    _with_client(workspace, body)
