"""Console entry point for the optional MCP server.

MCP support is an optional extra.  Without it this entry point explains how to
install it instead of failing with an import traceback.
"""

from __future__ import annotations

import importlib
import sys

INSTALL_HINT = '景语 MCP 依赖未安装；请运行: python -m pip install "jingyu[mcp]"'


def main() -> int:
    try:
        server = importlib.import_module("jingyu.mcp_server")
    except ModuleNotFoundError as exc:
        missing = exc.name or ""
        if missing in ("mcp", "mcp_types", "anyio") or missing.startswith("mcp."):
            print(INSTALL_HINT, file=sys.stderr)
            return 2
        raise
    server.main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
