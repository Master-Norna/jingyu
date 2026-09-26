"""Generated reference files: error codes, tools, the generator catalogue and the
exported scene JSON Schema.

These files are rendered from the registries, so they cannot drift from the
code: ``python -m jingyu.reference --write`` regenerates them and
``--check`` (run by the tests and CI) fails when one is stale.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

from .errors import ERROR_CODES
from .generator import GeneratorDef
from .geometry import GEOMETRY
from .materials import MATERIALS
from .scene import scene_schema
from .tools import REGISTRY

GENERATED_NOTE = "<!-- 本文件由 `python -m jingyu.reference --write` 生成，请勿手改。 -->"


def _cell(text: Any) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ")


def _table(headers: Iterable[str], rows: Iterable[Iterable[Any]]) -> list[str]:
    head = list(headers)
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    lines += ["| " + " | ".join(_cell(c) for c in row) + " |" for row in rows]
    return lines


def _json_inline(value: Any) -> str:
    return "`" + json.dumps(value, ensure_ascii=False, separators=(", ", ": ")) + "`"


def _param_type(schema: Mapping[str, Any]) -> str:
    if "enum" in schema:
        return " / ".join(str(v) for v in schema["enum"])
    kind = schema.get("type", "")
    if kind == "array":
        items = schema.get("prefixItems") or []
        if items:
            return "[" + ", ".join(str(i.get("type", "")) for i in items) + "]"
        return f"array of {schema.get('items', {}).get('type', 'value')}"
    bounds = []
    if "minimum" in schema:
        bounds.append(f">= {schema['minimum']}")
    if "exclusiveMinimum" in schema:
        bounds.append(f"> {schema['exclusiveMinimum']}")
    if "maximum" in schema:
        bounds.append(f"<= {schema['maximum']}")
    if "exclusiveMaximum" in schema:
        bounds.append(f"< {schema['exclusiveMaximum']}")
    return f"{kind} {', '.join(bounds)}".strip()


def error_codes_markdown() -> str:
    lines = [
        GENERATED_NOTE,
        "",
        "# 错误码",
        "",
        "每个跨越进程或工具边界的失败都带一个稳定的 `code`。调用方（人、CLI、模型）按",
        "`code` 分支；`message` 给人看，版本之间可能改写。校验问题（issue）还带",
        "`pointer`（RFC 6901 JSON Pointer，指向出问题的字段）和可选的 `hint`。",
        "",
        "新增错误码只加不改：已发布的码不改名、不改含义。",
        "",
    ]
    groups: dict[str, list[tuple[str, str]]] = {}
    for code, meaning in ERROR_CODES.items():
        groups.setdefault(code.split(".", 1)[0], []).append((code, meaning))
    for group, entries in groups.items():
        lines += [f"## {group}", ""]
        lines += _table(["code", "含义"], ((f"`{c}`", m) for c, m in entries))
        lines.append("")
    return "\n".join(lines)


def tools_markdown() -> str:
    lines = [
        GENERATED_NOTE,
        "",
        "# 工具",
        "",
        "CLI 和 MCP 服务器共用同一个工具注册表，参数和返回值都按下列 JSON Schema 校验。",
        '失败时返回 `{"error": {"code", "message", "hint"?, "details"?}}`，',
        "错误码见 [错误码](错误码.md)。",
        "",
    ]
    lines += _table(
        ["工具", "说明", "只读", "返回图像"],
        (
            (
                f"`{t.name}`",
                t.description,
                "是" if t.read_only else "否",
                "是" if t.returns_images else "",
            )
            for t in REGISTRY
        ),
    )
    lines.append("")
    for tool in REGISTRY:
        lines += [f"## `{tool.name}`", "", tool.description, ""]
        props: Mapping[str, Any] = tool.input_schema.get("properties", {})
        required = set(tool.input_schema.get("required", []))
        if props:
            lines += _table(
                ["参数", "类型", "默认", "说明"],
                (
                    (
                        f"`{name}`" + (" *" if name in required else ""),
                        _param_type(schema),
                        _json_inline(schema["default"]) if "default" in schema else "",
                        schema.get("description", ""),
                    )
                    for name, schema in props.items()
                ),
            )
            if tool.input_schema.get("oneOf"):
                alternatives = [
                    " + ".join(o.get("required", [])) for o in tool.input_schema["oneOf"]
                ]
                lines += [
                    "",
                    "以下参数恰好给出一组：" + "；".join(f"`{a}`" for a in alternatives) + "。",
                ]
        else:
            lines.append("无参数。")
        outputs = list(tool.output_schema.get("properties", {}))
        if outputs:
            lines += ["", "返回字段：" + "、".join(f"`{o}`" for o in outputs) + "。"]
        lines.append("")
    return "\n".join(lines)


def _generator_section(definition: GeneratorDef[Any]) -> list[str]:
    lines = [f"### `{definition.name}`", "", definition.summary, ""]
    if definition.params:
        lines += _table(
            ["参数", "类型", "默认", "说明"],
            (
                (
                    f"`{name}`" + (" *" if name in definition.required else ""),
                    _param_type(schema),
                    _json_inline(schema["default"]) if "default" in schema else "",
                    schema.get("description", ""),
                )
                for name, schema in definition.params.items()
            ),
        )
    else:
        lines.append("无参数。")
    for example in definition.examples:
        lines += ["", "示例：" + _json_inline(dict(example))]
    lines.append("")
    return lines


def generators_markdown() -> str:
    lines = [
        GENERATED_NOTE,
        "",
        "# 生成器目录",
        "",
        "景语收集的是生成器，不是资产：形状来自几何算子，表面来自材质族，每个都是带参数的",
        "原语。差异再小，也只是参数值不同。带 `*` 的参数必填，其余有默认值。单位：长度米、",
        "角度度，其他单位写在字段名后缀里。",
        "",
        f"## 几何算子（`geometry.{GEOMETRY.discriminator}`）",
        "",
    ]
    for op in GEOMETRY:
        lines += _generator_section(op)
    lines += [f"## 材质族（`material.{MATERIALS.discriminator}`）", ""]
    for family in MATERIALS:
        lines += _generator_section(family)
    return "\n".join(lines)


def scene_schema_json() -> str:
    return json.dumps(scene_schema(), ensure_ascii=False, indent=2) + "\n"


def _with_newline(render: Callable[[], str]) -> Callable[[], str]:
    return lambda: render().rstrip("\n") + "\n"


#: repository-relative path -> renderer
GENERATED: dict[str, Callable[[], str]] = {
    "docs/错误码.md": _with_newline(error_codes_markdown),
    "docs/工具.md": _with_newline(tools_markdown),
    "docs/生成器目录.md": _with_newline(generators_markdown),
    "schemas/scene.v1.schema.json": scene_schema_json,
}


def stale_files(root: Path) -> list[str]:
    """Generated files under *root* that are missing or differ from the code."""

    stale = []
    for relative, render in GENERATED.items():
        path = root / relative
        if not path.is_file() or path.read_text(encoding="utf-8") != render():
            stale.append(relative)
    return stale


def write_all(root: Path) -> list[str]:
    written = []
    for relative, render in GENERATED.items():
        path = root / relative
        text = render()
        if path.is_file() and path.read_text(encoding="utf-8") == text:
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8", newline="\n")
        written.append(relative)
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m jingyu.reference")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true", help="regenerate the reference files")
    mode.add_argument("--check", action="store_true", help="fail if any reference file is stale")
    parser.add_argument("--root", default=".", help="repository root (default: current directory)")
    args = parser.parse_args(argv)
    root = Path(args.root)
    if args.write:
        for relative in write_all(root):
            print(f"wrote {relative}")
        return 0
    stale = stale_files(root)
    for relative in stale:
        print(f"stale: {relative}", file=sys.stderr)
    if stale:
        print("run: python -m jingyu.reference --write", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
