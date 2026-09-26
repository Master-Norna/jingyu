"""Static checks of the layering rules in docs/工程规范.md, section 2.

* The pure core (errors, canonical_json, conventions, generator, idmask, geometry,
  materials, environments, placement) uses only the standard library, because it
  also runs inside Blender.
* jingyu/blender/ uses only the pure core, the standard library and bpy/bmesh/mathutils.
* Only jingyu/mcp_server.py imports the optional MCP stack.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from functools import cache
from pathlib import Path

import pytest
from helpers import PACKAGE_ROOT

SOURCE_ROOT = PACKAGE_ROOT.parent
STDLIB = frozenset(sys.stdlib_module_names)

CORE_MODULES = frozenset(
    {
        "jingyu.errors",
        "jingyu.canonical_json",
        "jingyu.conventions",
        "jingyu.generator",
        "jingyu.idmask",
        "jingyu.geometry",
        "jingyu.materials",
        "jingyu.environments",
        "jingyu.placement",
    }
)
BLENDER_THIRD_PARTY = frozenset({"bpy", "bmesh", "mathutils"})
MCP_STACK = frozenset({"mcp", "mcp_types", "anyio"})


def _module_name(path: Path) -> str:
    parts = list(path.relative_to(SOURCE_ROOT).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _is_module(name: str) -> bool:
    base = SOURCE_ROOT.joinpath(*name.split("."))
    return base.with_suffix(".py").is_file() or (base / "__init__.py").is_file()


@cache
def _imports(path: Path) -> frozenset[str]:
    """Absolute names of every module *path* imports, at any depth in the file."""

    module = _module_name(path)
    package = module if path.name == "__init__.py" else module.rpartition(".")[0]
    found: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"), filename=str(path))):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                parts = package.split(".")
                base = ".".join(parts[: len(parts) - node.level + 1])
                target = f"{base}.{node.module}" if node.module else base
            else:
                target = node.module or ""
            if target == "__future__":
                continue
            found.add(target)
            # `from package import submodule` imports the submodule too.
            found.update(
                f"{target}.{alias.name}"
                for alias in node.names
                if _is_module(f"{target}.{alias.name}")
            )
    return frozenset(found)


def _sources(*relative: str) -> list[Path]:
    paths: list[Path] = []
    for item in relative:
        path = PACKAGE_ROOT / item
        paths += sorted(path.rglob("*.py")) if path.is_dir() else [path]
    return paths


def _within(name: str, allowed: frozenset[str]) -> bool:
    return any(name == a or name.startswith(a + ".") for a in allowed)


def _params(paths: list[Path]) -> list[object]:
    return [pytest.param(p, id=str(p.relative_to(SOURCE_ROOT).as_posix())) for p in paths]


CORE_FILES = _sources(
    "__init__.py",
    "errors.py",
    "canonical_json.py",
    "conventions.py",
    "generator.py",
    "idmask.py",
    "geometry",
    "materials",
)
BLENDER_FILES = _sources("blender")
ALL_FILES = sorted(PACKAGE_ROOT.rglob("*.py"))


def test_the_scan_sees_the_package() -> None:
    assert len(CORE_FILES) >= 12
    assert {p.name for p in BLENDER_FILES} >= {"worker.py", "build.py", "kit.py", "compat.py"}
    # Relative, nested and `from . import x` imports all resolve.
    assert {"jingyu.blender.compat", "jingyu.blender.kit"} <= _imports(
        PACKAGE_ROOT / "blender" / "build.py"
    )
    assert "jingyu.blender.compat" in _imports(PACKAGE_ROOT / "blender" / "worker.py")
    assert "jingyu.conventions" in _imports(PACKAGE_ROOT / "tools" / "builtin.py")


@pytest.mark.parametrize("path", _params(CORE_FILES))
def test_pure_core_imports_only_the_stdlib_and_itself(path: Path) -> None:
    bad = sorted(
        name
        for name in _imports(path)
        if name.split(".")[0] not in STDLIB and not _within(name, CORE_MODULES)
    )
    assert bad == [], f"{path.name} must stay standard-library only; it imports {bad}"


@pytest.mark.parametrize("path", _params(BLENDER_FILES))
def test_blender_code_imports_only_what_blender_provides(path: Path) -> None:
    allowed = CORE_MODULES | {"jingyu.blender"}
    bad = sorted(
        name
        for name in _imports(path)
        if name.split(".")[0] not in STDLIB | BLENDER_THIRD_PARTY
        and name != "jingyu"
        and not _within(name, allowed)
    )
    assert bad == [], f"{path.name} runs inside Blender and must not import {bad}"


@pytest.mark.parametrize("path", _params(ALL_FILES))
def test_only_the_mcp_server_imports_the_mcp_stack(path: Path) -> None:
    if path == PACKAGE_ROOT / "mcp_server.py":
        return
    bad = sorted(name for name in _imports(path) if name.split(".")[0] in MCP_STACK)
    assert bad == [], f"{path.name} imports the optional MCP stack: {bad}"


def test_nothing_outside_the_blender_package_imports_bpy() -> None:
    offenders = {
        str(path.relative_to(SOURCE_ROOT)): sorted(
            n for n in _imports(path) if n.split(".")[0] in BLENDER_THIRD_PARTY
        )
        for path in ALL_FILES
        if path not in BLENDER_FILES
    }
    assert {k: v for k, v in offenders.items() if v} == {}


def test_importing_the_core_and_the_cli_loads_no_optional_dependency() -> None:
    probe = (
        "import sys\n"
        "import jingyu.errors, jingyu.canonical_json, jingyu.conventions, jingyu.generator\n"
        "import jingyu.idmask, jingyu.geometry, jingyu.materials\n"
        "core = sorted({m.split('.')[0] for m in sys.modules} & {'jsonschema', 'PIL', 'bpy'})\n"
        "import jingyu.cli\n"
        "cli = sorted({m.split('.')[0] for m in sys.modules} & {'mcp', 'mcp_types', 'anyio'})\n"
        "print(core, cli)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        check=True,
        cwd=SOURCE_ROOT,
        timeout=60,
    )
    assert result.stdout.strip() == "[] []"
