"""Find a Blender runtime to run the worker in.

Two kinds are supported:

* ``executable``: an installed Blender (``blender`` / ``blender.exe``), run with
  ``--background --factory-startup``.
* ``module``: a Python interpreter that can ``import bpy`` (the ``bpy`` wheel).

Search order, first match wins:

1. ``JINGYU_BLENDER``: path to a Blender executable.
2. ``JINGYU_BPY_PYTHON``: path to a Python interpreter with ``bpy``.
3. ``blender`` on ``PATH``.
4. The current interpreter, when ``bpy`` is importable in it.
5. Default install locations on Windows and macOS.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from ..errors import JingyuError

RuntimeKind = Literal["executable", "module"]

PACKAGE_PARENT = Path(__file__).resolve().parents[2]
WORKER_SCRIPT = Path(__file__).resolve().parents[1] / "blender" / "worker.py"


@dataclass(frozen=True)
class BlenderRuntime:
    kind: RuntimeKind
    path: str
    source: str

    def worker_command(self, request: Path, response: Path) -> list[str]:
        return self._command([str(request), str(response)])

    def serve_command(self) -> list[str]:
        """A worker that stays alive and reads one request per line of stdin."""

        return self._command(["--serve"])

    def _command(self, arguments: list[str]) -> list[str]:
        if self.kind == "executable":
            return [
                self.path,
                "--background",
                "--factory-startup",
                "--python-exit-code",
                "3",
                "--python",
                str(WORKER_SCRIPT),
                "--",
                *arguments,
            ]
        return [self.path, "-m", "jingyu.blender.worker", *arguments]

    def environment(self, base: Mapping[str, str]) -> dict[str, str]:
        env = dict(base)
        env["PYTHONUTF8"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        if self.kind == "module":
            existing = env.get("PYTHONPATH")
            env["PYTHONPATH"] = str(PACKAGE_PARENT) + (os.pathsep + existing if existing else "")
        return env

    def to_dict(self) -> dict[str, str]:
        return {"kind": self.kind, "path": self.path, "source": self.source}


def _default_locations() -> list[Path]:
    if sys.platform == "win32":
        roots = [os.environ.get("PROGRAMFILES", r"C:\Program Files")]
        found: list[Path] = []
        for root in roots:
            base = Path(root) / "Blender Foundation"
            if base.is_dir():
                found += sorted(base.glob("Blender*/blender.exe"), reverse=True)
        return found
    if sys.platform == "darwin":
        return [Path("/Applications/Blender.app/Contents/MacOS/Blender")]
    return []


def discover_runtime(env: Mapping[str, str] | None = None) -> BlenderRuntime:
    """Return the first usable runtime or raise ``blender.not_found``."""

    env = os.environ if env is None else env
    tried: list[str] = []

    explicit = env.get("JINGYU_BLENDER")
    if explicit:
        if Path(explicit).is_file():
            return BlenderRuntime("executable", explicit, "env:JINGYU_BLENDER")
        tried.append(f"JINGYU_BLENDER={explicit} (not a file)")

    bpy_python = env.get("JINGYU_BPY_PYTHON")
    if bpy_python:
        if Path(bpy_python).is_file():
            return BlenderRuntime("module", bpy_python, "env:JINGYU_BPY_PYTHON")
        tried.append(f"JINGYU_BPY_PYTHON={bpy_python} (not a file)")

    on_path = shutil.which("blender", path=env.get("PATH"))
    if on_path:
        return BlenderRuntime("executable", on_path, "PATH")
    tried.append("blender on PATH")

    if importlib.util.find_spec("bpy") is not None:
        return BlenderRuntime("module", sys.executable, "current-interpreter")
    tried.append("bpy in the current interpreter")

    for candidate in _default_locations():
        if candidate.is_file():
            return BlenderRuntime("executable", str(candidate), "default-location")
    tried.append("default install locations")

    raise JingyuError(
        "blender.not_found",
        "no Blender runtime found",
        hint=(
            "Install Blender 4.2+ and put it on PATH, or set JINGYU_BLENDER to the "
            "executable, or set JINGYU_BPY_PYTHON to a Python with the bpy module."
        ),
        details={"tried": tried},
    )


__all__ = ["BlenderRuntime", "RuntimeKind", "discover_runtime"]
