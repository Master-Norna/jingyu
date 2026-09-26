"""Environment diagnostics: what can this machine render, and if not, why."""

from __future__ import annotations

import platform
import sys
import tempfile
import time
from importlib import metadata
from pathlib import Path
from typing import Any

from . import __version__
from .bridge import BlenderRuntime, discover_runtime, run_worker
from .errors import JingyuError
from .scene import validate_scene

ENGINES = ("cycles", "eevee", "workbench")


def _dependency_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in ("jsonschema", "pillow", "mcp"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _tiny_scene(engine: str) -> dict[str, Any]:
    return {
        "schema": "jingyu.scene.v1",
        "id": f"doctor-{engine}",
        "objects": [{"id": "cube", "geometry": {"op": "box", "size": [1, 1, 1]}}],
        "lights": [{"id": "sun", "kind": "sun", "location": [0, 0, 5], "rotation": [30, 0, 30]}],
        "cameras": [{"id": "cam", "location": [3, -3, 2.5], "look_at": [0, 0, 0.5]}],
        "render": {"camera": "cam", "engine": engine, "resolution": [32, 32], "samples": 1},
    }


def _engine_check(runtime: BlenderRuntime, engine: str, timeout_s: float) -> dict[str, Any]:
    normalized = validate_scene(_tiny_scene(engine)).require_valid()
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="jingyu-doctor-") as tmp:
        try:
            run = run_worker(
                runtime,
                {
                    "action": "render",
                    "scene": normalized,
                    "output_dir": tmp,
                    "passes": {"id_mask": False},
                },
                Path(tmp),
                timeout_s=timeout_s,
            )
        except JingyuError as exc:
            return {
                "ok": False,
                "error": {"code": exc.code, "message": exc.message, "hint": exc.hint},
            }
    return {
        "ok": True,
        "seconds": round(time.perf_counter() - started, 2),
        "device": run.response["render"]["device"],
    }


def diagnose(*, deep: bool = False, timeout_s: float = 180.0) -> dict[str, Any]:
    """Report versions, the Blender runtime and (with *deep*) per-engine render checks."""

    report: dict[str, Any] = {
        "jingyu_version": __version__,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "dependencies": _dependency_versions(),
        "runtime": None,
        "blender": None,
        "engines": {},
        "problems": [],
    }
    try:
        runtime = discover_runtime()
    except JingyuError as exc:
        report["problems"].append({"code": exc.code, "message": exc.message, "hint": exc.hint})
        report["ok"] = False
        return report
    report["runtime"] = runtime.to_dict()

    with tempfile.TemporaryDirectory(prefix="jingyu-doctor-") as tmp:
        try:
            probe = run_worker(
                runtime, {"action": "probe"}, Path(tmp), timeout_s=timeout_s
            ).response
        except JingyuError as exc:
            report["problems"].append({"code": exc.code, "message": exc.message, "hint": exc.hint})
            report["ok"] = False
            return report
    report["blender"] = {
        **probe["blender"],
        "engines_registered": probe["engines_registered"],
        "gpu_devices": probe["gpu_devices"],
    }

    if deep:
        for engine in ENGINES:
            result = _engine_check(runtime, engine, timeout_s)
            report["engines"][engine] = result
            if not result["ok"]:
                report["problems"].append(
                    {
                        "code": result["error"]["code"],
                        "message": f"{engine}: {result['error']['message']}",
                        "hint": result["error"].get("hint"),
                    }
                )
    report["ok"] = not report["problems"]
    return report


__all__ = ["ENGINES", "diagnose"]
