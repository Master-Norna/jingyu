"""Worker process entry point (runs inside Blender).

Invocation, either::

    blender --background --factory-startup --python-exit-code 3 \\
        --python jingyu/blender/worker.py -- REQUEST.json RESPONSE.json

or, with the ``bpy`` module::

    python -m jingyu.blender.worker REQUEST.json RESPONSE.json

The worker always tries to write RESPONSE.json, ``{"ok": true, ...}`` on
success and ``{"ok": false, "error": {...}}`` on failure, and the host trusts
only that file.  A missing response means the process crashed.
"""

from __future__ import annotations

import sys
from pathlib import Path

if not globals().get("__package__"):
    # Executed as a script by the Blender executable: make ``jingyu`` importable.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import json
import time
import traceback
from typing import Any

PROTOCOL = "jingyu.worker.v1"


def _write_response(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _probe() -> dict[str, Any]:
    import bpy

    from jingyu.blender import compat, kit

    scene = kit.reset_to_empty()
    devices: list[dict[str, str]] = []
    try:
        prefs = compat.cycles_preferences()
        for backend in compat.GPU_BACKENDS:
            try:
                prefs.compute_device_type = backend
                compat.refresh_cycles_devices(prefs)
            except (TypeError, RuntimeError):
                continue
            devices += [
                {"backend": backend, "name": d.name, "type": d.type}
                for d in prefs.devices
                if d.type == backend
            ]
    except KeyError:
        pass
    return {
        "blender": compat.version_info(),
        "engines_registered": compat.available_engines(scene),
        "gpu_devices": devices,
        "python": sys.version.split()[0],
        "binary_path": bpy.app.binary_path,
    }


def _render(request: dict[str, Any]) -> dict[str, Any]:
    from jingyu.blender import build, compat, diagnostics, kit, passes
    from jingyu.errors import JingyuError

    spec = request.get("scene")
    output_dir = Path(str(request.get("output_dir", "")))
    if not isinstance(spec, dict) or not output_dir.is_dir():
        raise JingyuError("worker.bad_request", "render needs 'scene' and an existing 'output_dir'")
    overrides = request.get("overrides") or {}
    want_id_mask = bool((request.get("passes") or {}).get("id_mask", True))
    want_light = bool((request.get("passes") or {}).get("light", False))

    timings: dict[str, int] = {}
    started = time.perf_counter()

    scene = kit.reset_to_empty()
    built = build.build_scene(scene, spec)
    applied = build.configure_render(scene, spec["render"], overrides)
    timings["build_ms"] = _ms_since(started)

    mark = time.perf_counter()
    image = output_dir / "image.png"
    try:
        kit.render_still(scene, image)
    except JingyuError:
        raise
    except Exception as exc:  # Blender raises RuntimeError for render failures
        raise JingyuError("blender.render_failed", str(exc)) from exc
    timings["render_ms"] = _ms_since(mark)

    outputs = {"image": image.name}
    light_map: dict[str, Any] | None = None
    if want_light:
        mark = time.perf_counter()
        light_mask = output_dir / "light_mask.png"
        light_map = diagnostics.light_pass(
            scene, built.visible, built.lights, built.transmissive, light_mask
        )
        outputs["light_mask"] = light_mask.name
        timings["light_ms"] = _ms_since(mark)

    id_map: dict[str, Any] | None = None
    if want_id_mask:
        mark = time.perf_counter()
        id_mask = output_dir / "id_mask.png"
        id_map = passes.render_id_mask(scene, built.visible, id_mask)
        outputs["id_mask"] = id_mask.name
        timings["id_mask_ms"] = _ms_since(mark)

    timings["total_ms"] = _ms_since(started)
    return {
        "blender": compat.version_info(),
        "render": applied,
        "outputs": outputs,
        "id_map": id_map,
        "light_map": light_map,
        "timings": timings,
        "warnings": built.warnings,
    }


def _ms_since(mark: float) -> int:
    return round((time.perf_counter() - mark) * 1000)


def main(argv: list[str]) -> int:
    args = argv[argv.index("--") + 1 :] if "--" in argv else argv[1:]
    if len(args) != 2:
        print("usage: worker REQUEST.json RESPONSE.json", file=sys.stderr)
        return 64
    request_path, response_path = Path(args[0]), Path(args[1])

    from jingyu.errors import JingyuError

    action = "unknown"
    try:
        from jingyu.blender import compat

        compat.require_supported_version()
        request = json.loads(request_path.read_text(encoding="utf-8"))
        if not isinstance(request, dict) or request.get("protocol") != PROTOCOL:
            raise JingyuError("worker.bad_request", f"expected protocol {PROTOCOL!r}")
        action = str(request.get("action"))
        if action == "probe":
            result = _probe()
        elif action == "render":
            result = _render(request)
        else:
            raise JingyuError("worker.bad_request", f"unknown action {action!r}")
    except JingyuError as exc:
        _write_response(
            response_path,
            {"protocol": PROTOCOL, "action": action, "ok": False, "error": exc.to_dict()},
        )
        return 2
    except Exception as exc:
        _write_response(
            response_path,
            {
                "protocol": PROTOCOL,
                "action": action,
                "ok": False,
                "error": {
                    "code": "internal.unexpected",
                    "message": f"{type(exc).__name__}: {exc}",
                    "details": {"traceback": traceback.format_exc()},
                },
            },
        )
        return 3
    _write_response(response_path, {"protocol": PROTOCOL, "action": action, "ok": True, **result})
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
