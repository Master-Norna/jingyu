"""Runtime discovery and the worker runner, without Blender (fake worker scripts)."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest
from helpers import script_runtime

from jingyu.bridge import discover, run_worker
from jingyu.bridge.discover import PACKAGE_PARENT, WORKER_SCRIPT, BlenderRuntime, discover_runtime
from jingyu.bridge.runner import LOG_NAME, PROTOCOL
from jingyu.errors import JingyuError


@pytest.fixture
def no_fallbacks(monkeypatch: pytest.MonkeyPatch) -> None:
    """Hide bpy in this interpreter and any default install location."""

    real_find_spec = importlib.util.find_spec

    def find_spec(name: str, *args: object) -> object:
        return None if name == "bpy" else real_find_spec(name, *args)  # type: ignore[arg-type]

    monkeypatch.setattr(importlib.util, "find_spec", find_spec)
    monkeypatch.setattr(discover, "_default_locations", lambda: [])


def _file(tmp_path: Path, name: str) -> Path:
    path = tmp_path / name
    path.write_text("", encoding="utf-8")
    return path


def test_explicit_blender_executable_wins(tmp_path: Path) -> None:
    blender = _file(tmp_path, "blender-4.2")
    python = _file(tmp_path, "python")
    runtime = discover_runtime({"JINGYU_BLENDER": str(blender), "JINGYU_BPY_PYTHON": str(python)})
    assert runtime == BlenderRuntime("executable", str(blender), "env:JINGYU_BLENDER")


def test_bpy_python_is_used_when_no_executable_is_named(tmp_path: Path) -> None:
    python = _file(tmp_path, "python")
    env = {"JINGYU_BLENDER": str(tmp_path / "missing"), "JINGYU_BPY_PYTHON": str(python)}
    assert discover_runtime(env) == BlenderRuntime("module", str(python), "env:JINGYU_BPY_PYTHON")


@pytest.mark.skipif(sys.platform == "win32", reason="PATH lookup needs an executable bit")
def test_blender_on_path_is_found(tmp_path: Path, no_fallbacks: None) -> None:
    blender = _file(tmp_path, "blender")
    blender.chmod(0o755)
    runtime = discover_runtime({"PATH": str(tmp_path)})
    assert (runtime.kind, runtime.source) == ("executable", "PATH")
    assert Path(runtime.path) == blender


def test_bpy_in_the_current_interpreter_is_found(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a: object())
    runtime = discover_runtime({"PATH": str(tmp_path)})
    assert runtime == BlenderRuntime("module", sys.executable, "current-interpreter")


def test_nothing_found_lists_what_was_tried(tmp_path: Path, no_fallbacks: None) -> None:
    env = {"PATH": str(tmp_path), "JINGYU_BLENDER": str(tmp_path / "nope")}
    with pytest.raises(JingyuError) as info:
        discover_runtime(env)
    error = info.value
    assert error.code == "blender.not_found"
    assert error.hint and "JINGYU_BLENDER" in error.hint
    tried = error.details["tried"]
    assert tried[0].startswith("JINGYU_BLENDER=")
    assert "bpy in the current interpreter" in tried


def test_worker_commands(tmp_path: Path) -> None:
    request, response = tmp_path / "request.json", tmp_path / "response.json"
    executable = BlenderRuntime("executable", "/opt/blender", "PATH")
    command = executable.worker_command(request, response)
    assert command[0] == "/opt/blender"
    assert {"--background", "--factory-startup"} <= set(command)
    assert command[-3:] == ["--", str(request), str(response)]
    assert str(WORKER_SCRIPT) in command
    module = BlenderRuntime("module", "/usr/bin/python3", "env:JINGYU_BPY_PYTHON")
    assert module.worker_command(request, response) == [
        "/usr/bin/python3",
        "-m",
        "jingyu.blender.worker",
        str(request),
        str(response),
    ]


def test_worker_environment(tmp_path: Path) -> None:
    module = BlenderRuntime("module", "python", "x")
    env = module.environment({"PYTHONPATH": "extra"})
    assert env["PYTHONPATH"] == str(PACKAGE_PARENT) + os.pathsep + "extra"
    assert env["PYTHONUTF8"] == "1"
    executable = BlenderRuntime("executable", "blender", "x")
    assert "PYTHONPATH" not in executable.environment({})
    assert module.to_dict() == {"kind": "module", "path": "python", "source": "x"}


# ------------------------------------------------------------ run_worker


def _ok(extra: str = "") -> str:
    return (
        "import json, sys\n"
        "request = json.load(open(sys.argv[1], encoding='utf-8'))\n"
        f"response = {{'protocol': request['protocol'], 'ok': True, 'echo': request}}\n{extra}"
        "json.dump(response, open(sys.argv[2], 'w', encoding='utf-8'))\n"
    )


def test_run_worker_returns_the_response(tmp_path: Path) -> None:
    script = "print('hello from the worker')\n" + _ok()
    run = run_worker(script_runtime(script), {"action": "probe"}, tmp_path / "work")
    assert run.exit_code == 0
    assert run.response["echo"] == {"protocol": PROTOCOL, "action": "probe"}
    assert run.log_path == tmp_path / "work" / LOG_NAME
    assert "hello from the worker" in run.log_path.read_text(encoding="utf-8")


def _failure(script: str, tmp_path: Path, timeout_s: float = 60.0) -> JingyuError:
    with pytest.raises(JingyuError) as info:
        run_worker(script_runtime(script), {"action": "render"}, tmp_path, timeout_s=timeout_s)
    return info.value


def test_a_worker_without_a_response_crashed(tmp_path: Path) -> None:
    error = _failure("import sys\nprint('segfault, probably')\nsys.exit(139)\n", tmp_path)
    assert error.code == "blender.crashed"
    assert error.details["exit_code"] == 139
    assert error.details["log_tail"] == ["segfault, probably"]


def test_a_crash_mentioning_egl_gets_an_opengl_hint(tmp_path: Path) -> None:
    error = _failure("print('Unable to open libEGL.so')\n", tmp_path)
    assert error.code == "blender.crashed"
    assert error.hint and "OpenGL" in error.hint


def test_a_worker_error_keeps_its_code(tmp_path: Path) -> None:
    script = (
        "import json, sys\n"
        "print('rendering...')\n"
        "error = {'code': 'blender.engine_unavailable', 'message': 'no eevee',"
        " 'hint': 'use cycles'}\n"
        "json.dump({'protocol': 'jingyu.worker.v1', 'ok': False, 'error': error},"
        " open(sys.argv[2], 'w'))\n"
    )
    error = _failure(script, tmp_path)
    assert (error.code, error.message, error.hint) == (
        "blender.engine_unavailable",
        "no eevee",
        "use cycles",
    )
    assert error.details["log_tail"] == ["rendering..."]


@pytest.mark.parametrize(
    "payload",
    [
        "not json",
        '{"protocol": "jingyu.worker.v1", "ok": true, "ok": true}',
        '{"protocol": "jingyu.worker.v0", "ok": true}',
        "[1, 2]",
        '{"protocol": "jingyu.worker.v1", "ok": false, "error": {"code": "made.up"}}',
    ],
)
def test_malformed_responses_are_bad_responses(tmp_path: Path, payload: str) -> None:
    script = f"import sys\nopen(sys.argv[2], 'w').write({payload!r})\n"
    assert _failure(script, tmp_path).code == "blender.bad_response"


def test_a_slow_worker_times_out(tmp_path: Path) -> None:
    error = _failure("import time\ntime.sleep(60)\n", tmp_path, timeout_s=0.5)
    assert error.code == "blender.timeout"
    assert "log_tail" in error.details
