"""Run the Blender worker in an isolated child process.

Rendering happens in a separate process so that a Blender crash, a hang or a
GPU driver failure can never take down the host (a CLI, an MCP server).  The
child writes a response file; its absence is itself a diagnosis.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..canonical_json import load_file_strict
from ..errors import JingyuError
from .discover import BlenderRuntime

PROTOCOL = "jingyu.worker.v1"
LOG_NAME = "blender.log"
DEFAULT_TIMEOUT_S = 900.0
LOG_TAIL_LINES = 40


@dataclass(frozen=True)
class WorkerRun:
    response: dict[str, Any]
    log_path: Path
    exit_code: int


def run_worker(
    runtime: BlenderRuntime,
    request: dict[str, Any],
    work_dir: Path,
    *,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> WorkerRun:
    """Run one worker request in *work_dir* and return its successful response.

    Raises :class:`JingyuError` with the worker's own code when it reports a
    failure, ``blender.timeout`` when it overruns and ``blender.crashed`` when it
    exits without a response.
    """

    request_path, response_path = write_request(work_dir, request)
    log_path = work_dir / LOG_NAME
    command = runtime.worker_command(request_path, response_path)
    popen_kwargs = isolated_process()

    with log_path.open("wb") as log:
        process = subprocess.Popen(
            command,
            cwd=work_dir,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=runtime.environment(os.environ),
            **popen_kwargs,
        )
        try:
            exit_code = process.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            kill_tree(process)
            raise JingyuError(
                "blender.timeout",
                f"the Blender worker did not finish within {timeout_s:g} s",
                hint="Lower samples or resolution, or raise the timeout.",
                details={"log_tail": tail(log_path)},
            ) from None

    if not response_path.is_file():
        raise JingyuError(
            "blender.crashed",
            f"the Blender worker exited with code {exit_code} and wrote no response",
            hint=crash_hint(log_path),
            details={"exit_code": exit_code, "log_tail": tail(log_path)},
        )
    return WorkerRun(read_response(response_path, log_path), log_path, exit_code)


def write_request(work_dir: Path, request: dict[str, Any]) -> tuple[Path, Path]:
    """Write *request* into *work_dir*; return the request and response paths."""

    work_dir.mkdir(parents=True, exist_ok=True)
    request_path = work_dir / "request.json"
    request_path.write_text(
        json.dumps({"protocol": PROTOCOL, **request}, ensure_ascii=False), encoding="utf-8"
    )
    return request_path, work_dir / "response.json"


def isolated_process() -> dict[str, Any]:
    """Popen options that give the worker its own process group, to kill it whole."""

    if sys.platform == "win32":
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}  # type: ignore[attr-defined]
    return {"start_new_session": True}


def read_response(response_path: Path, log_path: Path) -> dict[str, Any]:
    """The worker's successful response, or its failure raised with its own code."""

    try:
        response = load_file_strict(response_path)
    except JingyuError as exc:
        raise JingyuError("blender.bad_response", exc.message) from exc
    if not isinstance(response, dict) or response.get("protocol") != PROTOCOL:
        raise JingyuError("blender.bad_response", "response is not a jingyu.worker.v1 document")
    if not response.get("ok"):
        error = response.get("error")
        failure = JingyuError.from_dict(error if isinstance(error, dict) else {})
        details = dict(failure.details)
        details.setdefault("log_tail", tail(log_path))
        raise JingyuError(failure.code, failure.message, hint=failure.hint, details=details)
    return response


def kill_tree(process: subprocess.Popen[bytes]) -> None:
    try:
        if sys.platform == "win32":
            process.kill()
        else:
            os.killpg(process.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass
    process.wait()


def tail(log_path: Path, lines: int = LOG_TAIL_LINES) -> list[str]:
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    return text.splitlines()[-lines:]


def crash_hint(log_path: Path) -> str:
    text = "\n".join(tail(log_path, 200))
    if "libEGL" in text or "EGL" in text:
        return (
            "Blender could not create an OpenGL/EGL context. EEVEE and Workbench need a "
            "GPU or a software OpenGL driver (for example Mesa); Cycles does not."
        )
    return "See details.log_tail; run diagnose_environment for a full check."


__all__ = [
    "DEFAULT_TIMEOUT_S",
    "LOG_NAME",
    "PROTOCOL",
    "WorkerRun",
    "crash_hint",
    "isolated_process",
    "kill_tree",
    "read_response",
    "run_worker",
    "tail",
    "write_request",
]
