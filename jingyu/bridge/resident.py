"""A Blender worker that stays alive between renders.

Starting Blender and importing jingyu inside it costs about a second, which a
model iterating on previews pays on every render.  A long-lived host (the MCP
server) keeps one worker process instead and hands it one request at a time.

Isolation is kept: the worker is still a separate process.  A timeout kills it
and the next request starts a fresh one; if it dies, the request fails with
``blender.crashed`` and the next one starts a fresh one too.  Each request's
share of the worker's output is copied into that request's ``blender.log``,
and the process is recycled after a number of requests so a slow leak inside
Blender cannot build up.
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from pathlib import Path
from typing import IO, Any

from ..errors import JingyuError
from .discover import BlenderRuntime
from .runner import (
    DEFAULT_TIMEOUT_S,
    LOG_NAME,
    WorkerRun,
    crash_hint,
    isolated_process,
    kill_tree,
    read_response,
    run_worker,
    tail,
    write_request,
)

#: A worker serves this many requests, then a fresh one takes over.
MAX_REQUESTS = 64
_POLL_S = 0.02


class ResidentWorker:
    """One long-lived worker process for one runtime."""

    def __init__(self, runtime: BlenderRuntime, log_path: Path):
        self.runtime = runtime
        self.log_path = log_path
        self.served = 0
        self._process: subprocess.Popen[bytes] | None = None
        self._log: IO[bytes] | None = None

    @property
    def alive(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def _start(self) -> subprocess.Popen[bytes]:
        self.close()
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log = self.log_path.open("ab")
        self._process = subprocess.Popen(
            self.runtime.serve_command(),
            stdin=subprocess.PIPE,
            stdout=self._log,
            stderr=subprocess.STDOUT,
            env=self.runtime.environment(os.environ),
            **isolated_process(),
        )
        self.served = 0
        return self._process

    def close(self) -> None:
        if self._process is not None:
            if self._process.poll() is None:
                try:
                    assert self._process.stdin is not None
                    self._process.stdin.close()
                    self._process.wait(timeout=5)
                except (OSError, subprocess.TimeoutExpired):
                    kill_tree(self._process)
            self._process = None
        if self._log is not None:
            self._log.close()
            self._log = None

    def run(self, request: dict[str, Any], work_dir: Path, timeout_s: float) -> WorkerRun:
        if not self.alive or self.served >= MAX_REQUESTS:
            self._start()
        process = self._process
        assert process is not None and process.stdin is not None
        request_path, response_path = write_request(work_dir, request)
        response_path.unlink(missing_ok=True)
        start = self.log_path.stat().st_size
        line = json.dumps({"request": str(request_path), "response": str(response_path)})
        try:
            process.stdin.write(line.encode("utf-8") + b"\n")
            process.stdin.flush()
        except OSError:  # the worker died between requests
            process = self._start()
            assert process.stdin is not None
            process.stdin.write(line.encode("utf-8") + b"\n")
            process.stdin.flush()
        self.served += 1
        deadline = time.monotonic() + timeout_s
        log_path = work_dir / LOG_NAME
        while not response_path.is_file():
            if process.poll() is not None:
                self._copy_log(start, log_path)
                self.close()
                raise JingyuError(
                    "blender.crashed",
                    f"the Blender worker exited with code {process.returncode} and wrote no "
                    "response",
                    hint=crash_hint(log_path),
                    details={"exit_code": process.returncode, "log_tail": tail(log_path)},
                )
            if time.monotonic() > deadline:
                kill_tree(process)
                self._copy_log(start, log_path)
                self.close()
                raise JingyuError(
                    "blender.timeout",
                    f"the Blender worker did not finish within {timeout_s:g} s",
                    hint="Lower samples or resolution, or raise the timeout.",
                    details={"log_tail": tail(log_path)},
                )
            time.sleep(_POLL_S)
        self._copy_log(start, log_path)
        return WorkerRun(read_response(response_path, log_path), log_path, 0)

    def _copy_log(self, start: int, target: Path) -> None:
        """Copy what the worker printed since *start* into this request's log."""

        if self._log is not None:
            self._log.flush()
        time.sleep(_POLL_S)  # let the worker's last lines of this request land
        with self.log_path.open("rb") as log:
            log.seek(start)
            target.write_bytes(log.read())


class WorkerPool:
    """Resident workers by runtime; a busy one is not waited for (a fresh process runs)."""

    def __init__(self, log_dir: Path):
        self.log_dir = log_dir
        self._workers: dict[tuple[str, str], ResidentWorker] = {}
        self._locks: dict[tuple[str, str], threading.Lock] = {}
        self._guard = threading.Lock()

    def run(
        self,
        runtime: BlenderRuntime,
        request: dict[str, Any],
        work_dir: Path,
        *,
        timeout_s: float = DEFAULT_TIMEOUT_S,
    ) -> WorkerRun:
        key = (runtime.kind, runtime.path)
        with self._guard:
            lock = self._locks.setdefault(key, threading.Lock())
            worker = self._workers.get(key)
            if worker is None:
                log = self.log_dir / f"worker-{os.getpid()}-{len(self._workers)}.log"
                worker = self._workers[key] = ResidentWorker(runtime, log)
        if not lock.acquire(blocking=False):
            return run_worker(runtime, request, work_dir, timeout_s=timeout_s)
        try:
            return worker.run(request, work_dir, timeout_s)
        finally:
            lock.release()

    def close(self) -> None:
        with self._guard:
            for worker in self._workers.values():
                worker.close()
            self._workers.clear()


__all__ = ["MAX_REQUESTS", "ResidentWorker", "WorkerPool"]
