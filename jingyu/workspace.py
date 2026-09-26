"""The workspace: the one directory tools may read from and write to.

Every path a tool accepts is resolved inside the workspace root; anything that
would escape it (``..``, absolute paths elsewhere, symlinks pointing out) is
rejected with ``workspace.path_escape``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from .errors import JingyuError

ENV_VAR = "JINGYU_WORKSPACE"
CANDIDATES_DIR = "candidates"
STAGING_DIR = ".staging"


@dataclass(frozen=True)
class Workspace:
    root: Path

    @classmethod
    def at(cls, root: str | Path | None = None) -> Workspace:
        """Use *root*, else ``$JINGYU_WORKSPACE``, else the current directory."""

        chosen = root or os.environ.get(ENV_VAR) or Path.cwd()
        path = Path(chosen).expanduser().resolve()
        path.mkdir(parents=True, exist_ok=True)
        return cls(path)

    @property
    def candidates_dir(self) -> Path:
        return self.root / CANDIDATES_DIR

    @property
    def staging_dir(self) -> Path:
        return self.candidates_dir / STAGING_DIR

    def resolve(self, relative: str | Path, *, must_exist: bool = True) -> Path:
        """Resolve a user-supplied path strictly inside the workspace."""

        candidate = Path(relative)
        target = (candidate if candidate.is_absolute() else self.root / candidate).resolve()
        if target != self.root and self.root not in target.parents:
            raise JingyuError(
                "workspace.path_escape",
                f"{relative!s} is outside the workspace",
                hint=f"Use a path inside {self.root}.",
            )
        if must_exist and not target.exists():
            raise JingyuError("workspace.not_found", f"{relative!s} does not exist")
        return target

    def relative(self, path: Path) -> str:
        return path.resolve().relative_to(self.root).as_posix()


__all__ = ["CANDIDATES_DIR", "ENV_VAR", "STAGING_DIR", "Workspace"]
