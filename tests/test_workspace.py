from __future__ import annotations

import os
from pathlib import Path

import pytest

from jingyu.errors import JingyuError
from jingyu.workspace import ENV_VAR, Workspace


def _code(workspace: Workspace, path: str | Path, **kwargs: bool) -> str:
    with pytest.raises(JingyuError) as info:
        workspace.resolve(path, **kwargs)
    return info.value.code


def test_at_creates_and_resolves_the_root(tmp_path: Path) -> None:
    workspace = Workspace.at(tmp_path / "a" / "b")
    assert workspace.root == (tmp_path / "a" / "b").resolve()
    assert workspace.root.is_dir()
    assert workspace.candidates_dir == workspace.root / "candidates"
    assert workspace.staging_dir == workspace.root / "candidates" / ".staging"


def test_at_falls_back_to_the_environment_then_the_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(ENV_VAR, str(tmp_path / "from-env"))
    assert Workspace.at().root == (tmp_path / "from-env").resolve()
    monkeypatch.delenv(ENV_VAR)
    monkeypatch.chdir(tmp_path)
    assert Workspace.at().root == tmp_path.resolve()


def test_resolve_accepts_paths_inside(workspace: Workspace) -> None:
    (workspace.root / "scenes").mkdir()
    scene = workspace.root / "scenes" / "a.json"
    scene.write_text("{}", encoding="utf-8")
    assert workspace.resolve("scenes/a.json") == scene
    assert workspace.resolve("scenes/../scenes/a.json") == scene
    assert workspace.resolve(scene) == scene
    assert workspace.resolve(".") == workspace.root
    assert workspace.relative(scene) == "scenes/a.json"


@pytest.mark.parametrize("path", ["..", "../x.json", "a/../../x.json", "scenes/../../.."])
def test_resolve_rejects_parent_escapes(workspace: Workspace, path: str) -> None:
    assert _code(workspace, path) == "workspace.path_escape"
    assert _code(workspace, path, must_exist=False) == "workspace.path_escape"


def test_resolve_rejects_absolute_paths_outside(workspace: Workspace, tmp_path: Path) -> None:
    outside = tmp_path / "outside.json"
    outside.write_text("{}", encoding="utf-8")
    assert _code(workspace, outside) == "workspace.path_escape"
    assert _code(workspace, str(outside)) == "workspace.path_escape"
    # A sibling whose name merely starts with the root's name is still outside.
    sibling = Path(str(workspace.root) + "-evil")
    sibling.mkdir()
    assert _code(workspace, sibling) == "workspace.path_escape"


def test_resolve_rejects_symlinks_pointing_out(workspace: Workspace, tmp_path: Path) -> None:
    target = tmp_path / "secret.json"
    target.write_text("{}", encoding="utf-8")
    link = workspace.root / "link.json"
    try:
        os.symlink(target, link)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"symlinks unavailable: {exc}")
    assert _code(workspace, "link.json") == "workspace.path_escape"
    outside_dir = tmp_path / "elsewhere"
    outside_dir.mkdir()
    os.symlink(outside_dir, workspace.root / "dir-link", target_is_directory=True)
    assert _code(workspace, "dir-link/new.json", must_exist=False) == "workspace.path_escape"


def test_resolve_reports_missing_files(workspace: Workspace) -> None:
    assert _code(workspace, "missing.json") == "workspace.not_found"
    assert workspace.resolve("missing.json", must_exist=False) == workspace.root / "missing.json"
