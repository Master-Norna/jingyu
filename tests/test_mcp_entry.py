"""The jingyu-mcp entry point, which must work without the optional mcp extra."""

from __future__ import annotations

import importlib
from typing import Any

import pytest

from jingyu import mcp_entry


def test_mcp_entry_prints_the_install_hint_without_mcp(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def missing(name: str) -> Any:
        raise ModuleNotFoundError("No module named 'mcp'", name="mcp")

    monkeypatch.setattr(importlib, "import_module", missing)
    assert mcp_entry.main() == 2
    assert mcp_entry.INSTALL_HINT in capsys.readouterr().err


@pytest.mark.parametrize("name", ["mcp_types", "anyio", "mcp.server"])
def test_mcp_entry_treats_any_mcp_dependency_as_missing_extra(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], name: str
) -> None:
    def missing(module: str) -> Any:
        raise ModuleNotFoundError(f"No module named {name!r}", name=name)

    monkeypatch.setattr(importlib, "import_module", missing)
    assert mcp_entry.main() == 2
    assert "jingyu[mcp]" in capsys.readouterr().err


def test_mcp_entry_reraises_unrelated_import_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing(name: str) -> Any:
        raise ModuleNotFoundError("No module named 'something_else'", name="something_else")

    monkeypatch.setattr(importlib, "import_module", missing)
    with pytest.raises(ModuleNotFoundError):
        mcp_entry.main()


def test_mcp_entry_runs_the_server_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    started: list[str] = []

    class FakeServer:
        @staticmethod
        def main() -> None:
            started.append("served")

    def load(name: str) -> Any:
        assert name == "jingyu.mcp_server"
        return FakeServer

    monkeypatch.setattr(importlib, "import_module", load)
    assert mcp_entry.main() == 0
    assert started == ["served"]
