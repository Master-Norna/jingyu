"""Shared fixtures: a temporary workspace and synthetic candidates that need no Blender."""

from __future__ import annotations

import itertools
from collections.abc import Callable
from pathlib import Path

import pytest
from helpers import CandidateFactory

from jingyu.candidate import Candidate
from jingyu.workspace import Workspace


@pytest.fixture
def workspace(tmp_path: Path) -> Workspace:
    return Workspace.at(tmp_path / "workspace")


@pytest.fixture
def make_candidate(workspace: Workspace) -> CandidateFactory:
    return CandidateFactory(workspace, itertools.count(1))


@pytest.fixture
def candidate(make_candidate: Callable[..., Candidate]) -> Candidate:
    return make_candidate()
