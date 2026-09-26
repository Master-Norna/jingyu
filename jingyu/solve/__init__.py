"""Solve scene parameters from a goal instead of guessing and re-rendering.

A model usually knows what it wants ("the still life fills the left half of
the frame", "the sun lands on the table through the window") but can only
reach it by changing a number, rendering and looking again.  These solvers work
backwards from the goal on the host, with the same placement, camera model and
rays the renderer uses, and return the edit that achieves it together with a
prediction of the result.
"""

from __future__ import annotations

from .framing import FramingRequest, frame_subject
from .sun import SunRequest, aim_sun

__all__ = ["FramingRequest", "SunRequest", "aim_sun", "frame_subject"]
