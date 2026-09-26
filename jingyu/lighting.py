"""Queries over a candidate's light pass: which lights reach a point, what shades it.

The light pass (``light_mask.png`` plus ``light_map.json``) is written by the
Blender worker; see :mod:`jingyu.blender.diagnostics`.  Coordinates are the same
as for locating: pixels of the image or normalised ``u``/``v`` from the top-left.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from typing import Any

from PIL import Image

from .candidate import LIGHT_MAP_NAME, LIGHT_MASK_NAME, Candidate
from .canonical_json import load_file_strict
from .errors import JingyuError

LIGHT_MAP_SCHEMA = "jingyu.light-map.v1"


class LightMap:
    def __init__(self, mask: Image.Image, document: Mapping[str, Any]):
        if document.get("schema") != LIGHT_MAP_SCHEMA:
            raise JingyuError(
                "candidate.integrity_mismatch", "light_map.json has an unknown schema"
            )
        self.document = dict(document)
        self.width, self.height = mask.size
        self._rgba = mask.convert("RGBA").tobytes()
        self.lights = [dict(light) for light in document["lights"]]

    @classmethod
    def load(cls, candidate: Candidate) -> LightMap:
        if not candidate.has_light_map:
            raise JingyuError(
                "candidate.missing_pass",
                f"candidate {candidate.id} has no light pass",
                hint="Render it again; candidates from before the light pass lack it.",
            )
        with Image.open(candidate.file(LIGHT_MASK_NAME)) as image:
            image.load()
            mask = image.convert("RGBA")
        return cls(mask, load_file_strict(candidate.file(LIGHT_MAP_NAME)))

    def bits_at_uv(self, u: float, v: float) -> int | None:
        """The lit-light bits of the sample under (u, v); None where there is no surface."""

        x = min(self.width - 1, max(0, int(u * self.width)))
        y = min(self.height - 1, max(0, int(v * self.height)))
        offset = (y * self.width + x) * 4
        r, g, b, a = self._rgba[offset : offset + 4]
        if a == 0:
            return None
        return (r << 16) | (g << 8) | b

    def lights_at(self, bits: int) -> list[str]:
        return [light["id"] for light in self.lights if bits & (1 << int(light["bit"]))]

    def at_uv(self, u: float, v: float) -> dict[str, Any]:
        bits = self.bits_at_uv(u, v)
        if bits is None:
            return {"surface": False, "lit_by": [], "not_reached_by": []}
        lit = self.lights_at(bits)
        return {
            "surface": True,
            "lit_by": lit,
            "not_reached_by": [light["id"] for light in self.lights if light["id"] not in lit],
        }

    def in_region(self, u0: float, v0: float, u1: float, v1: float) -> dict[str, Any]:
        """Share of the region's surface samples each light reaches."""

        x0, x1 = int(u0 * self.width), max(int(u0 * self.width) + 1, int(u1 * self.width))
        y0, y1 = int(v0 * self.height), max(int(v0 * self.height) + 1, int(v1 * self.height))
        counts: Counter[str] = Counter()
        surface = 0
        for y in range(max(0, y0), min(self.height, y1)):
            for x in range(max(0, x0), min(self.width, x1)):
                bits = self.bits_at_uv((x + 0.5) / self.width, (y + 0.5) / self.height)
                if bits is None:
                    continue
                surface += 1
                counts.update(self.lights_at(bits))
        return {
            "surface_samples": surface,
            "lit_fraction": {
                light["id"]: round(counts[light["id"]] / surface, 4) if surface else 0.0
                for light in self.lights
            },
        }

    def lit_mask(self, light_id: str) -> Image.Image:
        """A greyscale mask (255 = reached directly by *light_id*) at the pass resolution."""

        bit = next((int(light["bit"]) for light in self.lights if light["id"] == light_id), None)
        if bit is None:
            raise JingyuError(
                "light.unknown_light",
                f"the light pass has no light {light_id!r}",
                hint=f"lights: {', '.join(light['id'] for light in self.lights) or 'none'}",
            )
        data = bytearray(self.width * self.height)
        for i in range(self.width * self.height):
            r, g, b, a = self._rgba[i * 4 : i * 4 + 4]
            if a and ((r << 16) | (g << 8) | b) & (1 << bit):
                data[i] = 255
        return Image.frombytes("L", (self.width, self.height), bytes(data))

    def summary(self) -> dict[str, Any]:
        doc = self.document
        return {
            "lights": self.lights,
            "objects": doc["objects"],
            "shadows": doc["shadows"],
            "in_view": doc["in_view"],
            "openness": doc["openness"],
        }


__all__ = ["LIGHT_MAP_SCHEMA", "LightMap"]
