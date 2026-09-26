"""Point at the image, get back the scene: queries over a candidate's id mask.

This is how "I can see something is wrong here, but cannot say what" becomes
an edit: a model names a pixel or a region, and the answer is the object, its
material and the JSON Pointers to edit in the scene description.

Coordinates: pixels ``x`` (left to right) and ``y`` (top to bottom) from the
top-left corner, or normalised ``u``/``v`` in [0, 1] from the same corner.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from PIL import Image

from .candidate import ID_MAP_NAME, ID_MASK_NAME, Candidate
from .canonical_json import load_file_strict
from .errors import JingyuError, pointer_join
from .idmask import BACKGROUND, ID_MAP_SCHEMA, decode_rgba


@dataclass(frozen=True)
class Box:
    x0: int
    y0: int
    x1: int  # exclusive
    y1: int  # exclusive

    @property
    def area(self) -> int:
        return max(0, self.x1 - self.x0) * max(0, self.y1 - self.y0)


class IdMask:
    """A decoded id mask plus the entries it indexes."""

    def __init__(self, image: Image.Image, id_map: Mapping[str, Any], scene: Mapping[str, Any]):
        if image.mode != "RGBA":
            image = image.convert("RGBA")
        if id_map.get("schema") != ID_MAP_SCHEMA:
            raise JingyuError("candidate.integrity_mismatch", "id_map.json has an unknown schema")
        self.width, self.height = image.size
        self._rgba = image.tobytes()
        self._objects: dict[int, dict[str, Any]] = {
            int(k): dict(v) for k, v in id_map["objects"].items()
        }
        material_index = {m["id"]: i for i, m in enumerate(scene.get("materials", []))}
        self._material_pointer = {
            m: pointer_join("materials", i) for m, i in material_index.items()
        }
        for entry in self._objects.values():
            entry["material_pointer"] = self._material_pointer.get(entry.get("material"))
            if entry.get("part") is not None:
                entry["part_pointer"] = entry["pointer"] + pointer_join(
                    "part_materials", entry["part"]
                )
        #: The object's own material (the parts of an assembled object may differ).
        self._own_material = {
            o.get("id"): o.get("material") for o in scene.get("objects", []) if isinstance(o, dict)
        }
        self._scene_objects = [
            o["id"]
            for o in scene.get("objects", [])
            if o.get("visible", True) and o.get("camera_visible", True)
        ]

    @classmethod
    def load(cls, candidate: Candidate) -> IdMask:
        if not candidate.has_id_mask:
            raise JingyuError("candidate.missing_pass", f"candidate {candidate.id} has no id mask")
        with Image.open(candidate.file(ID_MASK_NAME)) as image:
            image.load()
            decoded = image.convert("RGBA")
        id_map = load_file_strict(candidate.file(ID_MAP_NAME))
        return cls(decoded, id_map, candidate.scene())

    # ------------------------------------------------------------ primitives

    def index_at(self, x: int, y: int) -> int | None:
        offset = (y * self.width + x) * 4
        r, g, b, a = self._rgba[offset : offset + 4]
        return decode_rgba(r, g, b, a)

    def entry(self, index: int) -> dict[str, Any] | None:
        return self._objects.get(index)

    def to_pixel(self, point: Mapping[str, Any]) -> tuple[int, int]:
        if "x" in point and "y" in point:
            x, y = int(point["x"]), int(point["y"])
        elif "u" in point and "v" in point:
            u, v = float(point["u"]), float(point["v"])
            # u = 1 or v = 1 is the far edge of the last pixel; beyond it is outside.
            x = self.width - 1 if u == 1 else math.floor(u * self.width)
            y = self.height - 1 if v == 1 else math.floor(v * self.height)
        else:
            raise JingyuError("tool.invalid_arguments", "a point needs x/y or u/v")
        if not (0 <= x < self.width and 0 <= y < self.height):
            raise JingyuError(
                "locate.out_of_bounds",
                f"({x}, {y}) is outside the {self.width}x{self.height} image",
            )
        return x, y

    def to_box(self, region: Mapping[str, Any]) -> Box:
        if all(k in region for k in ("x0", "y0", "x1", "y1")):
            box = Box(int(region["x0"]), int(region["y0"]), int(region["x1"]), int(region["y1"]))
        elif all(k in region for k in ("u0", "v0", "u1", "v1")):
            box = Box(
                math.floor(float(region["u0"]) * self.width),
                math.floor(float(region["v0"]) * self.height),
                round(float(region["u1"]) * self.width),
                round(float(region["v1"]) * self.height),
            )
        else:
            raise JingyuError("tool.invalid_arguments", "a region needs x0/y0/x1/y1 or u0/v0/u1/v1")
        if not (0 <= box.x0 < box.x1 <= self.width and 0 <= box.y0 < box.y1 <= self.height):
            raise JingyuError(
                "locate.out_of_bounds",
                f"region {box} is empty or outside the {self.width}x{self.height} image",
            )
        return box

    def background_pixels(self, box: Box) -> list[tuple[int, int]]:
        """Pixels of *box* that show no object: the world behind everything."""

        return [
            (x, y)
            for y in range(box.y0, box.y1)
            for x in range(box.x0, box.x1)
            if self.index_at(x, y) == BACKGROUND
        ]

    def counts(self, box: Box) -> Counter[int | None]:
        counts: Counter[int | None] = Counter()
        for y in range(box.y0, box.y1):
            for x in range(box.x0, box.x1):
                counts[self.index_at(x, y)] += 1
        return counts

    # -------------------------------------------------------------- queries

    def describe_entry(self, index: int) -> dict[str, Any]:
        entry = self._objects.get(index)
        if entry is None:
            raise JingyuError("locate.ambiguous_pixels", f"mask index {index} is not in the id map")
        described = {
            "object": entry["object"],
            "pointer": entry["pointer"],
            "material": entry.get("material"),
            "material_pointer": entry.get("material_pointer"),
        }
        if entry.get("part") is not None:
            described["part"] = entry["part"]
            described["part_pointer"] = entry["part_pointer"]
        return described

    def object_of(self, index: int) -> str | None:
        entry = self._objects.get(index)
        return None if entry is None else str(entry["object"])

    def locate_point(self, point: Mapping[str, Any], radius: int = 3) -> dict[str, Any]:
        x, y = self.to_pixel(point)
        index = self.index_at(x, y)
        box = Box(
            max(0, x - radius),
            max(0, y - radius),
            min(self.width, x + radius + 1),
            min(self.height, y + radius + 1),
        )
        hit: dict[str, Any] | None
        if index is None or index == BACKGROUND:
            hit = None
        else:
            hit = self.describe_entry(index)
        return {
            "point": {"x": x, "y": y, "u": (x + 0.5) / self.width, "v": (y + 0.5) / self.height},
            "hit": hit,
            "background": index == BACKGROUND,
            "neighborhood": self._breakdown(self.counts(box), box.area, None),
        }

    def locate_region(self, region: Mapping[str, Any]) -> dict[str, Any]:
        box = self.to_box(region)
        counts = self.counts(box)
        totals = self._visible_totals() if counts else {}
        return {
            "region": {"x0": box.x0, "y0": box.y0, "x1": box.x1, "y1": box.y1},
            "pixels": box.area,
            "background_fraction": counts.get(BACKGROUND, 0) / box.area,
            "objects": self._breakdown(counts, box.area, totals),
        }

    def layout(self) -> dict[str, Any]:
        """Where each object sits in the frame: area, bounding box and centroid."""

        stats: dict[int, list[float]] = {}
        unassigned = 0
        background = 0
        for y in range(self.height):
            for x in range(self.width):
                index = self.index_at(x, y)
                if index is None or (index != BACKGROUND and index not in self._objects):
                    unassigned += 1
                    continue
                if index == BACKGROUND:
                    background += 1
                    continue
                s = stats.get(index)
                if s is None:
                    stats[index] = [1, x, y, x, y, x, y]
                else:
                    s[0] += 1
                    s[1] = min(s[1], x)
                    s[2] = min(s[2], y)
                    s[3] = max(s[3], x)
                    s[4] = max(s[4], y)
                    s[5] += x
                    s[6] += y
        total = self.width * self.height
        merged: dict[str, list[float]] = {}
        parts: dict[str, list[dict[str, Any]]] = {}
        for index, (count, x0, y0, x1, y1, sx, sy) in stats.items():
            entry = self._objects[index]
            name = str(entry["object"])
            if entry.get("part") is not None:
                parts.setdefault(name, []).append(
                    {
                        "part": entry["part"],
                        "material": entry.get("material"),
                        "part_pointer": entry["part_pointer"],
                        "pixels": int(count),
                    }
                )
            m = merged.get(name)
            if m is None:
                merged[name] = [count, x0, y0, x1, y1, sx, sy]
            else:
                m[0] += count
                m[1], m[2] = min(m[1], x0), min(m[2], y0)
                m[3], m[4] = max(m[3], x1), max(m[4], y1)
                m[5] += sx
                m[6] += sy
        first = {str(e["object"]): e for e in self._objects.values()}
        objects = []
        for name, (count, x0, y0, x1, y1, sx, sy) in merged.items():
            own = self._own_material.get(name)
            entry = {
                "object": name,
                "pointer": first[name]["pointer"],
                "material": own,
                "material_pointer": self._material_pointer.get(own),
            }
            if name in parts:
                entry["parts"] = sorted(
                    (
                        {**p, "fraction_of_object": round(p["pixels"] / count, 4)}
                        for p in parts[name]
                    ),
                    key=lambda p: -p["pixels"],
                )
            objects.append(
                {
                    **entry,
                    "pixels": int(count),
                    "area_fraction": count / total,
                    "bbox": {"x0": int(x0), "y0": int(y0), "x1": int(x1) + 1, "y1": int(y1) + 1},
                    "bbox_uv": {
                        "u0": x0 / self.width,
                        "v0": y0 / self.height,
                        "u1": (x1 + 1) / self.width,
                        "v1": (y1 + 1) / self.height,
                    },
                    "centroid_uv": {
                        "u": (sx / count + 0.5) / self.width,
                        "v": (sy / count + 0.5) / self.height,
                    },
                    "touches_frame_edge": x0 == 0
                    or y0 == 0
                    or x1 == self.width - 1
                    or y1 == self.height - 1,
                }
            )
        objects.sort(key=lambda o: o["pixels"], reverse=True)
        seen = {o["object"] for o in objects}
        return {
            "width": self.width,
            "height": self.height,
            "background_fraction": background / total,
            "unassigned_pixels": unassigned,
            "objects": objects,
            "not_in_frame": [oid for oid in self._scene_objects if oid not in seen],
        }

    def _visible_totals(self) -> dict[int, int]:
        """Per index, the pixels of the whole object it belongs to (all its parts)."""

        full = self.counts(Box(0, 0, self.width, self.height))
        by_object: dict[str | None, int] = {}
        for index, count in full.items():
            if index:
                name = self.object_of(index)
                by_object[name] = by_object.get(name, 0) + count
        return {k: by_object[self.object_of(k)] for k in full if k}

    def _breakdown(
        self, counts: Counter[int | None], area: int, totals: Mapping[int, int] | None
    ) -> list[dict[str, Any]]:
        rows = []
        for index, count in counts.most_common():
            if index is None or index == BACKGROUND or index not in self._objects:
                continue
            row = {
                **self.describe_entry(index),
                "pixels": count,
                "fraction_of_region": count / area,
            }
            if totals is not None and totals.get(index):
                row["fraction_of_object"] = count / totals[index]
            rows.append(row)
        return rows


__all__ = ["Box", "IdMask"]
