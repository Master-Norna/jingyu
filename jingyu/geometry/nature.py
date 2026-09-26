"""Natural forms: terrain, rocks, trees and grass.

Nature is where parameters beat assets most clearly: no two rocks are alike,
and a seed gives another one of the same kind.  Everything is built from the
deterministic noise in :mod:`.noise`, on the host, so placement (``rest_on`` a
hillside) and physics see the same shape the renderer draws.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from ..generator import GeneratorDef, ParamProblem, integer, number
from .mesh import MeshData, Vec3
from .noise import fbm, random
from .pieces import Piece, assemble, at
from .shapes import tube

_SEED = integer(
    "Another number, another one of the same kind.", default=0, minimum=0, maximum=100000
)


# -------------------------------------------------------------------- terrain


def _terrain(p: Mapping[str, Any]) -> MeshData:
    width, depth = (float(v) for v in p["size"])
    relief, base = float(p["relief"]), float(p["base"])
    feature = float(p["feature_size"])
    seed, octaves, ridged = int(p["seed"]), int(p["detail"]), float(p["ridges"])
    falloff = float(p["edge_falloff"])
    cells = int(p["resolution"])
    nx = max(2, round(cells * width / max(width, depth)))
    ny = max(2, round(cells * depth / max(width, depth)))

    def height(i: int, j: int) -> float:
        x, y = (i / nx - 0.5) * width, (j / ny - 0.5) * depth
        h = 0.5 + 0.5 * fbm(x / feature, y / feature, octaves=octaves, seed=seed, ridged=ridged)
        if falloff > 0:
            edge = min(i / nx, 1 - i / nx, j / ny, 1 - j / ny) * 2.0
            t = min(1.0, edge / falloff)
            h *= t * t * (3 - 2 * t)
        return base + relief * h

    vertices: list[Vec3] = [
        ((i / nx - 0.5) * width, (j / ny - 0.5) * depth, height(i, j))
        for j in range(ny + 1)
        for i in range(nx + 1)
    ]

    def top(i: int, j: int) -> int:
        return j * (nx + 1) + i

    faces: list[tuple[int, ...]] = [
        (top(i, j), top(i + 1, j), top(i + 1, j + 1), top(i, j + 1))
        for j in range(ny)
        for i in range(nx)
    ]
    rim = (
        [top(i, 0) for i in range(nx)]
        + [top(nx, j) for j in range(ny)]
        + [top(i, ny) for i in range(nx, 0, -1)]
        + [top(0, j) for j in range(ny, 0, -1)]
    )
    below = []
    for index in rim:
        x, y, _ = vertices[index]
        below.append(len(vertices))
        vertices.append((x, y, 0.0))
    count = len(rim)
    sharp: list[tuple[int, int]] = []
    for k in range(count):
        a, b = rim[k], rim[(k + 1) % count]
        faces.append((b, a, below[k], below[(k + 1) % count]))
        sharp.append((a, b))
    faces.append(tuple(reversed(below)))
    return MeshData(tuple(vertices), tuple(faces), smooth=True, sharp_edges=tuple(sharp))


TERRAIN = GeneratorDef[MeshData](
    name="terrain",
    summary=(
        "A piece of land: rolling hills to ridged mountains, as a solid slab standing on "
        "its origin. Things stand on it with rest_on; water is a plane at the right height."
    ),
    params={
        "size": {
            "type": "array",
            "description": "Width (x) and depth (y) in metres.",
            "items": {"type": "number", "exclusiveMinimum": 0},
            "minItems": 2,
            "maxItems": 2,
        },
        "relief": number(
            "Height from the lowest dip to the highest top, metres.", default=2.0, minimum=0
        ),
        "feature_size": number(
            "Typical width of a hill or valley in metres.", default=10.0, exclusive_minimum=0
        ),
        "detail": integer(
            "Layers of finer bumps on the big shapes.", default=5, minimum=1, maximum=8
        ),
        "ridges": number(
            "0 rounded hills, 1 sharp crests and ravines.", default=0.0, minimum=0, maximum=1
        ),
        "edge_falloff": number(
            "Lower the land toward the edges over this share of its half-size (an island, "
            "a patch that blends into a flat floor); 0 keeps the edges as they are.",
            default=0.0,
            minimum=0,
            maximum=1,
        ),
        "base": number(
            "Solid ground under the lowest dip, metres.", default=0.2, exclusive_minimum=0
        ),
        "resolution": integer(
            "Grid cells along the longer side; more is finer and heavier.",
            default=128,
            minimum=4,
            maximum=512,
        ),
        "seed": _SEED,
    },
    run=_terrain,
    examples=(
        {"op": "terrain", "size": [40, 30], "relief": 4, "feature_size": 15},
        {
            "op": "terrain",
            "size": [200, 200],
            "relief": 60,
            "feature_size": 80,
            "ridges": 0.8,
            "resolution": 160,
        },
    ),
)


# ----------------------------------------------------------------------- rock


def _cube_sphere(n: int) -> MeshData:
    """A unit sphere made from a subdivided cube: even cells, no pinched poles."""

    samples = [-1.0 + 2.0 * k / n for k in range(n + 1)]
    vertices: list[Vec3] = []
    index: dict[tuple[int, int, int], int] = {}

    def vertex(key: tuple[int, int, int]) -> int:
        if key not in index:
            x, y, z = (samples[k] for k in key)
            length = math.sqrt(x * x + y * y + z * z)
            index[key] = len(vertices)
            vertices.append((x / length, y / length, z / length))
        return index[key]

    faces: list[tuple[int, ...]] = []
    for axis in range(3):
        b, c = (axis + 1) % 3, (axis + 2) % 3
        for side in (0, n):
            for i in range(n):
                for j in range(n):
                    quad = []
                    for di, dj in ((0, 0), (1, 0), (1, 1), (0, 1)):
                        key = [0, 0, 0]
                        key[axis], key[b], key[c] = side, i + di, j + dj
                        quad.append(vertex((key[0], key[1], key[2])))
                    faces.append(tuple(quad) if side else tuple(reversed(quad)))
    return MeshData(tuple(vertices), tuple(faces), smooth=True)


def _blob(radii: Vec3, roughness: float, feature: float, seed: int, n: int = 20) -> list[Vec3]:
    """Points of a noisy ellipsoid around the origin."""

    sphere = _cube_sphere(n)
    points = []
    for x, y, z in sphere.vertices:
        bump = fbm(x / feature, y / feature, z / feature, octaves=5, seed=seed)
        r = 1.0 + 0.45 * roughness * bump
        points.append((x * radii[0] * r, y * radii[1] * r, z * radii[2] * r))
    return points


def _rock(p: Mapping[str, Any]) -> MeshData:
    sx, sy, sz = (float(v) / 2.0 for v in p["size"])
    sphere = _cube_sphere(24)
    points = _blob(
        (sx, sy, sz), float(p["roughness"]), float(p["feature_size"]), int(p["seed"]), 24
    )
    low = min(z for _, _, z in points)
    cut = low + float(p["sink"]) * (max(z for _, _, z in points) - low)
    moved = [(x, y, max(z, cut) - cut) for x, y, z in points]
    return MeshData(tuple(moved), sphere.faces, smooth=True)


ROCK = GeneratorDef[MeshData](
    name="rock",
    summary=(
        "A stone or boulder: a lumpy, irregular solid, flat where it sits on the ground. "
        "Every seed is another rock; pebbles are small rocks, scattered."
    ),
    params={
        "size": {
            "type": "array",
            "description": "Rough width (x), depth (y) and height (z) in metres.",
            "items": {"type": "number", "exclusiveMinimum": 0},
            "minItems": 3,
            "maxItems": 3,
            "default": [0.6, 0.45, 0.35],
        },
        "roughness": number(
            "0 a smooth river pebble, 1 a jagged broken block.", default=0.5, minimum=0, maximum=1
        ),
        "feature_size": number(
            "Size of the lumps relative to the rock: small is craggy, large is gently wavy.",
            default=0.6,
            exclusive_minimum=0,
        ),
        "sink": number(
            "How much of its height is sunk into the ground (cut flat): 0 resting on its "
            "lowest point, 0.3 settled in.",
            default=0.15,
            minimum=0,
            maximum=0.8,
        ),
        "seed": _SEED,
    },
    run=_rock,
    examples=(
        {"op": "rock"},
        {"op": "rock", "size": [0.05, 0.04, 0.025], "roughness": 0.2, "seed": 4},
    ),
)


# ----------------------------------------------------------------------- tree

_TREE_PARTS = {"trunk": "Trunk and branches.", "foliage": "Leaves or needles."}
_BARK = {"family": "wood", "color": "#5b4a3a", "grain_color": "#3a2e24", "grain_axis": "z",
         "ring_size": 0.015, "figure": 0.9, "pores": 1.0, "finish": 0.0}  # fmt: skip
_LEAVES = {"family": "plastic", "color": "#4d6b34", "gloss": 0.15, "texture": 0.9}


def _tree(p: Mapping[str, Any]) -> MeshData:
    height = float(p["height"])
    radius = float(p["trunk_radius"])
    crown = float(p["crown_radius"]) or height * 0.3
    start = float(p["crown_start"])
    seed = int(p["seed"])
    lean = math.radians(float(p["lean"]))
    heading = random(0, seed, 9) * math.tau
    dx, dy = math.cos(heading) * math.tan(lean), math.sin(heading) * math.tan(lean)

    def along(t: float, wiggle: float = 1.0) -> Vec3:
        z = height * t
        sway = 0.04 * height * wiggle * math.sin(t * 5.0 + seed)
        return (dx * z + sway * math.cos(heading + 1.3), dy * z + sway * math.sin(heading + 1.3), z)

    pieces: list[Piece] = []
    top = 0.92 if p["kind"] == "conifer" else 0.8
    stem = [along(k / 10.0 * top) for k in range(11)]
    radii = [radius * (1.0 - 0.8 * k / 10.0) for k in range(11)]
    pieces.append(Piece(tube(stem, radii, 16), "trunk"))
    centre = along(start + (1.0 - start) * 0.5)
    if p["kind"] == "conifer":
        tiers = max(3, int(p["clumps"]))
        for k in range(tiers):
            t = start + (1.0 - start) * k / tiers
            r = crown * (1.0 - k / tiers) * (0.85 + 0.3 * random(k, seed, 3))
            tier_height = (1.0 - start) * height / tiers * 2.6
            points = _cone(r, tier_height, seed + k)
            x, y, z = along(t)
            pieces.append(Piece(points, "foliage", at((x, y, z))))
    else:
        branches = 4
        for k in range(branches):
            t = start + (0.9 - start) * k / branches
            angle = heading + k * 2.4 + random(k, seed, 1)
            base = along(t)
            tip = (
                centre[0] + math.cos(angle) * crown * 0.6,
                centre[1] + math.sin(angle) * crown * 0.6,
                base[2] + crown * 0.5,
            )
            mid = (
                (base[0] + tip[0]) / 2.0,
                (base[1] + tip[1]) / 2.0,
                (base[2] + tip[2]) / 2.0 + crown * 0.1,
            )
            branch = tube([base, mid, tip], [radius * 0.45, radius * 0.3, radius * 0.15], 10)
            pieces.append(Piece(branch, "trunk"))
        clumps = max(1, int(p["clumps"]))
        for k in range(clumps):
            angle = heading + k * 2.39996
            reach = crown * 0.55 * math.sqrt((k + 0.5) / clumps)
            lift = (random(k, seed, 2) - 0.3) * crown * 0.6
            x = centre[0] + math.cos(angle) * reach
            y = centre[1] + math.sin(angle) * reach
            z = centre[2] + lift
            r = crown * (0.42 + 0.2 * random(k, seed, 4))
            blob = MeshData(
                tuple(_blob((r, r, r * 0.8), 0.8, 0.35, seed + k * 7, 12)),
                _cube_sphere(12).faces,
                smooth=True,
            )
            pieces.append(Piece(blob, "foliage", at((x, y, z))))
    tree = assemble(pieces, list(_TREE_PARTS))
    # The leaning trunk's first cross-section tilts a little below the ground: lift the
    # whole tree so it stands on its origin.
    low = tree.bounds()[0][2]
    lifted = tuple((x, y, z - low) for x, y, z in tree.vertices)
    return MeshData(
        lifted,
        tree.faces,
        tree.smooth,
        tree.sharp_edges,
        parts=tree.parts,
        face_parts=tree.face_parts,
    )


def _cone(radius: float, height: float, seed: int) -> MeshData:
    """One tier of a conifer: a ragged skirt of drooping branches, hollow underneath.

    The sides curve in (the branch tips droop), the rim is jagged with branch
    ends, and the underside rises into the tier, so seen from the side it is a
    skirt rather than a flat disc.
    """

    segments = 32
    rings = 7
    phase = random(0, seed, 11) * math.tau
    vertices: list[Vec3] = []
    for ring in range(rings - 1):  # the last ring is the tip, a single point
        t = ring / (rings - 1)
        for s in range(segments):
            angle = math.tau * s / segments
            rag = (
                1.0
                + 0.35 * (random(ring * segments + s, seed, 7) - 0.5)
                + 0.12 * math.sin(9.0 * angle + phase)
            )
            r = radius * (1.0 - t) ** 1.4 * rag
            vertices.append((math.cos(angle) * r, math.sin(angle) * r, height * t))
    tip = len(vertices)
    vertices.append((0.0, 0.0, height))
    hollow = len(vertices)
    vertices.append((0.0, 0.0, height * 0.35))
    faces: list[tuple[int, ...]] = []
    for ring in range(rings - 2):
        for s in range(segments):
            a = ring * segments + s
            b = ring * segments + (s + 1) % segments
            faces.append((a, b, b + segments, a + segments))
    last = (rings - 2) * segments
    for s in range(segments):
        faces.append((last + s, last + (s + 1) % segments, tip))
        faces.append((hollow, (s + 1) % segments, s))
    return MeshData(tuple(vertices), tuple(faces), smooth=True)


def _check_tree(p: Mapping[str, Any]) -> list[ParamProblem]:
    if float(p["trunk_radius"]) * 8 > float(p["height"]):
        return [("trunk_radius", "the trunk is too thick for the tree's height")]
    return []


TREE = GeneratorDef[MeshData](
    name="tree",
    summary=(
        "A tree: a leaning trunk with branches under a crown of leafy clumps (broadleaf), "
        "or a tapering spire of needle tiers (conifer). Seen from a distance; parts trunk "
        "and foliage have bark and leaf materials of their own."
    ),
    params={
        "kind": {
            "enum": ["broadleaf", "conifer"],
            "default": "broadleaf",
            "description": "Broad crown or spire.",
        },
        "height": number("Height in metres.", default=6.0, exclusive_minimum=0),
        "trunk_radius": number(
            "Radius of the trunk at the ground, metres.", default=0.15, exclusive_minimum=0
        ),
        "crown_radius": number(
            "Radius of the crown in metres; 0 is 30 % of the height.", default=0.0, minimum=0
        ),
        "crown_start": number(
            "Where the crown begins, as a share of the height.",
            default=0.35,
            minimum=0.05,
            maximum=0.9,
        ),
        "clumps": integer(
            "Leaf clumps (broadleaf) or needle tiers (conifer).", default=12, minimum=1, maximum=40
        ),
        "lean": number("Lean of the trunk in degrees.", default=3.0, minimum=0, maximum=30),
        "seed": _SEED,
    },
    run=_tree,
    check=_check_tree,
    examples=(
        {"op": "tree"},
        {"op": "tree", "kind": "conifer", "height": 9, "trunk_radius": 0.2, "clumps": 7},
    ),
    parts=_TREE_PARTS,
    part_defaults={"trunk": _BARK, "foliage": _LEAVES},
)


# ---------------------------------------------------------------------- grass


def _grass(p: Mapping[str, Any]) -> MeshData:
    width, depth = (float(v) for v in p["size"])
    density = float(p["density"])
    count = min(40000, int(width * depth * density))
    tall, jitter = float(p["height"]), float(p["height_jitter"])
    blade = float(p["blade_width"])
    bend = float(p["bend"])
    seed = int(p["seed"])
    patchy = float(p["patchiness"])
    vertices: list[Vec3] = []
    faces: list[tuple[int, ...]] = []
    for k in range(count):
        x = (random(k, seed, 1) - 0.5) * width
        y = (random(k, seed, 2) - 0.5) * depth
        if patchy > 0 and (0.5 + 0.5 * fbm(x / 0.6, y / 0.6, seed=seed)) < patchy * 0.7:
            continue
        h = tall * (1.0 + jitter * (random(k, seed, 3) * 2 - 1))
        facing = random(k, seed, 4) * math.tau
        droop = bend * h * random(k, seed, 5)
        c, s = math.cos(facing), math.sin(facing)
        o = len(vertices)
        half = blade / 2.0
        vertices += [
            (x - s * half, y + c * half, 0.0),
            (x + s * half, y - c * half, 0.0),
            (x + c * half * 0.6, y + s * half * 0.6, 0.0),
            (x + c * droop, y + s * droop, h),
        ]
        faces += [(o, o + 2, o + 1), (o, o + 1, o + 3), (o + 1, o + 2, o + 3), (o + 2, o, o + 3)]
    return MeshData(tuple(vertices), tuple(faces), smooth=False)


GRASS = GeneratorDef[MeshData](
    name="grass",
    summary=(
        "A patch of grass: thousands of thin blades of varied height, bending a little, "
        "optionally in clumps. Give it a green, matte material; rest it on the ground."
    ),
    params={
        "size": {
            "type": "array",
            "description": "Width (x) and depth (y) of the patch in metres.",
            "items": {"type": "number", "exclusiveMinimum": 0},
            "minItems": 2,
            "maxItems": 2,
        },
        "density": number("Blades per square metre.", default=1500.0, exclusive_minimum=0),
        "height": number("Typical blade height in metres.", default=0.12, exclusive_minimum=0),
        "height_jitter": number(
            "How much heights vary, 0 to 0.9.", default=0.5, minimum=0, maximum=0.9
        ),
        "blade_width": number("Blade width in metres.", default=0.004, exclusive_minimum=0),
        "bend": number(
            "How far blades lean over, as a share of their height.",
            default=0.3,
            minimum=0,
            maximum=1,
        ),
        "patchiness": number(
            "0 an even lawn, 1 separate clumps with bare ground between.",
            default=0.0,
            minimum=0,
            maximum=1,
        ),
        "seed": _SEED,
    },
    run=_grass,
    examples=({"op": "grass", "size": [1.0, 1.0], "density": 800},),
)

NATURE: tuple[GeneratorDef[MeshData], ...] = (TERRAIN, ROCK, TREE, GRASS)

__all__ = ["GRASS", "NATURE", "ROCK", "TERRAIN", "TREE"]
