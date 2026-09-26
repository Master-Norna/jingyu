"""Assembled geometry: walls with windows, rooms, tables, chairs and shelves.

Each is still one generator call and one object, built from pieces that belong
to named parts (a table's top and base, a room's walls, floor, ceiling, window
frames and glass), so every part can carry its own material and every pixel can
be traced to the part it shows.  Dimensions follow how furniture is described:
overall size, thicknesses, insets.  Nothing is placed by hand-computed
coordinates: a window is an opening in a named wall of the room.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from ..generator import GeneratorDef, ParamProblem, integer, number
from .mesh import MeshData, Portal
from .pieces import Piece, assemble, at, cylinder, rounded_box, tapered_leg
from .transform import Mat4, multiply

_GLASS = 0.006
_FRAME_PROUD = 0.01
_CLEAR_GLASS: Mapping[str, Any] = {"family": "glass", "thin": True}

_BEVEL = number(
    "Radius of rounded edges in metres; a few millimetres catch the light like a real, "
    "handled object. 0 is razor-sharp.",
    default=0.003,
    minimum=0.0,
)


def _size3(description: str) -> dict[str, Any]:
    return {
        "type": "array",
        "description": description,
        "items": {"type": "number", "exclusiveMinimum": 0},
        "minItems": 3,
        "maxItems": 3,
    }


def _opening_schema(frame_default: float, with_wall: bool) -> dict[str, Any]:
    properties: dict[str, Any] = {
        "x": {
            "type": "number",
            "description": "Centre of the opening along the wall from the wall's middle, metres"
            + (", positive to the right seen from inside the room." if with_wall else "."),
        },
        "sill": {
            "type": "number",
            "minimum": 0,
            "description": "Height of the opening's bottom edge above the floor; 0 = door.",
        },
        "width": {"type": "number", "exclusiveMinimum": 0, "description": "Opening width."},
        "height": {"type": "number", "exclusiveMinimum": 0, "description": "Opening height."},
        "frame": {
            "type": "number",
            "minimum": 0,
            "default": frame_default,
            "description": "Width of the frame around the opening in metres (part frames); "
            "0 leaves a bare hole.",
        },
        "glass": {
            "type": "boolean",
            "default": False,
            "description": "true fills the opening with a pane (part glass, clear glass "
            "unless part_materials says otherwise).",
        },
    }
    required = ["x", "sill", "width", "height"]
    if with_wall:
        properties["wall"] = {
            "enum": ["back", "front", "left", "right"],
            "description": "Which wall: back (+y), front (-y), left (-x) or right (+x), in the "
            "room's own axes.",
        }
        required.insert(0, "wall")
    return {
        "type": "object",
        "additionalProperties": False,
        "required": required,
        "properties": properties,
    }


# ------------------------------------------------------------------- walls


Hole = tuple[float, float, float, float]  # x0, x1, z0, z1


def _holes(openings: Sequence[Mapping[str, Any]]) -> list[Hole]:
    return [
        (
            float(o["x"]) - float(o["width"]) / 2.0,
            float(o["x"]) + float(o["width"]) / 2.0,
            float(o["sill"]),
            float(o["sill"]) + float(o["height"]),
        )
        for o in openings
    ]


def _slab_with_holes(
    width: float, thickness: float, height: float, holes: Sequence[Hole]
) -> MeshData:
    """An upright slab spanning x, standing on z = 0, with rectangular holes."""

    x_min, x_max = -width / 2.0, width / 2.0
    y_front, y_back = -thickness / 2.0, thickness / 2.0
    xs = sorted({x_min, x_max, *(c for h in holes for c in h[:2] if x_min < c < x_max)})
    zs = sorted({0.0, height, *(c for h in holes for c in h[2:] if 0.0 < c < height)})

    def filled(i: int, j: int) -> bool:
        if not (0 <= i < len(xs) - 1 and 0 <= j < len(zs) - 1):
            return False
        cx, cz = (xs[i] + xs[i + 1]) / 2.0, (zs[j] + zs[j + 1]) / 2.0
        return not any(x0 < cx < x1 and z0 < cz < z1 for x0, x1, z0, z1 in holes)

    vertices: list[tuple[float, float, float]] = []
    index: dict[tuple[int, int, int], int] = {}

    def v(side: int, i: int, j: int) -> int:
        key = (side, i, j)
        if key not in index:
            index[key] = len(vertices)
            vertices.append((xs[i], y_back if side else y_front, zs[j]))
        return index[key]

    faces: list[tuple[int, ...]] = []
    for i in range(len(xs) - 1):
        for j in range(len(zs) - 1):
            if not filled(i, j):
                continue
            f00, f10, f11, f01 = v(0, i, j), v(0, i + 1, j), v(0, i + 1, j + 1), v(0, i, j + 1)
            b00, b10, b11, b01 = v(1, i, j), v(1, i + 1, j), v(1, i + 1, j + 1), v(1, i, j + 1)
            faces.append((f00, f10, f11, f01))
            faces.append((b10, b00, b01, b11))
            if not filled(i - 1, j):
                faces.append((b00, f00, f01, b01))
            if not filled(i + 1, j):
                faces.append((f10, b10, b11, f11))
            if not filled(i, j - 1):
                faces.append((f00, b00, b10, f10))
            if not filled(i, j + 1):
                faces.append((f01, f11, b11, b01))
    return MeshData(tuple(vertices), tuple(faces), smooth=False)


def _wall_pieces(
    width: float,
    thickness: float,
    height: float,
    openings: Sequence[Mapping[str, Any]],
    keys: Sequence[str],
    place: Mat4,
) -> tuple[list[Piece], list[Portal]]:
    """A wall slab, its frames and panes placed by *place*, and the clear openings."""

    holes = _holes(openings)
    pieces = [Piece(_slab_with_holes(width, thickness, height, holes), "walls", place)]
    portals: list[Portal] = []
    depth = thickness + 2.0 * _FRAME_PROUD
    for opening, (x0, x1, z0, z1), key in zip(openings, holes, keys, strict=True):
        f = float(opening["frame"])
        bottom = f if z0 > 0.0 else 0.0  # a door has no frame across its threshold
        if f > 0.0:
            frame_bevel = min(0.003, f / 4.0)
            jamb = rounded_box((f, depth, z1 - z0), frame_bevel)
            pieces.append(Piece(jamb, "frames", multiply(place, at((x0 + f / 2.0, 0.0, z0)))))
            pieces.append(Piece(jamb, "frames", multiply(place, at((x1 - f / 2.0, 0.0, z0)))))
            rail = rounded_box((x1 - x0 - 2.0 * f, depth, f), frame_bevel)
            pieces.append(
                Piece(rail, "frames", multiply(place, at(((x0 + x1) / 2.0, 0.0, z1 - f))))
            )
            if bottom:
                pieces.append(
                    Piece(rail, "frames", multiply(place, at(((x0 + x1) / 2.0, 0.0, z0))))
                )
        cx0, cx1, cz0, cz1 = x0 + f, x1 - f, z0 + bottom, z1 - f
        if opening["glass"]:
            pane = rounded_box((cx1 - cx0, _GLASS, cz1 - cz0), 0.0)
            pieces.append(Piece(pane, "glass", multiply(place, at(((cx0 + cx1) / 2.0, 0.0, cz0)))))
        local = Portal(
            key,
            ((cx0 + cx1) / 2.0, 0.0, (cz0 + cz1) / 2.0),
            ((cx1 - cx0) / 2.0, 0.0, 0.0),
            (0.0, 0.0, (cz1 - cz0) / 2.0),
        )
        portals.append(local.transformed(place))
    return pieces, portals


def _check_openings(
    openings: Sequence[Mapping[str, Any]], width: float, height: float, prefix: str = ""
) -> list[ParamProblem]:
    problems: list[ParamProblem] = []
    for n, (opening, (x0, x1, z0, z1)) in enumerate(zip(openings, _holes(openings), strict=True)):
        where = f"{prefix}opening {n}"
        if x0 < -width / 2.0 - 1e-9 or x1 > width / 2.0 + 1e-9 or z1 > height + 1e-9:
            problems.append(("openings", f"{where} extends beyond the wall"))
        f = float(opening["frame"])
        if 2.0 * f >= x1 - x0 or (2.0 if z0 > 0 else 1.0) * f >= z1 - z0:
            problems.append(("openings", f"the frame of {where} fills the whole opening"))
    return problems


_WALL_PARTS = {
    "walls": "The wall itself.",
    "frames": "Frames around the openings.",
    "glass": "Window panes.",
}


def _wall(p: Mapping[str, Any]) -> MeshData:
    width, thickness, height = (float(v) for v in p["size"])
    keys = [f"openings/{n}" for n in range(len(p["openings"]))]
    pieces, portals = _wall_pieces(
        width, thickness, height, p["openings"], keys, at((0.0, 0.0, 0.0))
    )
    return assemble(pieces, list(_WALL_PARTS), portals)


def _check_wall(p: Mapping[str, Any]) -> list[ParamProblem]:
    width, _, height = (float(v) for v in p["size"])
    problems = _check_openings(p["openings"], width, height)
    if not problems and not _slab_with_holes(width, 0.1, height, _holes(p["openings"])).faces:
        problems.append(("openings", "the openings remove the whole wall"))
    return problems


WALL = GeneratorDef[MeshData](
    name="wall",
    summary=(
        "Upright slab standing on its origin, spanning x, with rectangular openings "
        "cut through it (windows, doors), optionally framed and glazed. Light passes "
        "through the openings. For a whole room use room."
    ),
    params={
        "size": _size3("Width (x), thickness (y) and height (z) in metres."),
        "openings": {
            "type": "array",
            "items": _opening_schema(0.0, with_wall=False),
            "maxItems": 64,
            "default": [],
            "description": "Rectangular holes through the wall; x is measured from its centre.",
        },
    },
    run=_wall,
    check=_check_wall,
    examples=(
        {
            "op": "wall",
            "size": [4.0, 0.15, 2.6],
            "openings": [{"x": -0.6, "sill": 0.9, "width": 1.2, "height": 1.3}],
        },
        {
            "op": "wall",
            "size": [3.0, 0.2, 2.6],
            "openings": [
                {"x": 0.5, "sill": 0.9, "width": 1.0, "height": 1.2, "frame": 0.06, "glass": True}
            ],
        },
    ),
    parts=_WALL_PARTS,
    part_defaults={"glass": _CLEAR_GLASS},
)


# -------------------------------------------------------------------- room

_ROOM_PARTS = {
    "walls": "The four walls.",
    "floor": "The floor slab; its top is the room's origin plane.",
    "ceiling": "The ceiling slab.",
    "frames": "Frames around doors and windows.",
    "glass": "Window panes.",
}

#: Per wall: rotation about z so the wall's own x runs to the right seen from inside.
_WALL_TURN = {"back": 0.0, "front": 180.0, "left": 90.0, "right": -90.0}


def _room(p: Mapping[str, Any]) -> MeshData:
    width, depth, height = (float(v) for v in p["size"])
    t = float(p["wall_thickness"])
    slab = float(p["slab_thickness"])
    lengths = {"back": width, "front": width, "left": depth + 2 * t, "right": depth + 2 * t}
    centres = {
        "back": (0.0, depth / 2.0 + t / 2.0),
        "front": (0.0, -depth / 2.0 - t / 2.0),
        "left": (-width / 2.0 - t / 2.0, 0.0),
        "right": (width / 2.0 + t / 2.0, 0.0),
    }
    pieces: list[Piece] = []
    portals: list[Portal] = []
    for wall in ("back", "front", "left", "right"):
        chosen = [(n, o) for n, o in enumerate(p["openings"]) if o["wall"] == wall]
        x, y = centres[wall]
        place = at((x, y, 0.0), (0.0, 0.0, _WALL_TURN[wall]))
        more, found = _wall_pieces(
            lengths[wall],
            t,
            height,
            [o for _, o in chosen],
            [f"openings/{n}" for n, _ in chosen],
            place,
        )
        pieces += more
        portals += found
    footprint = (width + 2 * t, depth + 2 * t, slab)
    if p["floor"]:
        pieces.append(Piece(rounded_box(footprint, 0.0), "floor", at((0.0, 0.0, -slab))))
    if p["ceiling"]:
        pieces.append(Piece(rounded_box(footprint, 0.0), "ceiling", at((0.0, 0.0, height))))
    portals.sort(key=lambda portal: int(portal.key.split("/")[1]))
    return assemble(pieces, list(_ROOM_PARTS), portals)


def _check_room(p: Mapping[str, Any]) -> list[ParamProblem]:
    width, depth, height = (float(v) for v in p["size"])
    t = float(p["wall_thickness"])
    lengths = {"back": width, "front": width, "left": depth + 2 * t, "right": depth + 2 * t}
    problems: list[ParamProblem] = []
    for wall, length in lengths.items():
        chosen = [o for o in p["openings"] if o["wall"] == wall]
        problems += _check_openings(chosen, length, height, prefix=f"{wall} wall ")
    return problems


ROOM = GeneratorDef[MeshData](
    name="room",
    summary=(
        "A closed room: four walls with doors and windows, a floor and a ceiling, in one "
        "call. Its origin is the middle of the floor surface; things stand on it with "
        "rest_on. Openings are named by wall, so no wall has to be turned or placed by hand."
    ),
    params={
        "size": _size3("Inside width (x), depth (y) and height (z) in metres."),
        "wall_thickness": number(
            "Thickness of the walls in metres; windows get deep reveals in thick walls.",
            default=0.15,
            exclusive_minimum=0,
        ),
        "slab_thickness": number(
            "Thickness of the floor and ceiling slabs in metres.",
            default=0.12,
            exclusive_minimum=0,
        ),
        "floor": {"type": "boolean", "default": True, "description": "Include the floor slab."},
        "ceiling": {
            "type": "boolean",
            "default": True,
            "description": "Include the ceiling; without it the sky pours in from above.",
        },
        "openings": {
            "type": "array",
            "items": _opening_schema(0.05, with_wall=True),
            "maxItems": 64,
            "default": [],
            "description": "Doors and windows; each names its wall.",
        },
    },
    run=_room,
    check=_check_room,
    examples=(
        {
            "op": "room",
            "size": [4.0, 5.0, 2.7],
            "openings": [
                {"wall": "left", "x": 0.6, "sill": 0.9, "width": 1.2, "height": 1.3, "glass": True},
                {"wall": "front", "x": -1.2, "sill": 0.0, "width": 0.9, "height": 2.1},
            ],
        },
    ),
    parts=_ROOM_PARTS,
    part_defaults={"glass": _CLEAR_GLASS},
)


# ------------------------------------------------------------------- table

_TABLE_PARTS = {"top": "The table top.", "base": "Legs and the apron under the top."}


def _leg(style: str, thickness: float, height: float, bevel: float) -> MeshData:
    if style == "round":
        return cylinder(thickness / 2.0, height)
    if style == "tapered":
        return tapered_leg(thickness * 0.3, thickness / 2.0, height)
    return rounded_box((thickness, thickness, height), bevel)


def _table(p: Mapping[str, Any]) -> MeshData:
    width, depth, height = (float(v) for v in p["size"])
    top_t = float(p["top_thickness"])
    leg = float(p["leg_thickness"])
    inset = float(p["leg_inset"])
    bevel = float(p["bevel"])
    under = height - top_t
    pieces = [Piece(rounded_box((width, depth, top_t), bevel), "top", at((0.0, 0.0, under)))]
    lx = width / 2.0 - inset - leg / 2.0
    ly = depth / 2.0 - inset - leg / 2.0
    leg_mesh = _leg(p["legs"], leg, under, min(bevel, leg / 4.0))
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            pieces.append(Piece(leg_mesh, "base", at((sx * lx, sy * ly, 0.0))))
    apron = float(p["apron"])
    if apron > 0.0:
        board = 0.02
        z = under - apron
        along_x = rounded_box((2 * lx - leg, board, apron), min(bevel, 0.002))
        along_y = rounded_box((board, 2 * ly - leg, apron), min(bevel, 0.002))
        for s in (-1.0, 1.0):
            pieces.append(Piece(along_x, "base", at((0.0, s * ly, z))))
            pieces.append(Piece(along_y, "base", at((s * lx, 0.0, z))))
    return assemble(pieces, list(_TABLE_PARTS))


def _check_table(p: Mapping[str, Any]) -> list[ParamProblem]:
    width, depth, height = (float(v) for v in p["size"])
    leg, inset = float(p["leg_thickness"]), float(p["leg_inset"])
    problems: list[ParamProblem] = []
    if 2 * (inset + leg) >= min(width, depth):
        problems.append(("leg_inset", "the legs do not fit under the top"))
    if float(p["top_thickness"]) + float(p["apron"]) >= height:
        problems.append(("apron", "the top and apron are taller than the table"))
    if 2 * float(p["bevel"]) >= min(float(p["top_thickness"]), leg):
        problems.append(("bevel", "the rounding is larger than the parts it rounds"))
    return problems


TABLE = GeneratorDef[MeshData](
    name="table",
    summary=(
        "A table: a top on four legs, with an optional apron. size is the overall width, "
        "depth and height (the top surface)."
    ),
    params={
        "size": {
            **_size3("Overall width (x), depth (y) and height (z) in metres."),
            "default": [1.2, 0.75, 0.75],
        },
        "top_thickness": number(
            "Thickness of the top in metres.", default=0.035, exclusive_minimum=0
        ),
        "leg_thickness": number("Width of a leg in metres.", default=0.05, exclusive_minimum=0),
        "leg_inset": number(
            "Distance from the top's edge to the legs in metres.", default=0.04, minimum=0
        ),
        "legs": {
            "enum": ["square", "round", "tapered"],
            "default": "square",
            "description": "Leg shape: square posts, round posts or round and tapering.",
        },
        "apron": number(
            "Height of the rail under the top in metres; 0 for none.", default=0.07, minimum=0
        ),
        "bevel": _BEVEL,
    },
    run=_table,
    check=_check_table,
    examples=(
        {"op": "table", "size": [1.4, 0.8, 0.75]},
        {"op": "table", "size": [0.6, 0.6, 0.5], "legs": "tapered", "apron": 0.0},
    ),
    parts=_TABLE_PARTS,
)


# ------------------------------------------------------------------- chair

_CHAIR_PARTS = {
    "seat": "The seat.",
    "frame": "Legs, back posts and the rails under the seat.",
    "back": "Back slats or panel and the top rail.",
}


def _chair(p: Mapping[str, Any]) -> MeshData:
    width, depth = (float(v) for v in p["seat_size"])
    seat_h, seat_t = float(p["seat_height"]), float(p["seat_thickness"])
    back_h = float(p["back_height"])
    leg = float(p["leg_thickness"])
    bevel = float(p["bevel"])
    under = seat_h - seat_t
    pieces = [Piece(rounded_box((width, depth, seat_t), bevel), "seat", at((0.0, 0.0, under)))]
    lx, ly = width / 2.0 - leg / 2.0 - 0.01, depth / 2.0 - leg / 2.0 - 0.01
    edge = min(bevel, leg / 4.0)
    front = rounded_box((leg, leg, under), edge)
    post_h = seat_h + back_h
    post = rounded_box((leg, leg, post_h if back_h > 0 else under), edge)
    for sx in (-1.0, 1.0):
        pieces.append(Piece(front, "frame", at((sx * lx, -ly, 0.0))))
        pieces.append(Piece(post, "frame", at((sx * lx, ly, 0.0))))
    rail_h = 0.05
    rail_z = under - rail_h
    along_x = rounded_box((2 * lx - leg, 0.02, rail_h), min(edge, 0.002))
    along_y = rounded_box((0.02, 2 * ly - leg, rail_h), min(edge, 0.002))
    for s in (-1.0, 1.0):
        pieces.append(Piece(along_x, "frame", at((0.0, s * ly, rail_z))))
        pieces.append(Piece(along_y, "frame", at((s * lx, 0.0, rail_z))))
    style = p["back"]
    if back_h > 0 and style != "none":
        top_rail_h = min(0.07, back_h * 0.25)
        inner = 2 * lx - leg
        top = rounded_box((inner, leg * 0.8, top_rail_h), min(edge, 0.002))
        pieces.append(Piece(top, "back", at((0.0, ly, post_h - top_rail_h))))
        low, high = seat_h + 0.06, post_h - top_rail_h
        if style == "panel":
            panel = rounded_box((inner, 0.015, high - low), min(edge, 0.002))
            pieces.append(Piece(panel, "back", at((0.0, ly, low))))
        else:
            count = int(p["slats"])
            slat_w = min(0.05, inner / (2 * count + 1))
            slat = rounded_box((slat_w, 0.015, high - low), min(edge, 0.002))
            step = inner / count
            for k in range(count):
                x = -inner / 2.0 + step * (k + 0.5)
                pieces.append(Piece(slat, "back", at((x, ly, low))))
    return assemble(pieces, list(_CHAIR_PARTS))


def _check_chair(p: Mapping[str, Any]) -> list[ParamProblem]:
    width, depth = (float(v) for v in p["seat_size"])
    leg = float(p["leg_thickness"])
    problems: list[ParamProblem] = []
    if 2 * leg + 0.04 >= min(width, depth):
        problems.append(("leg_thickness", "the legs do not fit under the seat"))
    if float(p["seat_thickness"]) + 0.05 >= float(p["seat_height"]):
        problems.append(("seat_height", "the seat is too low for its thickness and rails"))
    if p["back"] != "none" and 0.0 < float(p["back_height"]) < 0.15:
        problems.append(("back_height", "a back needs at least 0.15 m above the seat"))
    return problems


CHAIR = GeneratorDef[MeshData](
    name="chair",
    summary=(
        "A dining chair: four legs, a seat, and a back of slats or a panel. It faces -y; "
        "its back is on the +y side. Turn it with rotation."
    ),
    params={
        "seat_height": number(
            "Height of the seat's top in metres.", default=0.45, exclusive_minimum=0
        ),
        "seat_size": {
            "type": "array",
            "description": "Seat width (x) and depth (y) in metres.",
            "items": {"type": "number", "exclusiveMinimum": 0},
            "minItems": 2,
            "maxItems": 2,
            "default": [0.44, 0.42],
        },
        "seat_thickness": number("Seat thickness in metres.", default=0.03, exclusive_minimum=0),
        "back_height": number(
            "Height of the back above the seat in metres; 0 makes a stool.",
            default=0.42,
            minimum=0,
        ),
        "back": {
            "enum": ["slats", "panel", "none"],
            "default": "slats",
            "description": "Vertical slats, a solid panel, or only the top rail and posts.",
        },
        "slats": integer("Number of slats in a slatted back.", default=3, minimum=1, maximum=12),
        "leg_thickness": number("Width of a leg in metres.", default=0.035, exclusive_minimum=0),
        "bevel": _BEVEL,
    },
    run=_chair,
    check=_check_chair,
    examples=({"op": "chair"}, {"op": "chair", "back": "panel", "seat_height": 0.47}),
    parts=_CHAIR_PARTS,
)


# ------------------------------------------------------------------- shelf

_SHELF_PARTS = {
    "carcass": "Sides, top, bottom and plinth.",
    "shelves": "The shelf boards inside.",
    "back": "The back panel.",
}


def _shelf(p: Mapping[str, Any]) -> MeshData:
    width, depth, height = (float(v) for v in p["size"])
    board = float(p["board_thickness"])
    plinth = float(p["plinth"])
    bevel = float(p["bevel"])
    edge = min(bevel, board / 4.0)
    inner_w = width - 2 * board
    pieces: list[Piece] = []
    side = rounded_box((board, depth, height), edge)
    for s in (-1.0, 1.0):
        pieces.append(Piece(side, "carcass", at((s * (width / 2.0 - board / 2.0), 0.0, 0.0))))
    across = rounded_box((inner_w, depth, board), edge)
    pieces.append(Piece(across, "carcass", at((0.0, 0.0, height - board))))
    pieces.append(Piece(across, "carcass", at((0.0, 0.0, plinth))))
    if plinth > 0.0:
        kick = rounded_box((inner_w, board, plinth), edge)
        pieces.append(Piece(kick, "carcass", at((0.0, -depth / 2.0 + 0.02 + board / 2.0, 0.0))))
    boards = int(p["shelves"])
    shelf = rounded_box((inner_w, depth - 0.01, board), edge)
    bottom, top = plinth + board, height - board
    gap = (top - bottom - boards * board) / (boards + 1)
    for k in range(boards):
        z = bottom + gap * (k + 1) + board * k
        pieces.append(Piece(shelf, "shelves", at((0.0, -0.005, z))))
    if p["back_panel"]:
        back = rounded_box((inner_w, 0.006, height - plinth - 2 * board), 0.0)
        pieces.append(Piece(back, "back", at((0.0, depth / 2.0 - 0.003, plinth + board))))
    return assemble(pieces, list(_SHELF_PARTS))


def _check_shelf(p: Mapping[str, Any]) -> list[ParamProblem]:
    width, _, height = (float(v) for v in p["size"])
    board = float(p["board_thickness"])
    problems: list[ParamProblem] = []
    if 3 * board >= width:
        problems.append(("board_thickness", "the boards are too thick for the width"))
    room = height - float(p["plinth"]) - 2 * board - int(p["shelves"]) * board
    if room <= 0.05 * (int(p["shelves"]) + 1):
        problems.append(("shelves", "the shelves do not fit in the height"))
    return problems


SHELF = GeneratorDef[MeshData](
    name="shelf",
    summary=(
        "A bookcase or open shelf: two sides, top and bottom, evenly spaced shelf boards "
        "and a back panel. Its open front faces -y. Put things on a board with rest_on: "
        "they fall from where they are onto the board below."
    ),
    params={
        "size": {
            **_size3("Overall width (x), depth (y) and height (z) in metres."),
            "default": [0.8, 0.3, 1.8],
        },
        "shelves": integer(
            "Number of shelf boards between bottom and top.", default=4, minimum=0, maximum=40
        ),
        "board_thickness": number("Board thickness in metres.", default=0.02, exclusive_minimum=0),
        "plinth": number(
            "Height of the recessed base under the bottom board in metres; 0 for none.",
            default=0.06,
            minimum=0,
        ),
        "back_panel": {
            "type": "boolean",
            "default": True,
            "description": "Include a thin back panel.",
        },
        "bevel": {**_BEVEL, "default": 0.002},
    },
    run=_shelf,
    check=_check_shelf,
    examples=(
        {"op": "shelf"},
        {"op": "shelf", "size": [1.2, 0.35, 0.9], "shelves": 1, "plinth": 0.0},
    ),
    parts=_SHELF_PARTS,
)


ASSEMBLIES: tuple[GeneratorDef[MeshData], ...] = (WALL, ROOM, TABLE, CHAIR, SHELF)

__all__ = ["ASSEMBLIES", "CHAIR", "ROOM", "SHELF", "TABLE", "WALL"]
