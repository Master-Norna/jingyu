"""The picture charter: the identity a series of pictures shares.

A series (a set of illustrations for one story, a product line, a calendar)
must look like it belongs together.  The charter writes down what makes it so:
its subject, the skeleton of its composition, its palette, its value key and
contrast, the mood of its light and its painting style.  ``check_charter``
measures a render against it and turns every difference into a question.

Like the constitution, the charter is for asking, not for scoring: there is no
pass mark and no total, because a number shown to a model becomes the thing it
optimises instead of the picture.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, cast

from PIL import Image, ImageOps

from .conventions import COLOR_PATTERN, parse_hex, srgb_to_linear
from .errors import JingyuError
from .schema_validation import Validator
from .style import PRESETS

CHARTER_SCHEMA_ID = "jingyu.charter.v1"

SKELETONS: dict[str, str] = {
    "thirds": "the subject on a crossing of the thirds lines",
    "centered": "the subject in the middle",
    "golden": "the subject on a crossing of the golden-section lines",
    "diagonal": "the subject on a diagonal of the frame",
    "free": "no fixed place",
}

CHARTER_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["schema", "id", "subject"],
    "properties": {
        "schema": {"const": CHARTER_SCHEMA_ID},
        "id": {"type": "string", "pattern": r"^[a-z][a-z0-9_-]{0,62}$"},
        "title": {"type": "string", "maxLength": 200},
        "subject": {
            "type": "string",
            "maxLength": 1000,
            "description": "What every picture of the series is about.",
        },
        "subject_objects": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Object or group ids that play the subject in the scenes.",
        },
        "composition": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "skeleton": {"enum": list(SKELETONS)},
                "subject_size": {"type": "number", "exclusiveMinimum": 0, "maximum": 1},
            },
        },
        "palette": {
            "type": "array",
            "items": {"type": "string", "pattern": COLOR_PATTERN},
            "maxItems": 8,
            "description": "The colours that carry the series, most important first.",
        },
        "value_key": {"enum": ["low", "middle", "high"]},
        "contrast": {"enum": ["soft", "medium", "hard"]},
        "light_mood": {"type": "string", "maxLength": 500},
        "style": {"enum": list(PRESETS)},
        "notes": {"type": "string", "maxLength": 8000},
    },
}

#: Mean display luma bounds of the value keys, and luma spread of the contrasts.
_KEYS = {"low": (0.0, 0.33), "middle": (0.33, 0.6), "high": (0.6, 1.0)}
_CONTRASTS = {"soft": (0.0, 0.45), "medium": (0.45, 0.72), "hard": (0.72, 1.0)}
#: Colour difference (CIE76 delta E) below which a palette colour counts as present.
PRESENT_DELTA_E = 18.0


def validate_charter(document: Any) -> dict[str, Any]:
    errors = list(Validator(CHARTER_JSON_SCHEMA).iter_errors(document))
    if errors:
        first = errors[0]
        where = "/" + "/".join(str(p) for p in first.absolute_path)
        raise JingyuError(
            "charter.invalid",
            f"the charter is not valid at {where}: {first.message}",
            details={"errors": len(errors)},
        )
    return dict(document)


def _lab(rgb: Sequence[float]) -> tuple[float, float, float]:
    """CIE L*a*b* (D65) of an sRGB colour with channels in [0, 1]."""

    r, g, b = (srgb_to_linear(c) for c in rgb)
    x = (0.4124 * r + 0.3576 * g + 0.1805 * b) / 0.95047
    y = 0.2126 * r + 0.7152 * g + 0.0722 * b
    z = (0.0193 * r + 0.1192 * g + 0.9505 * b) / 1.08883

    def f(t: float) -> float:
        return t ** (1 / 3) if t > 0.008856 else 7.787 * t + 16 / 116

    fx, fy, fz = f(x), f(y), f(z)
    return 116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)


def dominant_colours(image: Image.Image, count: int = 6) -> list[dict[str, Any]]:
    """The main colours of an image (median cut) with their share of it."""

    small = image.convert("RGB")
    small.thumbnail((200, 200))
    quantised = small.quantize(colors=count, method=Image.Quantize.MEDIANCUT)
    palette = quantised.getpalette() or []
    # A palette image reports (count, palette index) pairs.
    counts = sorted(
        ((n, cast(int, index)) for n, index in quantised.getcolors() or []), reverse=True
    )
    total = sum(n for n, _ in counts) or 1
    return [
        {
            "hex": "#{:02x}{:02x}{:02x}".format(*palette[3 * i : 3 * i + 3]),
            "share": round(n / total, 4),
        }
        for n, i in counts
    ]


def _bucket(value: float, buckets: Mapping[str, tuple[float, float]]) -> str:
    return next(name for name, (low, high) in buckets.items() if low <= value <= high)


def check_charter(
    charter: Mapping[str, Any],
    image: Image.Image,
    scene: Mapping[str, Any],
    layout: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Measurements of a render next to the charter, and the questions they raise."""

    gray = ImageOps.grayscale(image.convert("RGB"))
    gray.thumbnail((256, 256))
    histogram = gray.histogram()
    total = sum(histogram) or 1
    mean = sum(v * n for v, n in enumerate(histogram)) / total / 255.0
    cumulative = 0
    p05 = p95 = 0.0
    for value, n in enumerate(histogram):
        cumulative += n
        if not p05 and cumulative >= 0.05 * total:
            p05 = value / 255.0
        if cumulative >= 0.95 * total:
            p95 = value / 255.0
            break
    spread = p95 - p05
    facts: dict[str, Any] = {
        "value": {"mean_luma": round(mean, 3), "key": _bucket(mean, _KEYS)},
        "contrast": {"spread_5_95": round(spread, 3), "contrast": _bucket(spread, _CONTRASTS)},
        "colours": dominant_colours(image),
        "style": scene["render"].get("style", {}).get("preset", "none"),
    }
    questions: list[str] = []

    key = charter.get("value_key")
    if key and key != facts["value"]["key"]:
        questions.append(
            f"The charter keeps a {key} key; this frame is {facts['value']['key']} "
            f"(mean value {mean:.2f}). Is the difference the picture's own choice, or has the "
            "light drifted from the series?"
        )
    contrast = charter.get("contrast")
    if contrast and contrast != facts["contrast"]["contrast"]:
        questions.append(
            f"The charter asks for {contrast} contrast; the frame's values spread "
            f"{spread:.2f} ({facts['contrast']['contrast']}). Where would the series' "
            "contrast come from here: the light, or the materials?"
        )

    palette = charter.get("palette") or []
    if palette:
        measured = [(_lab(parse_hex(c["hex"])), c) for c in facts["colours"]]
        matches = []
        for colour in palette:
            lab = _lab(parse_hex(colour))
            distance, nearest = min(
                ((math.dist(lab, m), c) for m, c in measured), key=lambda t: t[0]
            )
            matches.append(
                {
                    "charter": colour,
                    "nearest": nearest["hex"],
                    "delta_e": round(distance, 1),
                    "present": distance <= PRESENT_DELTA_E,
                }
            )
        facts["palette"] = matches
        missing = [m["charter"] for m in matches if not m["present"]]
        if missing:
            questions.append(
                f"The series' colours {', '.join(missing)} do not appear among this frame's "
                "main colours. Should one of the objects, or the light, carry them?"
            )

    style = charter.get("style")
    if style and style != facts["style"]:
        questions.append(
            f"The charter paints the series as {style}; this scene renders as "
            f"{facts['style']}. Set render.style.preset, or is this picture the exception?"
        )

    composition = charter.get("composition") or {}
    subject_ids = charter.get("subject_objects") or []
    if layout is not None and subject_ids:
        rows = [o for o in layout.get("objects", []) if o["object"] in subject_ids]
        if not rows:
            questions.append(
                f"The subject ({', '.join(subject_ids)}) is not in the frame. Is this picture "
                "about something else?"
            )
        else:
            u0 = min(r["bbox_uv"]["u0"] for r in rows)
            v0 = min(r["bbox_uv"]["v0"] for r in rows)
            u1 = max(r["bbox_uv"]["u1"] for r in rows)
            v1 = max(r["bbox_uv"]["v1"] for r in rows)
            centre = ((u0 + u1) / 2, (v0 + v1) / 2)
            size = max(u1 - u0, v1 - v0)
            facts["subject"] = {"centre_uv": [round(c, 3) for c in centre], "size": round(size, 3)}
            skeleton = str(composition.get("skeleton", "free"))
            spot = _skeleton_spot(skeleton, centre)
            if spot is not None:
                facts["subject"]["skeleton_point"] = [round(c, 3) for c in spot]
                if math.dist(spot, centre) > 0.08:
                    questions.append(
                        f"The series puts {SKELETONS[skeleton]}; here the subject's centre is "
                        f"at u {centre[0]:.2f}, v {centre[1]:.2f}. frame_subject can place it "
                        f"at u {spot[0]:.2f}, v {spot[1]:.2f}; is the break intended?"
                    )
            wanted = composition.get("subject_size")
            if wanted and abs(size - wanted) > 0.1:
                questions.append(
                    f"The series gives the subject {wanted:.2f} of the frame; here it has "
                    f"{size:.2f}. Closer or farther, or does this picture need the change?"
                )
    if charter.get("light_mood"):
        questions.append(
            f'Does the light read as the series\' mood: "{charter["light_mood"]}"? Look at '
            "the values view and describe_light before answering."
        )
    return {"charter": charter.get("id"), "facts": facts, "questions": questions}


def _skeleton_spot(skeleton: str | None, centre: tuple[float, float]) -> tuple[float, float] | None:
    """The point of the skeleton nearest to the subject's centre."""

    if skeleton in (None, "free"):
        return None
    if skeleton == "centered":
        return (0.5, 0.5)
    if skeleton == "diagonal":
        # the nearer of the two diagonals, at the subject's own u
        u = centre[0]
        return min(((u, u), (u, 1 - u)), key=lambda p: math.dist(p, centre))
    lines = (1 / 3, 2 / 3) if skeleton == "thirds" else (0.382, 0.618)
    points = [(u, v) for u in lines for v in lines]
    return min(points, key=lambda p: math.dist(p, centre))


__all__ = [
    "CHARTER_JSON_SCHEMA",
    "CHARTER_SCHEMA_ID",
    "SKELETONS",
    "check_charter",
    "dominant_colours",
    "validate_charter",
]
