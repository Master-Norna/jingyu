"""Differences between supported Blender versions, kept in one place."""

from __future__ import annotations

import math
import warnings
from typing import Any

import bpy

from ..errors import JingyuError

MIN_VERSION: tuple[int, int, int] = (4, 2, 0)

ENGINE_IDS: dict[str, tuple[str, ...]] = {
    "cycles": ("CYCLES",),
    # EEVEE Next is BLENDER_EEVEE_NEXT in 4.2-4.5 and BLENDER_EEVEE again from 5.0.
    "eevee": ("BLENDER_EEVEE_NEXT", "BLENDER_EEVEE"),
    "workbench": ("BLENDER_WORKBENCH",),
}

VIEW_TRANSFORMS: dict[str, str] = {
    "standard": "Standard",
    "agx": "AgX",
    "filmic": "Filmic",
    "khronos_pbr_neutral": "Khronos PBR Neutral",
}

#: Principled BSDF input names, newest first.
PRINCIPLED_INPUTS: dict[str, tuple[str, ...]] = {
    "base_color": ("Base Color",),
    "metallic": ("Metallic",),
    "roughness": ("Roughness",),
    "ior": ("IOR",),
    "alpha": ("Alpha",),
    "transmission": ("Transmission Weight", "Transmission"),
    "coat_weight": ("Coat Weight", "Clearcoat"),
    "coat_roughness": ("Coat Roughness", "Clearcoat Roughness"),
    "emission_color": ("Emission Color", "Emission"),
    "emission_strength": ("Emission Strength",),
}

GPU_BACKENDS: tuple[str, ...] = ("OPTIX", "CUDA", "HIP", "METAL", "ONEAPI")


def version() -> tuple[int, int, int]:
    major, minor, patch = bpy.app.version
    return (int(major), int(minor), int(patch))


def version_info() -> dict[str, Any]:
    build_hash = bpy.app.build_hash
    if isinstance(build_hash, bytes):
        build_hash = build_hash.decode("ascii", "replace")
    return {
        "version": bpy.app.version_string,
        "version_tuple": list(version()),
        "build_hash": str(build_hash),
        "background": bool(bpy.app.background),
    }


def require_supported_version() -> None:
    if version() < MIN_VERSION:
        raise JingyuError(
            "blender.unsupported_version",
            f"Blender {bpy.app.version_string} is older than {'.'.join(map(str, MIN_VERSION))}",
            hint="Install Blender 4.2 LTS or newer.",
        )


def set_engine(scene: Any, engine: str) -> str:
    """Select a render engine by jingyu name and return Blender's identifier."""

    for identifier in ENGINE_IDS.get(engine, ()):
        try:
            scene.render.engine = identifier
        except TypeError:
            continue
        return identifier
    raise JingyuError(
        "blender.engine_unavailable",
        f"render engine {engine!r} is not available in Blender {bpy.app.version_string}",
    )


def available_engines(scene: Any) -> list[str]:
    """Jingyu engine names this runtime registers (registering is not rendering)."""

    current = scene.render.engine
    names = []
    for name in ENGINE_IDS:
        try:
            set_engine(scene, name)
        except JingyuError:
            continue
        names.append(name)
    scene.render.engine = current
    return names


def set_view_transform(scene: Any, key: str) -> str:
    name = VIEW_TRANSFORMS[key]
    try:
        scene.view_settings.view_transform = name
    except TypeError as exc:
        raise JingyuError(
            "blender.render_failed",
            f"view transform {name!r} is not available in this Blender",
        ) from exc
    return name


def ensure_node_tree(material: Any) -> None:
    """Make sure *material* has a node tree with a principled BSDF."""

    if material.node_tree is None or version() < (5, 0, 0):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            material.use_nodes = True


def set_principled_input(node: Any, key: str, value: Any) -> None:
    for name in PRINCIPLED_INPUTS[key]:
        socket = node.inputs.get(name)
        if socket is not None:
            socket.default_value = value
            return
    raise JingyuError(
        "blender.build_failed",
        f"principled BSDF has no input for {key!r} in Blender {bpy.app.version_string}",
    )


def enable_eevee_refraction(scene: Any, material: Any) -> None:
    """Let EEVEE render refraction through *material* (a no-op where unsupported)."""

    if hasattr(material, "use_raytrace_refraction"):
        material.use_raytrace_refraction = True
    if hasattr(scene.eevee, "use_raytracing"):
        scene.eevee.use_raytracing = True


def configure_sky(node: Any, elevation_deg: float, azimuth_deg: float, haze: float) -> str:
    """Set a sky texture node to a physical daylight sky without a sun disc.

    Blender 5.0 renamed the Nishita model to multiple scattering and its dust
    density to aerosol density.  The sky's sun rotation is measured clockwise
    from +Y, so azimuth 0 (+X) is a rotation of 90 degrees.
    """

    types = {item.identifier for item in node.bl_rna.properties["sky_type"].enum_items}
    sky_type = "MULTIPLE_SCATTERING" if "MULTIPLE_SCATTERING" in types else "NISHITA"
    node.sky_type = sky_type
    node.sun_disc = False
    node.sun_elevation = math.radians(elevation_deg)
    node.sun_rotation = math.radians(90.0 - azimuth_deg) % math.tau
    node.altitude = 0.0
    node.air_density = 1.0
    node.ozone_density = 1.0
    if hasattr(node, "aerosol_density"):
        node.aerosol_density = float(haze)
    else:
        node.dust_density = float(haze)
    return sky_type


def cycles_preferences() -> Any:
    return bpy.context.preferences.addons["cycles"].preferences


def refresh_cycles_devices(prefs: Any) -> None:
    if hasattr(prefs, "refresh_devices"):
        prefs.refresh_devices()
    else:  # pragma: no cover - pre-3.0 API kept for clarity
        prefs.get_devices()


__all__ = [
    "ENGINE_IDS",
    "GPU_BACKENDS",
    "MIN_VERSION",
    "VIEW_TRANSFORMS",
    "available_engines",
    "configure_sky",
    "cycles_preferences",
    "enable_eevee_refraction",
    "ensure_node_tree",
    "refresh_cycles_devices",
    "require_supported_version",
    "set_engine",
    "set_principled_input",
    "set_view_transform",
    "version",
    "version_info",
]
