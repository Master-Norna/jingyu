"""Turn a normalised scene description into Blender data and render settings."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from ..conventions import kelvin_to_linear, srgb_hex_to_linear, sun_rotation_deg
from ..environments import ENVIRONMENTS, EnvironmentRecipe
from ..errors import JingyuError, pointer_join
from ..geometry import GEOMETRY
from ..materials import MATERIALS, Recipe
from ..placement import aim_matrix, resolve_placement
from . import compat, kit

DEFAULT_MATERIAL_NAME = "__jingyu_default__"
DEFAULT_MATERIAL_COLOR = "#bfbfbf"
SUN_NAME = "__jingyu_sun__"
#: How the environment's sun is named in diagnostics (never a valid scene id).
ENVIRONMENT_SUN_ID = "environment.sun"


@dataclass(frozen=True)
class BuiltObject:
    id: str
    pointer: str
    material: str | None
    blender_object: Any


@dataclass
class BuildResult:
    objects: list[BuiltObject] = field(default_factory=list)
    refractive: bool = False
    sky_model: str | None = None
    #: (light id, kind, Blender object) for every light, the environment's sun included.
    lights: list[tuple[str, str, Any]] = field(default_factory=list)
    #: Ids of objects whose material transmits light (glass).
    transmissive: set[str] = field(default_factory=set)
    transmissive_materials: set[str] = field(default_factory=set)
    warnings: list[dict[str, str]] = field(default_factory=list)

    @property
    def visible(self) -> list[BuiltObject]:
        return [o for o in self.objects if not o.blender_object.hide_render]


def build_scene(scene: Any, spec: Mapping[str, Any]) -> BuildResult:
    """Populate *scene* (assumed empty) from a validated, normalised *spec*."""

    result = BuildResult()
    world = spec["world"]
    try:
        environment = (
            ENVIRONMENTS.run(world["environment"])
            if "environment" in world
            else EnvironmentRecipe(srgb_hex_to_linear(world["color"]), 1.0)
        )
    except (KeyError, ValueError) as exc:
        raise _build_error("/world/environment", exc) from exc
    strength = 1.0 if "environment" in world else float(world["strength"])
    result.sky_model = kit.set_world(scene, environment, strength)
    sun = environment.sun
    if sun is not None:
        lamp = kit.new_light(
            scene, SUN_NAME, "sun", sun.color, strength=sun.strength, angle=sun.angle
        )
        kit.set_world_matrix(
            lamp, aim_matrix((0.0, 0.0, 0.0), None, sun_rotation_deg(sun.elevation, sun.azimuth))
        )
        result.lights.append((ENVIRONMENT_SUN_ID, "sun", lamp))

    try:
        placement = resolve_placement(spec, build_meshes=False)
    except (KeyError, ValueError) as exc:
        raise _build_error("", exc) from exc
    if placement.issues:
        issue = placement.issues[0]
        raise JingyuError(
            "blender.build_failed",
            f"cannot place {issue.pointer}: {issue.message}",
            details={"pointer": issue.pointer},
        )

    materials: dict[str, Any] = {}
    refractive_materials: list[Any] = []
    for index, entry in enumerate(spec["materials"]):
        params = {k: v for k, v in entry.items() if k != "id"}
        try:
            recipe = MATERIALS.run(params)
        except (KeyError, ValueError) as exc:
            raise _build_error(pointer_join("materials", index), exc) from exc
        material = kit.principled_material(entry["id"], recipe)
        materials[entry["id"]] = material
        if recipe.refractive:
            refractive_materials.append(material)
            result.transmissive_materials.add(entry["id"])

    default_material: Any = None
    for index, entry in enumerate(spec["objects"]):
        pointer = pointer_join("objects", index)
        try:
            mesh = GEOMETRY.run(entry["geometry"])
        except (KeyError, ValueError) as exc:
            raise _build_error(pointer + "/geometry", exc) from exc
        obj = kit.new_mesh_object(scene, entry["id"], mesh)
        kit.set_world_matrix(obj, placement.world[entry["id"]])
        material_id = entry.get("material")
        if material_id is not None:
            material = materials[material_id]
        else:
            if default_material is None:
                default_material = kit.principled_material(
                    DEFAULT_MATERIAL_NAME,
                    Recipe(base_color=srgb_hex_to_linear(DEFAULT_MATERIAL_COLOR)),
                )
            material = default_material
        kit.assign_material(obj, material)
        obj.hide_render = not entry["visible"]
        if material_id in result.transmissive_materials:
            result.transmissive.add(entry["id"])
        result.objects.append(BuiltObject(entry["id"], pointer, material_id, obj))

    for entry in spec["lights"]:
        params = {
            k: entry[k]
            for k in (
                "power_w",
                "radius",
                "spot_size",
                "blend",
                "size",
                "size_y",
                "strength",
                "angle",
            )
            if k in entry
        }
        temperature = entry.get("temperature_k")
        color = (
            kelvin_to_linear(temperature)
            if temperature is not None
            else srgb_hex_to_linear(entry["color"])
        )
        light = kit.new_light(scene, entry["id"], entry["kind"], color, **params)
        result.lights.append((entry["id"], entry["kind"], light))
        kit.set_world_matrix(light, placement.oriented[entry["id"]])

    cameras: dict[str, Any] = {}
    for entry in spec["cameras"]:
        camera = kit.new_camera(scene, entry["id"], dict(entry))
        kit.set_world_matrix(camera, placement.oriented[entry["id"]])
        cameras[entry["id"]] = camera
    scene.camera = cameras[spec["render"]["camera"]]

    if refractive_materials:
        result.refractive = True
        for material in refractive_materials:
            compat.enable_eevee_refraction(scene, material)
    return result


def configure_render(
    scene: Any, render: Mapping[str, Any], overrides: Mapping[str, Any]
) -> dict[str, Any]:
    """Apply render settings (plus host overrides) and return what was applied."""

    engine = render["engine"]
    resolution = list(overrides.get("resolution", render["resolution"]))
    samples = int(overrides.get("samples", render["samples"]))

    blender_engine = compat.set_engine(scene, engine)
    scene.render.resolution_x, scene.render.resolution_y = resolution
    scene.render.resolution_percentage = 100
    scene.render.pixel_aspect_x = scene.render.pixel_aspect_y = 1.0
    scene.render.use_motion_blur = False
    scene.render.film_transparent = bool(render["transparent_background"])
    scene.render.dither_intensity = 1.0

    settings = scene.render.image_settings
    settings.file_format = "PNG"
    settings.color_mode = "RGBA" if render["transparent_background"] else "RGB"
    settings.color_depth = "8"

    view = compat.set_view_transform(scene, render["view_transform"])
    scene.view_settings.look = "None"
    scene.view_settings.exposure = float(render["exposure"])
    scene.view_settings.gamma = 1.0

    device = "n/a"
    if engine == "cycles":
        scene.cycles.samples = samples
        scene.cycles.use_denoising = bool(render["denoise"])
        scene.cycles.seed = int(render["seed"])
        scene.cycles.use_animated_seed = False
        device = kit.select_cycles_device(scene, render["device"])
    elif engine == "eevee":
        scene.eevee.taa_render_samples = samples

    return {
        "engine": engine,
        "blender_engine": blender_engine,
        "device": device,
        "resolution": resolution,
        "samples": samples,
        "denoise": bool(render["denoise"]) if engine == "cycles" else False,
        "seed": int(render["seed"]),
        "view_transform": view,
        "exposure": float(render["exposure"]),
    }


def _build_error(pointer: str, exc: Exception) -> JingyuError:
    if isinstance(exc, JingyuError):
        return exc
    return JingyuError(
        "blender.build_failed",
        f"cannot build {pointer}: {exc}",
        details={"pointer": pointer},
    )


__all__ = ["BuildResult", "BuiltObject", "build_scene", "configure_render"]
