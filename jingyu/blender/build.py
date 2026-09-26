"""Turn a normalised scene description into Blender data and render settings."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from mathutils import Vector

from ..canonical_json import canonical_bytes
from ..conventions import kelvin_to_linear, srgb_hex_to_linear, sun_rotation_deg
from ..environments import ENVIRONMENTS, EnvironmentRecipe
from ..errors import JingyuError, pointer_join
from ..geometry import GEOMETRY
from ..materials import MATERIALS
from ..materials.assign import PartMaterial, part_materials
from ..materials.weathering import material_recipe
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
    #: False for objects the camera must not see (reflector cards, flags).
    camera_visible: bool = True
    #: Material slot i dresses part parts[i] (a single None for a one-piece mesh)
    #: with the scene material slot_materials[i] (None for a default).
    parts: tuple[str | None, ...] = (None,)
    slot_materials: tuple[str | None, ...] = (None,)


@dataclass
class BuildResult:
    objects: list[BuiltObject] = field(default_factory=list)
    refractive: bool = False
    sky_model: str | None = None
    #: (light id, kind, Blender object) for every light, the environment's sun included.
    lights: list[tuple[str, str, Any]] = field(default_factory=list)
    #: Names of Blender materials that transmit light (glass).
    transmissive_materials: set[str] = field(default_factory=set)
    warnings: list[dict[str, str]] = field(default_factory=list)
    #: The box of scattering air, when the environment has mist; hidden from the
    #: diagnostic passes, which look at surfaces.
    air: Any = None

    @property
    def visible(self) -> list[BuiltObject]:
        """Objects in the render: seen by the camera or only lighting the scene."""

        return [o for o in self.objects if not o.blender_object.hide_render]

    @property
    def seen(self) -> list[BuiltObject]:
        """Rendered objects the camera sees."""

        return [o for o in self.visible if o.camera_visible]


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
        try:
            recipe = material_recipe(entry)
        except (KeyError, ValueError) as exc:
            raise _build_error(pointer_join("materials", index), exc) from exc
        material = kit.principled_material(entry["id"], recipe)
        materials[entry["id"]] = material
        if recipe.refractive:
            refractive_materials.append(material)
            result.transmissive_materials.add(entry["id"])

    defaults: dict[str, Any] = {}

    def default_for(assigned: PartMaterial) -> Any:
        """The Blender material for a part without a scene material."""

        spec_ = assigned.default or {"family": "principled", "base_color": DEFAULT_MATERIAL_COLOR}
        key = canonical_bytes(spec_).decode("utf-8")
        if key not in defaults:
            name = (
                DEFAULT_MATERIAL_NAME if assigned.default is None else f"__jingyu_{len(defaults)}__"
            )
            recipe = MATERIALS.run(spec_)
            defaults[key] = kit.principled_material(name, recipe)
            if recipe.refractive:
                refractive_materials.append(defaults[key])
                result.transmissive_materials.add(name)
        return defaults[key]

    for index, entry in enumerate(spec["objects"]):
        pointer = pointer_join("objects", index)
        try:
            mesh = GEOMETRY.run(entry["geometry"])
        except (KeyError, ValueError) as exc:
            raise _build_error(pointer + "/geometry", exc) from exc
        assignment = part_materials(entry, GEOMETRY.get(entry["geometry"]["op"]))
        slots = [
            materials[a.material] if a.material is not None else default_for(a)
            for a in assignment.values()
        ]
        obj = kit.new_mesh_object(scene, entry["id"], mesh, slots)
        kit.set_world_matrix(obj, placement.world[entry["id"]])
        obj.hide_render = not entry["visible"]
        obj.visible_camera = bool(entry["camera_visible"])
        result.objects.append(
            BuiltObject(
                entry["id"],
                pointer,
                entry.get("material"),
                obj,
                bool(entry["camera_visible"]),
                tuple(assignment),
                tuple(a.material for a in assignment.values()),
            )
        )

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

    if environment.air is not None:
        result.air = _fill_air(scene, environment.air, result, scene.camera)

    if refractive_materials:
        result.refractive = True
        for material in refractive_materials:
            compat.enable_eevee_refraction(scene, material)
    return result


def _fill_air(scene: Any, air: Any, result: BuildResult, camera: Any) -> Any:
    """Fill the space around everything visible, and the camera, with air."""

    points = [camera.matrix_world.translation]
    for built in result.visible:
        obj = built.blender_object
        points += [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    low = [min(p[k] for p in points) for k in range(3)]
    high = [max(p[k] for p in points) for k in range(3)]
    margin = [0.05 * (h - lo) + 0.25 for lo, h in zip(low, high, strict=True)]
    return kit.new_air(
        scene,
        air,
        (low[0] - margin[0], low[1] - margin[1], low[2] - margin[2]),
        (high[0] + margin[0], high[1] + margin[1], high[2] + margin[2]),
    )


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
