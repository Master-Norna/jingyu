"""A small, typed wrapper over ``bpy.data``.

Raw ``bpy`` is large and much of it depends on UI context (selection, active
object, edit mode), which makes it fragile in background processes.  This kit
creates and edits data blocks directly, speaks jingyu's conventions (metres,
degrees, Z up, sRGB hex converted by the caller to linear) and reports failures
as :class:`~jingyu.errors.JingyuError` with stable codes.

The only operators used are ``wm.read_factory_settings`` (to start from an empty
file) and ``render.render`` (there is no data-level render call).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import bpy
from mathutils import Euler, Vector

from ..errors import JingyuError
from ..geometry.mesh import MeshData
from ..materials.recipe import RGB, Recipe
from . import compat

Vec3 = Sequence[float]


def reset_to_empty() -> Any:
    """Start from factory settings with an empty scene and return that scene."""

    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.data.scenes[0]
    for extra in list(bpy.data.scenes)[1:]:
        bpy.data.scenes.remove(extra)
    return scene


def _link(scene: Any, obj: Any, expected_name: str) -> Any:
    scene.collection.objects.link(obj)
    if obj.name != expected_name:
        raise JingyuError(
            "blender.build_failed",
            f"Blender renamed {expected_name!r} to {obj.name!r}; ids must be unique",
        )
    return obj


def new_mesh_object(scene: Any, name: str, data: MeshData) -> Any:
    """Create a mesh object from renderer-independent mesh data."""

    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata([tuple(v) for v in data.vertices], [], [tuple(f) for f in data.faces])
    if data.smooth:
        mesh.polygons.foreach_set("use_smooth", [True] * len(mesh.polygons))
    if data.sharp_edges:
        sharp = {tuple(sorted(edge)) for edge in data.sharp_edges}
        for edge in mesh.edges:
            if tuple(sorted(edge.vertices)) in sharp:
                edge.use_edge_sharp = True
    mesh.update()
    if mesh.validate(verbose=False):
        raise JingyuError(
            "blender.build_failed",
            f"generated mesh for {name!r} was malformed and had to be repaired",
        )
    return _link(scene, bpy.data.objects.new(name, mesh), name)


def set_transform(obj: Any, location: Vec3, rotation_deg: Vec3, scale: Vec3) -> None:
    obj.location = Vector(location)
    obj.rotation_mode = "XYZ"
    obj.rotation_euler = Euler([math.radians(a) for a in rotation_deg], "XYZ")
    obj.scale = Vector(scale)


def aim(obj: Any, location: Vec3, look_at: Vec3 | None, rotation_deg: Vec3 | None) -> None:
    """Place *obj* and point its -Z axis at *look_at* (Y up), or apply a rotation."""

    obj.location = Vector(location)
    obj.rotation_mode = "XYZ"
    if look_at is not None:
        direction = Vector(look_at) - Vector(location)
        obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler("XYZ")
    elif rotation_deg is not None:
        obj.rotation_euler = Euler([math.radians(a) for a in rotation_deg], "XYZ")


def principled_material(name: str, recipe: Recipe) -> Any:
    """Create a principled BSDF material from a recipe (scene-linear values)."""

    material = bpy.data.materials.new(name)
    if material.name != name:
        raise JingyuError("blender.build_failed", f"material name {name!r} is not unique")
    compat.ensure_node_tree(material)
    node = next(
        (n for n in material.node_tree.nodes if n.bl_idname == "ShaderNodeBsdfPrincipled"),
        None,
    )
    if node is None:
        raise JingyuError("blender.build_failed", "the new material has no principled BSDF")
    compat.set_principled_input(node, "base_color", (*recipe.base_color, 1.0))
    compat.set_principled_input(node, "metallic", recipe.metallic)
    compat.set_principled_input(node, "roughness", recipe.roughness)
    compat.set_principled_input(node, "ior", recipe.ior)
    compat.set_principled_input(node, "alpha", recipe.alpha)
    compat.set_principled_input(node, "transmission", recipe.transmission)
    compat.set_principled_input(node, "coat_weight", recipe.coat_weight)
    compat.set_principled_input(node, "coat_roughness", recipe.coat_roughness)
    compat.set_principled_input(node, "emission_color", (*recipe.emission_color, 1.0))
    compat.set_principled_input(node, "emission_strength", recipe.emission_strength)
    material.diffuse_color = (*recipe.base_color, recipe.alpha)
    material.metallic = recipe.metallic
    material.roughness = recipe.roughness
    return material


def assign_material(obj: Any, material: Any) -> None:
    obj.data.materials.clear()
    obj.data.materials.append(material)


def new_light(scene: Any, name: str, kind: str, color: RGB, **params: Any) -> Any:
    """Create a light.  ``params`` use jingyu units (watts, metres, degrees)."""

    light = bpy.data.lights.new(name, kind.upper())
    light.color = color
    if kind == "sun":
        light.energy = float(params["strength"])
        light.angle = math.radians(float(params["angle"]))
    else:
        light.energy = float(params["power_w"])
    if kind in ("point", "spot"):
        light.shadow_soft_size = float(params["radius"])
    if kind == "spot":
        light.spot_size = math.radians(float(params["spot_size"]))
        light.spot_blend = float(params["blend"])
    if kind == "area":
        light.shape = "SQUARE"
        light.size = float(params["size"])
    return _link(scene, bpy.data.objects.new(name, light), name)


def new_camera(scene: Any, name: str, params: dict[str, Any]) -> Any:
    camera = bpy.data.cameras.new(name)
    camera.sensor_fit = "HORIZONTAL"
    camera.sensor_width = float(params["sensor_width_mm"])
    if params["projection"] == "orthographic":
        camera.type = "ORTHO"
        camera.ortho_scale = float(params["ortho_scale"])
    else:
        camera.type = "PERSP"
        camera.lens = float(params["lens_mm"])
    camera.clip_start = float(params["clip_start"])
    camera.clip_end = float(params["clip_end"])
    camera.shift_x, camera.shift_y = (float(v) for v in params["shift"])
    return _link(scene, bpy.data.objects.new(name, camera), name)


def set_world(scene: Any, color: RGB, strength: float) -> None:
    world = bpy.data.worlds.new("world")
    world.color = color
    scene.world = world
    if world.node_tree is None or compat.version() < (5, 0, 0):
        world.use_nodes = True
    background = next(
        (n for n in world.node_tree.nodes if n.bl_idname == "ShaderNodeBackground"), None
    )
    if background is None:
        background = world.node_tree.nodes.new("ShaderNodeBackground")
        output = next(
            (n for n in world.node_tree.nodes if n.bl_idname == "ShaderNodeOutputWorld"), None
        ) or world.node_tree.nodes.new("ShaderNodeOutputWorld")
        world.node_tree.links.new(background.outputs["Background"], output.inputs["Surface"])
    background.inputs["Color"].default_value = (*color, 1.0)
    background.inputs["Strength"].default_value = float(strength)


def select_cycles_device(scene: Any, want: str) -> str:
    """Choose a Cycles device; returns ``"CPU"`` or ``"GPU:<backend>"``."""

    if want == "cpu":
        scene.cycles.device = "CPU"
        return "CPU"
    try:
        prefs = compat.cycles_preferences()
    except KeyError:
        prefs = None
    if prefs is not None:
        for backend in compat.GPU_BACKENDS:
            try:
                prefs.compute_device_type = backend
                compat.refresh_cycles_devices(prefs)
            except (TypeError, RuntimeError):
                continue
            gpus = [d for d in prefs.devices if d.type == backend]
            if gpus:
                for device in prefs.devices:
                    device.use = device.type == backend
                scene.cycles.device = "GPU"
                return f"GPU:{backend}"
    if want == "gpu":
        raise JingyuError(
            "blender.gpu_unavailable",
            "render.device is 'gpu' but Cycles found no usable GPU",
            hint="Use device 'auto' to fall back to the CPU.",
        )
    scene.cycles.device = "CPU"
    return "CPU"


def render_still(scene: Any, filepath: Path) -> None:
    scene.render.filepath = str(filepath)
    result = bpy.ops.render.render(write_still=True, scene=scene.name)
    if "FINISHED" not in result or not filepath.exists():
        raise JingyuError("blender.render_failed", f"render did not produce {filepath.name}")


__all__ = [
    "aim",
    "assign_material",
    "new_camera",
    "new_light",
    "new_mesh_object",
    "principled_material",
    "render_still",
    "reset_to_empty",
    "select_cycles_device",
    "set_transform",
    "set_world",
]
