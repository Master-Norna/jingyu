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
from mathutils import Matrix

from ..environments import EnvironmentRecipe
from ..environments.recipe import Air
from ..errors import JingyuError
from ..geometry.mesh import MeshData
from ..geometry.pieces import rounded_box
from ..materials.recipe import RGB, Recipe
from . import compat, shading

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


def new_mesh_object(scene: Any, name: str, data: MeshData, materials: Sequence[Any]) -> Any:
    """Create a mesh object from renderer-independent mesh data.

    *materials* fill the slots in order: slot i dresses part i of an assembled mesh
    (a one-piece mesh takes one material).
    """

    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata([tuple(v) for v in data.vertices], [], [tuple(f) for f in data.faces])
    for material in materials:
        mesh.materials.append(material)
    if data.smooth:
        mesh.polygons.foreach_set("use_smooth", [True] * len(mesh.polygons))
    if data.parts:
        mesh.polygons.foreach_set("material_index", list(data.face_parts))
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


def set_world_matrix(obj: Any, matrix: Sequence[float]) -> None:
    """Place *obj* by a row-major 4x4 world matrix (jingyu.geometry.transform)."""

    obj.rotation_mode = "XYZ"
    obj.matrix_world = Matrix([matrix[0:4], matrix[4:8], matrix[8:12], matrix[12:16]])


def principled_material(name: str, recipe: Recipe) -> Any:
    """Create a principled BSDF material from a recipe (scene-linear values)."""

    material = bpy.data.materials.new(name)
    if material.name != name:
        raise JingyuError("blender.build_failed", f"material name {name!r} is not unique")
    compat.ensure_node_tree(material)
    if recipe.thin:
        _thin_sheet(material, recipe)
        return material
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
    if recipe.sheen_weight > 0:
        compat.set_principled_input(node, "sheen_weight", recipe.sheen_weight)
    if recipe.textured:
        shading.build_textured(material, node, recipe)
    material.diffuse_color = (*recipe.base_color, recipe.alpha)
    material.metallic = recipe.metallic
    material.roughness = recipe.roughness
    return material


def _thin_sheet(material: Any, recipe: Recipe) -> None:
    """A pane: see-through (tinted) where the Fresnel term does not reflect.

    A principled BSDF with transmission refracts, and sunlight through refraction
    is a caustic path the renderer barely samples: the window would shade the
    room.  A thin sheet does not bend light, so it is a transparent surface mixed
    with a glossy reflection, and shadow rays pass through it.
    """

    tree = material.node_tree
    tree.nodes.clear()
    output = tree.nodes.new("ShaderNodeOutputMaterial")
    clear = tree.nodes.new("ShaderNodeBsdfTransparent")
    clear.inputs["Color"].default_value = (*recipe.base_color, 1.0)
    gloss = tree.nodes.new("ShaderNodeBsdfGlossy")
    gloss.inputs["Roughness"].default_value = recipe.roughness
    fresnel = tree.nodes.new("ShaderNodeFresnel")
    fresnel.inputs["IOR"].default_value = recipe.ior
    mix = tree.nodes.new("ShaderNodeMixShader")
    tree.links.new(fresnel.outputs["Fac"], mix.inputs["Fac"])
    tree.links.new(clear.outputs["BSDF"], mix.inputs[1])
    tree.links.new(gloss.outputs["BSDF"], mix.inputs[2])
    tree.links.new(mix.outputs["Shader"], output.inputs["Surface"])
    material.diffuse_color = (*recipe.base_color, 0.1)
    compat.enable_transparency(material)


def new_light(scene: Any, name: str, kind: str, color: RGB, **params: Any) -> Any:
    """Create a light.  ``params`` use jingyu units (watts, metres, degrees)."""

    light = bpy.data.lights.new(name, "AREA" if kind == "fill" else kind.upper())
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
    if kind in ("area", "fill"):
        light.size = float(params["size"])
        if params.get("size_y") is not None:
            light.shape = "RECTANGLE"
            light.size_y = float(params["size_y"])
        else:
            light.shape = "SQUARE"
    obj = _link(scene, bpy.data.objects.new(name, light), name)
    if kind == "fill":
        # A helper light: no shadow of its own, no highlight, never seen by the camera.
        light.use_shadow = False
        light.specular_factor = 0.0
        obj.visible_glossy = False
        obj.visible_camera = False
    return obj


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


def set_world(scene: Any, environment: EnvironmentRecipe, strength: float) -> str | None:
    """Create the world from an environment recipe; returns the sky model used, if any.

    Above the horizon the background is a uniform fill plus, when the recipe has
    one, a physical sky; below it, the recipe's ground when it has one.  All are
    scaled by *strength*.  The recipe's sun is added separately as a lamp.
    """

    world = bpy.data.worlds.new("world")
    world.color = environment.fill
    scene.world = world
    if world.node_tree is None or compat.version() < (5, 0, 0):
        world.use_nodes = True
    nodes, links = world.node_tree.nodes, world.node_tree.links
    for node in list(nodes):
        nodes.remove(node)
    output = nodes.new("ShaderNodeOutputWorld")

    def background(color: RGB, level: float) -> Any:
        node = nodes.new("ShaderNodeBackground")
        node.inputs["Color"].default_value = (*color, 1.0)
        node.inputs["Strength"].default_value = float(level) * float(strength)
        return node

    upper = background(environment.fill, environment.fill_strength).outputs["Background"]
    model = None
    if environment.sky is not None:
        sky_texture = nodes.new("ShaderNodeTexSky")
        model = compat.configure_sky(
            sky_texture, environment.sky.elevation, environment.sky.azimuth, environment.sky.haze
        )
        sky = background((1.0, 1.0, 1.0), environment.sky.strength)
        links.new(sky_texture.outputs["Color"], sky.inputs["Color"])
        add = nodes.new("ShaderNodeAddShader")
        links.new(sky.outputs["Background"], add.inputs[0])
        links.new(upper, add.inputs[1])
        upper = add.outputs["Shader"]
    if environment.ground is None:
        links.new(upper, output.inputs["Surface"])
        return model

    coordinates = nodes.new("ShaderNodeTexCoord")
    separate = nodes.new("ShaderNodeSeparateXYZ")
    above = nodes.new("ShaderNodeMath")
    above.operation = "GREATER_THAN"
    above.inputs[1].default_value = 0.0
    links.new(coordinates.outputs["Generated"], separate.inputs[0])
    links.new(separate.outputs["Z"], above.inputs[0])
    mix = nodes.new("ShaderNodeMixShader")
    links.new(above.outputs[0], mix.inputs["Fac"])
    links.new(background(environment.ground, 1.0).outputs["Background"], mix.inputs[1])
    links.new(upper, mix.inputs[2])
    links.new(mix.outputs["Shader"], output.inputs["Surface"])
    return model


AIR_NAME = "__jingyu_air__"


def new_air(scene: Any, air: Air, low: Vec3, high: Vec3) -> Any:
    """A box of scattering air between *low* and *high* (world corners).

    The air is bounded on purpose: an infinite medium would swallow the sun on its
    way in from infinity and hide the sky behind every window.
    """

    material = bpy.data.materials.new(AIR_NAME)
    compat.ensure_node_tree(material)
    tree = material.node_tree
    tree.nodes.clear()
    output = tree.nodes.new("ShaderNodeOutputMaterial")
    scatter = tree.nodes.new("ShaderNodeVolumeScatter")
    scatter.inputs["Color"].default_value = (*air.color, 1.0)
    scatter.inputs["Density"].default_value = air.density
    scatter.inputs["Anisotropy"].default_value = air.anisotropy
    tree.links.new(scatter.outputs["Volume"], output.inputs["Volume"])
    x0, y0, z0 = low
    x1, y1, z1 = high
    size = (x1 - x0, y1 - y0, z1 - z0)
    box = rounded_box(size, 0.0)
    obj = new_mesh_object(scene, AIR_NAME, box, [material])
    obj.location = ((x0 + x1) / 2.0, (y0 + y1) / 2.0, z0)
    return obj


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
    "new_camera",
    "new_light",
    "new_mesh_object",
    "principled_material",
    "render_still",
    "reset_to_empty",
    "select_cycles_device",
    "set_world",
    "set_world_matrix",
]
