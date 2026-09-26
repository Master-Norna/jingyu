"""Auxiliary render passes.

The object id mask lets a model point at a region of the image and get back
which object, material and scene entry it is looking at.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import bpy

from ..idmask import ENCODING, ID_MAP_SCHEMA, encode_index
from . import compat, kit
from .build import BuiltObject

_OVERRIDE_NAME = "__jingyu_id_mask__"


def _override_material() -> Any:
    material = bpy.data.materials.new(_OVERRIDE_NAME)
    compat.ensure_node_tree(material)
    tree = material.node_tree
    tree.nodes.clear()
    info = tree.nodes.new("ShaderNodeObjectInfo")
    emission = tree.nodes.new("ShaderNodeEmission")
    output = tree.nodes.new("ShaderNodeOutputMaterial")
    emission.inputs["Strength"].default_value = 1.0
    tree.links.new(info.outputs["Color"], emission.inputs["Color"])
    tree.links.new(emission.outputs["Emission"], output.inputs["Surface"])
    return material


def render_id_mask(scene: Any, objects: list[BuiltObject], filepath: Path) -> dict[str, Any]:
    """Render the id mask for the visible *objects* and return the id map.

    This pass reconfigures the scene for exact, unlit output and must run after
    the beauty render.
    """

    entries: dict[str, Any] = {}
    for index, built in enumerate(objects, start=1):
        r, g, b = encode_index(index)
        built.blender_object.color = (r / 255.0, g / 255.0, b / 255.0, 1.0)
        entries[str(index)] = {
            "object": built.id,
            "pointer": built.pointer,
            "material": built.material,
        }

    for obj in scene.objects:
        if obj.type == "LIGHT":
            obj.visible_camera = False

    scene.view_layers[0].material_override = _override_material()
    compat.set_engine(scene, "cycles")
    scene.cycles.device = "CPU"
    scene.cycles.samples = 1
    scene.cycles.use_adaptive_sampling = False
    scene.cycles.use_denoising = False
    scene.cycles.filter_width = 0.01
    scene.cycles.max_bounces = 0
    scene.cycles.seed = 0
    scene.render.film_transparent = True
    scene.render.dither_intensity = 0.0
    scene.render.use_motion_blur = False
    scene.view_settings.view_transform = "Raw"
    scene.view_settings.look = "None"
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0
    settings = scene.render.image_settings
    settings.file_format = "PNG"
    settings.color_mode = "RGBA"
    settings.color_depth = "8"

    kit.render_still(scene, filepath)
    return {
        "schema": ID_MAP_SCHEMA,
        "encoding": ENCODING,
        "objects": entries,
    }


__all__ = ["render_id_mask"]
