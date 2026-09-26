"""Translate a recipe's fields, layers and bumps into shader nodes.

Fields read the object's own coordinates (metres, origin at its base), so a
texture sticks to the object and keeps its real size whatever the object is.
Only nodes that exist from Blender 4.2 to 5.0 are used; socket names that
differ between versions are resolved in :mod:`.compat`.
"""

from __future__ import annotations

import math
from typing import Any

from ..materials.recipe import Bump, Field, Layer, Recipe
from . import compat

#: Blender's wave texture advances 20 radians per unit before scaling.
_WAVE_RADIANS = 20.0


class _Graph:
    def __init__(self, material: Any):
        self.tree = material.node_tree
        self.nodes = self.tree.nodes
        self.links = self.tree.links
        self._coords: Any = None
        self._geometry: Any = None

    def node(self, kind: str, **settings: Any) -> Any:
        node = self.nodes.new(kind)
        for key, value in settings.items():
            setattr(node, key, value)
        return node

    def link(self, output: Any, node: Any, socket: str | int) -> None:
        self.links.new(output, node.inputs[socket])

    def value(self, number: float) -> Any:
        node = self.node("ShaderNodeValue")
        node.outputs[0].default_value = float(number)
        return node.outputs[0]

    def math(self, operation: str, a: Any, b: Any | float | None = None) -> Any:
        node = self.node("ShaderNodeMath", operation=operation)
        for index, operand in enumerate((a, b)):
            if operand is None:
                continue
            if isinstance(operand, float | int):
                node.inputs[index].default_value = float(operand)
            else:
                self.links.new(operand, node.inputs[index])
        return node.outputs[0]

    def coords(self) -> Any:
        if self._coords is None:
            self._coords = self.node("ShaderNodeTexCoord").outputs["Object"]
        return self._coords

    def geometry(self) -> Any:
        if self._geometry is None:
            self._geometry = self.node("ShaderNodeNewGeometry")
        return self._geometry

    # ------------------------------------------------------------ fields

    def field(self, spec: Field) -> Any:
        raw = self._raw(spec)
        remap = self.node("ShaderNodeMapRange", interpolation_type="SMOOTHSTEP", clamp=True)
        self.links.new(raw, remap.inputs["Value"])
        remap.inputs["From Min"].default_value = spec.low
        remap.inputs["From Max"].default_value = spec.high
        remap.inputs["To Min"].default_value = 0.0
        remap.inputs["To Max"].default_value = 1.0
        out = remap.outputs["Result"]
        return self.math("SUBTRACT", 1.0, out) if spec.invert else out

    def _placed(self, spec: Field) -> Any:
        """Object coordinates, shifted by the seed and stretched per axis."""

        mapping = self.node("ShaderNodeMapping", vector_type="POINT")
        self.links.new(self.coords(), mapping.inputs["Vector"])
        s = float(spec.seed)
        mapping.inputs["Location"].default_value = (s * 13.71, s * 7.33, s * 3.17)
        mapping.inputs["Scale"].default_value = spec.stretch
        return mapping.outputs["Vector"]

    def _raw(self, spec: Field) -> Any:
        kind = spec.kind
        if kind in ("noise", "ridges"):
            noise = self.node("ShaderNodeTexNoise", noise_dimensions="3D")
            self.links.new(self._placed(spec), noise.inputs["Vector"])
            noise.inputs["Scale"].default_value = 1.0 / spec.size
            noise.inputs["Detail"].default_value = spec.detail
            noise.inputs["Roughness"].default_value = spec.roughness
            noise.inputs["Distortion"].default_value = spec.distortion
            fac = noise.outputs["Fac"]
            if kind == "noise":
                return fac
            centred = self.math(
                "ABSOLUTE", self.math("SUBTRACT", self.math("MULTIPLY", fac, 2.0), 1.0)
            )
            return self.math("SUBTRACT", 1.0, centred)
        if kind in ("cells", "cracks"):
            feature = "F1" if kind == "cells" else "DISTANCE_TO_EDGE"
            cells = self.node("ShaderNodeTexVoronoi", voronoi_dimensions="3D", feature=feature)
            self.links.new(self._placed(spec), cells.inputs["Vector"])
            cells.inputs["Scale"].default_value = 1.0 / spec.size
            return cells.outputs["Distance"]
        if kind in ("rings", "bands"):
            wave = self.node("ShaderNodeTexWave", wave_type=kind.upper())
            direction = spec.axis.upper()
            if kind == "rings":
                wave.rings_direction = direction
            else:
                wave.bands_direction = direction
            self.links.new(self._placed(spec), wave.inputs["Vector"])
            wave.inputs["Scale"].default_value = 2.0 * math.pi / (_WAVE_RADIANS * spec.size)
            wave.inputs["Distortion"].default_value = spec.distortion
            wave.inputs["Detail"].default_value = spec.detail
            return wave.outputs["Fac"]
        if kind == "facing_up":
            separate = self.node("ShaderNodeSeparateXYZ")
            self.links.new(self.geometry().outputs["Normal"], separate.inputs[0])
            return separate.outputs["Z"]
        if kind in ("edges", "cavity"):
            occlusion = self.node("ShaderNodeAmbientOcclusion", inside=kind == "edges", samples=8)
            occlusion.inputs["Distance"].default_value = spec.distance
            return self.math("SUBTRACT", 1.0, occlusion.outputs["AO"])
        if kind == "low":
            separate = self.node("ShaderNodeSeparateXYZ")
            self.links.new(self.coords(), separate.inputs[0])
            height = self.math("DIVIDE", separate.outputs["Z"], spec.distance)
            return self.math(
                "SUBTRACT", 1.0, self.math("MINIMUM", self.math("MAXIMUM", height, 0.0), 1.0)
            )
        raise ValueError(f"unknown field kind {kind!r}")

    def mask(self, layer: Layer) -> Any:
        out = self.field(layer.mask[0])
        for spec in layer.mask[1:]:
            out = self.math("MULTIPLY", out, self.field(spec))
        return self.math("MULTIPLY", out, layer.amount)

    # ------------------------------------------------------------ mixing

    def mix_color(self, factor: Any, below: Any, color: tuple[float, float, float]) -> Any:
        node = self.node("ShaderNodeMix", data_type="RGBA", blend_type="MIX")
        self.links.new(factor, compat.mix_socket(node, "Factor", "FLOAT"))
        self.links.new(below, compat.mix_socket(node, "A", "RGBA"))
        compat.mix_socket(node, "B", "RGBA").default_value = (*color, 1.0)
        return compat.mix_output(node, "RGBA")

    def mix_value(self, factor: Any, below: Any, value: float) -> Any:
        node = self.node("ShaderNodeMix", data_type="FLOAT")
        self.links.new(factor, compat.mix_socket(node, "Factor", "FLOAT"))
        self.links.new(below, compat.mix_socket(node, "A", "FLOAT"))
        compat.mix_socket(node, "B", "FLOAT").default_value = float(value)
        return compat.mix_output(node, "FLOAT")


def build_textured(material: Any, principled: Any, recipe: Recipe) -> None:
    """Wire the recipe's layers and bumps into the principled BSDF's inputs."""

    graph = _Graph(material)
    color_node = graph.node("ShaderNodeRGB")
    color_node.outputs[0].default_value = (*recipe.base_color, 1.0)
    channels: dict[str, Any] = {"base_color": color_node.outputs[0]}
    for name in ("roughness", "metallic", "coat_weight"):
        channels[name] = graph.value(getattr(recipe, name))
    touched: set[str] = set()
    for layer in recipe.layers:
        factor = graph.mask(layer)
        if layer.color is not None:
            channels["base_color"] = graph.mix_color(factor, channels["base_color"], layer.color)
            touched.add("base_color")
        for name in ("roughness", "metallic", "coat_weight"):
            value = getattr(layer, name)
            if value is not None:
                channels[name] = graph.mix_value(factor, channels[name], value)
                touched.add(name)
    for name in touched:
        graph.links.new(channels[name], compat.principled_socket(principled, name))
    normal: Any = None
    for bump in recipe.bumps:
        normal = _bump(graph, bump, normal)
    if normal is not None:
        graph.links.new(normal, compat.principled_socket(principled, "normal"))
        coat_normal = compat.principled_socket(principled, "coat_normal", required=False)
        if coat_normal is not None:
            graph.links.new(normal, coat_normal)


def _bump(graph: _Graph, bump: Bump, normal: Any) -> Any:
    node = graph.node("ShaderNodeBump")
    node.inputs["Strength"].default_value = bump.strength
    node.inputs["Distance"].default_value = bump.distance
    graph.links.new(graph.field(bump.field), node.inputs["Height"])
    if normal is not None:
        graph.links.new(normal, node.inputs["Normal"])
    return node.outputs["Normal"]


__all__ = ["build_textured"]
