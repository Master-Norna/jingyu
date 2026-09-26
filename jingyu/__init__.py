"""Jingyu: a controllable, editable image-making toolkit built on Blender.

This top-level module must stay import-light.  The Blender worker imports
``jingyu`` inside Blender's bundled Python, where only the standard library,
``bpy``, ``bmesh`` and ``mathutils`` are available.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
