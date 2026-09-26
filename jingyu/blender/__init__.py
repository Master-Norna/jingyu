"""Code that runs inside Blender.

Everything in this package executes in Blender's own Python (the ``blender``
executable or the ``bpy`` module).  It may import only the standard library,
``bpy``, ``bmesh``, ``mathutils`` and the standard-library-only parts of
``jingyu`` (:mod:`jingyu.errors`, :mod:`jingyu.conventions`,
:mod:`jingyu.canonical_json`, :mod:`jingyu.generator`, :mod:`jingyu.geometry`,
:mod:`jingyu.materials`).  A test enforces this boundary.

Layers:

* :mod:`.compat` hides differences between supported Blender versions.
* :mod:`.kit` is the thin, typed wrapper over ``bpy.data`` (units, axes and
  names follow :mod:`jingyu.conventions`; no context-dependent operators).
* :mod:`.build` turns a normalised scene description into Blender data.
* :mod:`.passes` renders auxiliary passes such as the object id mask.
* :mod:`.worker` is the process entry point speaking the worker protocol.
"""
