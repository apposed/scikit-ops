"""The bundled op collection.

Ops are grouped into namespaces by what they do -- ``skop.ops.threshold``,
``skop.ops.segment`` -- and are imported from the namespace regardless of how
it is laid out inside::

    from skop.ops.threshold import otsu
    from skop.ops.segment import cellpose

A namespace whose ops are variations on one call sharing one environment is a
single module (``threshold.py``). A namespace whose ops span environments and
carry their own models and types is a package with a module per op
(``segment/``), re-exporting them in its ``__init__.py``. Both read the same
from the outside, so a namespace can grow from one into the other without
breaking a caller's import.

Discovery walks this package recursively, skipping underscore-prefixed
modules; see ``skop.discovery``.
"""
