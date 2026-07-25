"""The op collection.

Each op is either a bare module here (``ops/myop.py``) or a subpackage
(``ops/myop/``) when one file is not enough. Both are discovered the same way.

This file exists so the collection is a namespace of its own. Without it, the
directory would have to go on ``sys.path`` directly, where a module named
``json.py`` or ``types.py`` would shadow the standard library inside every
worker process.
"""
