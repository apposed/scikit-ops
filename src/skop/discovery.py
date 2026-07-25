"""Finding the ops in a collection.

Discovery works by importing op modules and reading their live signatures,
rather than by consulting a generated manifest that could drift. That is only
possible because ops keep heavy imports inside their function bodies -- which
means discovery doubles as enforcement of that rule. An op whose module scope
drags in tensorflow simply fails to load, and says so.
"""

from __future__ import annotations

import importlib
import pkgutil
from dataclasses import dataclass

from ._spec import OpSpec, is_op, spec


@dataclass(frozen=True)
class LoadFailure:
    """An op module that could not be imported, and why."""

    module: str
    error: str
    heavy_imports: tuple[str, ...]

    def __str__(self) -> str:
        lines = [f"Op module '{self.module}' could not be loaded: {self.error}"]
        if self.heavy_imports:
            lines.append(
                "  Module-scope imports found: "
                + ", ".join(self.heavy_imports)
                + "\n  Move heavy imports inside the op function body -- ops are "
                "discovered in a minimal environment."
            )
        return "\n".join(lines)


def discover(package: str = "skop.ops") -> tuple[list[OpSpec], list[LoadFailure]]:
    """Import every module in a collection and return the ops it declares.

    Returns:
        The specs of every op found, and a failure record for each module
        that could not be imported.
    """
    # A collection that does not import at all is a bad argument rather than
    # a bad op, so it raises instead of joining the failure list.
    root = importlib.import_module(package)
    specs: list[OpSpec] = []
    failures: list[LoadFailure] = []

    for module_name, module in _walk(package, root, failures):
        for name in dir(module):
            obj = getattr(module, name)
            # NB: the name check keeps a re-exported op from being counted
            # once per module that imports it -- which is what makes a
            # namespace's __init__.py free to re-export its ops.
            if is_op(obj) and obj.__module__ == module_name:
                specs.append(spec(obj))

    specs.sort(key=lambda s: s.name)
    return specs, failures


def _walk(package: str, module, failures: list[LoadFailure]):
    """Yield a collection and every module beneath it, depth first.

    An op module may sit directly in the collection (``skop/ops/toy.py``),
    inside a namespace (``skop/ops/segment/cellpose.py``), or be a package of
    its own when one file is not enough (``skop/ops/segment/unseg/``). All
    three are the same walk: import what is there, then descend into whatever
    has submodules. Underscore-prefixed names are private helpers --
    ``_util.py``, ``_algorithm.py`` -- and are never imported.
    """
    yield package, module

    paths = getattr(module, "__path__", None)
    if paths is None:
        return
    for info in pkgutil.iter_modules(paths):
        if info.name.startswith("_"):
            continue
        name = f"{package}.{info.name}"
        child = _import(name, failures)
        if child is not None:
            yield from _walk(name, child, failures)


def _import(module_name: str, failures: list[LoadFailure]):
    try:
        return importlib.import_module(module_name)
    except Exception as exc:  # noqa: BLE001 -- one bad op must not hide the rest
        failures.append(
            LoadFailure(
                module=module_name,
                error=f"{type(exc).__name__}: {exc}",
                heavy_imports=_module_scope_imports(module_name),
            )
        )
        return None


def _module_scope_imports(module_name: str) -> tuple[str, ...]:
    """Parse a module that would not import, to explain what it pulled in.

    The import is the source of truth; this only produces the diagnostic, so
    a best-effort answer beats a precise one.
    """
    import ast

    try:
        spec_ = importlib.util.find_spec(module_name)
        if spec_ is None or spec_.origin is None:
            return ()
        with open(spec_.origin, encoding="utf-8") as f:
            tree = ast.parse(f.read())
    except Exception:  # noqa: BLE001 -- diagnostics must never raise
        return ()

    found = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            found.extend(f"{alias.name} (line {node.lineno})" for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.append(f"{node.module} (line {node.lineno})")
    return tuple(found)
