"""What a front end in another language asks skop for.

An in-process front end reads ``OpSpec`` as a Python object and calls
``skop.plan`` directly. A front end in another language cannot: it reaches
skop over Appose, so everything it needs has to be expressible as JSON. This
module is that surface, and nothing else belongs in it.

Three things live here:

* the **strings** an out-of-process host needs in order to drive a worker at
  all -- the service init script and the task script it posts. They are data
  rather than something each front end transcribes, so that changing how a
  worker is started changes one place instead of one place per language;
* ``describe``, which is ``discover`` with its result flattened to JSON;
* ``plan``, which is ``skop.plan`` reached by op name rather than by function
  object, so that the axis arithmetic is computed once here rather than
  reimplemented in every host language.

Like the rest of skop's core this module is austere -- standard library only
-- because the service that answers these questions runs in the ``minimal``
environment, and because ``discover`` importing an op module is the whole
point: an op that will not load in a minimal environment must be reported,
not crash the description.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

from . import _adapt, _spec

# The script sent for every op call. Its last statement is an expression
# yielding a dict, which Appose turns into the task's outputs.
CALL = "skop_invoke(task, module, function, kwargs, plans)"

# Installed into every op worker via the service init script. Names defined
# here become worker exports, and so are in scope for every task.
#
# NB: The search path is extended here rather than via PYTHONPATH. Appose
# gained per-service env vars in 0.12; setting them on the builder instead
# would fold a machine-specific path into the environment's identity, causing
# a rebuild whenever the checkout moves.
#
# The insertion is at position 0, so a checkout shadows the copy of skop that
# the environment installs. That is deliberate: a developer editing an op sees
# the edit without bumping the pin every environment's identity depends on.
# NB: MPLBACKEND is dropped because a worker inherits its caller's environment,
# and a Jupyter kernel sets MPLBACKEND=module://matplotlib_inline.backend_inline
# to get inline plots. matplotlib reads it at import and raises if it names a
# backend it cannot load -- and matplotlib_inline is a Jupyter package no op
# environment carries.
#
# So an op that imports pyplot (ultralytics does, deep in its chain) died from a
# notebook and worked from a script, surviving only where something had pulled
# matplotlib_inline in transitively. Popped before any op module is imported;
# nothing in a worker draws, so none needs a backend chosen by its caller.
INIT = """
import os
import sys
os.environ.pop("MPLBACKEND", None)
sys.path.insert(0, {root!r})
import numpy  # NB: must precede the worker's I/O loop on Windows.
import skop.worker
skop_invoke = skop.worker.invoke
"""

# Installed into the one service that answers metadata questions. It runs in
# the minimal environment and never touches numpy or an op's dependencies.
METADATA_INIT = """
import sys
sys.path.insert(0, {root!r})
import skop.host
skop_describe = skop.host.describe
skop_plan = skop.host.plan
skop_constants = skop.host.constants
"""

# The same two scripts, appending rather than inserting.
#
# `root` is only worth putting first when it is a checkout, which is the case
# the shadowing above exists for. When skop is installed normally, `root` is
# the host's site-packages -- and putting *that* first hands the worker every
# package the host has, built for the host's Python. A worker on a different
# Python then fails at `import numpy`, before skop_invoke is ever defined, and
# reports only `NameError: name 'skop_invoke' is not defined`.
#
# Appending keeps the one case that still needs the path: envs/unseg-cv has no
# pinned scikit-ops at all, because scikit-ops requires Python 3.10 and UNSEG's
# pins 3.9, so it reaches skop through this path alone. Last on sys.path is
# enough for that, and lets every environment that does install skop use its
# own copy.
INIT_APPEND = """
import os
import sys
os.environ.pop("MPLBACKEND", None)
sys.path.append({root!r})
import numpy  # NB: must precede the worker's I/O loop on Windows.
import skop.worker
skop_invoke = skop.worker.invoke
"""

METADATA_INIT_APPEND = """
import sys
sys.path.append({root!r})
import skop.host
skop_describe = skop.host.describe
skop_plan = skop.host.plan
skop_constants = skop.host.constants
"""

DESCRIBE = "skop_describe(package)"
PLAN = "skop_plan(op, param, shape, axes, position, mapping, dispositions)"


def is_checkout(root: "Path | str") -> bool:
    """Whether *root* is a source tree rather than an install directory.

    A checkout's ``root`` holds only skop, so putting it first on a worker's
    ``sys.path`` shadows nothing else. An install's ``root`` is the host's
    site-packages, which holds everything -- see the note on ``INIT_APPEND``.

    Compared against this interpreter's own package directories rather than
    matched by name, because "site-packages" is not the only spelling: a
    virtualenv, a conda env and a Debian system install each place them
    differently.
    """
    import sysconfig

    root = Path(root).resolve()
    installed = {
        Path(path).resolve()
        for key in ("purelib", "platlib")
        if (path := sysconfig.get_paths().get(key))
    }
    return root not in installed


def init_script(root: "Path | str", metadata: bool = False) -> str:
    """The init script for a worker, given where this skop lives.

    Chooses between inserting and appending ``root``; see ``INIT_APPEND``.
    """
    if is_checkout(root):
        template = METADATA_INIT if metadata else INIT
    else:
        template = METADATA_INIT_APPEND if metadata else INIT_APPEND
    return template.format(root=str(root))
CONSTANTS = "skop_constants()"


def constants() -> dict:
    """The scripts a host needs in order to drive workers.

    A host transcribes only enough to start *this* service; everything it
    uses afterwards it reads from here.
    """
    return {
        "init": INIT,
        "call": CALL,
        "metadata_init": METADATA_INIT,
        # Additive: a host that only knows the two above keeps the behaviour
        # it had. One that can tell a checkout from an install should prefer
        # these when skop is installed -- see the note on INIT_APPEND.
        "init_append": INIT_APPEND,
        "metadata_init_append": METADATA_INIT_APPEND,
        "describe": DESCRIBE,
        "plan": PLAN,
        "wire_types": list(_spec.WIRE_TYPES),
        "roles": [role.value for role in _spec.Role],
        "dispositions": list(_adapt.DISPOSITIONS),
        "forms": [_spec.FUNCTION, _spec.COMPUTER, _spec.INPLACE],
    }


def describe(package: str = "skop.ops") -> dict:
    """Every op in a collection, and every module that would not load.

    Failures are part of the answer rather than an exception, for the same
    reason ``discover`` collects them: an environment missing one dependency
    must still yield the other fifty-eight ops.
    """
    from .discovery import discover

    specs, failures = discover(package)
    return {
        "package": package,
        "ops": [spec.to_dict() for spec in specs],
        "failures": [
            {
                "module": failure.module,
                "error": failure.error,
                "heavy_imports": list(failure.heavy_imports),
                "message": str(failure),
            }
            for failure in failures
        ],
    }


def plan(
    op: str,
    param: str,
    shape: list,
    axes: list,
    position: dict | None = None,
    mapping: list | None = None,
    dispositions: dict | None = None,
) -> dict:
    """Fit an array to an op's declared axes, by op name rather than function.

    Args:
        op: The op's ID, ``"<module>:<function>"`` -- the same string
            ``OpSpec.name`` carries, treated as opaque.
        param: Name of the parameter the array is destined for.
        shape: The array's shape.
        axes: What that array's axes actually are, one label per axis. A host
            works this out for itself; skop does not guess. ``None`` in place
            of a label means that axis has no name.

            **Order matters, and it is numpy's**: last axis fastest-moving. A
            host whose own arrays are x-fastest -- ImgLib2's are -- reverses
            its axis list before it gets here. Nothing downstream can detect
            the mistake, because a transposed array is still a valid array.
        position: Current position along each axis, keyed by axis name or
            index, for any axis being indexed down to a single plane.
        mapping: One input-axis index (or None) per declared slot, overriding
            skop's own assignment.
        dispositions: What to do with each leftover axis, keyed by axis index
            as a string or an int: ``"iterate"``, ``"select"`` or ``"pass"``.

    Returns:
        The plan, as ``AdaptationPlan.to_dict`` gives it -- warnings included.
        Raises only when no mapping could work at all.
    """
    fn = op_function(op)
    return _adapt.plan(
        fn,
        param,
        tuple(shape),
        [None if label is None else str(label) for label in axes],
        _int_keyed(position),
        None if mapping is None else list(mapping),
        _int_keyed(dispositions),
    ).to_dict()


def op_function(op: str) -> Any:
    """Resolve an op ID to the function it names.

    The ID is opaque to a host, but not to skop: it is ``module:function``,
    and this is the one place that says so.
    """
    module_name, _, function_name = op.partition(":")
    if not module_name or not function_name:
        raise ValueError(f"Not an op ID: {op!r}. Expected '<module>:<function>'.")
    fn = getattr(importlib.import_module(module_name), function_name, None)
    if fn is None or not _spec.is_op(fn):
        raise ValueError(f"No op named {op!r}")
    return fn


def _int_keyed(mapping: dict | None) -> dict | None:
    """Restore integer keys that JSON turned into strings.

    Both ``position`` and ``dispositions`` are keyed by axis index, and JSON
    object keys are strings, so ``{0: "iterate"}`` arrives as
    ``{"0": "iterate"}``. ``position`` may legitimately be keyed by axis
    *name* as well, so only the numeric-looking keys are converted.
    """
    if mapping is None:
        return None
    return {
        int(key) if isinstance(key, str) and key.lstrip("-").isdigit() else key: value
        for key, value in mapping.items()
    }
