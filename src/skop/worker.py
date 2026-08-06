"""The worker-side half of skop.

This module is loaded inside every op environment, via an Appose service init
script. It must therefore stay austere: standard library, numpy, and appose
only -- never magicgui, napari, or anything an op happens to depend on.

Its whole job is to turn an Appose task into an ordinary Python function call
and turn the return value back into Appose outputs.
"""

from __future__ import annotations

import importlib
from enum import Enum
from pathlib import Path
from typing import Any, get_args, get_origin

from . import _adapt, _codec, _progress, _spec


def invoke(
    task: Any,
    module: str,
    function: str,
    kwargs: dict,
    plans: list | None = None,
) -> dict:
    """Run an op in this worker process, on behalf of a host.

    Args:
        task: The Appose task, used for progress and cancellation.
        module: Module to import the op from, e.g. ``"skop.ops.segment.starfun3d"``.
        function: Name of the op function within that module.
        kwargs: Op arguments, as decoded by Appose.
        plans: Serialized ``AdaptationPlan``s, one per parameter needing one.
            Iteration runs here rather than on the host so that forty planes
            cost one task and one trip through shared memory, not forty.

    Returns:
        A dict of task outputs. Computer- and inplace-form ops return an
        empty dict, since their results land in buffers the host already
        owns.
    """
    fn = getattr(importlib.import_module(module), function)
    spec = _spec.spec(fn)
    adaptations = {
        plan["param"]: _adapt.AdaptationPlan.from_dict(plan) for plan in plans or []
    }

    refs: list = []
    outbound: list = []
    try:
        args = _coerce(spec, _codec.decode(kwargs, refs))
        _check_args(spec, args)

        token = _progress._bind(task)
        try:
            result = _adapt.execute(spec, fn, args, adaptations)
        finally:
            _progress._unbind(token)

        return _pack(spec, result, outbound)
    finally:
        # Release our mappings of the host's shared memory. Per Appose's
        # rule, only the host process unlinks; we merely close our view.
        _codec.release(refs, unlink=False)


def _coerce(spec: _spec.OpSpec, args: dict) -> dict:
    """Rebuild the rich argument types that JSON flattened in transit.

    An Enum travels as its value and a Path as a string, so an op called
    remotely would otherwise receive something different from what an op
    called directly receives. The declared parameter types say how to put
    them back.
    """
    for param in spec.params:
        if param.name not in args:
            continue
        value = args[param.name]
        if value is None:
            # An optional parameter left empty. Nothing to rebuild, and
            # Path(None) would raise.
            continue
        declared = _unwrap_optional(param.type)
        if isinstance(declared, type) and issubclass(declared, Enum):
            if not isinstance(value, declared):
                args[param.name] = declared(value)
        elif declared is Path and not isinstance(value, Path):
            args[param.name] = Path(value)
    return args


def _unwrap_optional(annotation: Any) -> Any:
    """``X`` from ``X | None``, and anything else unchanged.

    A parameter that may be left empty is spelled ``X | None``, which is a
    union rather than ``X`` -- so without this, an optional Path or Enum
    would travel as a string and arrive as one, while the same op called
    directly received the real type.
    """
    if get_origin(annotation) is None:
        return annotation
    args = [a for a in get_args(annotation) if a is not type(None)]
    return args[0] if len(args) == 1 else annotation


def _check_args(spec: _spec.OpSpec, args: dict) -> None:
    if spec.form is _spec.FUNCTION:
        return
    missing = [
        p.name for p in spec.params if p.direction is not None and p.name not in args
    ]
    if missing:
        raise TypeError(
            f"Op {spec.name} is {spec.form} form, so its buffers must be "
            f"supplied by the caller. Missing: {', '.join(missing)}. "
            "Automatic allocation is not implemented yet."
        )


def _pack(spec: _spec.OpSpec, result: Any, refs: list) -> dict:
    names = spec.outputs
    if spec.form is not _spec.FUNCTION or not names:
        # The host owns the output buffers; nothing to send back.
        return {}
    if names == ("result",):
        return {"result": _codec.encode(result, refs)}
    if result is None:
        raise TypeError(f"Op {spec.name} declares outputs {names} but returned None")
    if len(result) != len(names):
        raise TypeError(
            f"Op {spec.name} declares {len(names)} outputs {names} "
            f"but returned {len(result)} values"
        )
    return {name: _codec.encode(value, refs) for name, value in zip(names, result)}
