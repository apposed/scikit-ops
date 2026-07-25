"""The worker-side half of opkit.

This module is loaded inside every op environment, via an Appose service init
script. It must therefore stay austere: standard library, numpy, and appose
only -- never magicgui, napari, or anything an op happens to depend on.

Its whole job is to turn an Appose task into an ordinary Python function call
and turn the return value back into Appose outputs.
"""

from __future__ import annotations

import importlib
from typing import Any

from . import _codec, _progress, _spec


def invoke(task: Any, module: str, function: str, kwargs: dict) -> dict:
    """Run an op in this worker process, on behalf of a host.

    Args:
        task: The Appose task, used for progress and cancellation.
        module: Module to import the op from, e.g. ``"ops.starfun3d"``.
        function: Name of the op function within that module.
        kwargs: Op arguments, as decoded by Appose.

    Returns:
        A dict of task outputs. Computer- and inplace-form ops return an
        empty dict, since their results land in buffers the host already
        owns.
    """
    fn = getattr(importlib.import_module(module), function)
    spec = _spec.spec(fn)

    refs: list = []
    outbound: list = []
    try:
        args = _codec.decode(kwargs, refs)
        _check_args(spec, args)

        token = _progress._bind(task)
        try:
            result = fn(**args)
        finally:
            _progress._unbind(token)

        return _pack(spec, result, outbound)
    finally:
        # Release our mappings of the host's shared memory. Per Appose's
        # rule, only the host process unlinks; we merely close our view.
        _codec.release(refs, unlink=False)


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
