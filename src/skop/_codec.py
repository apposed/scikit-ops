"""Conversion between plain Python values and Appose wire values.

Runs on both sides of the process boundary, so it stays austere: standard
library, numpy, and appose only.

Conversion is done explicitly here rather than by registering a global
``appose.util.message`` codec for ``numpy.ndarray``. A global registration
would reinterpret every bare array passed by any Appose user in the same
process, including code that has nothing to do with skop.
"""

from __future__ import annotations

import contextlib
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
from appose import NDArray, SharedMemory

# dtypes whose element size Appose can derive from the dtype name.
_UNSUPPORTED_HINT = (
    "Supported dtypes are the sized numeric types (int8/16/32/64, "
    "uint8/16/32/64, float16/32/64). Cast before returning; note that "
    "'bool' has no size in its name and so cannot cross the boundary."
)


def encode(value: Any, refs: list) -> Any:
    """Convert a Python value into something Appose can put on the wire.

    Any shared memory allocated along the way is appended to ``refs``, which
    the caller must keep alive until the message has been sent.
    """
    if isinstance(value, np.ndarray):
        return _to_ndarray(value, refs)
    if isinstance(value, NDArray):
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {k: encode(v, refs) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [encode(v, refs) for v in value]
    if isinstance(value, np.generic):
        # numpy scalar -- json cannot encode it, but its Python peer is fine.
        return value.item()
    return value


def decode(value: Any, refs: list) -> Any:
    """Convert a value received from Appose back into plain Python.

    Arrays are returned as views onto shared memory, with no copy. The
    backing ``NDArray`` objects are appended to ``refs``; the caller must
    keep them alive for as long as the views are in use.
    """
    if isinstance(value, NDArray):
        refs.append(value)
        return value.ndarray()
    if isinstance(value, dict):
        return {k: decode(v, refs) for k, v in value.items()}
    if isinstance(value, list):
        return [decode(v, refs) for v in value]
    return value


def _to_ndarray(array: np.ndarray, refs: list) -> NDArray:
    dtype = str(array.dtype)
    _check_dtype(dtype)
    shape = list(array.shape)
    if array.size == 0:
        # A shared memory block must have a positive size, but an empty result
        # is ordinary -- a segmentation that finds nothing returns one. Carry
        # the shape and dtype over a token block; numpy is happy to view zero
        # elements of it.
        nda = NDArray(dtype, shape, SharedMemory(create=True, rsize=1))
    else:
        nda = NDArray(dtype, shape)
    refs.append(nda)
    # The one unavoidable copy: an arbitrary array is not already in shared
    # memory. Callers who allocate via skop can skip it.
    nda.ndarray()[...] = array
    return nda


def _check_dtype(dtype: str) -> None:
    if not any(ch.isdigit() for ch in dtype):
        raise TypeError(
            f"Cannot transfer arrays of dtype '{dtype}'. {_UNSUPPORTED_HINT}"
        )


def copy_out(value: Any) -> Any:
    """Deep-copy shared-memory views into ordinary process-local arrays.

    Used on results before their shared memory is released.
    """
    if isinstance(value, np.ndarray):
        return np.array(value, copy=True)
    if isinstance(value, dict):
        return {k: copy_out(v) for k, v in value.items()}
    if isinstance(value, list):
        return [copy_out(v) for v in value]
    return value


def release(refs: list, unlink: bool) -> None:
    """Dispose of shared memory blocks held by ``refs``.

    Appose's rule is that the service process cleans up shared memory
    regardless of which process allocated it, so the host passes
    ``unlink=True`` and the worker never unlinks.
    """
    for ref in refs:
        shm = getattr(ref, "shm", None)
        if shm is None:
            continue
        # Cleanup is best effort: a block the peer already released must not
        # take down an otherwise successful call.
        with contextlib.suppress(Exception):
            shm.unlink_on_dispose(unlink)
            shm.dispose()
