"""Workflows: ops built out of other ops.

A workflow is an op with no environment. ``@op(env=...)`` is what pins an op to
a worker, and a workflow has nothing to pin -- the ops it calls each bring
their own -- so it runs on the host, where the runner lives, and dispatches
from there. See docs/spec/workflow-ops.md.

**They live here rather than beside the ops they compose.** That is a
navigation decision before it is a technical one: somebody arriving at a
collection of sixty ops should be able to see what has already been assembled,
and where to put what they assemble next, without reading sixty signatures. The
subdirectories mirror ``skop.ops`` itself, so a workflow sits under the name of
what it *produces* -- ``workflows/mask/`` makes masks, whatever it ran to get
there.

It falls out that the dependency runs one way. Everything here imports from
``skop.ops``; nothing in ``skop.ops`` imports from here. Left mixed in, a
``mask/__init__.py`` re-exporting a workflow would import ``skop.ops.detect``
and then reach back into its own half-initialized package -- which works until
somebody writes the import the other way round and it does not. A separate
tree makes that layering physical, and a cycle an obvious mistake rather than a
subtle one.

Nothing else distinguishes them. Discovery finds a workflow the same way it
finds any op, ``spec()`` describes it the same way, and a front end runs it
through the same call. What tells them apart is ``OpSpec.is_workflow``, which
reads the missing environment -- never the module path.
"""

from __future__ import annotations

from .deconvolve.with_psf import Deconvolved, deconvolve_with_psf
from .mask.detect_then_mask import Detected, detect_then_mask
from .segment.connect_2d_in_3d import Connected, connect_2d_in_3d

__all__ = [
    "Connected",
    "Deconvolved",
    "Detected",
    "connect_2d_in_3d",
    "deconvolve_with_psf",
    "detect_then_mask",
]
