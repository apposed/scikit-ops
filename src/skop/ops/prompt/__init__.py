"""Prompted segmentation: the object under the user's clicks.

For interactive tools. A click costs milliseconds but the image's embedding
costs seconds, so a prompt op comes with an embed op, and the module keeps
the embeddings the embed op made in the worker, which lives as long as the
runner. ``microsam_embed`` once per image, then ``microsam_points`` per click.

Not ``skop.ops.mask``: those take a detector's boxes and return overlapping
masks for a workflow. These take a person's points and return the one object
they meant.
"""

from __future__ import annotations

from .microsam import microsam_embed, microsam_points

__all__ = ["microsam_embed", "microsam_points"]
