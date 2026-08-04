# 0012 — Rendering skop results in a notebook

**Status:** placeholder, written 2026-08-03 alongside the first notebook that
plots detector output
([`notebooks/detection/detect-then-mask.ipynb`](../../notebooks/detection/detect-then-mask.ipynb)).
It records the question, and the reason it is not yet answered.

Numbered 0012 rather than filling the gap at 0009: that number was
`per-object-features`, which moved to [`docs/spec/`](../spec/per-object-features.md)
unnumbered, and reusing it would make the older commits read wrong.

## The question

A notebook is the second caller of skop, after skop-napari, and the first one
with no viewer. It needs the same two things a panel needs — boxes drawn on an
image, and an `(N, Y, X)` mask stack drawn on one — using matplotlib rather
than layers.

Half of that is already settled. `skop.masks` holds the projections from a
stack onto something displayable (`to_labels_2d`, `to_labels_3d`,
`order_by_area`), and they are utilities rather than ops precisely because
every front end needs them. What is missing is the last step: turning a
projection into an artist.

## Where it lives today

In tnia-python, in `tnia.plotting.plt_helper`:

| function | does |
| --- | --- |
| `draw_boxes(ax, boxes, order="yxyx", ...)` | rectangles onto an axes |
| `mask_outline_overlay(img, labels, color, thickness)` | instance boundaries into an RGB array |
| `mask_overlay(img, labels)` | filled masks, greyscaling the image to do it |

The notebook calls them directly and does the skop-to-array glue inline. That
glue is currently two lines — `order_by_area` then `to_labels_2d` — which is
the whole reason this is a placeholder rather than a decision.

## The tension

[`docs/spec/front-ends.md`](../spec/front-ends.md) gives the rule: *if a change
to skop would only make sense to a napari author, it belongs in the front end.*
A matplotlib renderer is a front end by that test, and the clean answer is a
`skop-mpl` package mirroring `skop_napari/_roles.py` — twenty lines of
`Role` → artist lookup, same shape as the napari one.

The argument against is that these functions are useful well beyond skop:
Cellpose, StarDist, YOLO and `regionprops` all produce boxes and label images,
and none of them should need skop installed to get them drawn. Moved into
`skop-mpl`, they would only be reachable by skop users.

Note that keeping them in tnia-python costs nothing architecturally today,
because none of them mentions skop. `draw_boxes` takes an `(N, 4)` array and an
`order` string; that skop's canonical order is one of the two it accepts is a
coincidence of both following `regionprops`, not a coupling. The dependency
points one way — `scikit-ops[examples]` requires `tnia-python[plotting]` — and
nothing here reverses it.

## What would settle it

The size of the inline glue, measured across more than one notebook. Two lines
means tnia-python was the right home and this document can be closed. If it
grows, or if the same twenty lines appear in three notebooks, that is the
evidence for `skop-mpl`.

If it does move, the array-level functions should stay in tnia-python and
`skop-mpl` should wrap them — mapping `Role` and `OutputSpec` to a call, which
is the part that is genuinely skop-shaped — rather than absorb them.

## Not in scope

Interactivity. A panel restyles a layer; a notebook redraws a figure. Anything
that wants a slider on `strategy` or on the mask view is asking for a widget,
and that is a different document.
