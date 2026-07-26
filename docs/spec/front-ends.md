# Spec — what a front end needs from skop

**Status:** one front end exists ([skop-napari](https://github.com/apposed/skop-napari));
a Fiji one is the next target. This file records what the boundary between skop
and a front end is, so that the second one does not require changing the first.

## The contract, as it stands

A front end may read `OpSpec` and nothing else. Specifically:

| From `OpSpec` | For |
| --- | --- |
| `name`, `module`, `function` | identifying and grouping ops |
| `doc` | summary line and per-parameter tooltips (Google-style `Args:`) |
| `env` | telling the user what will be built |
| `params` → `ParamSpec` | one input widget each |
| `ParamSpec.type` | which widget |
| `ParamSpec.ui` | that widget's type, range, step |
| `ParamSpec.role` | whether it is a layer/image selector instead |
| `ParamSpec.axes` | which shapes it accepts, and what may be done to others |
| `ParamSpec.direction` | `Out` params are never shown |
| `output_specs` → `OutputSpec` | where each result goes |
| `OutputSpec.role` | which display type |

Plus `Runner.run(..., on_progress=, on_start=)`, the three `subscribe_build_*`
methods ([design 0004](../design/0004-build-feedback.md)), and `skop.plans`
([design 0005](../design/0005-dimensional-adaptation.md)) — which is the
sharpest case of the rule below, since a front end must work out an array's
axes itself before skop will fit it to anything.

Nothing in that list is napari-shaped. That is the test: if a change to skop
would only make sense to a napari author, it belongs in the front end.

## What skop-napari proved

Building the first front end changed skop exactly once, and productively: it
forced [semantic roles](../design/0003-semantic-roles.md) into existence.
Everything else the panel needed was already readable off `OpSpec`. That is
weak evidence the boundary is in the right place, and it is only weak because
one front end cannot distinguish "general" from "coincidentally napari-shaped".

The one napari-specific decision — mapping `Role` to a napari type — lives in
`skop_napari/_roles.py` and is about twenty lines of lookup table.

## What a Fiji front end will probably want

Unresolved, and the reason this is a spec rather than a design doc:

- **A stable op ID across languages.** `skop.ops.threshold:otsu` is currently a
  Python import path. A Java front end enumerating ops over Appose needs it to
  be an opaque string it can round-trip, which it already is — but nothing
  currently *says* that, and someone will eventually be tempted to derive one
  from the other.
- **`OpSpec` over the wire.** Front ends today are in-process with skop and
  read `OpSpec` as a Python object. Fiji is not, so `OpSpec` needs a serialized
  form. `outputs` is a `tuple[str, ...]` today precisely because it crosses the
  Appose boundary; the rest of `OpSpec` has never had to.
- **Role vocabulary in Java.** `Role` mirrors `napari.types` by name, which was
  free for napari and will not be for ImageJ. Expect the mapping to be less
  tidy; that is fine, it is the same kind of lookup table.
- **Type mapping.** `np.ndarray` ↔ ImgLib2 is a real conversion, not an
  annotation, and belongs in the Fiji layer or in Appose — not here.

## The rule to hold

Roles are never guessed in skop ([0003](../design/0003-semantic-roles.md)).
Every front end will want to guess, and every front end should — differently,
in its own module, where its own display model justifies it. The first time a
guess migrates into skop because "both front ends do it anyway", this boundary
is gone.
