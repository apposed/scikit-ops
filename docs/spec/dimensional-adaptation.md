# Spec — dimensional adaptation

**Status:** not implemented. An op receives whatever array the caller hands it,
and fails inside its own body if the shape is wrong.

## What exists

An image parameter is annotated with a [role](../design/0003-semantic-roles.md)
and nothing else:

```python
def stardist2d(image: ImageData) -> LabelsData: ...
```

That says the array is a picture. It does not say the op is 2-D only, so a
front end handing it a `(z, y, x)` stack gets a traceback out of StarDist
rather than a useful refusal — or worse, silently wrong output. `to_gray` in
`skop/ops/_util.py` is the current state of the art: an op guessing, in its own
body, whether a trailing extent of 3 or 4 means RGB.

## What is proposed

Three concerns, deliberately kept apart, because they belong to three different
parties.

### 1. Declaration — the op author, in `Annotated`

The same mechanism roles use:

```python
@op(env="stardist-tf")
def stardist2d(
    image: Annotated[ImageData, Axes("yxc?", extra=Extra.iterate)],
) -> LabelsData: ...
```

`Axes` takes a pattern over the vocabulary `x y z c t` — the intersection of
OME-NGFF, bioimage.io and ImgLib2's `AxisType`, so that the Fiji mapping stays
a lookup table rather than a translation. A trailing `?` marks an axis
optional: `"yxc?"` is "two spatial axes, and I will cope with a channel axis if
one is there".

`extra` is the policy for axes the op does not consume:

| Policy | Meaning |
| --- | --- |
| `Extra.reject` (default) | the op has nothing to say about them |
| `Extra.iterate` | slices along them are independent; map and stack |
| `Extra.passthrough` | the op handles leading axes itself |

`Extra.iterate` is a scientific claim, not a convenience. Only the op author
can make it, which is why it lives in the annotation and not in the runner.
The default is `reject` for the same reason roles are never guessed.

Note what is *not* gated on this: taking one `(y, x)` plane out of a stack is
always allowed, whatever `extra` says, because the op declared it accepts a
plane and a plane is what it gets. Iterating is the op author's claim;
selecting is the user's decision.

### 2. Inference — the front end, never skop

Which axes a given array *has* is a guess, and
[front-ends.md](front-ends.md) already holds the line: skop does not guess, a
front end does, because a front end has a viewer to guess for. skop-napari
resolves in order, and reports which rung it landed on:

1. a user override on `layer.metadata["skop_axes"]`
2. `xarray.DataArray.dims`, or NGFF `multiscales` axes
3. napari layer state: `layer.rgb` is decisive for a trailing `c`;
   `axis_labels` when they are not still the default `"0"`, `"1"`, …
4. `viewer.dims.order` and `ndisplay` — which axes form the displayed plane,
   positionally, without saying whether the slider axis is `z` or `t`
5. shape heuristics
6. fail, with an axis editor

Two rules make that friendly rather than nagging. The plugin **writes its
resolution back** onto the layer and stamps axes onto the layers it creates, so
inference improves as a session goes on rather than repeating. And it
**confirms based on the consequence, not the confidence**: a weak guess is fine
when the op will iterate over the unknown axis anyway (`z` versus `t` changes
nothing), and needs confirming when the plan drops data.

### 3. Adaptation — skop, as a value rather than as control flow

Declared pattern plus actual axes yields an `AdaptationPlan`: which axes to
index down to a single position, which to iterate, the transpose that puts the
rest in declared order, the resulting output axes, and a call count.

Making the plan a first-class value is what buys the user experience. When
several plans are valid, a front end can *offer* them — "run on the current Z
slice" against "run on all 41 slices, about 4 minutes" — instead of picking
silently, and can show the chosen one before anything runs. `skop.plans()`
enumerates the candidates; `Runner.run(plans=...)` executes a chosen one.

Automatic selection stays deliberately timid: skop picks a plan on its own only
when the plan is lossless — an exact match, or an iteration the op author
opted into. A plan that discards data is never chosen without being asked for.

## Iteration runs in the worker

The one piece that touches the wire protocol, so it is settled here. Iterating
40 Z slices host-side would mean 40 Appose round trips, 40 trips through shared
memory, and 40 calls to `StarDist2D.from_pretrained`. The plan crosses the
boundary with the call instead, and `skop.worker.invoke` runs the loop: one
task, one encode of the whole stack, and per-slice progress for free through
the existing `progress()` channel.

## Stacking is role-aware

Iterating `stardist2d` over `z` and stacking the results gives object `1` on
every slice, each a different cell. Correct stacking has to offset label IDs
per slice — which the adapter can only know to do because `OutputSpec.role`
says `labels`. This is the first place roles and axes have to meet, and it is
why they are one feature and not two.

## Known limits, accepted

**Axis requirements that depend on another parameter.** `stardist2d`'s `model`
selects between a fluorescence model wanting one channel and an H&E model
wanting three. The pattern declares the union — `c` optional — and the op body
keeps sorting it out, as it does today. A conditional-shape language would be
the tail wagging the dog.

**Outputs that do not stack.** Iteration reassembles array outputs whose shape
matches the consumed core, and scalars, into one array. A `points` output has a
different N per slice and no meaningful stacked form; that combination raises
rather than inventing one.

## Where numpydantic fits, and where it does not

It cannot be the op-facing vocabulary. `_spec._resolve_hints` evaluates
annotations *inside the environment the op runs in*, so an annotation spelled
in numpydantic would put pydantic-core into `stardist-tf` (py3.10, TensorFlow),
`unseg-cv` (py3.9, opencv pinned to 4.7.0.72), and every environment added
later — a per-environment solver risk paid to serve a host-side concern, and
against `_spec.py`'s standard-library-only rule.

It earns its place one layer up, on the host, where it is optional: validating
an actual array against the declared constraint, including the lazy cases its
interface layer understands (dask, zarr, HDF5, xarray) so that a 40 GB layer
need not be materialized to be checked; and emitting JSON Schema for `OpSpec`,
which [front-ends.md](front-ends.md) already names as the unresolved problem
for a Fiji front end. The austere marker travels; the heavy interpretation
stays home.

## Open questions

- Whether `Extra.passthrough` earns its keep, or whether an op that handles its
  own leading axes should simply declare them (`"zyx"`) and be done.
- Whether output axes need declaring too. Today they are derived: core in,
  core out, iterated axes prepended. `points` outputs already do not fit that,
  and neither would a projection.
- Whether a chunked array should iterate along its own chunk boundaries rather
  than one plane at a time.

Graduate this file into `design/` when it lands.
