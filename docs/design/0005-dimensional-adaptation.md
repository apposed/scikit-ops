# 0005 — Dimensional adaptation

> **Superseded in part by [0006](0006-axis-mapping.md).** The `Extra` policy
> (`reject` / `iterate` / `passthrough`), the requirement that declared axis
> names *match*, and the `plans()` enumeration described below have all been
> replaced: an op now declares arity plus name *hints*, and the mapping belongs
> to the user. 0006 records why. The rest of this document — roles meeting
> axes, iteration running in the worker, role-aware stacking, and the refusal
> to guess what an array's axes are — still holds, and the reasoning is kept
> here as written.

## The problem

`stardist2d` is 2-D only. Its signature said so nowhere:

```python
def stardist2d(image: ImageData) -> LabelsData: ...
```

[Roles](0003-semantic-roles.md) told a front end the array was a picture, and
nothing about its shape. A napari user with a `(z, y, x)` stack selects the
layer, runs the op, and gets a traceback out of StarDist — when what they
wanted was obvious: run it on the slice they are looking at, or run it on every
slice. Neither is hard. Nothing knew enough to offer either.

`to_gray` in `skop/ops/_util.py` was the state of the art: an op guessing, in
its own body, whether a trailing extent of 3 or 4 means RGB.

## The decision

Three concerns, kept apart, because they belong to three different parties.

### Declaration — the op author, in `Annotated`

The same mechanism roles use, so the two compose:

```python
@op(env="stardist-tf")
def stardist2d(
    image: Annotated[ImageData, Axes("y", "x", "c?", extra=Extra.iterate)],
) -> LabelsData: ...
```

`Axes` takes one axis label per argument, where a trailing `?` marks that axis
optional: `Axes("y", "x", "c?")` is "two spatial axes, and I cope with a
channel axis if one is there".

**An axis label is any string**, and that is deliberate. `CANONICAL` — `x y z c
t`, the intersection of OME-NGFF, bioimage.io and ImageJ2/ImgLib2's `AxisType`
(ImageJ2's `AxisType` is an identity, like `name` below, not the coarser `type`
category two paragraphs down — the two are easy to conflate since both go by
"axis type") — is privileged only in that a viewer knows how to *display*
those; it is not the vocabulary. There is no agreed letter for a lifetime bin,
a well, an excitation wavelength or a polarization angle, and napari is n-D
precisely to carry them. `Axes("lifetime", "y", "x")` is as valid as
`Axes("z", "y", "x")`.

Almost nothing pays for that openness, because an axis an op does not consume
is one it never has to understand: the planner needs axis *identity and order*,
never meaning, so `Extra.iterate` maps over a lifetime axis exactly as it maps
over `z`. Only an op that genuinely consumes an exotic axis — a decay fit
wanting `("lifetime", "y", "x")` — needs to name one, and now it can.

We took names only, not OME-NGFF's `name` *plus* a closed `type` enum. A
non-extensible enum re-imposes the very ceiling this removes, and physical
units and scale factors want to arrive alongside axis types anyway. `Axes` can
grow a per-axis type later without disturbing the label syntax.

### Synonyms

A front end may report scikit-image's `(pln, row, col)`, ImageJ's slices and
frames, or Bio-Formats' upper-case `XYZCT`. Without resolution, an op declaring
`y` refuses an array labelled `row` with a confident, useless "no y axis" — so
the open vocabulary makes synonyms load-bearing rather than polish.

`ALIASES` resolves them, case-folded, in skop rather than in each front end.
That does not breach [0003](0003-semantic-roles.md)'s no-guessing rule, because
it is not a guess: `row` *is* `y`. Anything unrecognized is left exactly as it
is — an open vocabulary means `lifetime` has to survive the lookup unchanged.

The line held is **only true synonyms, never near-synonyms**:

- bioio's `s` (RGB samples) is *not* folded into `c`. bioio distinguishes the
  two deliberately, and an image with both would then collide.
- bioimage.io's `b` (batch), CZI's `m` (mosaic tile) and `v` (view), and
  SCIFIO's `lifetime`, `spectra`, `polarization` and `phase` have no canonical
  equivalent, so they stay opaque — and adapt perfectly well anyway, since an
  axis an op does not consume needs no meaning. Batch iterating by default is
  precisely right.
- `slice` *is* included: within the hyperstack model this vocabulary shares,
  ImageJ's slice means z. The genuinely ambiguous case is a plain stack, which
  has no axis labels to resolve in the first place.

Which spelling is canonical is a coin flip we resolve toward the shortest form,
since `CANONICAL` is what error messages and `output_axes` display.

### One string is one label

`Axes` is a frozen dataclass with `init=False` and a hand-written `__init__`,
since a generated one cannot take `*args`. It stores the *canonicalized* labels
rather than the raw text, which is what makes `Axes("z", "y", "x")` and
`Axes("pln", "row", "col")` compare equal.

A lone non-string argument is the whole sequence, so `Axes(list("zyx"))` works.
The `list` cast idiom does not cover an optional axis, though: `list("yxc?")`
is `["y", "x", "c", "?"]`, which splits the optional marker off the axis it
belongs to, so a lone `"?"` label raises rather than being silently attached to
the wrong axis.

### The extra-axis policy

`extra` is the policy for axes the op does not consume:

| Policy | Meaning |
| --- | --- |
| `Extra.reject` (default) | the op has nothing to say about them |
| `Extra.iterate` | slices along them are independent; map and stack |
| `Extra.passthrough` | the op handles the extra axes itself |

`Extra.iterate` is a scientific claim, not a convenience — it asserts the
slices can be treated separately. Only an op author can make that claim, which
is why it lives in the annotation and not in the runner, and why the default
does not make it for them.

Note what is deliberately *not* gated on it: indexing a stack down to one
plane is always available, whatever `extra` says, because an op declaring
`("y", "x")` said it accepts a plane and a plane is what it gets. Iterating is
the author's claim to make; selecting is the user's decision.

`ParamSpec.axes` carries it to front ends, alongside `role`.

### Inference — the front end, never skop

Which axes an array *has* is a guess.
[front-ends.md](../spec/front-ends.md) already holds this line for roles, and
it holds here for the same reason: a front end guesses because a front end has
a viewer to guess *for*. skop asks to be told, through
`axes={"image": ("z", "y", "x")}`, and refuses to invent one.

skop-napari has since built that resolver
([its 0006](https://github.com/apposed/skop-napari/blob/main/docs/design/0006-axis-awareness.md)),
which settled two things this file had guessed at.

**Writing the answer back is the load-bearing half.** Every resolution goes
onto the layer, and every result layer is stamped with `axis_labels` and the
plan's `output_axes`. Guessing then happens once per layer rather than once per
run, a correction stays corrected, and a chain of ops gets more certain as it
goes instead of re-deriving the same guess.

**Confirming "on the consequence, not the confidence" needed no prompt.**
Because `plans()` returns lossless candidates first, a combo box in that order
*is* the rule: the default never discards data, and the plan that does is one
visible click away rather than behind a modal. The confirmation dialog this
file expected turned out to be the ordering it already had.

And one hazard worth recording here, because it constrains what a *guess* may
contain rather than what skop does with it: convention must never invent an
axis name an op might consume. Every image op declares
`Axes("y", "x", "c?")`, so a guessed `c` is not iterated over — it is taken as the channel axis, and
`to_gray` averages across it. A guessed `t` has no such failure mode, since no
op declares `t`. An axis a resolver cannot name at all is likewise safe as
`"dim0"`: it satisfies no declaration, so it can only be iterated or rejected.

### Adaptation — skop, as a value rather than as control flow

`skop.plans(fn, param, array, axes, position=)` returns every workable
`AdaptationPlan`, lossless ones first: which axes to index down to a single
position, which to iterate, the transpose that puts the rest in declared order,
the derived output axes, and a call count.

Making the plan a value is what buys the user experience. A front end can
*offer* the choice — "run on the current Z slice" against "run on all 41
slices" — instead of making it, and can show what will happen before it
happens. `Runner.run(axes=...)` picks one automatically via `_adapt.choose`;
`Runner.run(plans=...)` takes one the caller already chose.

Automatic selection is timid on purpose: `choose` takes a plan only when it is
lossless. A plan that discards data raises instead, listing the candidates. So
the friendly path is fully automatic exactly when it is safe to be.

## Iteration runs in the worker

The plan crosses the Appose boundary with the call, as `plans` in the task
args, and `skop.worker.invoke` runs the loop. Doing it host-side would have
meant 41 tasks, 41 trips through shared memory, and 41 calls to
`StarDist2D.from_pretrained`. This way it is one task and one encode of the
whole stack, per-slice progress comes free through the existing `progress()`
channel, and cancellation is checked between slices.

## Stacking is role-aware

Iterating `stardist2d` over `z` and stacking as-is gives object `1` on every
plane, each a different cell. `_adapt._renumber` offsets label IDs per slice —
and it only knows to, because `OutputSpec.role` says `labels`. This is the
first place roles and axes have to meet, and it is why they are one feature
rather than two. `skop.ops.toy.quadrants` is the standing test case: four
quadrants per plane, twelve objects in a three-plane stack.

## Nothing changes for anyone who does not care

The same property that made roles work. An op with no `Axes` is planned for
never. An op *with* `Axes` that is called without naming its axes is passed
its array untouched, exactly as before. Naming axes is what opts in, and every
existing caller and test kept passing unchanged.

## Limits, accepted knowingly

**Axis requirements that depend on another parameter.** `stardist2d`'s `model`
selects between a fluorescence model wanting one channel and an H&E model
wanting three. The pattern declares the union — `c` optional — and the op body
keeps sorting it out. A conditional-shape language would be the tail wagging
the dog.

**One iterated parameter per call**, and **function form only**. A computer-form
op's buffer has the shape of the whole input, not of one slice, and reconciling
those needs [form adaptation](../spec/form-adaptation.md) to exist first.

**Outputs that do not stack.** Reassembly handles arrays whose shape is the
same on every slice, and numbers. A `points` output has a different N per
slice and no meaningful stacked form; that raises rather than inventing one.

## Where numpydantic fits, and where it does not

It was the obvious candidate for the annotation vocabulary, and it cannot be.
`_spec._resolve_hints` evaluates annotations *inside the environment the op
runs in*, so an annotation spelled in numpydantic would put pydantic-core into
`stardist-tf` (py3.10, TensorFlow), `unseg-cv` (py3.9, opencv pinned to
4.7.0.72) and every environment added later — a per-environment solver risk
paid to serve a host-side concern, and against `_spec.py`'s
standard-library-only rule. `Axes` is a frozen dataclass and a string.

It earns its place one layer up, on the host, where it is optional and not yet
built: validating an actual array against a declared constraint, including the
lazy cases its interface layer understands (dask, zarr, HDF5, xarray) so that a
40 GB layer need not be materialized to be checked; and emitting JSON Schema
for `OpSpec`, which [front-ends.md](../spec/front-ends.md) names as the
unresolved problem for a Fiji front end. The austere marker travels; the heavy
interpretation stays home.

## Still open

- Whether `Extra.passthrough` earns its keep, or whether an op that handles its
  own extra axes should simply declare them (`Axes("z", "y", "x")`) and be done.
  `otsu` is the only user.
- **Which spelling is canonical.** `ALIASES` normalizes toward the
  one-character forms because they are the shortest, not because they are the
  most descriptive; `output_axes` and every error message inherit that choice.
  Worth revisiting if front ends end up translating back.
- Whether output axes need declaring. Today `AdaptationPlan.output_axes` is
  derived — iterated axes, then the input's core axes — which is right for
  every current op and wrong for a projection.
- Whether a chunked array should iterate along its own chunk boundaries rather
  than one plane at a time.
