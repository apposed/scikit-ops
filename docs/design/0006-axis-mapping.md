# 0006 — Axis mapping

Supersedes the `Extra` policy and the plan-enumeration API of
[0005](0005-dimensional-adaptation.md). The rest of 0005 — roles meeting axes,
iteration running in the worker, role-aware stacking, the refusal to guess what
an array's axes *are* — still stands.

## What 0005 got wrong

0005 gave an op two things to say about axes it does not consume: an `Extra`
policy (`reject` / `iterate` / `passthrough`), and a fixed list of axis names it
required. Building front ends against it turned both into obstacles.

### `Extra.iterate` was never the author's claim to make

0005 argued that iterating is "a scientific claim — that slices along those axes
are independent — so only an op author can make it". That is wrong, and the
reason is worth stating precisely, because it looked right for a long time.

**If `Axes` names everything the op's output depends on, independence is a
consequence, not a claim.** "Extra" means the op's computation does not use that
axis. If that is true, the result for slice *i* cannot depend on slice *j*, and
there is nothing left for the author to vouch for. The only way iterating an
extra axis could be wrong is if the declaration were *incomplete* — the op
secretly depending on more than it named.

Every candidate counterexample turned out to be exactly that:

- **Contrast stretch** normalizing to each call's own min/max. It declares
  `("y", "x")` but depends on whatever array the call receives. That is a
  mis-declaration, not a case for a veto.
- **Otsu**, likewise: a global threshold is a histogram over whatever it is
  given. It consumes no particular axis at all — see `variadic` below.
- **A fit needing a population** for a stable estimate. That axis is genuinely
  consumed; declare it in the slots.
- **Stateful or recursive algorithms** (a Kalman filter, a running background).
  These are not expressible as an `Axes`-annotated function in the first place,
  since iteration calls the op afresh per index with no state threaded between
  calls. That is a gap in the execution model, not something `reject` protected.

And the decisive point: whether per-slice or whole-volume processing is
scientifically *valid* is a property of **the image and the experiment**, not of
the op. Contrast-stretching a calcium-imaging time series per frame destroys the
signal; contrast-stretching a plate of unrelated fields per field is right.
Nothing the op author can write distinguishes those, because the op cannot see
which one it has been handed. Only the person who acquired the data knows. So
the runner must offer both and forbid neither.

`Extra` is therefore deleted outright rather than renamed. What survives of it
is one genuinely different fact — see `variadic`.

### Exact name matching over-restricted

A 2-D Delaunay triangulation does not care whether it is handed `y x`, `z x` or
`z y`; it is all spatial, and there is no reason to stop someone feeding it `t`
either. Under 0005 an op naming `y` and `x` simply refused anything else, with a
confident and useless "no y axis".

Names are now **hints**. They bias which input axis fills which slot, and a
mismatch is reported through `AdaptationPlan.warnings` rather than refused. What
binds is the **arity**: how many axes the op consumes, which is the part the
op's own indexing genuinely depends on.

## The decision

An op declares arity plus name hints; **the mapping belongs to the user**.

```python
Axes("y", "x")        # two axes; prefers to call them y and x
Axes("y", "x", "c?")  # two, plus a channel axis if one is there
Axes("*", "*")        # two axes, no opinion which
Axes(variadic=True)   # any number, whatever they are
```

`ParamSpec.axes` still carries this to front ends alongside `role`, and `Axes`
still stores canonicalized labels, so `Axes("z","y","x")` and
`Axes("pln","row","col")` remain equal.

### `variadic` is what is left of `passthrough`

`passthrough` conflated two questions. The one that dissolved was scientific
validity. The one that remains is mechanical: **can this Python function's array
parameter physically accept extra dimensions?** `stardist2d` indexes its input
as exactly 2-D; `otsu` computes a histogram over whatever it is handed. That is
a property of the implementation, and a runner cannot infer it or override it.

So `variadic=True` says "I cope with any number of further axes myself", and it
is the only axis-related veto an op still holds — enforced as such: asking for
`PASS` on a non-variadic op raises.

`otsu` consequently declares `Axes(variadic=True)` with **no slots at all**,
which is the honest declaration. It works on a line, a plane or a volume, and
0005's `Axes("y","x","c?")` for it was simply wrong.

### The fill rules

Given the declared slots and the caller's labelled axes:

1. **Named slots claim by name**, through `canonical()`, so `row` still
   satisfies `y`.
2. **Required slots still unfilled** — wildcards included — claim what remains,
   **right-aligned**. This is what makes a plain unlabelled 3-D array feed a
   `("z","y","x")` op as `0→z, 1→y, 2→x`: the innermost axes are what an imaging
   op means by `y x`.
3. **Optional slots fill by name match and by nothing else.** Load-bearing:
   with `Axes("y","x","c?")` and a `(z,y,x)` stack, positional fallback would
   drop `z` into the channel slot, where `to_gray` averages across it instead of
   iterating over it. This is 0005's own hazard — "convention must never invent
   an axis name an op might consume" — now enforced in the planner rather than
   left as advice to front ends.
4. Everything left over gets a **disposition**: `iterate`, `select`, or `pass`.

Rule 3 is also why `"*?"` raises: a wildcard has no name to match on, so an
optional wildcard could never be filled by anything. Better to reject the
spelling than ship one that silently does nothing.

### Warnings, not refusals

A *named* slot fed a differently-named axis produces a warning
(`"lifetime is being fed the z axis"`). A wildcard slot never warns — it asked
for nothing. An unnamed axis never warns — it said nothing. Only genuine
impossibilities still raise: fewer axes than the op consumes, labels that do not
describe the array, or an explicit mapping that double-claims an axis.

This is the same "on the consequence, not the confidence" principle 0005 landed
on, applied one level further out: the run is never blocked, but the thing that
would make it wrong is visible before it happens.

### One editable plan, not an enumeration

0005's `plans()` returned every workable candidate, lossless first, and 0005's
napari counterpart put them in a combo box. That works only while the candidate
space is tiny. With user-chosen mappings the space is *permutations of input
axes × dispositions per leftover axis* — there is no list to show.

So `plans()` and `choose()` are replaced by one function:

```python
skop.plan(fn, param, array, axes, position=, mapping=, dispositions=)
```

Called with just the axes it returns skop's best effort. Called with `mapping`
or `dispositions` it returns what the caller asked for. A front end renders the
default, lets the user edit either half, and re-plans — which is how per-axis
control is offered without skop enumerating anything.

**`choose()`'s guarantee became structural.** It used to refuse at run time to
pick a plan that discarded data. Now the default plan simply never selects:
leftovers are iterated, or passed to a variadic op, both of which keep
everything. Data is discarded only when someone explicitly asks for it, so the
protection no longer needs enforcing — it cannot be reached by accident.

`AdaptationPlan` keys `select` and `iterate` by **input-axis index** rather than
name, since an axis may be unnamed, and adds `mapping`, `passed` and `warnings`.
`position` accepts an index key for the same reason: a front end whose axes are
mostly unnamed still knows where its sliders are, and name keys would collapse
every unnamed axis onto one entry.

### Splitting `stardist2d`

0005 listed "axis requirements that depend on another parameter" as a limit
accepted knowingly: `stardist2d`'s `model` selected between a fluorescence model
wanting one channel and an H&E model wanting three, and the declaration had to
be the union.

Hints-and-arity does not fix that, and it should not have to. The two models
genuinely accept different things, so they are now two ops —
`stardist2d_fluo` (`Axes("y","x","c?")`) and `stardist2d_he`
(`Axes("y","x","c")`) — over one shared private implementation. The differing
requirement becomes visible to a front end instead of surfacing as a traceback
from inside StarDist. The limit 0005 accepted is withdrawn: the answer to a
conditional shape is two ops, not a conditional-shape language.

## Consequences

- Nothing an op can declare forbids a user from running it. The only refusals
  left are arithmetic impossibilities.
- `Extra` and `skop.plans` are gone from the public surface; `skop.plan`,
  `skop.Slot` and the disposition constants (`ITERATE`, `SELECT`, `PASS`) join it.
- A front end must now render a mapping *and* dispositions, which is more work
  than a combo box — the cost of the flexibility, paid once per front end. See
  skop-napari's 0007.
- `output_axes` is still derived, and now uses the **caller's** vocabulary
  throughout, so a remapped axis keeps the name its owner gave it.

## Still open

- **Whether output axes need declaring.** Unchanged from 0005, and now slightly
  sharper: with remapping, "iterated axes, then the slots' input labels" is
  right for every current op and still wrong for a projection.
- **Whether a name hint should carry a category** (space / time / channel, in
  OME-NGFF's sense of `type` as distinct from `name`) so that `Axes("*", "*")`
  could prefer the two spatial axes rather than merely the innermost two. It
  would sharpen the default mapping without ever becoming a requirement. Note
  the terminology trap: ImageJ2's `AxisType` is the *name* concept, not this one.
- Whether a chunked array should iterate along its own chunk boundaries rather
  than one plane at a time.
- Iteration still calls the op with no state between calls, so genuinely
  recursive algorithms remain inexpressible. That is an execution-model gap and
  may deserve its own design.
