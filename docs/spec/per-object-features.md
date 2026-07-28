# Spec — per-object features

**Status:** open question, raised by a real failure. Nothing decided, nothing
implemented. This records the problem and the shape of the choices while it is
fresh; it does not pick one.

## What happened

Both box detectors ([0007](../design/0007-box-detector-ops.md)) originally
returned a confidence per box:

```python
class Boxes(NamedTuple):
    boxes: BoxesData      # (N, 4), Role.shapes
    scores: np.ndarray    # (N,), no role
```

The boxes drew correctly in napari. The scores raised:

```
ValueError: Image data must have at least 2 dimensions.
```

skop-napari's `layer_type_for` falls back to `"image"` for any unannotated
`np.ndarray`, so a `(24,)` vector of confidences was handed to an Image layer.
The fallback is a reasonable guess — most role-less arrays *are* pictures — but
this array is not a picture, and it is not any other layer either.

`scores` has been dropped from the return type for now. Every detector still
computes it; it just has nowhere to go. That is the cost being carried until
this is decided.

## Why it is not just a missing role

The tempting fix is to add a `Role` member and move on. It does not work,
because of what the vocabulary is:

Every existing role — `image`, `labels`, `points`, `shapes`, `surface`,
`tracks`, `vectors` — names a napari layer type. That was deliberate
([0003](../design/0003-semantic-roles.md)): it makes the front end's job a
lookup rather than a judgement. A role answers exactly one question, *which layer does this
become?*

For a per-box confidence the honest answer is **none**. It is not a layer; it
belongs *to* one. A `Role.features` would be the first member that does not
name a layer type, and every front end's role table would need a special case
for it — which is precisely the property 0003 was buying.

There is a second gap underneath. Features are features *of* something, and
`OutputSpec` has no way to say of what. It carries `name`, `type`, `role` and
nothing relational. An op returning boxes, scores and masks would need to state
that scores attach to boxes and not to masks. Today it cannot.

## What we would get for it

Worth stating, because the effort has to be worth something. napari's Shapes
and Points layers take a `features` table and can render it:

```python
viewer.add_shapes(boxes, features={"score": scores}, text="{score:.2f}")
```

That gives confidence numbers drawn on the boxes, and it makes the layer
filterable and colourable by score. It is also the natural home for anything
else per-object we will produce later: an area per mask, a class per detection
if a classifying detector ever appears, a track ID.

So this is not only about `scores`. It is about whether skop can express
"N values, one per object" at all.

## The choices

1. **Leave it out; make the front end refuse.** Have skop-napari's fallback
   return `None` for arrays with fewer than two dimensions, so they land in the
   results panel instead of a layer. Cheapest, no new vocabulary — but a panel
   of 24 numbers next to 24 boxes is not useful, and it puts the fix in one
   front end rather than in the contract.

2. **`Role.features`, plus a way to say what it attaches to.** Either a
   convention (features attach to the nearest preceding layer output) or a new
   `OutputSpec` field. Honest and general, and it breaks the
   role-equals-layer-type property, so it needs 0003 amended rather than
   quietly contradicted.

3. **Fold the values into the geometry.** A fifth column on the boxes array.
   One output, no new vocabulary — but every consumer has to know the column
   layout, `skop.boxes` converters all have to preserve it, and it does not
   generalize past one scalar.

4. **A features-bearing output type.** Return something richer than an array —
   a NamedTuple of geometry plus a dict of columns, encoded as separate arrays
   over the wire. Most expressive, most machinery, and it strains the rule that
   what crosses the Appose boundary stays trivially serializable.

## What would settle it

The mask detector of [0008](../design/0008-mask-detector-ops.md) produces an area
and possibly a stability score per mask, and a segmenter workflow
([workflow-ops](workflow-ops.md)) will want to filter boxes by confidence before
passing them on. Two more real cases, both arriving soon, and both of them will
say more about which of the four is right than more argument will.

The thing to avoid in the meantime is the fifth option: letting each op invent
its own way of returning per-object values, so that the front end acquires one
special case per op.
