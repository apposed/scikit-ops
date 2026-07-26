# 0003 — Semantic roles

## The problem

Every op in this collection passes `np.ndarray` around. That is correct — it is
the only type all four run modes agree on — and it tells a front end nothing.
An op returns a 2-D integer array: is that a label image, a grayscale result, a
mask, an array of point coordinates? napari would display each of those
completely differently, and the type system has no opinion.

This came up while designing the napari front end, and the first instinct was
to solve it *there*: let the napari layer inspect dtype and shape and guess.
That fails immediately — `otsu` returns labels and `synthetic_nuclei` returns an
image, and no amount of squinting at a `uint16` array distinguishes them.

## The decision

A `Role` enum in `skop._spec`, attached to types via `Annotated`, exposed as
aliases in `skop.types`:

```python
ImageData  = Annotated[np.ndarray, Role.image]
LabelsData = Annotated[np.ndarray, Role.labels]
PointsData = Annotated[np.ndarray, Role.points]
VectorsData = Annotated[np.ndarray, Role.vectors]
TracksData = Annotated[np.ndarray, Role.tracks]
```

```python
@op(env="skimage")
def otsu(image: ImageData) -> LabelsData: ...
```

`ParamSpec.role` and `OutputSpec.role` carry it to front ends.

Three properties made this the right shape:

**Nothing changes for anyone who does not care.** `Annotated[np.ndarray, ...]`
*is* `np.ndarray` to the codec, to a direct caller, and to `ParamSpec.type`. An
op can be annotated with roles or not; the machinery is identical either way.

**They compose.** Nested `Annotated` flattens, so `Out[LabelsData]` and
`Annotated[LabelsData, {"label": "Nuclei"}]` both work, in either order. This
mattered: roles had to coexist with the existing UI-hint dicts and the
`Out`/`Mut` form markers, all of which already used `Annotated`.

**They import no GUI.** `skop.types` is numpy and `typing`. The names
deliberately *mirror* `napari.types` so that the napari mapping is a lookup
table rather than a judgement call — but the vocabulary belongs to skop, and a
Fiji or plain-magicgui front end reads exactly the same `Role`.

## Roles are never guessed

This is the load-bearing part. skop reports `role is None` for an unannotated
array. It does not fall back to `Role.image`, even though that would be right
most of the time.

A guess made here would be indistinguishable, downstream, from a declaration.
A front end receiving `Role.image` cannot tell whether the op author meant it
or whether skop invented it, so it cannot decide how much to trust it. A front
end receiving `None` knows exactly where it stands and can apply whatever
default suits its own display model — which the napari layer does, mapping a
role-less `np.ndarray` to an Image layer. That is a front end's business,
because a front end has a viewer to make the assumption *for*.

`skop.ops.toy.scale` is left unannotated on purpose, as the standing test case
for the no-role path.

## Outputs needed a spec of their own

`OpSpec.outputs` is a `tuple[str, ...]` and stays that way — it is the wire
view, used on both sides of the Appose boundary, and it must stay trivially
serializable.

Front ends need more, so `OpSpec.output_specs` derives `OutputSpec(name, type,
role)` on demand, from three different places depending on the op:

- a **computer/inplace** op: from the matching `Out`/`Mut` parameter
- a **function** op returning one value: from the return annotation
- a **function** op returning a `NamedTuple`: from that class's field
  annotations, resolved with `get_type_hints(..., include_extras=True)`

That last resolution can fail — a `NamedTuple` defined in an awkward scope,
a forward reference that does not resolve — and when it does, `_field_hints`
swallows the exception. The reasoning is in the comment there: any failure
costs a role, and a missing role is a degraded GUI, not a broken op. Nothing
about running the op depends on this working.

## Alternatives considered

**A separate `roles=` argument to `@op`.** Rejected: a second source of truth
keyed by parameter name, which drifts the moment anyone renames a parameter.

**Wrapper classes** (`class LabelsData(np.ndarray)`). Rejected: it changes what
the op actually receives, breaking mode B and forcing conversions at every
boundary. The whole appeal of `Annotated` is that it is invisible at runtime.

**Reusing `napari.types` directly.** Rejected: it would make napari a
dependency of the base package, to serve one of four run modes.
