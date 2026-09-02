# 0018 — Explicit array carriers

**Status:** proposed. Nothing built. Touches every op signature, which is why
it is written down before anyone starts.

## The problem

`skop.types` attaches a role to one carrier and one only:

```python
ImageData = Annotated[np.ndarray, Role.image]      # types.py:32
```

Two requirements pull against that.

**One.** An op writer who knows numpy and nothing else must be able to write
an op without learning anything new. That works today.

**Two.** An op writer who wants cupy, zarr, xarray or dask must be able to say
so. That does not work today, and cannot: the carrier is baked into the alias.

The role is already orthogonal to the carrier -- `Annotated[T, Role.image]` is
valid for any `T`, and the module docstring already says roles may attach
directly as `Annotated[T, Role.surface]`. So the vocabulary needs almost
nothing. What needs deciding is what op authors write.

## The decision

**Make the carrier explicit in every signature, and drop the `…Data` aliases
for the bulk-pixel roles.**

```python
A = TypeVar("A")

ImageOf  = Annotated[A, Role.image]
LabelsOf = Annotated[A, Role.labels]
MasksOf  = Annotated[A, Role.masks]
```

so an op declares its carrier where a reader can see it:

```python
def threshold(image: ImageOf[np.ndarray]) -> LabelsOf[np.ndarray]: ...   # numpy only
def gpu_thing(image: ImageOf[cp.ndarray]) -> LabelsOf[cp.ndarray]: ...   # cupy only
def general  (image: ImageOf[Array])      -> LabelsOf[Array]:      ...   # any array
```

`ImageOf[np.ndarray]` is the *same object* as today's `ImageData` --
`Annotated[np.ndarray, Role.image]` -- so nothing downstream changes. Only
what authors type does.

### Why not keep `ImageData` as a shorthand

It hides the decision this document exists to make visible. If most ops write
`ImageData`, most ops never state a carrier, and "which ops are numpy-only" is
answerable by assumption rather than by grep. That is the same
*we-cannot-tell-which-it-is* problem, relocated.

The argument for keeping it -- that the names mirror `napari.types` -- is
weaker than it looks. skop-napari maps on the **role**, not the alias name
(`skop-napari/_roles.py:65`), so renaming breaks nothing mechanical. The
mirroring buys familiarity, not behaviour.

The real cost is signature noise, since the carrier repeats:

```python
def deconvolve(image: ImageOf[np.ndarray], psf: ImageOf[np.ndarray]) -> ImageOf[np.ndarray]: ...
```

Judged worth paying. It is three more words in exchange for every op stating
what it can actually handle, and `npt.NDArray[np.float32]` set that precedent
years ago. If it proves unbearable in practice the alias can return as pure
sugar; adding explicitness *after* everyone has written `ImageData` is the
migration worth avoiding.

## The `Array` protocol

`Role` says what an array *means*. It cannot say what *counts* as an array --
an image and a label image are structurally identical, and conversely numpy
and cupy arrays are structurally alike but nominally unrelated. That second
question is exactly what a `Protocol` is for, and it is the one place in skop
where one belongs.

```python
@runtime_checkable
class Array(Protocol):
    @property
    def shape(self) -> tuple[int, ...]: ...
    @property
    def dtype(self) -> object: ...
    def __getitem__(self, key: object) -> object: ...
```

**Deliberately minimal, and `__array_namespace__` is deliberately absent.**
That hook is the Python array API standard's entry point and would be the
obvious basis -- but it landed in **numpy 2.0**, and the stardist environments
run **numpy 1.26.4**, where it does not exist:

```
pytorch_napari   numpy 2.4.6    __array_namespace__ True
stardist         numpy 1.26.4   __array_namespace__ False    <- TensorFlow pins it
```

Requiring it would exclude every TensorFlow environment in the project.
`shape` / `dtype` / `__getitem__` is the honest intersection that numpy 1.26,
numpy 2.x, cupy, dask and zarr all satisfy. An op needing real arithmetic
across carriers wants `__array_namespace__` as well; that is a second,
narrower protocol, declared by the ops that need it.

Nothing off the shelf does this job today. `array-api-typing` is a
`0.0.0.dev0` placeholder reading "coming soon"; `optype` is real but requires
Python 3.12, above this project's 3.10 floor. Six lines here is the cheaper
answer, and reaching for the standard later is not blocked by writing them.

## Which roles this applies to

Only the bulk-pixel ones. The roles split by kind:

| | roles | size | wants another carrier? |
| --- | --- | --- | --- |
| bulk pixel data | image, labels, masks | large | yes |
| small structured data | boxes, points, vectors, tracks | (N, 4)-ish | no |

A bounding-box array is fifty numbers; there is no reason to put it on a GPU.
`BoxesData`, `PointsData`, `VectorsData` and `TracksData` stay
`Annotated[np.ndarray, Role.…]` unchanged.

`skop.boxes.as_boxes` (`boxes.py:40`) is worth noting as the model the
pixel roles lack: it takes `object`, coerces, validates shape and raises at
the boundary rather than mid-body. Images have no equivalent and fail inside
`np.asarray` instead.

## What this does not solve

- **Op bodies.** 35 call sites do `np.asarray(image, dtype=np.float32)` --
  `smooth.py`, `edges.py`, `normalize.py` and others. Widening a signature does
  not widen the body, and cupy refuses implicit conversion to numpy. Declaring
  `ImageOf[Array]` is a promise the implementation must keep, per op.
  `array-api-compat` (maintained, `>=3.10`) is the tool for that half.
- **Marshalling.** `runner.py:334` branches on `isinstance(value, np.ndarray)`
  for the shared-memory path and silently falls to the generic encoder
  otherwise. A non-numpy carrier needs either its own codec or a rule that it
  is in-process only. Making the carrier explicit is what lets the runner
  decide deliberately instead of by accident.
- **Enforcement.** `Annotated` is inert; `_validate` (`runner.py:405`) checks
  parameter names, never types. If fail-fast at the boundary is wanted, that
  function is where an `isinstance` against the declared carrier belongs.

## Verified

The sketch runs unchanged on both ends of the supported range -- Python 3.11 /
numpy 1.26.4 and Python 3.12 / numpy 2.4.6 -- with identical output:

```
numpy_only  carrier: <class 'numpy.ndarray'> | role: Role.image
any_array   carrier: Array                   | role: Role.image
erases to  : <class 'numpy.ndarray'>
numpy array isinstance(Array): True
a list      isinstance(Array): False
```

The third line is the property that matters: to any reader not asking for
extras, an op still takes a plain `ndarray`. The role is opt-in, so the type
checker, the codec and a notebook caller are all unaffected.

## Related

- [0017](0017-memory-and-tiled-processing.md) — the other declaration an op
  owes its caller. zarr and dask carriers are usually lazy or out-of-core,
  which is 0017's problem arriving from the other direction.
- [0003](0003-semantic-roles.md) — what `Role` is and why skop never guesses.
- napari's `types.py` is the cautionary tale: `ArrayLike` is a hardcoded
  `Union[ndarray, dask, zarr]` under a comment calling it a "WOEFULLY
  inadequate stub" that "should probably be replaced by a typing.Protocol".
  The light package meant to fix it, `napari/image-types`, was created in 2019,
  never released to PyPI, and archived in 2025.
