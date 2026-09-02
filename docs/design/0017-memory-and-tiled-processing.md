# 0017 — Memory requirements and tiled processing

**Status:** proposed, and barely that. Nothing built, nothing designed. This
document exists so the problem is written down rather than rediscovered.

## The problem

An op that fits in memory for a 512³ crop does not fit for the 4096³ volume it
was really meant for. Today nothing in skop says so — the caller finds out
when the process dies.

Richardson-Lucy is the example that makes it concrete. `richardson_lucy`
(design 0014) is not an in-place filter: it holds the estimate, the blurred
estimate, the ratio, the correction and the FFT plans, all as float32 or
complex64, and `noncirc=True` pads on top of that. The working set is roughly
**7–10× the input array**, so a 1 GB image wants 7–10 GB. A user with 8 GB of
VRAM asking for `richardson_lucy_cupy` on a 2 GB stack is asking for something
that cannot happen, and there is currently no way for anything to know that
before it fails.

## What is missing

Two halves, and the first is the one skop owns.

**1. The op declares what it needs.** Something answering "for an input of
this shape and dtype, how much working memory will you use" — a multiplier, a
function, or a small structure; undecided. It has to cover the compute buffers,
not just the input and output, because the buffers are the whole point.

**2. The caller inverts it.** Given a memory budget, solve for the largest
tile that fits, then run the op tile by tile with whatever overlap the op
needs. The tiling loop is generic and belongs beside the other adaptation
machinery; the overlap is not, since a convolution needs a halo and a
per-pixel op needs none. So the declaration probably carries both: cost and
halo.

## Why it is not just a decon problem

Deep-learning inference has the same shape — a UNet's activations dwarf the
patch — which is why tiled prediction is hand-rolled in every framework. If
this hook exists, `can_process`-style questions ("will this run here at all?")
and tiled execution are the same declaration read two ways: axes for whether
the op applies, memory for whether it fits.

## A candidate form

Not settled, but concrete enough to argue with. The declaration rides on the
parameter that gets tiled, which is how it says *which* array to tile as well
as how much it costs. It sits beside `Axes`, which ops already carry:

```python
@dataclass(frozen=True)
class WorkingSet:
    scale: float          # peak bytes / (n_elements * itemsize(dtype))
    dtype: Any = None     # dtype the buffers are held in; None = same as input
    fixed: int = 0        # bytes that do not scale with tile size


@op(env="cupy")
def richardson_lucy_cupy(
    image: Annotated[
        ImageOf[cp.ndarray],
        Axes("z", "y", "x"),
        WorkingSet(scale=8, dtype=np.float32),
    ],
    psf: ImageOf[cp.ndarray],
    num_iters: Annotated[int, {"min": 1, "max": 1000}] = 10,
    noncirc: bool = False,
) -> ImageOf[cp.ndarray]: ...
```

Spelled inline here to show the shape; ops in the tree pull the repeated
part out into an alias, as `edges.py` does with `_Plane` and `_Volume`.
`WorkingSet` would not travel with such an alias though -- it is a fact
about one op's buffers, not about a family of them, and a return value has
no working set at all.

That one annotation now answers three questions with three independent pieces
of metadata: what the value *means* (`Role.image`, carried by `ImageOf`), what
shape of input the op consumes (`Axes`), and what it will cost to run
(`WorkingSet`). PEP 593 flattens the nesting, so they end up as one flat tuple
and no consumer needs to know about the others -- `_spec.py` already selects
by `isinstance` rather than by position, which is why `Axes` and `Role`
coexist today.

`Axes("z", "y", "x")` rather than `Axes(variadic=True)` is deliberate here.
The op deconvolves a volume; a 4D input is looped over, not handed in whole.
That is also what makes the memory question answerable, since the thing being
budgeted is one ZYX volume rather than whatever the caller happened to load.

The caller inverts it in one line, which is the test of whether the
declaration is well formed:

```
n_elements <= (budget - fixed) / (scale * itemsize(dtype))
```

Three things this shape is reacting to, all visible in RL:

- **It is not a count of copies.** FFT plans and `noncirc` padding are not
  copies of anything. A multiplier on bytes covers them; "five copies" does
  not. `noncirc=True` is most of the spread between 7x and 10x, which is an
  argument for `scale` being a callable of the arguments rather than a
  constant.
- **dtype promotion has to be named.** cupy's FFT is single precision, so this
  op works in float32 whatever it is handed; uint16 in is 2x before a buffer
  is allocated. "8x the array *as float32*" is unambiguous; "8x" alone is not.
- **Some of it does not scale.** Cached FFT plans and model weights do not
  shrink with the tile. Folding them into the multiplier makes the tiler
  choose a tile that does not fit, which is the failure this is meant to
  prevent.

Halo stays a separate annotation rather than a field here: different units
(pixels per axis, not bytes), and a per-pixel op has a working set but no
halo. dask calls it `depth` in `map_overlap`, which is the name to borrow.

Two things this example takes for granted that are not settled anywhere yet.
`ImageOf[cp.ndarray]` is [0018](0018-explicit-array-carriers.md)'s proposal,
and it changes who owns the transfer: today `richardson_lucy_cupy` declares
`np.ndarray` and calls `cp.asarray` itself, so the caller hands host memory
and the upload is invisible. Declaring a cupy carrier says the caller arrives
with device arrays. Whichever way that lands, it decides the second thing --
that `budget` here is **VRAM, not host RAM**. A cupy op tiled against free
system memory is tiled against the wrong number.

## Not decided

Everything. In rough order of what to settle first:

- The form of the declaration: constant multiplier, callable, or a structure
  with a halo beside it. `WorkingSet` above is the first candidate; whether
  a scalar `scale` is enough, or it has to be a callable of shape and dtype,
  is the open half.
- Whether the budget comes from the caller, from probing the device, or both —
  CPU RAM and VRAM are different questions and cupy ops care about the second.
- Whether tiling is a wrapper the caller applies, or something an op can
  advertise it does internally.
- Whether the multiplier is measured or declared. Declared is a guess that
  drifts; measured needs a benchmark that runs somewhere.

## Related

- [0014](0014-make-decon-ops.md) — Richardson-Lucy, the motivating op.
- [0006](0006-axis-mapping.md) — the precedent for an op declaring something
  about its inputs and the caller acting on it.
- napari-ai-lab spec 0006 (batch segmentation over a sequence), whose
  `can_process` question is the same declaration seen from the axis side.
