# 0017 design — declaring memory and tiling

Status: proposed. Nothing built. The cases, how to run the test scripts, and
the file list are in [README.md](README.md).

## The problem

An op that fits in memory for a 512³ crop does not fit for the 4096³ volume
it was really meant for. Today nothing in skop says so; the caller finds out
when the process dies.

Richardson-Lucy makes it concrete. It holds the estimate, the blurred
estimate, the ratio, the correction and the FFT plans, all float32 or
complex64, and non-circulant mode pads on top. That is roughly 10x the input,
so a 50 GB image wants 500 GB, and the GPU has 24.

Deep-learning inference has the same shape -- a UNet's activations dwarf the
patch -- and a detector adds a second reason to tile, scale (case 4 in the
README), which has nothing to do with memory.

The fix in every case: cut the input into tiles, run the op on each, put the
results back together. The tiling loop is generic. What differs per op is
how much memory a tile needs, how far a tile must reach past its edge, and how
the results merge. The op hints at those; the runner, or the user, decides.

## What the op declares

On the function, in `@op(...)`, not on one input. All of it optional.

```python
@op(env="cupy", tile=("image", "mask"),
    overlap=..., peak_memory=..., merge="blend")
def richardson_lucy(image, psf, mask=None, iterations=100, noncirc=True): ...
```

| Declaration | Says | Example |
|---|---|---|
| `tile` | which inputs are cut; all cut at the same places | image and mask yes, PSF no |
| `overlap` | how far each tile reaches past its edge, per axis | gaussian `4 * sigma`; decon `psf / 2` |
| `peak_memory` | a multiple of the tile's size, in a named dtype | decon 11x, as float32 |
| `merge` | how the tiles go back together | crop, blend, boxes, labels |
| unsplit axes | never cut along these | channels; z for a 3D model |
| native size | the size the op works at, whatever it is given | YOLO `imgsz` 640 |
| global | looks at the whole image, so tiling changes the result; the runner warns | Otsu, min/max normalising |

**Hints, not rules.** Every declaration is information the op offers, so
the runner can choose a good default -- a tile that fits, an overlap that
hides seams -- without the user knowing anything about the op. Nothing is
enforced. A GUI or a script can override any of it: the tile size, the
overlap, the merge, the budget, or tiling at all. An op that declares
nothing still runs, whole, as it does today.

**Why the function and not the input.** An earlier draft put `PeakMemory` on
the tiled parameter, beside `Axes`. Three things moved it:

- Both depend on other parameters: overlap on `sigma` or the PSF, memory on
  the PSF size and `noncirc`. An annotation on `image` would have to reach
  across to its siblings.
- An op with several arrays tiles only some of them: decon tiles `image` and
  `mask`, not `psf`. Naming which ones is a statement about the call.
- Per-input annotations stay what they are now, what an array *means*
  (`Role`, `Axes`), not how the call is run.

## Simplest to most complicated

`overlap` and `peak_memory` can each be given at any of these levels. Most
ops need only the first.

1. **A fixed number.** `overlap=10`; `peak_memory=PeakMemory(scale=8)`.
2. **A fraction of the tile.** Overlap 10% of the tile, as Cellpose and
   StarDist do.
3. **A formula of the op's parameters**, including other arrays' shapes:
   `4 * sigma`, `psf.shape // 2`. For memory, a multiplier that depends on
   the parameters: decon's padding grows the tile by the PSF size and rounds
   it up to a fast FFT size, so the multiple is bigger for a small tile than
   a large one. The runner never has to know about FFTs.
4. **From the model.** Deep-learning weights carry their own halo and input
   size; bioimage.io model files have `halo`, `min` and `step` per axis.
5. **Measured.** Run on a small tile, read the peak, scale up.
   `zarr-idr0079-memory.py` already reads the peak with `memory.peak`.

Levels 1, 2 and 4 are plain data that any front end can read, Fiji included.
Level 3 as a Python lambda can only be evaluated by a Python host, so prefer a
declarative form where one exists -- `Overlap(param="sigma", scale=4)` covers
gaussian and most filters -- and keep the lambda as the escape hatch.

### Formulas run on the host

Whatever is in the decorator runs in the host, not the op's environment:
the arguments when the op's module is imported, a lambda's body when the
runner plans the call. A formula that imports cupy fails in either place. So:

1. **A formula uses only what the host has: Python and numpy.** The same
   rule ops already follow for module-level imports, and discovery already
   reports a module that breaks it.
2. **skop provides the helpers**, so authors do not write their own:
   `next_smooth` for fast FFT sizes, `padded_size` for decon's padding. Pure
   Python, safe on the host by construction.
3. **A failing hint degrades, it does not block.** If a formula raises, the
   runner says so -- "could not work out richardson_lucy's memory: ..." --
   and falls back to no tiling, or to what the user set. The failure is
   shown, and the op still runs.
4. **Data over code.** `Overlap(param="sigma", scale=4)` cannot import
   anything.

### PeakMemory

```python
@dataclass(frozen=True)
class PeakMemory:
    scale: float      # peak memory as a multiple of the tile, in dtype
    dtype: Any = None # dtype the buffers are held in; None = same as input
    fixed: int = 0    # bytes on top that do not grow with the tile
```

The op says a multiple, "8x the tile as float32", not a number of bytes;
the runner turns it into bytes. Inverting it gives the biggest tile that fits,
in one line, which is the test of whether it is well formed:

```
n_elements <= (budget - fixed) / (scale * itemsize(dtype))
```

- **Not a count of copies.** FFT plans and padding are not copies of
  anything. A multiplier on bytes covers them.
- **dtype is named.** cupy's FFT is float32 whatever it is given, so uint16
  in is 2x before a buffer is allocated. "8x as float32" is unambiguous.
  numpy decon in tnia-python runs float64, so dtype depends on the backend.
- **Some of it does not grow.** Model weights and FFT plans stay the same
  size whatever the tile. YOLO is almost all `fixed`: the model at `imgsz`,
  plus `scale` for reading and resizing the tile.
- **A constant is not enough for decon.** Padding adds the PSF size and then
  rounds up to a 7-smooth number, so a 1000-pixel tile can pad to 1050. That
  needs level 3.

### Merge

| Merge | For | How |
|---|---|---|
| crop | filters | drop the overlap, keep each tile's middle; exact if the overlap is big enough |
| blend | decon, deep learning | weight each tile down towards its edges, average where they overlap; for ops where no overlap is exact |
| boxes | detectors | shift boxes back to image coordinates, then NMS |
| labels | segmentation | relabel, merge objects cut by a seam |

The merge follows the op's kind more than the op, so it can default from the
kind and only be declared where an op differs.

**The runner does only crop and blend.** Those are the same for every op.
Boxes and labels merges are **ops**, named by the declaration
(`merge=merge_boxes`), because they carry choices the user should see and
change: the IoU threshold, IoU or IoS, NMS or NMM, class-aware or not. A
user can swap one merge op for another. Tiled detection is then a workflow
op: cut, run the detector per tile, `merge_boxes`.

[SAHI](https://github.com/obss/SAHI) does tiled detection, 2D boxes only.
It cannot be the runner -- no decon, no filters, no 3D, and it wants to call
the model itself -- but its box merges (NMS, NMM, IoS) are what
`merge_boxes` should do, and a `merge_boxes` op could use it in its own
environment. A detector op that tiles internally with SAHI is a useful
baseline to compare against.

**Global is a fact, not a ban.** Tiled Otsu gives each tile its own
threshold, which is adaptive thresholding -- maybe exactly what the user
wants. So the runner tiles a global op only when it has to or is asked to,
and warns, for example "otsu looks at the whole image; tiled, each tile gets
its own threshold". The warning comes from the declaration, not a list of op
names. Detectors and models that need context are milder versions of the
same thing: tiling changes their result too, and the merge softens it.

## What the runner does

1. **Find the budget.** The device's memory -- VRAM for a cupy op, RAM
   otherwise -- or the user's override. Free or total is a choice:
   tnia-python uses free, clij2-fft uses total.
2. **Find the tile.** For memory, the largest tile whose peak fits. For scale
   (case 4), from the user's object size and the op's native size, then
   capped by memory. Round to the op's size rules (a multiple of 32, a
   minimum).
3. **Cut.** Overlap between neighbouring tiles. **At the real image edge add
   nothing**, and let the op handle its own boundary -- non-circulant decon
   is built for exactly that, and padding the edge with mirror or zeros
   undoes it.
4. **Run each tile and merge.**

## Lazy inputs (case 1)

A zarr or dask array too big for RAM. Decided: the runner reads it one tile
at a time, converts each tile to numpy, and runs the op on it, with the same
cut, overlap and merge as any other case. The op never sees the zarr. Later,
an op that declares it can take any array (`ImageOf[Array]`) could get a
proxy that fetches chunks from the host as it reads them. The alternatives,
and the proxy in detail, are in [lazy-array-input.md](lazy-array-input.md).

## Tiling for scale (case 4)

A detector resizes its input to its native size, `imgsz`, so the tile size
comes from the objects, not from memory:

```
tile ≈ imgsz × (object size in the image / object size the model expects)
```

The user gives the object size, by drawing a box; the model's training
decides the size it expects. Memory only caps the tile. The overlap must be
at least the largest object, so every object is whole in some tile, and the
boxes merge with `merge_boxes`. A small image with big objects goes the
other way: fewer, larger tiles, or a downsample. JDLL's rules, and SAM, which
has the same problem, are in [yolo-sam-tiling.md](yolo-sam-tiling.md).

## Who says what

Some of it is the op's, some only the user knows, as with axis mapping
([0006](../0006-axis-mapping.md)).

| Who | Says |
|---|---|
| op | which inputs tile, overlap, peak memory, merge, unsplit axes, native size |
| user | object size (a drawn box) for detectors; and any override of the defaults: budget, tile size, overlap, merge, or no tiling |
| runner | defaults for all of it, from the op's hints and the device |

For a detector the overlap must be at least the largest object, so every
object is whole in some tile -- the user's number again, not the op's.

## Checked against real decon code

clij2-fft and tnia-python both tile Richardson-Lucy. Everything they do fits
the mechanisms above:

| What they do | Expressed by |
|---|---|
| overlap fixed at 10 px, or `floor(psf / 2)` | levels 1 and 3 |
| only y and x split, never z | unsplit axes |
| crop, no blending | `merge="crop"` |
| tile count solved from GPU memory | the runner inverting `peak_memory` |
| GPU memory free (tnia-python) or total (clij2-fft); user override | the budget |
| pad by the PSF, round up to 7-smooth sizes | `peak_memory` at level 3 |
| a bad-pixel mask tiled with the image | `tile=("image", "mask")` |

And four things they do that this design should avoid:

- **Nobody knows the multiplier.** clij2-fft uses x11 and tnia-python x41
  for the same algorithm, and both have comments that disagree with their
  code. That is the case for level 5, measured.
- **Reflect padding at the outer edge.** Both use dask's default boundary,
  which reflect-pads the image edge and then hands it to non-circulant decon.
  Hence step 3's edge rule.
- **Overlap not from the PSF.** A fixed 10 px whatever the PSF.
- **Crop, not blend.** Decon's influence reaches past any overlap, so every
  tile edge is slightly wrong, and cropping shows it as a seam.

## Not now

- The boxes and labels merge ops, in detail; crop and blend first.
- Several GPUs: both decon libraries queue tiles across devices.
- Parameters that change per tile, such as a PSF that varies across the
  image (`richardson_lucy_variable.py` in tnia-python). A tile would need to
  know where it sits.
- Whether an op can say it tiles internally, so the runner leaves it alone.

## Not decided

- Evaluating a formula in the worker instead, where the op's own
  libraries are, at the cost of a round trip before every plan. Only if a
  formula turns up that really needs them.

- The exact declarative form of level 3, and whether `Overlap` and
  `PeakMemory` are objects or keyword arguments on `@op`.
- Free or total memory as the default budget.
- Whether measured `peak_memory` is cached, and where.
- Who owns the GPU transfer. [0018](../0018-explicit-array-carriers.md)
  proposes `ImageOf[cp.ndarray]`; whichever way it lands decides that the
  budget for a cupy op is VRAM, not RAM.

## Related

- [examples.md](examples.md) -- gaussian, decon and YOLO, declared in full.
- [0014](../0014-make-decon-ops.md) -- Richardson-Lucy, the motivating op.
- [0006](../0006-axis-mapping.md) -- an op declaring something about its
  inputs and the caller acting on it.
- [lazy-array-input.md](lazy-array-input.md) -- case 1, where the input does
  not even fit in RAM.
- [yolo-sam-tiling.md](yolo-sam-tiling.md) -- case 4, tiling for scale.
- napari-ai-lab spec 0006 (batch segmentation over a sequence), whose
  `can_process` question is the same declaration seen from the axis side.
