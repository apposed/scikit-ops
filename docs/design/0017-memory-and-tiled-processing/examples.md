# 0017 examples — three ops, declared in full

Three real ops in skop, with the declarations [design.md](design.md)
proposes. The parameter annotations are the kind skop has today -- `ImageOf`,
`Axes`, widget ranges -- and the tiling arguments on `@op` are new: none of
that syntax exists yet; it is a sketch to argue with. The numbers are counted
from the code, and want measuring.

Today `gaussian` and `yolo` take `ImageData` and the decon op a bare
`np.ndarray`. Gaussian and decon are written here as `ImageOf[np.ndarray]`,
where skop is heading. YOLO is shown as it is, then as `ImageOf[Array]`,
because it slices its input itself and never needs all of it as numpy.

The two halves answer different questions. The annotations say what each
value *is*: an image, which axes the op takes, what range a number has. The
tiling arguments say how a call can be *split*, and they refer to the
parameters by name.

## Gaussian: a filter, simplest case

`skop.ops.smooth.gaussian` (env `skimage`).

```python
@op(env="skimage", tile="image",
    overlap=Overlap(param="sigma", scale=4),
    peak_memory=PeakMemory(scale=2, dtype=np.float32),
    merge="crop")
def gaussian(
    image: Annotated[ImageOf[np.ndarray], Axes(variadic=True)],
    sigma: Annotated[float, {"min": 0.0, "max": 50.0, "step": 0.5}] = 1.0,
) -> ImageOf[np.ndarray]: ...
```

- **`Axes(variadic=True)`**: any number of axes, so any axis can be tiled.
- **Peak memory, 2x as float32.** The op converts the input to float32 (one
  buffer) and scikit-image writes the result into another. The final
  `astype(np.float32)` copies again, but by then the first buffer is free, so
  two at a time. On the 177 MB uint8 idr0079 channel that is about 1.4 GB,
  which is why `zarr-idr0079-memory.py` fails at 1G and succeeds at 2G.
- **Plain scikit-image is 2x as float64**, double that: called directly on
  the uint8 image, it converts to float64 (`zarr-idr0079-memory.py
  --skimage`). Same multiple, different dtype, so the dtype belongs in the
  declaration: "2x" alone means nothing.
- **Overlap `4 * sigma`.** scikit-image cuts the kernel off at `truncate *
  sigma`, and `truncate` is 4 and not exposed by this op. That is the
  declarative form, level 3 without a lambda: a front end in any language can
  evaluate it.
- **Crop** is exact: with that overlap, a tiled result equals the untiled one.

## Richardson-Lucy on the GPU: decon

`skop.ops.deconvolve.richardson_lucy_cupy` (env `cupy`).

```python
@op(env="cupy", tile=("image", "mask"),
    overlap=lambda psf: [s // 2 for s in psf.shape],
    peak_memory=lambda tile, psf, noncirc: PeakMemory(
        scale=13 * padded_size(tile, psf, noncirc) / size(tile),
        dtype=np.float32),
    merge="blend")
def richardson_lucy_cupy(
    image: Annotated[ImageOf[np.ndarray], Axes("z", "y", "x")],
    psf: Annotated[ImageOf[np.ndarray], Axes("z", "y", "x")],
    num_iters: Annotated[int, {"min": 1, "max": 1000}] = 10,
    noncirc: bool = False,
    mask: Annotated[ImageOf[np.ndarray], Axes("z", "y", "x")] | None = None,
) -> ImageOf[np.ndarray]: ...
```

- **`Axes("z", "y", "x")`**: the op takes one volume. A t,z,y,x stack is
  first sliced into volumes by the axis machinery, and each volume is then
  tiled for memory if it needs to be. Slicing and tiling stack; they are not
  the same thing.
- **`tile=("image", "mask")`.** The mask is cut with the image, at the same
  places. The PSF is not cut: every tile gets all of it.
- **Overlap half the PSF**, per axis -- the "Rule of Brian" in clij2-fft.
  Only a default. Each iteration spreads a pixel's influence by another PSF
  radius, so in theory every pixel affects every other and no overlap is
  exact. In practice the influence fades fast; how much overlap is enough is
  an art, and the user can raise or lower it.
- **Blend, not crop.** Since no overlap is exact, the edge of every tile is
  a little wrong, and cropping leaves a visible seam where two tiles meet.
  A linear blend across the overlap -- each tile weighted down towards its
  edge, the two averaged -- hides it. clij2-fft and tnia-python both crop.
- **Peak memory about 13x the *padded* tile, as float32.** Held on the GPU
  through the loop: image, PSF, weights, estimate (4), two complex OTFs (4),
  and per iteration two or three complex FFT temporaries and the ratio. With
  `noncirc` the tile is padded by the PSF and rounded up to a fast FFT size,
  so the multiple of the *unpadded* tile depends on the tile -- level 3, a
  formula. `padded_size` and `size` are skop helpers, pure Python, so the
  formula runs on the host without cupy (design.md, "Formulas run on the
  host"). clij2-fft says 11x and tnia-python 41x for the same algorithm,
  which is why this should be measured.
- **Budget is VRAM**, not RAM.
- **Real image edges** get no overlap, and `noncirc` handles them; the
  runner must not reflect-pad there.

## YOLO: a detector that tiles itself

`skop.ops.detect.yolo` (env `pytorch`). Unlike the two above, the op does
the tiling, so it can batch tiles through the model (design.md, "Ops that
tile themselves"). Its signature today:

```python
@op(env="pytorch")
def yolo(
    image: Annotated[ImageData, Axes("y", "x", "c?")],
    weights: Path,
    object_size: float | None = None,
    overlap: Annotated[float, {"min": 0.0, "max": 0.95}] = 0.15,
    gpu_fraction: Annotated[float, {"min": 0.01, "max": 1.0}] = 0.9,
    conf: Annotated[float, {"min": 0.0, "max": 1.0}] = 0.25,
    iou: Annotated[float, {"min": 0.0, "max": 1.0}] = 0.5,
    max_det: int = 300,
    merge_threshold: Annotated[float, {"min": 0.0, "max": 1.0}] = 0.5,
) -> Detections: ...
```

- **Nothing on `@op` but the environment.** No `overlap`, `peak_memory` or
  `merge`: the tiling is the op's own, so the runner has nothing to tile it
  with and runs it whole.
- **The tiling knobs are parameters.** `object_size`, an area in pixels,
  sets the tile: about `sqrt(1.5 * object_size / 0.001)` on a side, so an
  object covers about a thousandth of a tile. `overlap` is a fraction of the
  tile. The last tile in each row or column is pinned to the image edge, so
  all tiles are the same size and batch without padding.
- **Memory is measured, not declared.** The op runs one batch, reads the
  GPU's peak, and sizes the next batches to `gpu_fraction` of the free
  memory. A batch that runs out of memory is halved and retried.
- **The merge is inside the op**: boxes shifted back to image coordinates,
  then SAHI's greedy NMM across tiles, by intersection over the smaller box
  (`merge_threshold`), so a box cut by a tile edge merges into the whole one.
- **`Axes("y", "x", "c?")`**: a plane, with an optional colour axis. `c` is
  taken whole, never tiled.

### The same op on a lazy image

`ImageData` is numpy, so today the whole image must fit in RAM, and a bigger
one would need the runner to tile it on the outside too (design.md, "Two
levels"). Typed on `Array`, it would not:

```python
    image: Annotated[ImageOf[Array], Axes("y", "x", "c?")],
```

- **The op already only slices.** Its uses of `image` are `ndim`, `shape`
  and `image[y:y + side, x:x + side]`. On a zarr the slice is numpy already;
  on dask it needs `np.asarray(...)` around it.
- **Contrast limits come first.** Each tile is stretched into 8-bit by
  `to_rgb`, between its own 1st and 99.8th percentiles, so on 16-bit EM
  every tile is stretched differently. The limits want working out once,
  from a coarse level or a sample of tiles, and passing to `to_rgb` -- a fix
  that is needed with or without `Array`.
- **Getting the zarr to the worker** is the missing piece: the runner
  converts everything to numpy today.
