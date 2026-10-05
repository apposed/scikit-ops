# 0017 examples — three ops, declared in full

Three real ops in skop, with the declarations [design.md](design.md)
proposes. The parameter annotations are the kind skop has today -- `ImageOf`,
`Axes`, widget ranges -- and the tiling arguments on `@op` are new: none of
that syntax exists yet; it is a sketch to argue with. The numbers are counted
from the code, and want measuring.

Today `gaussian` and `jdll_yolo` take `ImageData` and the decon op a bare
`np.ndarray`; they are written here as `ImageOf[np.ndarray]`, where skop is
heading.

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

## YOLO: a detector, tiled for scale

`skop.ops.detect.jdll_yolo` (env `pytorch`).

```python
@op(env="pytorch", tile="image",
    native_size="imgsz",
    peak_memory=PeakMemory(scale=6, dtype=np.uint8, fixed=MEASURED),  # scale a guess
    merge=merge_boxes)
def jdll_yolo(
    image: Annotated[ImageOf[np.ndarray], Axes("y", "x", "c?")],
    weights: Path,
    conf: Annotated[float, {"widget_type": "FloatSlider", "min": 0.0, "max": 1.0}] = 0.25,
    iou: Annotated[float, {"widget_type": "FloatSlider", "min": 0.0, "max": 1.0}] = 0.7,
    imgsz: Annotated[int, {"min": 32, "step": 32}] = 640,
) -> Boxes: ...
```

- **`Axes("y", "x", "c?")`**: a plane, with an optional colour axis. `c` is
  something the op takes whole, so it is never tiled -- for this op the
  "unsplit axes" declaration comes free from `Axes`.
- **`native_size="imgsz"`** names the parameter that says what size the model
  works at. The runner needs it to choose tiles; today it is only a knob.
- **Tile size comes from the user**, not memory: an object about 30 px across
  in a 10000 x 10000 image, and a model that expects about 64 px, gives tiles
  of about `640 * 30 / 64 = 300` px, roughly 1100 tiles. Memory only caps it.
- **Overlap at least the largest object** -- the user's number again -- so
  every object is whole in some tile. No overlap declared by the op.
- **Peak memory is mostly `fixed`**: the model and its activations at
  `imgsz`, the same whatever the tile, so it is measured once per model. The
  part that grows is converting the tile to 8-bit RGB and resizing it, a few
  bytes a pixel.
- **`merge=merge_boxes`**, an op: shift boxes back to image coordinates, then
  NMS or NMM. The tiles also change the result -- an object cut by a tile edge
  gets a partial box -- which the merge softens and the user should be told.
