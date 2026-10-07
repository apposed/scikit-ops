# 0017 — Memory requirements and tiled processing

**Status:** partly built.

**Built**

- Tiling inside an op: `skop.ops.detect.yolo` picks tiles from
  `object_size`, batches them on the GPU with memory measured as it runs,
  and merges boxes across tiles.
- The declarations on `@op`: `tile`, `overlap`, `peak_memory`, `merge`
  (opspec), and the planner that turns them into tiles (`opspec.tiling`).
- The runner tiling an op on the host: `Runner.run(..., memory=)`, with a
  default budget of 85% of available memory, cgroup limits included. Crop
  merge only. `gaussian` declares its hints.
- A memory budget box in skop-napari, which says how a run will be tiled.
- Test scripts that run ops under a memory cap: `zarr-idr0079-*.py`,
  `yolo-*.py`.

**Not built yet**

- Blend merge, tiling several inputs together, and tiling combined with
  axis slicing.
- VRAM as the budget for a GPU op.
- Lazy input: getting a zarr into the worker, so an op can take
  `ImageOf[Array]`.
- Contrast limits for the whole image, before `yolo` stretches each tile.

**Reading order:** this page, then [tiling.md](tiling.md), which is the whole
design. [tiling-examples.md](tiling-examples.md) shows it on three real ops.
[lazy-array-input.md](lazy-array-input.md) and
[yolo-sam-tiling.md](yolo-sam-tiling.md) are optional background: the
alternatives considered and why the design came out as it did.

Four cases, one piece of code. In the first three the op declares its
`PeakMemory`, and the runner splits the input into chunks that fit the
memory available.

1. **Lazy input, too big for RAM.** A zarr or dask array that cannot be
   read into memory whole. Only this case touches zarr; see
   [lazy-array-input.md](lazy-array-input.md).
2. **In memory, but the op's copies are too big for RAM.** A 5 GB numpy
   array on a 32 GB machine, and an op that needs 8x the input.
3. **In memory, but too big for the GPU.** A 50 GB image in RAM, and
   deconvolution needing 8x the input, 400 GB, on a 24 GB GPU.
4. **Fits, but too big for the model's scale.** YOLO resizes its input so
   the longest side is `imgsz`, 640 or 1024. A 10000 x 10000 image squashed
   to 640 shrinks objects 16x, and small ones vanish, however much memory
   there is. Tiles about `imgsz` across keep objects near the size the
   model was trained at; the user's object size picks the tile, and memory
   only caps it. See [yolo-sam-tiling.md](yolo-sam-tiling.md).

## How to run

Each script's docstring has more options. The images go in `test_images/`.

Big 3D, case 2: idr0079, a light-sheet volume, 177 MB membrane channel.

```sh
D=docs/design/0017-memory-and-tiled-processing
uv run $D/zarr-idr0079-download.py
uv run $D/zarr-idr0079-memory.py --cap 1G     # gaussian fails
uv run $D/zarr-idr0079-memory.py --cap 2G     # gaussian succeeds
uv run $D/zarr-idr0079-napari.py              # one zarr layer, the skop panel
```

Big 2D with YOLO, case 4: EM (idr0083, 23k x 36k at level 2) or aerial
(28k x 28k RGB).

```sh
uv run $D/yolo-download.py aerial                    # or em
uv run $D/yolo-run.py aerial --level 4               # a coarse level, whole
uv run $D/yolo-run.py aerial --level 2 --tile 1024   # a fine level, tiled
uv run $D/yolo-napari.py aerial                      # levels as layers, YOLO ready
```

## Files

Everything for this design lives in this directory, until it is built:

| File | What |
|---|---|
| [tiling.md](tiling.md) | The design: what an op declares, what the runner does |
| [tiling-examples.md](tiling-examples.md) | Gaussian, Richardson-Lucy and YOLO, declared in full |
| [lazy-array-input.md](lazy-array-input.md) | Background for case 1: getting zarr and dask inputs into the worker |
| `zarr-skimage.py` | Which scikit-image functions accept a zarr array |
| `zarr-idr0079-download.py` | Copies idr0079 (3D light sheet, 355 MB) into `test_images/` |
| `zarr-idr0079-memory.py` | Runs gaussian or frangi on it under a memory cap |
| `zarr-idr0079-napari.py` | Opens it in napari as one zarr layer, optionally capped |
| `zarr-idr0083-download.py` | Copies levels 2-10 of idr0083 (2D EM, 1.7 GB at level 2) |
| [yolo-sam-tiling.md](yolo-sam-tiling.md) | Background for case 4: tiling detectors and SAM to match `imgsz` |
| `yolo_data.py` | Where the two big 2D images live; imported by the `yolo-*.py` scripts |
| `yolo-download.py` | Makes `test_images/yolo-{em,aerial}.zarr`, one 2D array per level |
| `yolo-run.py` | FastSAM or JDLL YOLO on one level, whole or tiled; box counts and sizes |
| `yolo-napari.py` | Several levels as aligned layers, with the skop panel on YOLO |

