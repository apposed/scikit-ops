# Lazy arrays across the process boundary

Status: proposed. Nothing here is built.

Background for case 1 of [0017](README.md): the input is lazy and too big
for RAM. The decision is summarised in [design.md](design.md); this is the
alternatives and the proxy in detail.

## The problem

Only numpy (through shared memory) and plain JSON cross into a worker. A
zarr, dask or xarray array depends on things that exist only in the host: a
store and its codecs, a task graph, GPU memory. The op environment may not
have zarr at all.

Today the runner converts any non-numpy input to numpy before sending
(`runner.py`, `np.asarray`), and `_codec` then copies it into shared memory.
So the whole array must fit in RAM, and probably twice over at the peak:
the numpy copy and the shared-memory copy. Not yet measured.

## Four approaches

1. **Convert to numpy, optionally in chunks.** Today's behaviour, plus
   chunking driven by `PeakMemory` and overlap (below). Works for any op.
   Global ops change when chunked: Otsu on each chunk picks a different
   threshold per chunk, so the mask shows seams at chunk edges -- adaptive
   thresholding, wanted or not, so the runner warns. A niche
   workaround is two passes: compute the threshold once (from a lower
   multiscale level, or a histogram built chunk by chunk), then apply it to
   each chunk at full resolution.
2. **Pass the store reference.** The codec sends where the store is (path
   or URL, array name, icechunk snapshot id) and the worker opens it with
   zarr. No round trips, but the op environment needs zarr, icechunk and any
   cloud credentials. The op gets a real zarr, which lacks most numpy
   methods (see `zarr-skimage.py`).
3. **Lazy proxy.** The codec sends a handle and the worker gets a
   `LazyArray` that fetches chunks from the host on demand. The op
   environment needs nothing extra. Costs round trips and a cache to tune.
   Detail below.
4. **Call the op directly in the host.** The zarr passes through untouched,
   with no codec. Needs the op's dependencies in the host, which may take
   some environment wrangling. This is the fallback that always exists:
   when the proxy has a bug, or is waiting on a fix, the ops are still a
   collection of plain functions, and work can continue.

## The annotation says which

`ImageOf[T]` says what array the op takes:

- `ImageOf[np.ndarray]`: the op needs numpy. The runner converts the whole
  input up front, or, if the op declares `PeakMemory`, tile by tile.
- `ImageOf[Array]`: the op can work on a more general array. It gets a
  `LazyArray` proxy that fetches chunks from the host on demand.

An op that calls `np.asarray(image)` itself should instead declare
`ImageOf[np.ndarray]`, so the runner can do the conversion, and tile it.

If an op is passed a lazy `Array` it cannot handle, raising an error is the
op's job, not the runner's: the op can be called outside the runner too, so
it keeps its own checks.

## Chunking

What happens to each kind of op when it is chunked -- overlap, global ops,
labels, detectors, axes that stay whole -- is the same as for any tiling,
and is in [design.md](design.md).

## Cross-environment lazy array

All in skop's `_codec`; Appose is unchanged.

| Host value | Wire | Worker gets |
|---|---|---|
| `np.ndarray` | shared-memory `NDArray` | numpy view |
| other `Array` | `{"lazy": id, "shape", "dtype", "chunks"}` | `LazyArray` |

The shape travels inside the record, so the op signature stays `op(image)`.
`LazyArray` satisfies the `Array` protocol (`types.py`).

Fetching a chunk, using only what Appose has today:

1. `LazyArray.__getitem__` maps the region to chunk indices. Chunks already
   in the cache are served locally.
2. For missing chunks it calls `task.update(info={"want": ..., "id": n})`
   and waits on an event, with a timeout.
3. The host listener looks up the real array by its lazy id, reads the
   chunks into a shared-memory `NDArray`, and starts a small delivery task:
   `service.task(..., inputs={"id": n, "data": nda})`.
4. The delivery task runs on its own worker thread. It puts the data into a
   registry in skop and sets the event.

The worker side needs the task handle to call `update`, and `worker.py`
already has it.

## Cache

- An LRU of chunks, so pixel-by-pixel access costs one round trip per chunk,
  not per pixel. The budget comes from the op's `PeakMemory` (0017).
- Fetch in transfer blocks of several chunks when the zarr's chunks are tiny.
  Read ahead when access is in order.
- If the whole array fits in the budget, fetch it all at once.
- Writes (`Out`/`Mut`) go to a write-back cache. Dirty chunks return to the
  host through `update(info={"tile": nda})` on eviction and at the end of the
  task.

## Open

- The protocol relies on Appose running tasks on concurrent threads, which
  is how it behaves, not something it documents.
- Random access across chunks thrashes the cache. Nothing fixes that but a
  bigger budget.
