# Lazy arrays across the process boundary

Status: proposed. Nothing here is built.

## The problem

Only numpy (through shared memory) and plain JSON cross into a worker. 

A zarr,
dask or xarray array depends on things that exist only in the host: a store
and its codecs, a task graph, GPU memory. The op environment may not have zarr
at all.

Current situation: runner converts any non-numpy input to numpy before sending
(`runner.py`, `np.asarray`), and `_codec` then copies it into shared memory.

two full copies at peak??
- array must fit in RAM.

## Four approaches

1. **Convert to numpy, optionally in chunks.** Today's behaviour, plus
   chunking driven by `PeakMemory` and overlap (below). Works for any op.
   Global ops cannot be chunked: Otsu on each chunk picks a different
   threshold per chunk, so the mask shows seams at chunk edges. A niche
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

##  Templated annotation

  ie ImageOf[T]
 
- `ImageOf[np.ndarray]`: Op expresses it needs numpy. By default op converts all of it up front.  
  
  Proposed:  op optionally declares `PeakMemory` (0017), the runner uses
  it to chunk the input instead of converting the whole array at once.
- 
`ImageOf[Array]`: the op expresses it can operate on more general array. It gets a `LazyArray` proxy
  that fetches chunks from the host on demand.

An op that explicitly calls `np.asarray(image)` should instead declare type is `ImageOf[np.ndarray]`.
Then the runner can handle the conversion (possibly with chunking).  

If op is passed a lazy `Array` we could raise an error but that error checking should be up to op.
For example op could be run outside of runner, so could still have option of error checking itself.  


## Chunking consideration:

- Pixel-wise ops (scaling, a fixed threshold) chunk freely.
- Neighbourhood ops (filters, morphology, deconvolution) need chunks that
  overlap by the kernel radius, with the overlap trimmed off afterwards.
  `PeakMemory` does not say how much overlap that is.
- Global ops (Otsu, normalising by min and max, histograms) give a
  different answer on each chunk. They cannot be chunked.
- Labelling and segmentation split objects at chunk edges, so labels
  repeat and need stitching. Detectors find boxes twice in the overlap.
- A model trained at one scale may see too little context in a small chunk.
- Some axes must stay whole, such as channels or z for a 3D model.

Alongside `PeakMemory` op also needs to declare overlap.



## Cross environment lazy array approach

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
