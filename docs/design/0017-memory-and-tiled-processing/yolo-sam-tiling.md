# Tiling for YOLO and SAM

Status: detector tiling built; runner-level tiling and SAM remain proposals.
Started from reading JDLL's `DetectionMerger`
([jdll-reuse.md](../../spec/jdll-reuse.md)).

Update: `skop.ops.detect.yolo:yolo` implements detector tiling locally to
the op, with object area, configurable overlap, checkpoint input size and
measured GPU batching. It shifts boxes into image coordinates and suppresses
same-class duplicates using intersection over the smaller box, preferring
whole boxes over fragments on internal tile edges. The general runner design
and tiled SAM below remain proposals.

**Tiling is not optional for detectors.** There are two separate reasons to
tile, and only the first one is about memory.

## Two reasons to tile

### 1. Memory

The image does not fit, so cut it up. This is
[0017](README.md): the op declares its
working set, the caller solves for the biggest tile that fits. The tile size
comes from the **budget**.

### 2. Scale — matching `imgsz`

A YOLO model does not see the image you give it. It sees the image **resized
so the longest side is `imgsz`** (640 for stock YOLO, 1024 for FastSAM and
the object-aware model). The model only finds objects at roughly the pixel
size it was trained on.

- **Big image, small objects.** A 4096² image with 20 px cells, resized to
  1024, turns the cells into 5 px blobs. The detector misses most of them.
  Raising `imgsz` to 4096 works if it fits, but then you're back at reason 1.
  The fix is **more tiles**, each about `imgsz` in size, so every cell stays
  20 px.
- **Small image, big objects.** A 512² crop holding three 300 px objects gets
  upscaled to 1024, so the objects come out bigger than anything in training.
  The fix is **fewer or larger tiles**, or a downsample. Here a tile can be
  bigger than `imgsz`.

So the tile size comes from **object size**, not the budget:

```
tile_side ≈ imgsz × (object_px_in_image / object_px_the_model_expects)
```

where an object in the image is `object_px_in_image` pixels across and should
end up `object_px_the_model_expects` pixels across after the resize. Only the
user knows the first number (by drawing a box, or from the pixel size), and
the model's training data decides the second.

The two reasons interact. Scale picks a tile, memory caps it, and if the
scale tile does not fit then the scale is wrong and nothing fixes that.

## How JDLL does it

`model/tiling/merger/DetectionMerger.java` has three ways to choose tiles:

| mode | tiles from |
| --- | --- |
| `TILE_MAKER` | JDLL's generic memory-driven tiler |
| `PATCH_SIZES` | a fixed tile size given by the caller |
| `OBJECT_SIZE` | **an example object** the user draws (`Yolo.setObjectSize(Rectangle)`) |

`OBJECT_SIZE` is reason 2, and it's the interesting one:

- If the object covers ≥ 0.1% of the image, use one tile, the whole image.
- Otherwise tile so the object covers about 1% of a tile:
  `tile_side = object_side × √(1/0.01) = 10 × object_side`.
- 15% overlap; the last tile in each row or column is pinned to the edge, not
  padded.

Merging: shift each tile's boxes to global coordinates, then **class-aware
NMS** at IoU 0.5 over everything.

Notes on that:

- The "1% of tile" rule ignores `imgsz`. A 10 × object tile is then resized
  to `imgsz` anyway, so the object ends up `imgsz / 10` px across (64 px at
  640, about 100 px at 1024) whatever the model was trained on. That's a
  reasonable default, but it's a constant standing in for the training scale.
- **Partial boxes at tile edges.** An object cut by a tile edge gets a
  truncated box. NMS only removes it if its IoU with the full box is > 0.5,
  and a box cut in half has IoU about 0.5 with the whole one. Whether
  `toGlobalDetection` drops edge-touching boxes first: not yet read. This is
  the standard tiled-detection problem; SAHI's fix is to keep the box whose
  centre is in the tile's interior, or merge with IoS instead of IoU.

## SAM has the same problem

`detect_then_mask` passes boxes to SAM (`mobilesam_masks`). SAM's
`set_image` also resizes the longest side to 1024. A 20 px box in a 4096²
image becomes a 5 px prompt on an embedding grid of 64², so each grid cell
covers 16 px, and the mask comes out as a blob.

Same fix, but the tiles come from the boxes: **crop around each box (or
cluster of boxes) with some margin, run SAM on the crop, paste the mask
back.** One embedding per crop rather than one per image, which costs more
encoder passes but keeps resolution. Crop size uses the same scale rule as
above, with SAM's 1024.

So a tiled YOLO → SAM pipeline is:

1. Choose detection tiles from object size (reason 2), capped by memory (reason 1).
2. Detect per tile, shift, merge (NMS, with edge handling).
3. Crop per box or box cluster at SAM's scale, mask per crop, paste back.

## How this fits skop

- **Boxes don't stitch, they merge.** 0017 imagines a dense op: tile,
  run, stitch with a halo. A box detector's output is a list, and merging is
  shift + NMS. So tiling needs a merge rule per output kind, and box
  detectors ([0007](../0007-box-detector-ops.md)) would declare
  theirs. JDLL has the same split: `DenseMerger` vs `DetectionMerger`.
- **`imgsz` is the op's native scale.** Right now it's an "implementation
  extra" parameter at the end of the signature (0007). For tiling it's the
  number the caller needs, so it may want to be a declaration rather than
  only a knob.
- **Object size is the caller's**, the same way axis mapping is
  ([0006](../0006-axis-mapping.md)). It's something the user knows
  about this image, and in napari it's a drawn box. JDLL's
  `setObjectSize(Rectangle)` is exactly that.
- Training is the other half. A model trained on 640 tiles of the user's own
  data at native resolution expects objects at that scale, so the tile rule at
  inference should mirror the tile rule at training (0011).

## Open

- Edge handling in the merge: drop edge-touching boxes, IoS, or both.
- Where the scale rule lives: a wrapper the caller applies, a workflow op,
  or inside each detector.
- Whether `imgsz` becomes a declaration on the op.
- Per-box SAM crops vs per-tile SAM: crops are sharper; tiles reuse one
  embedding for many boxes.
- 3-D: JDLL's YOLO is 2-D only (`// TODO add 3D`); so is ours.
