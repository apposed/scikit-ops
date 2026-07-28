# 0008 — Mask detector ops

**Status:** implemented. `skop.ops.mask` holds both detectors, `skop.masks`
holds the projections, and `Role.masks` is in the vocabulary (which amended
[0003](0003-semantic-roles.md)). What is *not* settled is how the per-mask
scalars travel; that is [per-object features](../spec/per-object-features.md),
deliberately kept separate.

## The kind

A mask detector takes an image and a set of boxes, and returns one mask per
box:

```python
@op(env=...)
def some_mask_detector(image: ImageData, boxes: BoxesData, ...) -> Masks
```

This is stage two of the segmenter in
[workflow-ops](../spec/workflow-ops.md), fed by a box detector
([0007](0007-box-detector-ops.md)). Because the two stages live in
different environments, what joins them is a workflow op.

It is a different kind from the instance segmenters in `ops/segment` — Cellpose
and StarDist take an image and hand back a label image, deciding for themselves
what the objects are. A mask detector is *told* where the objects are and only
draws their outlines. The prompt is the input, which is what makes the two
stages separable and the choice of detector interesting.

As with the box detectors, the two implementations must be substitutable, so
they share a signature and a return type. The shared core is `image` and
`boxes`; implementation-specific extras (`model_type`, `batch_size`) get
defaults and stay at the end.

## Two detectors, no new environments

Both ports are small, and both land in an environment that already exists.

**`mobilesam_masks`, in `envs/segment-everything`.** Joins `object_aware_yolo`,
which is the detector it was built to pair with — MobileSAMv2's prompt-guided
decoder consuming the object-aware model's boxes. The code is
`segment_everything/mask_detectors/mobilesam.py::segment_from_bbox`, a batched
loop over `SamPredictorV2`. We wrap `segment_everything` here rather than port
it, for the reason 0007 gives: the decoder is inside the vendored tree.

The model has to be assembled rather than requested, because
`weights_helper.create_mobile_sam_model` cannot be called: that module imports
toolz and `napari.utils.progress` at module scope and segment-everything
declares neither, which is the same wall `object_aware_yolo` hit. Assembling it
is six lines — an empty `vit_h` Sam, EfficientViT-L2 as its image encoder, and
MobileSAMv2's prompt encoder and mask decoder — and it needs two weight files
that both turned out to be reachable without gdown:

- `l2.pt`, 234 MB, from the same `huggingface.co/RogerQi/MobileSAMV2` mirror
  already in `WEIGHTS_URLS`, into skop's asset cache.
- `Prompt_guided_Mask_Decoder.pt`, 15 MB, which **ships inside the installed
  package** — it is the one entry in `WEIGHTS_URLS` marked `local:`. Nothing to
  download; the mirror carries a copy if the layout ever changes.

So the Google Drive path `weights_helper` uses is avoided entirely, and with it
the reason `gdown` was in the environment.

**`microsam_masks`, in `envs/pytorch`.** Joins `fastsam`. micro_sam is a normal
conda package, and `envs/pytorch` says in its own comments that
`ultralytics + cellpose >4 + micro_sam >1.6 + pytorch-gpu` resolves from
conda-forge alone — so this is the second occupant that environment was left
open for. Adding `micro_sam >1.6` to its `[dependencies]` is the whole
environment change. Weights are micro_sam's own pooch download into its own
cache (`MICROSAM_CACHEDIR`), so `skop.assets` is not involved.

This is also the first micro_sam op of what should become several — it carries
an automatic mask generator, a 3-D segmenter and finetuning. Only the mask
detector now; the point of putting it in `envs/pytorch` is that the rest cost
nothing to add later.

### What micro_sam actually needs

0008's earlier draft flagged `micro_sam.sam_annotator._state.AnnotatorState` —
a private singleton belonging to the napari annotator — and asked whether there
was a public path. There is, and it is the whole of what `AnnotatorState` was
doing:

```python
from micro_sam.util import get_sam_model, precompute_image_embeddings
from micro_sam.prompt_based_segmentation import segment_from_box

predictor = get_sam_model(model_type=model_type)
embeddings = precompute_image_embeddings(predictor, image, ndim=2, verbose=False)
mask = segment_from_box(predictor, box, image_embeddings=embeddings)
```

Use that. No singleton, no `reset_state()`, nothing private.

**Embeddings are the expensive part**, and they depend only on the image. One
op call per image, looping over boxes inside, pays that cost once and avoids a
round trip per box (workflow-ops, "round trips cost").
`precompute_image_embeddings` takes `pbar_init`/`pbar_update` callbacks,
which is where `skop.progress` hooks in.

### Boxes need no conversion for micro_sam

`prompt_based_segmentation._process_box` receives the box as
`[y0, x0, y1, x1]` and swaps it to SAM's `xyxy` itself. That is exactly the
canonical `BoxesData` order from 0007, so `microsam_masks` passes boxes
straight through. `mobilesam_masks` calls `boxes.to_xyxy` on the way in, the
same swap `object_aware_yolo` does on the way out.

`segment-everything`'s `microsam_detector` docstring says its boxes are
`[x1,y1,x2,y2]`; that is wrong, and only survives because its callers ran
`convert_yolo_boxes_to_microsam` first.

## What comes back

A SAM mask is boolean and SAM masks overlap, so a label image loses
information. The community answer is
`SamAutomaticMaskGenerator`'s list of annotation dicts — `segmentation`,
`area`, `bbox`, `predicted_iou`, `stability_score` — and both
segment-everything mask detectors already emit it. It is the right *shape* to
think in. It cannot be the wire format:

- `skop._codec` rejects `bool` outright. Appose derives element size from the
  dtype name, and `bool` has no digit in it. Masks have to be `uint8` to cross
  the boundary at all.
- Each array in each dict becomes its own shared-memory block. A 400-object
  detection would allocate 400 blocks and 400 `NDArray` handles where one
  stacked array allocates one.

So the collection is a stack, and the per-mask scalars are dropped for now:

```python
class Masks(NamedTuple):
    masks: MasksData    # (N, Y, X) uint8, 0 or 1; may overlap
    boxes: BoxesData    # (N, 4) the prompt each mask came from
```

`boxes` is not redundant. Both originals drop empty predictions, so N out is
not N in, and without the surviving prompts the correspondence to the detector
stage is gone.

`area` is `masks.sum(axis=(1, 2))` and does not need carrying. `predicted_iou`
and `stability_score` are MobileSAM's and have no micro_sam counterpart
(`segment_from_box(..., return_all=True)` gives a score, but not the same one),
so they cannot go in a shared return type until
[per-object features](../spec/per-object-features.md) says how per-object
values travel. Same reasoning that dropped `scores` from `Boxes`, and this is
the second real case that document was waiting on.

**The cost worth naming:** `(400, 1024, 1024)` uint8 is 400 MB. `StackedLabels`
had the same appetite and nobody minded, so this is not a regression — but RLE
is the escape hatch when it bites, and the community dict format already
supports it.

## The role

`Masks` needs a role, and it is a new one: `Role.masks`, exposed as `MasksData`
in `skop.types`.

This bends 0003's rule that a role names a napari layer type — but far less
than `Role.features` would. A mask stack is not a layer, yet it maps onto one
*deterministically*: a front end projects it to a Labels layer. The role still
answers "which layer does this become", it just answers with a conversion
rather than an identity. `Role.features` had no answer at all. Amend 0003 to
say so rather than let the contradiction sit quietly.

The alternative is annotating it `LabelsData` and letting napari open a 3-D
Labels layer the user scrolls through. That works and is honest about the
array, but it makes every front end guess whether a 3-D labels array is a
z-stack or a mask stack, which is the guessing 0003 exists to prevent.

## `skop.masks`, a module of utilities

Projection and ordering are **utilities, not ops** — the same call 0007 made
for `skop.boxes`, and for the same reason: making them ops buys a
shared-memory round trip and a worker to transpose an array. `src/skop/masks.py`,
numpy and standard library only, so a worker, the host, and skop-napari can all
import it freely.

That module is where the useful half of `segment_everything/stacked_labels.py`
lands, as functions over arrays instead of methods on a class:

| `StackedLabels` | `skop.masks` |
| --- | --- |
| `make_3d_labels()` | `to_labels_3d(masks)` — `(N, Y, X)`, slice *i* filled with *i+1* |
| `make_2d_labels(type=)` | `to_labels_2d(masks, strategy="min")` — the projection that loses overlap |
| `sort_largest_to_smallest()` | `order_by_area(masks)` — returns the permutation |
| `from_2d_label_image()` | `from_labels(labels)` — the inverse, for Cellpose or StarDist output |

Four functions, deliberately. `to_labels_2d` is what skop-napari calls to show
a `MasksData` output, and `strategy` is where the overlap goes: `"min"` lets
the lowest label win a contested pixel, `"max"` the highest.

Neither means anything without an order. Sorted largest-first by
`order_by_area` — the order `StackedLabels` imposed in its constructor — the
largest object holds label 1, so `"min"` gives contested pixels to the
**largest** mask and an object drawn wholly inside another vanishes from the
projection; `"max"` gives them to the **smallest**, and every mask still
appears somewhere.

`"min"` stays the default because it is what `StackedLabels` did, not because
it is the better answer — it suits a big mask whose nested masks are fragments
of it, and is wrong when the nested masks are the objects. That neither is
safe to assume is the argument for the front end exposing the choice rather
than skop picking one.

`order_by_area` returning a permutation rather than sorting in place is a
deliberate fix. `StackedLabels.__init__` sorted its `mask_list` by area, which
silently desynchronised it from any box array the caller still held. An index
array reorders `masks` and `boxes` together, or neither.

Filtering, likewise, is `masks[keep]` — no `keep` flag threaded through every
dict and no `filter_labels_3d_multi` writing zeros into a cached label image to
express it.

## Choosing the projection is the front end's call

The op always returns the collection. Never a label image, and never a
*parameter* asking which label image — `MasksData` out of every mask detector,
every time. Which projection to look at is a display decision, and the op does
not know what it is being displayed in.

skop-napari makes that decision, and makes it changeable: a dropdown on the
widget, attached to the `MasksData` output, offering

- **2-D labels** — `to_labels_2d`, one Labels layer, overlap resolved by
  `strategy`. The default, and what people usually mean by "the segmentation".
- **3-D stack** — `to_labels_3d`, one Labels layer of `(N, Y, X)`. Nothing is
  lost, overlapping objects sit on separate planes, and it is genuinely worth
  rotating.
- **both** — two layers from the one output.

with `strategy` (`"min"` / `"max"`) beside it, since that is the only knob that
changes what the 2-D picture actually says.

**The reason this belongs downstream of the op:** changing the dropdown is a
numpy call on a result the host already has. If the projection were an op
parameter, every change would be another worker round trip and another SAM run
for a decision that touches no model.

Two consequences worth naming:

**One output can become two layers.** "Both" breaks the output-to-layer
correspondence that skop-napari otherwise has, and whatever handles it needs to
name the layers distinguishably and replace the right pair on re-run.

**The stack's first axis is not space.** `N` is a mask index. napari will
happily rotate a `(N, Y, X)` Labels layer and it will look like a volume; it is
not one, and nothing measured along that axis means anything.

Which means its spacing has no true value, and 1 is a bad guess: ten masks over
a 512-pixel image at unit spacing is a pancake, and rotating it shows nothing.
skop-napari therefore gives the stack `scale=(z, 1, 1)` with `z` adjustable and
defaulting to 10, and leaves y and x at 1 to match the image. The number is a
viewing preference, not a property of the data.

Two things left open, both wanted and both bigger than they look:

- **It applies at layer creation.** Changing the spacing, or the projection,
  currently means running the op again to see the difference — when both are
  pure functions of a result the host already holds. Re-styling or re-projecting
  an existing layer is its own piece of work.
- **The axis still has no name.** It wants to read `index` rather than showing
  up as a `z` slider, which is [0006](0006-axis-mapping.md)'s business and has
  no per-layer answer in napari today.

None of this adds machinery to skop. The role already obliges a front end to
know that `MasksData` converts before it can be shown; the dropdown just
exposes that conversion's parameter. `OutputSpec` gains nothing. If a second
role ever wants display options, that is the point at which to generalize —
not before.

## Not porting

- `BaseDetector` / `BaseMaskDetector` — ops are functions (0001).
- `StackedLabels` the class. Its state is `image`, `mask_list`, two cached
  label images and a `keep` flag; as arrays plus four functions, all of that is
  the arrays.
- `create_mask_from_xywhn_bbox` / `create_mask_from_xyxy_bbox` /
  `read_yolo_txt` — `skop.boxes` plus one slice assignment.
- `add_properties_to_label_image`, `filter_labels_3d_multi`,
  `filter_labels_hue_inverse` — these are per-object features and filtering by
  them. They belong to [per-object features](../spec/per-object-features.md),
  and they are the best evidence in the old code for how much that decision is
  holding up.
- `stacked_label_dataset.py` — a torch `Dataset` that jitters prompt boxes for
  SAM finetuning. That is deep-learning training ops — a document not yet
  written — not this.
- `add_background_results` — also finetuning: empty masks to suppress false
  positives. Same document.

## Errors to fix on the way through

Small things, worth doing while the code is being retyped and not after:

- `area = sum(sum(segmentation))` in `segment_from_bbox` — Python's `sum` over
  a 2-D array, one row at a time. `.sum()`.
- `create_mask_from_segmentation` stores `indexes` — the full `np.where` of
  every mask, an int64 pair per foreground pixel, used by nothing that survives
  here.
- `StackedLabels.get_bbox` returns an inclusive `max`, while
  `skop.boxes.from_labels` uses the half-open convention of `regionprops` and
  of slicing. Half-open wins; it is the one that composes.
- `make_3d_labels` reads `mask['keep']`, which only exists if the masks came
  through `__init__` — `add_segmentation` and `add_background_results` both
  produce dicts without it.
- `segment_from_bbox` narrows its cached `image_embedding` and
  `prompt_embedding` *in place* on every iteration
  (`image_embedding = image_embedding[0 : boxes.shape[0]]`), so they can only
  ever shrink. It survives because the short batch is always the last one; the
  port slices from the full repeat instead. `test_mobilesam_batches_without_
  changing_the_answer` is the assertion that this made no difference.
- The same function moves boxes to the device with an `if cuda / elif cpu`
  chain, which leaves them a numpy array on `mps` — the one device branch
  `get_device` can return that it does not handle.
- It computes a stability score and a predicted IoU per mask, the first of
  which costs an extra thresholding pass. Neither is returned by `Masks`, so
  the port does not compute the stability score at all.
