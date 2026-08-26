# 0011 — Deep learning training ops

**Status:** built for StarDist 2D — `skop.ops.train.stardist2d`. Cellpose and
3D are not written; the open questions at the bottom stand. The considerations
below came first; the decisions after them came out of planning the first
real one, StarDist 2D, against the existing training code in
`../napari-ai-lab/src/napari_ai_lab/Segmenters/GlobalSegmenters/StardistSegmenter.py`.

Segmentation is the first case we will tackle, so the examples lean that way,
but the same considerations apply to restoration (denoising, deconvolution,
super-resolution), classification, and registration.

## What makes training different

1.  They require input/truth pairs. For segmentation these are images and the ground truth label. For restoration they are typically a low quality image and a high quality counterpart, for example a noisy acquisition paired with a long exposure or averaged version of the same field. For classification the truth is a label per image or per object.

2.  The truth can take many forms. For segmentation it can be a semantic label image, an instance label image, or shapes (for example bounding boxes or points representing the location of objects). For restoration it is another image of the same shape as the input. For classification it is a scalar or a table of values. The op needs to know which form it is getting.  For super-resolution the input is a low resolution image and the truth a high resolution image.

3.  This type of data could be represented in Napari as viewer layers, but is more often curated and stored on disk.

4.  If the data is in Napari, it is often convenient to use bounding boxes to mark good areas. For example, it may be tedious to label all the data, so you label some of it and mark which areas are labeled to satisfaction. The same trick is useful for restoration, where you may only have well registered input/truth pairs in part of the field.

5.  After step 4 there can be an intermediate step where the truth in Napari is augmented to create a set of patches with more variation. From a single label, hundreds of patches could be created, and this is often the data we want to use for training. Note that augmentation has to be applied consistently to the input and the truth whenever the truth is spatial, which covers both label images and restoration targets.

6.  Alternatively, the augmentation is often done inline with the training. The advantage of doing it pre-training is that you can sometimes use slower but more powerful augmentations, like warping and color augmentation. The disadvantage is that you may have a fixed number of augmentations. The advantage of inline is an infinite number of variations.

7.  So the training op often has to take a file location as input(s), where the truth will be put in a specific format compatible with the formats the training op can read. The format varies by task and by framework, so this is a per-op concern rather than something we can settle once.

## The decision: the caller resolves the files, the op receives a list

Point 7 is where the trouble is. A directory only means something once you
know a convention, and whoever ships that convention has made everyone else's
data wrong. The convention in napari-ai-lab today is `input0/`,
`ground_truth0/` and an `info.json` carrying `axes`. It is a perfectly good
convention. It is also *one* convention, and putting it in scikit-ops would
put the first opinion into a library that has managed to avoid having any.

So the op does not take a directory. It takes the resolved result:

```python
@op(env="stardist-tf")
def train_stardist2d(
    images: list[str],          # paths, in order
    labels: list[str],          # labels[k] is the truth for images[k]
    model_dir: str,
    name: str,
    image_axes: str = "YX",
    epochs: int = 100,
    ...
) -> str:                       # path to the saved model
```

The caller globs, matches, and orders; the op reads and trains. The layout
logic runs in the host process and its *return value* is what crosses the
worker boundary — two lists of strings, a few hundred KB for a few thousand
patches, serialized like any other argument. `_codec.py` already turns a
`Path` into a string, but pass `str` and be explicit.

Both of the obvious naming schemes collapse to the same argument:

```python
# same name, different folders
imgs = sorted((d / "input0").glob("*.tif"))
lbls = [d / "ground_truth0" / p.name for p in imgs]

# same folder, suffix distinguishes
imgs = sorted(d.glob("*_input.tif"))
lbls = [p.with_name(p.name.replace("_input", "_patch")) for p in imgs]
```

**Pairing is positional, and that is the only contract.** Derive the second
list from the first rather than sorting both independently — two independent
sorts agree until one directory has an extra file, and the symptom is silently
mispaired training data rather than an error.

`image_axes` crosses as a plain string. That is data *about* the data, not
a convention, and the op genuinely needs it: `YX` and `YXC` want different
`n_channel_in` and different reshaping.

Not named `axes`, though it wants to be. `Runner.run` has an `axes`
parameter of its own — labels for the *arrays being passed across*, so it
can transpose or iterate them. A bare `axes=` kwarg binds to that one and
the op never sees it, which surfaces two frames later as
`'str' object has no attribute 'items'`. The two are not the same concept
and cannot be merged: this op takes paths, not arrays, so there is nothing
for the adaptation machinery to act on and the op must carry its own.

### What this buys

- `input0` appears nowhere in scikit-ops. Not as a string, not as a `layout=`
  parameter, not as a default. The "why should scikit-ops carry my way"
  problem does not arise, because it does not carry one.
- A list of pairs *is* the general form. Every layout anyone can invent
  reduces to "these images go with these labels", so a second user writes
  eight lines in their own code rather than a pull request here.
- No arrays cross the boundary. The worker opens the files itself.
- Whether the worker reads eagerly or lazily is invisible in the signature,
  so it is not a decision owed to anyone up front. See *Not in memory* below.

### What it rules out

An op taking `(X, Y)` arrays directly. It was tempting as the format-free door,
but it pays a serialization tax on every call to buy something the list of
paths already provides. Someone doing genuinely in-memory training — a toy
problem, a notebook — calls the underlying function rather than the op.

## The output is a path

The op returns the directory the model was written to. A trained TensorFlow or
torch model cannot cross the worker boundary, so returning the object is not
an option that exists, and pretending otherwise would be a signature that
lies. `model_dir` and `name` come in as parameters: the caller owns the
project layout, so the caller decides where the model lands, not the op.

## Per-framework adapters are the part worth sharing

The manifest is the same for every framework. What each one wants from it is
not, and that difference is real work worth having in one place:

| | What the framework takes | Adapter work |
| --- | --- | --- |
| StarDist | `X, Y` as lists of arrays | read the tifs, append the trivial channel axis, `-1` offset when labels are sparse |
| Cellpose | `train_files`, `train_labels_files` — the paths themselves | pass them through |

`cellpose.train.train_seg` accepts lists of file paths natively with
`load_files=True`, so the manifest is already its input form. StarDist has no
file API at all — `StarDist2D.train()` takes `X` and `Y` and never reads from
disk — which is exactly why the convention question landed on the caller in
the first place.

Train/val splitting belongs on this side too. Everyone needs it and nobody has
a convention about it.

## Not in memory

StarDist's `train()` hands `X` and `Y` to `StarDistData2D`, which pulls one
item at a time. Every eager operation in it — the float32 cast, the shape
checks — sits behind `isinstance(X, (np.ndarray, tuple, list))`, and
`__getitem__` only ever does `self.X[k]`. So any object with `__len__` and
`__getitem__` returning an array works; a `keras.utils.Sequence` subclass is
documented but not required.

That makes "training data larger than RAM" an internal change to the op rather
than a signature change. Two gotchas when it is worth doing: `X[0]` is read
eagerly at construction for shape and channel inference, and `get_valid_inds`
re-reads `Y[k]` on every cache miss, so a lazy reader wants a small cache or
the disk reads multiply across epochs.

## Augmentation comes after, not before

[0015] argued augmentation should be built first, as the piece of this
document that needs no new machinery. That ordering is reversed, and 0015 now
records it: needing no new machinery is what makes augmentation *not* urgent.
albumentations installs in the host without a fight, so a caller who wants
augmented patches today just imports it. StarDist's TensorFlow stack cannot go
in the host at all, which is what ops are for. Training first.

One finding for 0015 when it does happen: StarDist's inline hook is
`augmenter=`, a Python callable `(x, y) -> (xt, yt)`. A callable cannot cross
the worker boundary, so "inline augmentation is a parameter on the training
op" cannot mean *passing a function*. It has to be a name or a spec the worker
resolves on its own side.

## What is still open

- **Paths mean the worker's filesystem.** Same machine, fine. The day a
  genuinely remote environment appears, a manifest of local paths is
  meaningless there and something has to move the bytes. Keeping the resolver
  a single small function on the caller's side is what makes that survivable.
- **Progress and cancel** are solved in principle — `skop.progress()` exists,
  and StarDist's Keras callback maps onto it — but a two-hour run surviving a
  restart is the caller's problem, not skop's.
- **3D.** `Config3D`, rays, and anisotropy are a second op, not a flag.
- **Truth that is not a label image.** Points 2 and 4 above are untouched by
  this design. Boxes-as-truth and "only these regions are labeled" still need
  an answer; the sparse `-1` convention is StarDist's version of the latter.

[0015]: 0015-augment-ops.md
