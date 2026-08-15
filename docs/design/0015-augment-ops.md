# 0015 — Augment ops

**Status:** proposed. Nothing built. `grep -i augment src/` returns nothing
today; the only mention anywhere is [design 0011], points 5 and 6.

Augmentation is named in 0011 as a step inside the training story. This
document argues it should not be — it is its own kind of op, useful before any
training op exists, and it is the one piece of the training picture that needs
no new machinery.

## The claim

**Augmentation is a separate step, not a phase of training.**

If you turn 40 annotated regions into 4000 patches on disk, then from the
training op's point of view there are simply 4000 pairs in a directory. It has
no interest in how they got there, and nothing in its signature should imply
it. The alternative — training that knows how to augment — makes every training
op carry a second, unrelated body of code, and makes the 4000 patches
unreproducible without re-running training.

The genuinely different case is augmentation *inside* the training loop, where
the framework's own dataset pipeline resamples every epoch (0011 point 6).
That is not this op. It is a parameter on a training op, expressed in whatever
vocabulary that framework already has, and the two should not be unified — one
returns arrays, the other is a knob on a process that runs for two hours.

So:

| | Where |
| --- | --- |
| Pre-training augmentation | An op. This document |
| Inline augmentation | A parameter on the training op. Per-framework |
| Writing N patches to a directory | The caller. Not skop |

## The kind

One pair in, one or more pairs out.

Its own environment rather than an existing one, because albumentations has no
anchor — no CUDA, no python pin, no hard numpy ceiling — and joining a heavy
environment would couple it to a solve it has no business in. The reasoning is
[0016](0016-choosing-an-environment.md), which uses this op as its worked
example.

```python
@op(env="albumentations")
def albumentations(
    image: Annotated[ImageData, Axes("y", "x", "c?")],
    truth: LabelsData,
    patch_size: tuple[int, int] = (256, 256),
    n: int = 1,
    seed: int = 0,
    horizontal_flip: bool = True,
    vertical_flip: bool = True,
    rotate90: bool = True,
    scale_limit: float = 0.1,
    brightness_contrast: float = 0.0,
) -> AugmentedPairs
```

```python
class AugmentedPairs(NamedTuple):
    images: ImageData   # (n, *patch_size, c?)
    truths: LabelsData  # (n, *patch_size)
```

Field names become output names, per [0001].

Three things about this signature are the whole design.

### 1. The truth's role does the work

0011 point 2 says the op must know what form the truth takes. It already can:
[`Role`][0003] carries exactly that, and needs no new members.

| Truth role | How the transform applies |
| --- | --- |
| `labels` | Same geometry as the image, nearest-neighbour only. Never interpolate a label |
| `image` | Same geometry, same interpolation — the restoration case |
| `shapes`, `points` | Coordinates transformed, not pixels |
| a scalar or table | Geometry does not apply; passed through — the classification case |

This is the argument for augment being an op at all: the hard part of
augmentation is keeping the truth consistent with the input, a flip applied to
one and not the other is silent corruption, and skop already has the vocabulary
that says which consistency rule applies. A caller passing `LabelsData` has
said, in the type, that nearest-neighbour is required.

The first op is segmentation-shaped (`ImageData`, `LabelsData`). The
restoration variant is the same body with a different truth role, and is the
test of whether the role really is carrying the weight.

### 2. Determinism is the caller's, so `seed` is a parameter

An op is an ordinary function ([0001]). One that seeds itself cannot be
replayed, cannot be cached, and produces a patch set that nobody can regenerate
— which makes a bad patch unfilable as a bug. Given `(pair, params, seed)` the
output must be fixed.

The caller owns the sequence. Producing 4000 patches means 4000 seeds it chose
and can write down, not 4000 calls hoping for variety.

This is a real change from the reference implementation, which calls
`np.random.randint` inside the augment body.

### 3. `n` exists because of the wire

An op may run in another environment, and 4000 round-trips of full arrays
through that boundary is the wrong granularity. `n` amortises it: ask for 64
variants of one pair in one call. The op stays stateless — `n` is a count, not
a schedule — and `n=1` is the honest default for a caller that does not care.

The alternative, a sticky `exclusive=True` worker called once per patch, keeps
the signature smaller but pays per-call overhead forever. Prefer `n`.

## What this op must not know

The line is the same one [0003] draws: skop must never learn what a project is.

| Stays with the caller | Why |
| --- | --- |
| Which pairs exist, and which are annotated well enough to use | Project knowledge. Often the bounding boxes of 0011 point 4 |
| Where a valid crop may start | Needs the sparse annotation to answer |
| How many patches, and the seed sequence | A schedule. An op has no schedule |
| Filenames, directory layout, a manifest | Not data |
| Normalization | Already an op — `skop.ops.normalize.percentile`. Composed by the caller, not folded in here |

The last row is worth stating out loud because the reference implementation does
normalize inside augmentation. Two ops composed beats one op with a `normalize`
flag, and skop already has the other one.

**Where the crop is drawn is the caller's; the crop *within* it is the op's.**
The caller hands over a region it knows is annotated; the op does the final
random crop to `patch_size` as part of the pipeline, because scale jitter and
warping need margin and cannot be done after a tight crop.

## Who this is for, and why the loop is not an op

The test that settles the granularity: **someone outside ai-lab, with their own
labeling tool, who wants to augment and train.**

What they cannot write themselves is the transform — image and truth kept
consistent, nearest-neighbour on the labels, intensity changes not applied to
them at all. A flip applied to one and not the other is silent corruption, and
that is worth borrowing.

What they can write themselves is the loop. It is three lines, and the parts of
ours that are not three lines are all things they do not want: our patch
directory layout, our manifest, our idea of which regions are annotated well
enough to crop from. Turning the loop into an op would export our project
conventions under the name of a computation.

So the op is one pair in, `n` pairs out, and the caller loops — in chunks of
`n`, which is what keeps the wire cost off them without giving them a schedule.

**The reusable asset is the recipe, not the loop.** If the parameter set is a
plain serializable value, then a form in ai-lab builds one, a script builds one,
and both get identical output from the same op and seed. That makes a training
set something you can ship as a config rather than as pixels, and it is what an
outside caller actually reuses. See the open question below — flat booleans get
there for one pipeline, and stop working as soon as order matters.

## Open questions

- **3-D.** Albumentations is 2-D. The reference implementation handles a volume
  by picking a random z-range and replaying one 2-D transform across the slices
  (`A.ReplayCompose`) — which is consistent, but is not a 3-D augmentation, and
  cannot do anything anisotropic. Whether the first op admits this by declaring
  `Axes("y", "x", "c?")` and letting the caller loop, or takes a volume and
  documents the per-slice behaviour, is unsettled. Declaring 2-D is more honest
  and is what [0006] would have us do.
- **A second implementation.** An n-D random crop with flips needs no
  albumentations at all, works in 3-D natively, and could live in the `skimage`
  environment. If there are two, they must be substitutable in the sense of
  [0007] — same output shape, same seed contract — which is a constraint on the
  first one's signature and so is worth deciding before it lands.
- **Environment.** A new `envs/albumentations`, or add it to an existing one.
- **How the params are expressed.** The flat booleans above are the reference
  implementation's parameter set flattened. A pipeline is really a list of
  transforms, and a front end generating a form from a flat signature cannot
  express "elastic, then flip, then crop". Flat is the right first cut; a
  composable spec is the thing it will eventually want — and per the section
  above, the recipe is the part outside callers reuse, so its shape matters
  more than the rest of the signature.

## Why this is worth doing before training ops

It needs nothing that does not exist. No path role, no session handles, no
long-running worker — arrays in, arrays out, in an environment. It is the one
part of 0011 that is buildable today, and building it settles the truth-role
question that the training ops will need answered anyway.

[design 0011]: 0011-deep-learning-training-ops.md
[0001]: 0001-ops-are-plain-functions.md
[0003]: 0003-semantic-roles.md
[0006]: 0006-axis-mapping.md
[0007]: 0007-box-detector-ops.md
