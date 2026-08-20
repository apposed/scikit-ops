# 0017 — Memory requirements and tiled processing

**Status:** proposed, and barely that. Nothing built, nothing designed. This
document exists so the problem is written down rather than rediscovered.

## The problem

An op that fits in memory for a 512³ crop does not fit for the 4096³ volume it
was really meant for. Today nothing in skop says so — the caller finds out
when the process dies.

Richardson-Lucy is the example that makes it concrete. `richardson_lucy`
(design 0014) is not an in-place filter: it holds the estimate, the blurred
estimate, the ratio, the correction and the FFT plans, all as float32 or
complex64, and `noncirc=True` pads on top of that. The working set is roughly
**7–10× the input array**, so a 1 GB image wants 7–10 GB. A user with 8 GB of
VRAM asking for `richardson_lucy_cupy` on a 2 GB stack is asking for something
that cannot happen, and there is currently no way for anything to know that
before it fails.

## What is missing

Two halves, and the first is the one skop owns.

**1. The op declares what it needs.** Something answering "for an input of
this shape and dtype, how much working memory will you use" — a multiplier, a
function, or a small structure; undecided. It has to cover the compute buffers,
not just the input and output, because the buffers are the whole point.

**2. The caller inverts it.** Given a memory budget, solve for the largest
tile that fits, then run the op tile by tile with whatever overlap the op
needs. The tiling loop is generic and belongs beside the other adaptation
machinery; the overlap is not, since a convolution needs a halo and a
per-pixel op needs none. So the declaration probably carries both: cost and
halo.

## Why it is not just a decon problem

Deep-learning inference has the same shape — a UNet's activations dwarf the
patch — which is why tiled prediction is hand-rolled in every framework. If
this hook exists, `can_process`-style questions ("will this run here at all?")
and tiled execution are the same declaration read two ways: axes for whether
the op applies, memory for whether it fits.

## Not decided

Everything. In rough order of what to settle first:

- The form of the declaration: constant multiplier, callable, or a structure
  with a halo beside it.
- Whether the budget comes from the caller, from probing the device, or both —
  CPU RAM and VRAM are different questions and cupy ops care about the second.
- Whether tiling is a wrapper the caller applies, or something an op can
  advertise it does internally.
- Whether the multiplier is measured or declared. Declared is a guess that
  drifts; measured needs a benchmark that runs somewhere.

## Related

- [0014](0014-make-decon-ops.md) — Richardson-Lucy, the motivating op.
- [0006](0006-axis-mapping.md) — the precedent for an op declaring something
  about its inputs and the caller acting on it.
- napari-ai-lab spec 0006 (batch segmentation over a sequence), whose
  `can_process` question is the same declaration seen from the axis side.
