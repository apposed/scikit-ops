# Spec — form adaptation

**Status:** not implemented. A computer or inplace op must currently be handed
its buffers by the caller.

## What exists

[Design 0001](../design/0001-ops-are-plain-functions.md) defines three
computation forms, inferred from the signature:

| Form | Signature |
| --- | --- |
| function | `def f(image) -> Result` |
| computer | `def f(image, out: Out[np.ndarray])` |
| inplace | `def f(image: Mut[np.ndarray])` |

`OpSpec.form` reports which. `Out` parameters are hidden from generated GUIs,
which means a GUI currently *cannot run a computer op at all* — it refuses to
ask the user for a buffer, and has nothing else to supply.

## What is proposed

Adapt between forms, as SciJava Ops does, so that a caller asks for the form it
wants and skop bridges the gap:

- **computer → function.** Allocate the output buffer, call, return it. Needs a
  way to know the output's shape and dtype, which is the hard part: in general
  it depends on the inputs.
- **function → computer.** Call, copy the result into the caller's buffer.
  Cheap and always correct, but wasteful.
- **inplace → function.** Copy the input, mutate the copy, return it.
- **function → inplace.** Call, copy the result over the input.

## The open question

Output allocation. `def f(image, out: Out[np.ndarray])` does not say how `out`
relates to `image`. Options, roughly in order of how much they ask of an op
author:

1. **Assume same shape and dtype as the first array input.** Correct for the
   large majority of image-processing ops, silently wrong for the rest
   (projections, resampling, anything producing coordinates).
2. **A declared relationship**, e.g. `Out[np.ndarray, SameAs("image")]`. Honest
   and checkable, but a new vocabulary to design and learn.
3. **An allocator hook** on the op: `@op(..., allocate=lambda image: ...)`.
   Fully general, and puts a lambda in the decorator, which
   [0001](../design/0001-ops-are-plain-functions.md) has otherwise avoided.

Option 1 as the default with option 2 as the escape hatch is the likely answer,
but it should not be built until there is a second computer op to check it
against. `skop.ops.toy.scale_into` is currently the only one, and one example
is not enough to design against.

## Why it has not been done

No op needs it yet. The forms exist because the SciJava Ops model is the target
and the annotations were cheap to add; adaptation is where the real design work
is, and doing it speculatively would mean guessing at the allocation question
above with a single data point.

Graduate this file into `design/` when it lands.
