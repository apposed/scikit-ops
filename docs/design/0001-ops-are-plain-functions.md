# 0001 — An op is an ordinary function

## The requirement

The same op has to run in four modes without changing:

- **A.** as a standalone script
- **B.** imported and called as a plain Python function
- **C.** inside napari, as an Appose task
- **D.** inside Fiji, the same way

Mode B is the constraint that kills most designs. If an op has to be a class,
or registered in a table, or invoked through a framework object, then calling
it directly stops being ordinary — and an op that is annoying to call directly
is an op nobody tests.

## The decision

An op is a module-level Python function carrying a `@op(env=...)` decorator.
Everything a front end needs is *read off the function*: its signature, its
annotations, its docstring. The decorator records the environment ID and marks
the function as discoverable; it does not wrap, replace or proxy the function.

```python
toy.add(2, 3)                          # mode A/B: just a function

with skop.Runner() as runner:
    runner.run(toy.add, a=2, b=3)      # mode C/D: same function, own process
```

`OpSpec` (in `_spec.py`) is the reified description — name, module, env, params,
outputs, docstring, form. It is produced by introspection at discovery time and
is the only thing a front end is allowed to look at. No front end imports an op
module in order to render it.

## What follows from it

**Discovery works by importing.** `discover()` imports every module under
`skop.ops` in the *host* environment, which has little more than numpy. That
gives two rules ops must follow, and both have the same cause:

1. Heavy imports go inside the function body. A module-scope `import tensorflow`
   makes the op undiscoverable, because the host cannot import tensorflow.
2. Annotations must be plain types. Annotations *are* evaluated at discovery
   time, so `-> tf.Tensor` fails even with the import in the body.

Rule 2 is the surprising one and it is worth restating: putting the import in
the body is not enough if the type escapes into the signature. This is why ops
type their arrays as `np.ndarray` rather than as anything from their own stack.

**Multiple outputs come from a `NamedTuple`.** Its field names become the op's
output names. This was chosen over returning a dict (no types, no field
annotations to hang roles on) and over out-parameters (see the forms below).

**Progress and cancellation are free functions**, `skop.progress(...)` and
`skop.cancel_requested()`, not arguments. An op that took a `progress` callback
would have a parameter that mode B callers must supply and front ends must hide
— and would break the rule that the signature is entirely about the science.
They do nothing when the op is called directly, which is what makes mode B
identical to mode C from inside the function.

## Computation forms

Following [SciJava Ops](https://ops.scijava.org/en/latest/Concepts.html), the
signature declares one of three forms:

| Form | Signature | Meaning |
| --- | --- | --- |
| function | `def f(image) -> Result` | allocates and returns its output |
| computer | `def f(image, out: Out[np.ndarray])` | fills a buffer the caller owns |
| inplace | `def f(image: Mut[np.ndarray])` | mutates its input |

`Out` and `Mut` are `Annotated` markers, so the form is inferred rather than
declared twice. `Out` parameters are hidden from generated GUIs — a user is
never asked to supply an output buffer.

Form adaptation is **not** implemented; a computer must currently be handed its
buffers. See [spec/form-adaptation.md](../spec/form-adaptation.md).

## Alternatives considered

**A registry with explicit metadata** (`register(fn, inputs=[...], outputs=[...])`).
Rejected: the metadata immediately drifts from the signature, and it makes the
op definition longer than the op.

**Ops as classes.** Rejected on mode B. `Otsu().run(image)` is not an ordinary
function call, and the class buys nothing — an op has no state between runs.

**Deriving `OpSpec` from a static manifest** rather than by importing. Tempting,
because it would remove both authoring rules above. Rejected because it means a
build step between writing an op and running it, and because the manifest would
be a second source of truth for exactly the information the signature already
carries. The cost is that discovery is only as reliable as the host environment
is minimal — worth watching.
