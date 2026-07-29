# Spec — workflow ops: ops that compose ops

**Status:** a first iteration is implemented — see *What landed* below. The
rest of this document is the use-case exploration it was built from, kept as
written. Most of it still stands; where the implementation settled a question,
the section says so. Read the **Why** and the **Considerations** as the
content, and expect the parts marked *not designed yet* to change.

## What landed

Two workflows and the machinery to render them, 2026-07-28.

- **The marker is the absence of an environment.** `@op()` with no `env`
  declares a workflow; `OpSpec.is_workflow` is `env is None`. No flag, no
  second decorator — of the three options left open below, this is the one
  that needed no new vocabulary.
- **`Choices(label=op, ...)`** on a `Callable` parameter, as sketched.
  `ParamSpec.choices` carries it, and `Choices.ids` gives `(label, "mod:fn")`
  pairs for a front end that cannot hold Python objects.
- **`ParamsFor("chooser", binds=(...))`** on a partnering dict parameter: the
  chosen op's settings live there, and `binds` names what the workflow supplies
  itself. This is the answer to *partial binding*, flagged below as the thing
  most likely to be got wrong first. It is declared, not inferred — matching on
  name would hide `image` for free but still not know that a mask generator's
  `boxes` come from the detector, and a rule covering half the cases is harder
  to explain than none.
- **The runner is ambient**, exactly as `progress()` is: `Runner.run` binds
  itself for the duration of a workflow, so `skop.run` inside the body reuses
  the warm workers, and a sub-op's progress and cancellation reach the panel.
  Mode B is unaffected — outside a run, `skop.run` falls through to the default
  runner.
- **skop-napari draws a workflow with one combo box per stage**, the chosen
  op's own widgets underneath, rebuilt on change. Nearly free: the widgets
  under a chooser are built by the same `build_inputs` that builds the panel's.
  It is a second widget contribution, `Workflows`, which is also what earns
  the plugin its `Plugins > scikit-ops > ...` submenu — napari nests a
  plugin's widgets only once there is more than one.

Three workflows landed:

- `deconvolve_with_psf` — a PSF op paired with a deconvolver.
- `detect_then_mask` — a box detector paired with a mask detector. The one
  that needed `binds`: both its stages take the image.
- `connect_2d_in_3d` — normalize the whole volume, segment plane by plane with
  the segmenter's *own* normalization off, then `connect` the planes. It binds
  `normalize=False` for every choice rather than offering it, because a
  checkbox there would let someone switch back on the exact behaviour the
  workflow exists to avoid. Making that possible meant adding a `normalize`
  parameter to `cellpose` and `cellpose3`, which had none — Cellpose
  normalized per plane unconditionally, so two of the three choices would have
  silently undone step one.

`connect_2d_in_3d` also settles how a workflow pins axes: it passes
`axes={"image": list("zyx")}` on each `skop.run`, so the segmenter is looped
over z inside its own worker while the variadic normalizer gets the volume
whole. The workflow declares `Axes("z", "y", "x")` on its own input so a front
end can ask which of the caller's axes is z, and passes nothing down — each
stage's handling is fixed, because getting it wrong is the bug being fixed.

They live in **`src/skop/ops/workflows/`**, mirroring `skop.ops`'s own
subdirectories, rather than beside the ops they compose. The first draft put
them by domain and let `is_workflow` do the telling-apart; that reads well in a
script and badly in a source tree, where somebody wanting to write a workflow
has to read sixty signatures to find the two that exist. The separate tree also
makes the dependency one-way and visible -- workflows import ops, never the
reverse -- which removed a real fragility: `mask/__init__.py` re-exporting a
workflow reached back into its own half-initialized package, and worked only
because the import happened to be spelled as a submodule.

Nothing keys off the path. `OpSpec.is_workflow` still reads the missing
environment, and the front end strips the `workflows.` prefix from a label so
the domain underneath it is what shows.

Still open, and deliberately so: no preflight (consideration 10) — picking an
op whose environment is not built simply builds it, and the panel says which
environment a choice will use so that is not a surprise. Axis adaptation does
not reach into a stage; a sub-op gets the array as the workflow passes it.

## Naming

An earlier draft called these *skcommands*. That name is out. "Command" is a
front-end word — in Fiji it is a menu entry, and in most GUIs it means "the
thing a button invokes" — and it says nothing about what is actually different
here, which is that one op is built out of other ops.

The word we are using is **workflow**. A workflow op is an op whose body calls
other ops.

## Why

Two things we want to build next are ops that call other ops:

- **`psf_from_beads`** — extract a PSF from a bead image. It works by running a
  deconvolution with the roles reversed: feed it the measured beads and a
  synthetic ground-truth bead, and solve for the PSF. So it *needs a
  deconvolver*, and which one depends on the machine.
- **A segmenter** — bounding box detector, then mask generator. Each stage will
  have several implementations, and we want to hand it to a researcher or an
  intern to experiment with.

Both need to pick between ops that live in different environments. An ordinary
op can't do that: `@op(env=...)` is fixed per function, and an op that called
another op would need to build environments from inside a worker — which may not
even be possible if the op's environment can't run skop itself.

So the distinguishing feature is not "it composes" — it is **where it runs**. A
workflow runs on the host, where the runner lives, and orchestrates ops that run
in workers.

Two kinds of person want one:

- **The scripter, tinkering.** Good GPU at work, none at home. They want to swap
  the deconvolver in their PSF experiment by changing one word, and they want to
  try an op nobody has blessed yet.
- **The plugin developer.** They want their segmenter panel to offer a chooser
  over a small list of ops they trust — *I have tested these and I am familiar
  with them* — not over everything that happens to be installed.

We would do the **explicit list first**, because that is the common case for
both, and because a curated list is information where a discovered one is only
an inventory. Discovering "every op of this kind" is a later design task, and
may turn out not to be what a plugin developer wants anyway.

## A workflow is still an op

The current thinking — and this is the part most worth attacking — is that a
workflow should not be a second category of thing. It is an op with:

- a **`workflow` marker**, saying its body calls other ops and it needs the
  runner, and
- **no environment**. `env` is what pins an op to a worker; a workflow has
  nothing to pin. The absence is the statement: *this one runs on the host.*

```python
@op(workflow=True)                 # illustrative spelling only
def psf_from_beads(beads, deconvolver=richardson_lucy, ...): ...
```

Why keep it an op at all: it discovers, specs, renders and is called exactly
like everything else. A napari panel or a Fiji dialog does not want to learn a
second protocol to run the segmenter, and a script certainly doesn't — mode B
(0001) is the same plain function call either way. Only the *runner* needs to
tell them apart, because a workflow is invoked in-process rather than
dispatched.

Open, and not answered here: whether the marker is a flag on `@op`, a distinct
decorator, or simply inferred from `env is None`; and whether a workflow may
call another workflow.

## Considerations

1. **Ops are leaves, workflows are nodes.** An op with an environment never
   calls the runner; a workflow never runs in a worker. Cross-environment
   composition happens on the host and nowhere else. Note the rule is about
   *environments*, not composition — an op may freely call plain functions
   inside its own env.

2. **Workflows run in the host environment, which stays minimal.** The host has
   little more than numpy, on purpose, so discovery works. That means a
   workflow's glue code is numpy-level and all real computation goes into ops
   with environments. There will be steady pressure to grow the host env; we
   should hold the line.

3. **Two users, both served by the same thing.** The *scripter* is a signal
   processing researcher who has a good GPU at work and doesn't at home; they
   want to change one word in their PSF experiment, not learn a dispatch
   framework. The *plugin developer* building the segmenter panel wants to offer
   a combo box with a few ops they've chosen — without hand-rolling it, and
   without magicgui deciding everything for them. One curated list in the
   workflow signature would give both: a default the scripter overrides
   positionally, and a menu the front end can render.

4. **Op-valued parameters: just pass the function.** This works today with no
   new machinery, because ops are plain functions that carry their own env
   (design 0001):

   ```python
   from skop.ops.deconvolve import richardson_lucy, richardson_lucy_cupy

   psf = psf_from_beads(beads, deconvolver=richardson_lucy_cupy)   # at work
   psf = psf_from_beads(beads)                                     # at home
   ```

   Default to the CPU op so it always runs; the GPU is the explicit opt-in.
   Importing `richardson_lucy_cupy` in the workflow module is safe on a machine
   with no CUDA — ops keep heavy imports in the function body, so the *function*
   is always importable in the host env.

5. **Substitutable ops must share a signature.** `richardson_lucy` and
   `richardson_lucy_cupy` already do, by design. Enforce it with a docstring line
   and discipline, not a type.

6. **Choosers are explicit, not discovered.** A curated list is information — the
   author tested these. A discovered list is just an inventory, and gets long and
   untrustworthy. Something like another `Annotated` marker, beside `Role` and
   `Out`:

   ```python
   deconvolver: Annotated[Callable, Choices(cpu=richardson_lucy, gpu=richardson_lucy_cupy)]
   ```

   The kwargs double as menu labels — "gpu" is a better thing to show a
   researcher than `richardson_lucy_cupy`. The spec would expose `(label, op_id)`
   pairs rather than function objects, so this survives going over the wire to a
   Fiji front end. A front end can also look up each choice's `env` and warn that
   picking "gpu" triggers a build.

7. **The list constrains the GUI, not the function.** Passing an op that isn't in
   `Choices` stays legal — annotations are advisory everywhere else in skop, and
   a researcher trying an untested solver in a script shouldn't hit a menu.

8. **The escape hatch is how the list grows.** The list means "tested, and I
   stand behind these". A scripter passes something outside it precisely because
   they are evaluating it; if the new mask generator turns out to work well, they
   add it to `Choices` and the next version of the workflow offers it in the GUI.
   So the loop is: experiment in a script → promote into the list → it appears in
   the combo box. Nobody edits somebody else's list at panel-construction time —
   curation lives in the workflow, in version control, where it can be reviewed.

9. **Unrenderable params are hidden and left at their default.** This is the rule
   that would let us ship without designing a chooser at all. A napari panel
   shows `psf_from_beads` with its image and physical parameters, runs the CPU
   path, and doesn't crash — exactly as `Out` params are already hidden today.
   Scripts keep the full flexibility. We can add the chooser widget whenever we
   want it.

10. **Workflows need three things leaf ops don't.** A runner, ambiently rather
    than as a parameter (same reasoning as `progress()` in 0001, so mode B still
    works); progress and cancellation that aggregate across several op calls; and
    a "can this run here?" preflight, so a GUI can grey out rather than fail after
    four minutes of pixi solving.

## Deliberately not designed yet

How the workflow marker is actually spelled, and whether it is inferred from a
missing `env`. Automatic selection (`backend="auto"`), capability tags, matching,
nested choosers where picking one op reveals its own parameters, saving a
configured workflow as a re-runnable recipe, and whether a workflow may call
another workflow.

The segmenter is how we find out which of these we actually need. Twenty
experiments from an intern will tell us more than another design conversation.

## Notes we don't want to lose

- **Partial binding.** A workflow binds some of a sub-op's parameters and exposes
  others. `psf_from_beads` binds the deconvolver's `image` and `psf` slots itself
  and would only ever expose something like `num_iters`. Whenever we do build a
  chooser UI, this is the part most likely to be got wrong first.

- **Roles are not invariant under use.** `richardson_lucy` declares
  `psf: PsfData` and returns an image; `psf_from_beads` uses it with those
  meanings swapped. Nothing breaks — roles are advisory — but it means we can
  never infer "is a deconvolver" structurally from a signature. Any capability
  tag has to be author-declared.

- **Round trips cost.** Every op call crosses shared memory. A workflow looping
  per-bead or per-bounding-box pays for each one. The fix is a batched op — one
  call over a stack, inside one env — not more cleverness in the workflow.

- **Pass functions, not lambdas or `partial`s.** A function has a stable
  importable id, so a saved recipe stays reachable as a future option. A lambda
  closes that door, and will creep in the first time someone wants to pin
  `num_iters`.

Graduate this file into `design/` when it lands. It was numbered `design/0005`
for a while, colliding with [dimensional adaptation](../design/0005-dimensional-adaptation.md);
older references to "0005" in the design documents and in the source mean
whichever of the two the surrounding text is about.
