# 0005 — skcommands: composing ops

**Status:** sketch, for comment. Nothing is implemented yet.

## Why

Two things we want to build next are not ops:

- **`psf_from_beads`** — extract a PSF from a bead image. It works by running a
  deconvolution with the roles reversed: feed it the measured beads and a
  synthetic ground-truth bead, and solve for the PSF. So it *needs a
  deconvolver*, and which one depends on the machine.
- **A segmenter** — bounding box detector, then mask generator. Each stage will
  have several implementations, and we want to hand it to a researcher or an
  intern to experiment with.

Both are workflows made of ops, and both need to pick between ops that live in
different environments. An op can't do that: `@op(env=...)` is fixed per
function, and an op that called another op would need to build environments from
inside a worker — which may not even be possible if the op's environment can't
run skop itself.

So: a second category. **skcommands run on the host and orchestrate ops.**

Two kinds of person want one:

- **The scripter, tinkering.** Good GPU at work, none at home. They want to swap
  the deconvolver in their PSF experiment by changing one word, and they want to
  try an op nobody has blessed yet.
- **The plugin developer.** They want their segmenter panel to offer a chooser
  over a small list of ops they trust — *I have tested these and I am familiar
  with them* — not over everything that happens to be installed.

We do the **explicit list first**, because that is the common case for both, and
because a curated list is information where a discovered one is only an
inventory. Discovering "every op of this kind" is a later design task, and may
turn out not to be what a plugin developer wants anyway.

## Considerations

1. **Ops are leaves, commands are nodes.** An op never calls the runner; a
   command never runs in a worker. Cross-environment composition happens on the
   host and nowhere else. Note the rule is about *environments*, not
   composition — an op may freely call plain functions inside its own env.

2. **Commands run in the host environment, which stays minimal.** The host has
   little more than numpy, on purpose, so discovery works. That means a command's
   glue code is numpy-level and all real computation goes into ops. There will be
   steady pressure to grow the host env; we should hold the line.

3. **Two users, both served by the same thing.** The *scripter* is a signal
   processing researcher who has a good GPU at work and doesn't at home; they
   want to change one word in their PSF experiment, not learn a dispatch
   framework. The *plugin developer* building the segmenter panel wants to offer
   a combo box with a few ops they've chosen — without hand-rolling it, and
   without magicgui deciding everything for them. One curated list in the command
   signature gives both: a default the scripter overrides positionally, and a
   menu the front end can render.

4. **Op-valued parameters: just pass the function.** This works today with no
   new machinery, because ops are plain functions that carry their own env
   (design 0001):

   ```python
   from skop.ops.deconvolve import richardson_lucy, richardson_lucy_cupy

   psf = psf_from_beads(beads, deconvolver=richardson_lucy_cupy)   # at work
   psf = psf_from_beads(beads)                                     # at home
   ```

   Default to the CPU op so it always runs; the GPU is the explicit opt-in.
   Importing `richardson_lucy_cupy` in the command module is safe on a machine
   with no CUDA — ops keep heavy imports in the function body, so the *function*
   is always importable in the host env.

5. **Substitutable ops must share a signature.** `richardson_lucy` and
   `richardson_lucy_cupy` already do, by design. Enforce it with a docstring line
   and discipline, not a type.

6. **Choosers are explicit, not discovered.** A curated list is information — the
   author tested these. A discovered list is just an inventory, and gets long and
   untrustworthy. Another `Annotated` marker, beside `Role` and `Out`:

   ```python
   deconvolver: Annotated[Callable, Choices(cpu=richardson_lucy, gpu=richardson_lucy_cupy)]
   ```

   The kwargs double as menu labels — "gpu" is a better thing to show a
   researcher than `richardson_lucy_cupy`. The spec exposes `(label, op_id)`
   pairs rather than function objects, so this survives going over the wire to a
   Fiji front end. A front end can also look up each choice's `env` and warn that
   picking "gpu" triggers a build.

7. **The list constrains the GUI, not the function.** Passing an op that isn't in
   `Choices` stays legal — annotations are advisory everywhere else in skop, and
   a researcher trying an untested solver in a script shouldn't hit a menu.

8. **The escape hatch is how the list grows.** The list means "tested, and I
   stand behind these". A scripter passes something outside it precisely because
   they are evaluating it; if the new mask generator turns out to work well, they
   add it to `Choices` and the next version of the command offers it in the GUI.
   So the loop is: experiment in a script → promote into the list → it appears in
   the combo box. Nobody edits somebody else's list at panel-construction time —
   curation lives in the command, in version control, where it can be reviewed.

9. **Unrenderable params are hidden and left at their default.** This is the rule
   that lets us ship without designing a chooser at all. A napari panel shows
   `psf_from_beads` with its image and physical parameters, runs the CPU path,
   and doesn't crash — exactly as `Out` params are already hidden today. Scripts
   keep the full flexibility. We can add the chooser widget whenever we want it.

10. **Commands need three things ops don't.** A runner, ambiently rather than as
    a parameter (same reasoning as `progress()` in 0001, so mode B still works);
    progress and cancellation that aggregate across several op calls; and a
    "can this run here?" preflight, so a GUI can grey out rather than fail after
    four minutes of pixi solving.

## Deliberately not designed yet

Automatic selection (`backend="auto"`), capability tags, matching, nested
choosers where picking one op reveals its own parameters, saving a configured
workflow as a re-runnable recipe, and whether a command may call another command.

The segmenter is how we find out which of these we actually need. Twenty
experiments from an intern will tell us more than another design conversation.

## Notes we don't want to lose

- **Partial binding.** A command binds some of a sub-op's parameters and exposes
  others. `psf_from_beads` binds the deconvolver's `image` and `psf` slots itself
  and would only ever expose something like `num_iters`. Whenever we do build a
  chooser UI, this is the part most likely to be got wrong first.

- **Roles are not invariant under use.** `richardson_lucy` declares
  `psf: PsfData` and returns an image; `psf_from_beads` uses it with those
  meanings swapped. Nothing breaks — roles are advisory — but it means we can
  never infer "is a deconvolver" structurally from a signature. Any capability
  tag has to be author-declared.

- **Round trips cost.** Every op call crosses shared memory. A command looping
  per-bead or per-bounding-box pays for each one. The fix is a batched op — one
  call over a stack, inside one env — not more cleverness in the command.

- **Pass functions, not lambdas or `partial`s.** A function has a stable
  importable id, so a saved recipe stays reachable as a future option. A lambda
  closes that door, and will creep in the first time someone wants to pin
  `num_iters`.
