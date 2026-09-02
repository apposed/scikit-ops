# 0019 — Model residency and GPU lifetime

**Status:** proposed. Nothing built. Written because `stardist2d_custom`
(commit 2d97205) reloads its model on every call, and the obvious fix — hold
it in a dict — is wrong in a way that only shows up once training lands.

## The problem

`stardist2d_custom` takes a directory and constructs `StarDist2D(None,
name=name, basedir=basedir)` every call. Called on a stack through skop's
slicewise adaptation, that is one full model load per slice: `_adapt.execute`
loops calling `fn(**args)`, so forty planes are forty loads of the same
weights. The pretrained ops do the same through `from_pretrained` — StarDist
caches the *downloaded files*, not the built network.

Nothing about this is forced. The worker is long-lived: `Runner.service`
keys workers by `(env, variant, name if exclusive)` and keeps them, so a
module-level variable in the op's module outlives both the slice loop and
every later call in the session. The reload is an accident, not a design.

Anyone writing this without skop keeps the model in a local variable and
reuses it. That is the behaviour to recover, and the reason it is not a
five-line change is what has to happen when something *else* wants the GPU.

## What the naive fix misses

Hold the model in a module dict and the sequence that breaks is the one this
project is heading straight for:

1. Predict with a trained model. It loads, and stays.
2. Keep predicting with it. This is the win — one load, many calls.
3. Train. The predict model is still resident, and training now starts on a
   GPU that is short by the size of a model nobody is using.

Step 3 is not hypothetical. `ops/train/stardist2d.py` and the predict ops both
declare `env="stardist-tf"`, and ops sharing an environment share a worker, so
training begins **in the same process** that is holding the predict model.

The op's module dict cannot fix this. It does not know training is about to
start, and the training op has no handle on it.

## Three triggers, three owners

The useful cut is by who can *see* the thing that should cause eviction.

**Predict with a different model — the op sees it.** Both directories are in
the op's own hands. One resident model, replaced when a call asks for another
one. This much needs no machinery and should just be written.

**Train after predicting — the worker sees it.** Two ops, one process, no
shared state between them today. Whatever answers this has to sit above the
op and below the host.

**Something else GPU-intensive — only the host sees it.** `richardson_lucy_cupy`
lives in another environment, which means another process. No amount of
in-process bookkeeping in the StarDist worker can observe it. `Runner` is the
only thing that knows what every environment is holding.

So the ladder ends at the host, and that is the finding: this is a runner
question wearing an op question's clothes.

## Freeing is process exit

The reason it cannot be pushed back down into the op: with TensorFlow,
dropping the last Python reference does not reliably return VRAM. `del model`
plus `clear_session()` is the usual recipe and it is still not dependable.
Torch is better — `del` then `empty_cache()` — but "better" is not "known".

Process exit is the only release that is certain. `Runner.close()` shuts down
every worker; there is no way to say *release this environment*. That, rather
than any cache API, is the primitive this needs: something like
`runner.release(env)`, whose implementation is closing that worker, because
for TensorFlow nothing weaker actually frees anything.

Cheap eviction is a lie. Design it as expensive and it stays honest.

## Which way to be wrong

Reloading a model costs seconds. Running a training job out of memory costs
minutes to hours, and it fails at the end rather than the start. The costs are
not symmetric, so the default should not be either: hold a model while
consecutive predicts keep asking for it, and drop it for anything else.

## No session object

"Session" is the tempting name for the thing a resident model belongs to, and
it should be resisted. The worker's lifetime already is one, and it is the
granularity VRAM actually respects — a Python object graph can outlive the
memory it thinks it owns, a process cannot. A session concept layered on top
invites the belief that residency survives a worker restart, which is the one
thing it can never do. The worker *is* the session; what is missing is a way
to end one without ending them all.

## Staleness

A second question the same state raises, smaller but real. Train writes into
a model directory; predict may already be holding the model that directory
used to contain. Keyed on the path alone, the resident model silently answers
for weights that no longer exist on disk. A modification time on the weights
file settles it for the cost of a `stat`, and matters exactly once: retraining
in place, which is the loop this repo is building.

## Interactions

- **`exclusive`** (`_spec.py:749`) gives an op its own worker. Marking the
  training op exclusive makes this *worse*, not better — two live processes,
  both wanting the GPU — unless releasing the predict worker comes first.
  Exclusivity separates ops; it does not sequence them.
- **[0017](0017-memory-and-tiled-processing.md)** already names model weights
  as memory that does not shrink with the tile, which is what its `fixed`
  field is for. A budget that ignores a resident model from an earlier call is
  computing against the wrong free-memory number.
- **[0011](0011-deep-learning-training-ops.md)** is where the train side of
  this is specified; this document is only about what predict may hold while
  train runs.

## Not decided

- Whether the worker-level answer is an explicit call an op makes ("release
  what you are holding"), or a rule the host applies before dispatching to a
  known-heavy op. The second needs ops to declare that they are heavy.
- Whether `runner.release(env)` is host policy or something a front end drives
  from a button. A GUI wants both: automatic before training, manual for
  "free the GPU".
- Whether a resident model should be dropped on a timer. Probably not — idle
  and finished look identical from inside the process.
- Whether the op-local slot is worth writing before the rest exists. It fixes
  the slicewise reload, which is real today, and nothing above it contradicts
  a single resident model.
