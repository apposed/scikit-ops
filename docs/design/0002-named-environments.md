# 0002 — Environments are named and shared

## The problem

Ops in this collection cannot coexist. `unseg` pins Python 3.9, numpy 1.24 and
an old scikit-image; `stardist2d` wants a TensorFlow build; `cellpose` wants
torch. There is no single environment that runs all of them, and there never
will be — that incompatibility is the reason this project exists rather than
being a normal library.

So each op runs in its own process, in its own environment, and talks to the
host over [Appose](https://github.com/apposed/appose).

## The decision

Environments are **named**, defined as data, and shared between ops:

```
envs/<env-id>/pixi.toml   the environment definition
envs/<env-id>/init.py     optional; runs in each worker before its I/O loop
```

An op declares one by ID: `@op(env="stardist-tf")`. Several ops may declare the
same one. `Runner` keys both its Appose environments and its warm worker
services on that ID, so ops sharing an environment share a process.

`stardist2d` and `segment_nuclei` share one TensorFlow build and one warm
worker. That sharing is the entire point of naming environments rather than
deriving one per op: TensorFlow takes minutes to install and seconds to import,
and neither cost should be paid twice.

## Why not derive the environment from the op

The obvious alternative is to let each op declare its dependencies inline
(`@op(deps=["tensorflow==2.15", ...])`) and synthesize an environment per op.
Rejected for three reasons:

1. **No sharing.** Two ops with identical dependency lists would still get two
   environments unless we hashed and deduplicated them — at which point the
   hash is the name, only unreadable.
2. **The definition is not a list.** Real pinning needs channels, platform
   markers, PyPI-vs-conda splits, and occasionally a build string. That is a
   `pixi.toml`, and inlining a `pixi.toml` in a decorator is worse than
   pointing at one.
3. **Nothing to edit.** When a build breaks, a named file is something a human
   can open, fix, and rebuild. A synthesized environment is not.

The cost is indirection: reading an op does not tell you what it depends on.
Accepted — `spec.env` names a file two directories away, and that file is the
thing you actually need to look at anyway.

## The host imports nothing heavy

`Runner` never imports an op's dependencies. It imports the op *module* to read
its signature, which is cheap because ops keep heavy imports in their function
bodies (see [0001](0001-ops-are-plain-functions.md)), and dispatches the call
into a worker.

Two details in `runner.py` that look arbitrary and are not:

- **The worker's `sys.path` is extended in the init script, not via an
  environment variable.** Appose 0.12 gained per-service env vars, and setting
  `PYTHONPATH` on the builder would be the tidier-looking choice. It would also
  fold a machine-specific checkout path into the environment's *identity*,
  forcing a full rebuild whenever the checkout moves.
- **`import numpy` precedes the worker's I/O loop.** On Windows it must, or the
  loop and numpy's own initialization interleave badly.

## Environment IDs are not op namespaces

`skop.ops.segment` spans three environments; `skop.ops.threshold` is one module
sharing one. The package layout tracks *what ops do*, the environment ID tracks
*what they need*, and conflating the two would force one of them to be wrong.
The practical rule that falls out: a namespace stays a single module while its
ops share an environment, and becomes a package when they stop.
