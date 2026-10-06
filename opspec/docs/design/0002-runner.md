# 0002 — Runner

## Background

A runner is what actually calls ops.  

## What a runner does

- (Optionally) **Provides the environment** the op asked for with `env=`.
- (Optionally) **Slices** — the op declared `Axes("z", "y", "x")` and the
  caller has ZYXT, so the runner loops over T and gathers the results.
- (Optionally) **Chunks** — the op declared a `PeakMemory`, so the runner
  divides the input into tiles that fit the memory it has.
- **Calls the op** with the caller's arguments.
- **Returns the result**, or an error.

opspec does the arithmetic for the two optional ones: how many slices,
which tile size fits the budget. That is shape and byte counting, so it
stays in opspec. Carrying it out — slicing the array, calling the op once
per tile, putting the pieces back — touches pixels, so it belongs to the
runner.

## What a runner is

A Protocol. opspec says what methods a runner has; it ships no runner.
The simplest one calls the function in this process. Others start a
worker, a conda environment, or a machine with a GPU.

## Two protocols

- `Builder` provides environments. Slow, rare, needs progress.
- `Runner` calls ops. One per call.

Structural, so a class satisfies them by having the methods. scikit-ops'
existing `Runner` satisfies both, unchanged and without importing opspec.
An in-process runner implements `Runner` only.

## Open

Left open on purpose until something needs them.

- Does `run` block, or return a task? scikit-ops blocks; napari#9347
  returns a `PluginTask`.
- Where an environment's recipe lives. npe2#498 puts dependencies in the
  manifest, scikit-ops in a pixi.toml. opspec names an environment and
  says nothing about its contents.
- Whether `Builder` also lists environments and in-flight tasks, as
  napari#9347 does.
- How a runner keys an environment directory. Two versions of the same
  runner writing one directory can leave it half-installed.
- Converting array types. The op declares what it needs, the runner
  converts. `__array__` is not a reliable test: cupy defines it and
  raises.
