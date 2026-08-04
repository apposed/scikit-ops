# Running the notebooks and interactive tests

Both run in the same *host* environment, which is why one file covers them.
`notebooks/` holds narrative examples with their outputs committed;
`../interactive_tests/` holds scripts that open a viewer and are meant to be
played with. Neither is run by `pytest`.

## The host is not where the ops run

This is the thing to understand first, because everything else follows from it.

The host holds only what a notebook itself imports: a kernel, matplotlib,
scikit-image for its sample images, tnia-python for the plotting helpers, and
napari for the interactive tests. **No torch, no cellpose, no CUDA.** Ops run in
their *own* environments, which skop builds under `envs/` the first time one is
called, and an op's dependencies never belong in the host — adding one rebuilds
the monolith this project exists to avoid.

So the first run of a notebook that calls a real op is slow: it builds one or
two PyTorch stacks, several gigabytes each, and then each model downloads its
weights. Build phases print as they happen. A long silence means something is
wrong, not that it is thinking.

## Building the host

Three ways, holding the same things. Pick whichever tool you already have.

**pixi** — `pixi/examples/pixi.toml`:

```sh
cd pixi/examples
pixi install
```

**uv** — the `examples` group in `pyproject.toml`, from the checkout root:

```sh
uv sync
```

**By hand**, into a conda env or venv you already have:

```sh
pip install -e .
pip install jupyterlab matplotlib scikit-image tifffile "tnia-python[plotting]"
```

## Working on tnia-python at the same time

`pixi/examples` always takes tnia-python from PyPI, so a notebook that runs
there is one anybody can run. To change its plotting helpers and a notebook
together, use `pixi/tnia-dev` instead — the same environment with that one
dependency pointed at the `../tnia-python` sibling checkout, editable:

```sh
cd pixi/tnia-dev && pixi run register-kernel
```

It registers its own kernel, *scikit-ops (tnia-dev, editable tnia)*, so both
show up in VS Code's picker and switching is picking one. An edit to
`plt_helper.py` then takes effect on the next kernel restart, with no release in
between.

`pixi/tnia-dev` is gitignored. It assumes tnia-python sits beside scikit-ops,
which is true of a development machine and of nothing else.

**Re-run against *scikit-ops (examples)* before committing a notebook**, or you
may commit one that only runs where that checkout exists.

For `uv`, the equivalent is `uv pip install -e ../tnia-python`, undone with
`uv sync --reinstall-package tnia-python`.

If you never open Jupyter Lab, ignore the `lab` task — `register-kernel` is the
one that matters, since VS Code starts the kernel itself.

## Version skew

A host can hold an older tnia-python than a notebook needs:

```
ImportError: cannot import name 'draw_boxes' from 'tnia.plotting.plt_helper'
```

A lock file does not move on its own, so upgrade explicitly:

```sh
pixi update tnia-python                   # in pixi/examples
uv sync --upgrade-package tnia-python     # or, for uv
pip install -U "tnia-python[plotting]"    # or, by hand
```

`draw_boxes` needs 0.2.2 or newer; the `[plotting]` extra needs 0.2.0, and does
not exist at all on 0.1.x.

## When an op fails to build

`BuildException: pixi build failed` on its own tells you nothing. Two things
get you the cause.

**Check whether the environment still solves** — seconds to a minute each, no
downloads:

```sh
python scripts/check_envs.py            # all of them
python scripts/check_envs.py pytorch    # just one
```

An environment built weeks ago goes on working from its lock file even if its
manifest stopped being solvable, so this is the check the test suite cannot
make: the op tests run *inside* an environment and cannot tell you whether that
environment could still be created today. Run it when a build fails for no
apparent reason, and after editing anything in `envs/`.

**Subscribe to the build's stderr**, which is where an unsatisfiable solve
explains itself:

```python
build_log = []
runner.subscribe_build_error(build_log.append)
```

Buffer it rather than printing it live — pixi writes ordinary status there too,
and a progress subscriber puts it in `-vv`, so the interesting lines drown.
Print the tail when a run raises. The runner cell of
[`detection/detect-then-mask.ipynb`](detection/detect-then-mask.ipynb) does
exactly this and is worth copying.

## Before committing a notebook

Outputs are committed on purpose — the figures are the product, and should
render for someone who has installed nothing. So:

- Restart and run all, so execution counts read 1, 2, 3.
- Run against the released environment, not `local`.
- Keep figures small. Base64 PNGs are permanent repo weight.

## Interactive tests

Scripts, not notebooks, and not collected by pytest — they open a napari window
and are meant to be edited while running. Same host:

```sh
cd pixi/examples && pixi run python ../../interactive_tests/interactive_test_yolo_coins.py
uv run python interactive_tests/interactive_test_yolo_coins.py
```

The automated equivalent is `pytest --build-envs -m env`, described in the
[README](../README.md#testing-the-ops-for-real).
