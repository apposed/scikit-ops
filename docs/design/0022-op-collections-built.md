# 0022 — Op collections: what is built

**Status:** partly built. Tested with
[tnia-ops](https://github.com/True-North-Intelligent-Algorithms/tnia-ops):
tnia-python's utilities as ops, a second collection whose op and environment
names clash with these on purpose. [0021](0021-op-collections.md)
proposes a `Collection` descriptor for the same problem; the two need
reconciling.

## Built

- **An op can come from any package.** Before, every op was in scikit-ops,
  and every environment had scikit-ops installed, so a worker could always
  import it. An op from another package, such as tnia-ops, may not be
  installed in its environment at all; the environment holds only what the op
  uses. So the runner sends each call with the folder the op's module came
  from on the host, and the worker imports it from there.
- **A collection brings its own environments.** The runner looks for
  `envs/<id>/pixi.toml` beside the op's files, and in the folders above,
  before its own `envs/`.
- **Same ID, separate environments.** Each is built as `<package>--<id>`
  (`scikit-ops--cupy`, `tnia-ops--cupy`), the package's name taken from its
  `pyproject.toml` or its installed distribution, and workers are keyed by the
  environment's file, so two collections' `cupy` never share a build or a
  worker.
- **`Runner.env_for(op)`** says which file defines an op's environment and
  where it is built, without building it.
- **Progress, cancel and `run` are opspec's** (`opspec.runtime`, opspec
  design 0004): a protocol, with skop's runner plugged in. tnia-ops' stack
  `rolling_ball` reports each plane and stops when cancelled, depending on
  opspec alone.
- **The helpers are their own package, `ophelper`** (`./ophelper`): boxes,
  masks and model downloads, numpy and opspec only. tnia-ops'
  `object_boxes` uses it.
- **A collection registers itself** with an entry point, and napari-skopkit
  lists every one installed (`NAPARI_SKOPKIT_COLLECTION=tnia-ops` narrows it):

  ```toml
  [project.entry-points."opspec.collections"]
  tnia-ops = "tnia_ops"
  ```

## Not built

- **`discover()` with no argument**, walking every registered collection, so
  each front end need not. 0021 also wants a fallback for an install made
  before the entry point existed, which today shows no scikit-ops ops at all.
- **Name clashes are not reported.** Two collections may declare an op of the
  same name; napari-skopkit keeps them apart by prefixing each label with its
  collection, and nothing warns.
- **Fiji sees only `skop.ops`.** Its runner has one ops folder and one
  `envs/`. Entry points do not help there: they exist only on the host, and
  Fiji's host is Java.
- **Ops in a script run directly** (`python my_script.py`), where the op is
  in `__main__`, which a worker cannot import. The worker could load the
  script's file instead; left out of the first version to keep it simple.
- **Other environment formats.** Appose also builds from `environment.yml`
  (mamba) and `requirements.txt` or `pyproject.toml` (uv); `appose.file()`
  picks the builder. skop only calls `appose.pixi`.
- **pixi follows `workspace = true` badly.** A pixi environment could not
  install scikit-ops from a path while opspec was a uv workspace member, so
  opspec is now a plain path source.
