# scikit-ops

A collection of image-processing **ops**, and `skop` — the machinery that runs
them.

An op is an ordinary Python function. The same function can be called directly,
run in its own isolated environment through
[Appose](https://github.com/apposed/appose), or wrapped in a GUI. Nothing about
the op changes between those modes.

## Layout

```
envs/<env-id>/pixi.toml   environment definitions, named and shared between ops
envs/<env-id>/init.py     optional; runs in each worker before its I/O loop
src/skop/                 op-independent machinery
src/skop/ops/<ns>.py      an op namespace, when its ops share one environment
src/skop/ops/<ns>/<op>.py an op namespace, when its ops do not
src/imgops/               napari/magicgui front ends (not published)
```

Environments are keyed by ID, and several ops may declare the same one. Ops
sharing an environment also share a warm worker process, unless one asks for
`exclusive=True`.

## What's here

| Op | Environment | Does |
| --- | --- | --- |
| `skop.ops.threshold:otsu` | `skimage` | Otsu thresholding |
| `skop.ops.segment.cellpose:cellpose` | `cellpose` | Cellpose segmentation |
| `skop.ops.segment.stardist2d:stardist2d` | `stardist-tf` | StarDist 2D, pretrained |
| `skop.ops.segment.starfun3d:segment_nuclei` | `stardist-tf` | StarDist 3D nuclei |
| `skop.ops.generate:synthetic_nuclei` | `skimage` | Synthetic 3D test volume |
| `skop.ops.segment.unseg:unseg` | `unseg-cv` | Unsupervised nuclei + cells |
| `skop.ops.toy:*` | `minimal` | Exercises for skop itself |

Those are the fully qualified op IDs, as `discover()` reports them. Callers
import from the namespace instead, and never need to know whether it is one
module or a package of them:

```python
from skop.ops.threshold import otsu
from skop.ops.segment import cellpose, stardist2d
```

`stardist2d` and `segment_nuclei` share one TensorFlow build and one warm
worker — the point of naming environments rather than tying them to ops.
`unseg-cv` shares nothing with anything: it pins Python 3.9, numpy 1.24 and an
old scikit-image, which is exactly why it needs an environment of its own.

To list what the collection currently offers:

```python
import skop
specs, failures = skop.discover()
```

## Writing an op

```python
from typing import Annotated, NamedTuple

from skop import op
from skop.types import ImageData, LabelsData


class Result(NamedTuple):
    labels: LabelsData
    count: int


@op(env="stardist-tf")
def segment(
    image: ImageData,
    prob_thresh: Annotated[
        float, {"widget_type": "FloatSlider", "min": 0.0, "max": 1.0}
    ] = 0.5,
) -> Result:
    """Segment nuclei."""
    from stardist.models import StarDist3D  # NB: heavy imports go in the body.

    ...
```

Two rules, and both exist for the same reason — ops are discovered by importing
them into a minimal environment that has little more than numpy:

1. **Heavy imports go inside the function body.** A module-scope
   `import tensorflow` makes the op undiscoverable.
2. **Type your inputs and outputs with plain types.** Annotations *are*
   evaluated at discovery time, so `-> tf.Tensor` fails even if the import is
   in the body.

Multiple outputs come from a `NamedTuple`: its field names become the names of
the op's outputs. `skop.progress(...)` and `skop.cancel_requested()` report
to whoever is running the op, and do nothing when it is called directly.

### Saying what an array is

Every op here passes `np.ndarray` around, which tells a front end nothing
about how to display it. `skop.types` adds that missing half:

| Alias | Role | napari shows it as |
| --- | --- | --- |
| `ImageData` | `Role.image` | an Image layer |
| `LabelsData` | `Role.labels` | a Labels layer |
| `PointsData` | `Role.points` | a Points layer |
| `VectorsData` | `Role.vectors` | a Vectors layer |
| `TracksData` | `Role.tracks` | a Tracks layer |

They are `Annotated[np.ndarray, ...]` aliases, so nothing else changes: the
codec still sees an array, direct callers still get an array, and
`ParamSpec.type` is still `np.ndarray`. They compose with the rest —
`Out[LabelsData]` and `Annotated[LabelsData, {"label": "Nuclei"}]` both work,
in either order.

Roles are advisory and never guessed. An unannotated array reports
`role is None`, and deciding what to do with that is the front end's business,
not skop's. Nothing in `skop.types` imports napari; the names simply mirror
`napari.types` so that mapping is a lookup rather than a judgement call, and
`Role` is equally the hook a Fiji or plain-magicgui front end reads.

### Where it goes

Ops live in a namespace under `skop/ops/`, named for what they do. How the
namespace is laid out inside depends on its ops, and the split point is the
environment:

- **One module** when the ops are variations on one call sharing one
  environment — every `skop.ops.threshold` op is a few lines against
  `skimage.filters`, so they share `threshold.py`.
- **A package, one module per op**, when they do not — `skop.ops.segment`
  spans three environments and each op carries its own models and types, so
  each gets a file and `segment/__init__.py` re-exports them.

An op needing several files of its own becomes a package
(`segment/unseg/__init__.py` plus siblings). Underscore-prefixed modules are
private and never imported by discovery, which is where shared helpers and
vendored code belong.

Because `from skop.ops.<ns> import <op>` reads the same either way, a
namespace can grow from a module into a package without breaking callers.

### Computation forms

Following [SciJava Ops](https://ops.scijava.org/en/latest/Concepts.html), an op
is one of three forms, declared through its signature:

| Form | Signature | Meaning |
| --- | --- | --- |
| function | `def f(image) -> Result` | allocates and returns its output |
| computer | `def f(image, out: Out[np.ndarray])` | fills a buffer the caller owns |
| inplace | `def f(image: Mut[np.ndarray])` | mutates its input |

`Out` parameters are hidden from generated GUIs — a user is never asked for an
output buffer. Adapting between forms is not implemented yet: a computer or
inplace op must be handed its buffers.

## Running an op

```python
import skop
from skop.ops import toy

# Directly, in this process, using this process's dependencies.
toy.add(2, 3)

# Or in the op's own environment, in a worker process.
with skop.Runner() as runner:
    runner.run(toy.add, a=2, b=3)
```

The first run of an environment builds it, which takes a while. Later runs
reuse it, since Appose keys environments by name.

Because that build happens inside `run()`, a caller with a progress bar can
subscribe to it:

```python
runner.subscribe_build_progress(lambda title, current, maximum: ...)
runner.subscribe_build_output(lambda text: ...)
runner.subscribe_build_error(lambda text: ...)
```

These pass straight through to Appose's builder. Progress titles name the
phase — `Solving`, `Installing conda packages`, `Downloading PyPI packages`,
`Installing PyPI packages`, `Done` — and come with a real denominator.

Two things worth knowing. Subscribing to progress is what *enables* it:
Appose's `PixiInstallMonitor` runs pixi under `-vv` to read its phase
transitions, and only wires up when a progress subscriber exists. And
`build_error` is the stderr *stream* rather than a failure report — pixi
writes its ordinary status there, and under `-vv` its whole debug log as
well. A build that actually fails raises out of `run()`.

This needs Appose newer than 0.11, which is why `pyproject.toml` currently
sources it from the main branch.

## Development

```sh
uv sync --all-groups
uv run pytest
uv run ruff check --fix && uv run ruff format
```

### Testing the ops for real

Ops are exercised end to end in their own environments, in `test_ops_e2e.py`.
Each such test declares what it needs with `@pytest.mark.env("<env-id>")`, and
is skipped when that environment is not built — so a plain `pytest` run stays
fast and offline, testing against whatever happens to be installed already.

To run them all, building whatever is missing:

```sh
uv run pytest --build-envs        # everything; slow the first time
uv run pytest --build-envs -m env # only the op tests
uv run pytest -m "not env"        # only the fast ones
```

CI runs `--build-envs`, since these ops are the whole point of the project and
nothing else validates them against the stacks they target. The flag is also
settable as `PYTEST_ADDOPTS=--build-envs`, which is handy when the pytest
invocation is not yours to edit.
