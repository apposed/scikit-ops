# naplari-hacking

A collection of image-processing **ops**, and `opkit` — the machinery that runs
them.

An op is an ordinary Python function. The same function can be called directly,
run in its own isolated environment through
[Appose](https://github.com/apposed/appose), or wrapped in a GUI. Nothing about
the op changes between those modes.

## Layout

```
envs/<env-id>/pixi.toml   environment definitions, named and shared between ops
envs/<env-id>/init.py     optional; runs in each worker before its I/O loop
ops/<name>.py             an op module (or ops/<name>/ when one file won't do)
opkit/                    op-independent machinery
```

Environments are keyed by ID, and several ops may declare the same one. Ops
sharing an environment also share a warm worker process, unless one asks for
`exclusive=True`.

## What's here

| Op | Environment | Does |
| --- | --- | --- |
| `ops.otsu:otsu` | `skimage` | Otsu thresholding |
| `ops.cellpose:cellpose` | `cellpose` | Cellpose segmentation |
| `ops.stardist2d:stardist2d` | `stardist-tf` | StarDist 2D, pretrained |
| `ops.starfun3d:segment_nuclei` | `stardist-tf` | StarDist 3D nuclei |
| `ops.starfun3d:synthetic_nuclei` | `skimage` | Synthetic 3D test volume |
| `ops.unseg:unseg` | `unseg-cv` | Unsupervised nuclei + cells |
| `ops.toy:*` | `minimal` | Exercises for opkit itself |

`stardist2d` and `segment_nuclei` share one TensorFlow build and one warm
worker — the point of naming environments rather than tying them to ops.
`unseg-cv` shares nothing with anything: it pins Python 3.9, numpy 1.24 and an
old scikit-image, which is exactly why it needs an environment of its own.

To list what the collection currently offers:

```python
import opkit
specs, failures = opkit.discover()
```

## Writing an op

```python
from typing import Annotated, NamedTuple

import numpy as np

from opkit import op


class Result(NamedTuple):
    labels: np.ndarray
    count: int


@op(env="stardist-tf")
def segment(
    image: np.ndarray,
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
the op's outputs. `opkit.progress(...)` and `opkit.cancel_requested()` report
to whoever is running the op, and do nothing when it is called directly.

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
import opkit
from ops import toy

# Directly, in this process, using this process's dependencies.
toy.add(2, 3)

# Or in the op's own environment, in a worker process.
with opkit.Runner() as runner:
    runner.run(toy.add, a=2, b=3)
```

The first run of an environment builds it, which takes a while. Later runs
reuse it, since Appose keys environments by name.

## Development

```sh
uv sync --all-groups
uv run pytest
uv run ruff check --fix && uv run ruff format
```
