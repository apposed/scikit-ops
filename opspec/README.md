# opspec

Describe image-processing ops so plugins, runners and hosts can share them.

opspec is a small Python package with no dependencies beyond the standard
library. It defines two things:

- **Ops**: how a plugin author declares what an op takes and returns
  (for example "a 2D image, numpy only").
- **Runners**: the interface a runner implements to run those ops, whether
  in the same process, in another environment, or on a GPU.

Hosts (napari, CellProfiler, a notebook) read the op declarations and hand
ops to a runner.

## Install

opspec lives in the [scikit-ops](../README.md) repository. Not on PyPI yet:

```sh
pip install "opspec @ git+https://github.com/apposed/scikit-ops#subdirectory=opspec"
```

## Docs

- [0001](docs/design/0001-opspec.md) — what an `OpSpec` is: `@op`, roles,
  `ImageOf`, `Axes`
- [0002](docs/design/0002-runner.md) — what a runner does: the `Runner` and
  `Builder` protocols
- [0003](docs/design/0003-reconciling-with-scikit-ops.md) — where opspec and
  scikit-ops' spec code differed, and which way each went

## AI use

Developed with AI assistance (Claude). A human reviews, understands and
tests every change.

## License

BSD-3-Clause
