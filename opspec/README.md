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

## AI use

Developed with AI assistance (Claude). A human reviews, understands and
tests every change.

## License

BSD-3-Clause
