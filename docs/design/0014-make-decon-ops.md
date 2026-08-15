# 0014 — Iterative deconvolution for numpy, cupy, and opencl

**Status:** implemented except OpenCL. `skop.ops.deconvolve` holds
`richardson_lucy` and `richardson_lucy_cupy`, `envs/cupy` is the GPU
environment, and the Gaussian PSF landed as `skop.ops.kernels.gaussian_psf` —
answering this plan's open question in favour of a general `kernels` namespace,
which [0010](0010-theoretical-psf-ops.md) then built on. **OpenCL was never
done**, so the "any machine" claim in the background below is still aspiration.
No separate design doc graduated out of this; this one is the record.

## Background

We want deconvolution to work on any machine, but be fast if we have GPU. GPU might be nvidia so we can use cuda, GPU might not have cuda though, so we can use opencl.  Very difficult to manage environment for each case so we are making scikit-ops for this, each op is associated with a pixi environment.  See src/skop/ops/segment/cellpose.py for an example op, see envs/cellpose for example pixi.

Now we want deconvolution to follow the same design and we port pre-existing deconvolution

1.  You can find an example of numpy cpu deconvolution here ../tnia-python/tnia/deconvolution/richardson_lucy_np.py

2.  You can find an example of cuda gpu deconvolution in same folder just called richardson_lucy.py

3.  You can find examples of psf calculation in the same folder, in ../tnia-python/tnia/deconvolution/gaussian_psf.py.  It has separate `gaussian_2d(xy_dim, xy_sigma)` and `gaussian_3d(xy_dim, z_dim, xy_sigma, z_sigma)` functions, but collapsing them into a single op is fine.

Eventually we want to port most of this code, however for now we want to start with cpu and cuda gpu deconvolution and guassian numpy psf.  port these to scikit under new deconvolution folder beside the segment folder.  base on design of cellpose.py.  

One open question is whether Gaussian PSF should be in deconvolution folder or new more general folder because it can be used in other context. 

GaussianPSF, and numpy deconvolution should be able to re-use the scikit pixi (because scikit will give us numpy).  cupy needs a new pixi for cupy. 

We also want an interactive_test and formal tests.  See interactive_tests/interactive_test_scikit.pyfor example interactive test, this light and informal and for people to play with, should follow that test except use napari instead of matplotlib to view result because easier to see 3d. 

# Plan to migrate old deconvolution to skops

## Decisions this plan assumes

These are the calls that shape everything below. Change any of them and the steps change.

- **Namespace name.** `src/skop/ops/deconvolve/`, a package with one module per op.
  A package rather than a single module because the two RL ops live in different
  environments, which is exactly the split point the README names. `deconvolve`
  rather than `deconvolution` to match the existing verb namespaces (`segment`,
  `threshold`, `generate`) -- say the word and it becomes `deconvolution`.
- **Gaussian PSF goes in a new `kernels` namespace, `src/skop/ops/kernels/psf.py`,**
  not under deconvolve. That answers the open question in the section above: a PSF
  is used for forward simulation and blur modelling too, so it is no more a
  deconvolution op than `generate.py` is a segmentation op. `kernels` is the
  namespace for things that generate a kernel to convolve with -- later PSFs
  (Gibson-Lanni, widefield) join `psf.py`, and other kernel families get their own
  module beside it, none of it touching the deconvolve package.
- **One op per backend, not one op with a `backend=` flag.** `@op(env=...)` is
  static per function, so the environment cannot be chosen at call time. So:
  `richardson_lucy` (numpy, env `skimage`) and `richardson_lucy_cupy` (env
  `cupy`). A dispatcher that picks the fastest available backend is a later
  concern, and belongs above the ops rather than inside one.
- **Two ops, one algorithm.** The numpy and cupy originals have drifted apart --
  see "Behaviour to unify" below. The ports reconcile them so both backends give
  the same answer, and the divergence from the originals gets an NB comment.

## What gets written

```
envs/cupy/pixi.toml                              new env: cupy + CUDA
src/skop/ops/kernels/__init__.py                 re-exports gaussian_psf
src/skop/ops/kernels/psf.py                      gaussian_psf            (env skimage)
src/skop/ops/deconvolve/__init__.py              re-exports the two ops
src/skop/ops/deconvolve/_pad.py                  vendored from tnia.deconvolution.pad
src/skop/ops/deconvolve/richardson_lucy.py       richardson_lucy         (env skimage)
src/skop/ops/deconvolve/richardson_lucy_cupy.py  richardson_lucy_cupy    (env cupy)
tests/test_pad.py                                host-only, no env needed
tests/test_ops.py                                add spec-level assertions
tests/test_ops_e2e.py                            add real runs for both backends
interactive_tests/interactive_test_decon.py      napari
README.md                                        add the ops to the table
```

## Steps

**1. Vendor the pad helpers.** Both RL versions call
`tnia.deconvolution.pad`, so it comes across as `deconvolve/_pad.py`
(`pad`, `unpad`, `pad_to_largest`, `next_smooth`, `get_next_smooth`). Underscore
prefix keeps discovery from importing it as an op module. It is pure numpy +
`math`, so it needs no environment of its own and can be unit-tested on the host.
`unpad` is already backend-agnostic -- it only slices -- so the cupy op uses it
unchanged.

**2. `kernels/psf.py` -- one op covering 2D and 3D.** Imported as
`from skop.ops.kernels import gaussian_psf`.

```python
@op(env="skimage")
def gaussian_psf(xy_dim=64, xy_sigma=2.0, z_dim=0, z_sigma=2.0) -> np.ndarray
```

`z_dim <= 1` gives the 2D PSF, anything larger gives the 3D one -- the collapse
sanctioned in item 3 above. Two things to get right while porting:

- The originals build the grid with triple nested Python loops. Vectorize with
  broadcasting; a 64x64x64 PSF is ~260k Python iterations otherwise.
- `gaussian_3d`'s meshgrid has that "could not get z order right" comment and
  indexes `x_[x,y,z]` while writing `gauss[z,y,x]`. Port to a clean
  `indexing="ij"` grid, then assert numerically against the original output so
  the reorientation is proven rather than assumed.

Both normalize to sum 1. Keep `+1e-12` on the 3D path (the 2D original has it
commented out) or drop it from both -- either way, make the two consistent.

**3. `richardson_lucy.py` -- the numpy op.** Signature:

```python
@op(env="skimage")
def richardson_lucy(image, psf, num_iters=10, noncirc=False,
                    mask: np.ndarray | None = None) -> np.ndarray
```

Changes from `richardson_lucy_np.py`:

- Drop `use_mkl` -- the `skimage` env has no `mkl_fft`, and a flag that silently
  falls back is worse than no flag.
- Drop `print_diagnostics`; the `print(i, end=" ")` progress loop becomes
  `progress(f"Iteration {i + 1} of {num_iters}", i, num_iters)`, and the loop
  checks `cancel_requested()` so a long run can be stopped.
- Keep `mask` semantics exactly: masked pixels are folded into `HTones` and the
  original values restored at the end.
- Note in the docstring that numpy FFT promotes float32 to float64, so the op
  returns float64 -- that is upstream behaviour, not a bug to fix here.

**4. `richardson_lucy_cupy.py` -- the cupy op.** Same signature and same return
type (`np.ndarray`, via `cp.asnumpy` at the end -- cupy arrays cannot cross the
Appose boundary, and the annotation must resolve during discovery in an env with
no cupy, so `import cupy` stays inside the body). Dropped from the original:
`truth`/`stats` and the RMSE tracking (it pulls in `tnia.metrics` and is a
benchmarking concern, not an op's), `do_unpad` (an op always returns something
the caller can use), and the diagnostics prints. The duplicated `if truth is not
None: stats['rmse'].append(...)` block does not survive the port either.

**5. Behaviour to unify between the two.** Decided -- the proposed column is what
both ports implement. It changes results relative to `richardson_lucy_np.py`,
which gets an NB comment where it happens:

| | numpy original | cupy original | proposed |
| --- | --- | --- | --- |
| circulant, shapes differ | pad psf to image | `pad_to_largest` both | `pad_to_largest` |
| initial estimate | `image` | flat `mean(image)` | flat `mean(image)` |
| non-circulant extent | plain extended size | `get_next_smooth` | `get_next_smooth` |
| negative clamp | `correction[<0] = delta` | `estimate[<0] = delta` | both |

The cupy column is the more recent thinking in each row, so the proposal follows
it. The cost, accepted, is that the numpy op no longer matches
`richardson_lucy_np.py` number-for-number.

**6. `envs/cupy/pixi.toml`.** `python 3.11`, `numpy`, `cupy`,
`cuda-version = "12.*"`, `appose >= 0.11`, plus
`[system-requirements] cuda = "12.0"` as `envs/cellpose` has. Platforms
`linux-64` and `win-64` only -- there is no CUDA on macOS, and listing a platform
the env cannot resolve on is a worse failure than not listing it. Python must
stay >= 3.10 for the `np.ndarray | None` annotation to resolve during discovery.

**7. Tests.**

- `tests/test_pad.py` -- host-only, no marker: pad/unpad round trip in 2D and 3D,
  odd and even size deltas, `next_smooth` against known values.
- `tests/test_ops.py` -- the new ops are discovered, `gaussian_psf` is in
  `skimage`, the two RL ops declare different envs, and bump the
  `len(SPECS) >= 12` floor.
- `tests/test_ops_e2e.py`, `@pytest.mark.env("skimage")`: PSF shape/sum/symmetry;
  and a real round trip -- blobs, convolve with a known PSF, add Poisson noise,
  deconvolve, and assert the result is closer to the truth than the blurred input
  is. That is the assertion that actually catches a broken port; a shape check
  does not.
- Same round trip under `@pytest.mark.env("cupy")`, plus a cross-backend test
  asserting the two agree to a loose tolerance when both envs exist.
- **Gap to close:** `conftest.py` skips on "is the env built", not "is there a
  GPU". A machine with the `cupy` env installed but no NVIDIA card will fail
  rather than skip. Add a hardware check -- simplest is for the cupy test to
  probe `cp.cuda.runtime.getDeviceCount()` in the worker and skip on 0.

**8. `interactive_tests/interactive_test_decon.py`.** Follows
`interactive_test_scikit.py` in spirit -- flat script, no pytest, prints as it
goes -- but napari instead of matplotlib, as asked. Flow: `synthetic_nuclei`
(3D, already an op) -> `gaussian_psf` 3D -> blur + Poisson noise -> both RL ops
-> `viewer.add_image` for truth / blurred / numpy result / cupy result, then
`napari.run()`. Two notes: the cupy run wraps in try/except so the script stays
useful on a machine without CUDA, and the blur is done inline with numpy FFT
rather than adding a `convolve` op (porting `forward.py` is future work).
`interactive_test_scikit.py` still imports the pre-rename `from opkit import
runner` / `from ops import ...` -- the new script uses `import skop` /
`from skop.ops.deconvolve import ...` rather than copying that.

**9. README.** Add `skop.ops.kernels.psf:gaussian_psf`, `skop.ops.deconvolve:*`
and the `cupy` env to the table.

## Sequencing

Steps 1-3 and 7's first three bullets are one unit of work and land together --
they need no new environment, so they are testable immediately on any machine.
Steps 4, 6 and the cupy tests are the second unit and need a CUDA box to verify.
Steps 8-9 close it out.

## Explicitly not in scope

`richardson_lucy_variable.py`, `fftdeconv.py`, `richardson_lucy_dask_cp.py`,
`forward.py`, `psfs.py` (Gibson-Lanni etc.), the OpenCL backend, and any
backend-dispatching op. The layout above is chosen so each of these is an added
file rather than a rewrite.

