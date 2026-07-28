# 0010 — Theoretical PSF ops

Second step in migrating deconvolution to skops. This is a follow-up spec,
written after testing and working with the first implementation a bit.

In the first iteration we implemented deconvolution and the Gaussian PSF. Now we
add the Gibson-Lanni PSF and the paraxial models, based on the code in
`../tnia-python/tnia/deconvolution/psfs.py`. That file has both psfmodels and
sdeconv options. In this implementation we can only do sdeconv, because of the
license (maybe later we figure out two projects, or dual licensing).

The file has several functions. Where it makes sense, make each one an op; where
a function is trivial, make it a utility instead. Functions that use sdeconv will
need an sdeconv pixi of their own. The other functions will probably work in
several environments, so for now perhaps the skimage environment (later we work
towards more complicated pixi schemes). Remember that we don't bother with any
psfmodels code or options, because of the license.

Everything here is a *theoretical* PSF: computed from optical parameters, no
measurement involved. Extracting a PSF from a bead image is deliberately not part
of this — it composes a deconvolution op whose backend depends on the machine, so
it is a workflow op rather than a plain op. See
[workflow-ops](../spec/workflow-ops.md).

We don't want an interactive test for this one, just some standard tests. What we
will try instead is to make a notebook to test it. Adapt
`../tnia-python/notebooks/Deconvolution/decon-bead-edge-handling.ipynb` to the new
API. Give recommendations on what technology to use to make the notebook easier
to maintain on GitHub — but it also has to render right, so if we use a more
text-based notebook format it still has to render correctly.

## Migration plan

### What was checked first

**The licensing holds.** sdeconv 1.0.4 is BSD 3-Clause; psfmodels 0.3.3 is
GPL-3.0. So the `use_psfm` branches get deleted outright rather than defaulted off
-- no GPL code, and no import path that could reach it.

**sdeconv drags in torch.** Its dependencies are torch, torchvision, scipy, numpy
and scikit-image. That is why the PSF code ends in `.cpu().numpy()`, and it means
sdeconv cannot share the skimage environment.

~~Generating a PSF is small work, so the pixi pins a CPU-only torch build:
hundreds of MB rather than multiple GB, and nothing here would use a GPU
anyway.~~ **Reversed during implementation — see "The environment, as built".**
A CPU pin is only cheap on a machine with no other torch environment, and costs
a full extra copy on one that has `envs/pytorch`.

**The sdeconv API moved between 0.x and 1.x.** The original sniffs
`sdeconv.__version__` and branches between `PSFGibsonLanni` and
`SPSFGibsonLanni`. The pixi pins `>=1,<2` and the port keeps only the 1.x path.
Note that sdeconv does not centre its PSF axially, which is what
`recenter_psf_axial` exists to fix.

**And it forces sdeconv to come from PyPI.** conda-forge carries exactly one
sdeconv, 0.1.0, built for python 3.8 — years old, no `ni`/`ns` parameters at
all, and wavelengths in nanometres. So the 1.x pin above is also a decision to
take sdeconv as a `[pypi-dependencies]` entry, which is what makes the
conda-versus-PyPI care below necessary rather than stylistic.

### Where each function goes

| tnia function | becomes | environment |
| --- | --- | --- |
| `gibson_lanni_3D` + `gibson_lanni_3D_partial_confocal` | one op, `gibson_lanni` | `sdeconv` |
| `paraxial_otf` | op | `skimage` |
| `paraxial_psf` | op | `skimage` |
| `recenter_psf_axial` | utility | -- |
| `wave_dictionary` | `Fluorophore` enum | -- |
| `psf_from_beads` | deferred, becomes a command | -- |
| `load_psf`, `load_and_resize_psf` | deferred | -- |
| `gibson_lanni_3D_old` | dropped | -- |

**The two Gibson-Lanni functions collapse into one op**, the same way
`gaussian_2d` and `gaussian_3d` did. `confocal=True` is just
`confocal_factor=2.0`, and the partial-confocal variant is the same computation
plus recentring:

```python
@op(env="sdeconv")
def gibson_lanni(
    xy_size=128, z_size=64, voxel_size_xy=0.1, voxel_size_z=0.2,
    NA=1.4, ni=1.518, ns=1.33, pz=0.0, wavelength=0.53,
    confocal_factor=1.0, recenter=True,
) -> np.ndarray
```

`confocal_factor` of 1 is widefield, 2 is confocal, and in between is the ad-hoc
partial approximation -- the original's warning that this should be used
carefully goes in the docstring, since it is the kind of caveat that gets lost in
a port. `recenter=True` computes taller and crops back, which is what the
partial-confocal function did and what makes a non-zero `pz` usable — though
~~at `z_size + z_size // 2`~~ **that flat headroom proved insufficient past
`pz` = 1 um, and now scales with depth; see "What the port turned up".**

`gibson_lanni_3D_old` uses microscPSF, whose import is already commented out
upstream. It is dead and does not come across.

`load_psf` and `load_and_resize_psf` are file I/O against a `psf.tif` plus
`psf.json` convention. skop has no I/O op convention yet and these would bake a
directory layout into one, so they wait until there is one.

`wave_dictionary` becomes a `Fluorophore` enum so a wavelength can be picked by
name in a GUI -- `stardist2d`'s model enum is the precedent. Its entries are
excitation/emission pairs; the op wants emission.

**`psf_from_beads` is out of scope here, and that is a decision rather than an
omission.** It deconvolves a bead image by a rendered centroid image, so it
composes `richardson_lucy` -- and which deconvolver it should use depends on
whether the machine has a GPU. Under workflow-ops that makes it a command with an
op-valued `deconvolver` parameter, not an op that hardcodes one backend. Two
things it will need when we get to it: `draw_centroids` from
`tnia.segmentation.rendering` vendored in, and a note that the original's
try/except falling back from cupy RL to clij2fft is replaced by an explicit
choice.

### What gets written

```
envs/sdeconv/pixi.toml                    python, sdeconv, CPU torch
src/skop/ops/kernels/gibson_lanni.py      gibson_lanni          (sdeconv)
src/skop/ops/kernels/paraxial.py          paraxial_otf, paraxial_psf  (skimage)
src/skop/ops/kernels/_recenter.py         recenter_psf_axial
src/skop/ops/kernels/__init__.py          re-export the new ops
notebooks/decon_bead_edge_handling.py     jupytext source of truth
notebooks/decon_bead_edge_handling.ipynb  generated, committed so GitHub renders it
```

`kernels` now spans two environments, which is exactly the README's rule for a
namespace being a package rather than a module. It already is one, so nothing has
to move.

### Tests

Standard tests only, no interactive test.

- **Spec level**, in `test_ops.py`: the new ops are discovered, `gibson_lanni`
  declares `sdeconv`, the paraxial ops declare `skimage`.
- **`@pytest.mark.env("skimage")`**: `paraxial_psf` sums to 1 and is symmetric; a
  paraxial OTF is 1 at DC and 0 past the cutoff; `recenter_psf_axial` puts the
  maximum on the centre plane.
- **`@pytest.mark.env("sdeconv")`**: `gibson_lanni` returns the requested shape,
  sums to 1 and peaks on axis; `confocal_factor=2` is strictly narrower than 1; a
  non-zero `pz` makes it axially asymmetric. Keep these small, 32x64x64 or so --
  it is a CPU torch computation.
- No `@pytest.mark.gpu` anywhere: CPU torch, so an NVIDIA card is not needed.

### The notebook

**Recommendation: jupytext pairing.** The tension asked about is real and does not
fully resolve. GitHub renders `.ipynb` natively and renders *nothing else* as a
notebook -- not jupytext `.py`, not MyST `.md`, not Quarto `.qmd`. But `.ipynb` is
JSON with base64 images inside, which is what makes it miserable in git.

So keep both, and be explicit about which is authoritative:

- `notebooks/decon_bead_edge_handling.py` in jupytext percent format is the source
  of truth, and the file to review.
- The `.ipynb` is generated from it and committed, so the GitHub page shows the
  pictures.
- `.gitattributes` gets `*.ipynb linguist-generated=true -diff`, so the notebook
  collapses in a pull request rather than dominating it.
- A CI step runs `jupytext --sync` then `git diff --exit-code`, so the two cannot
  drift.

Alternatives, and why not: `nbstripout` gives clean diffs but renders a notebook
with no images, which defeats the purpose when the output *is* the point.
Publishing MyST or Quarto to GitHub Pages renders well but is a second thing to
build and is invisible on the repo page. marimo is `.py`-native and reactive, but
does not render on GitHub at all.

**The notebook needs data it does not have.** The original reads
`D:\images\tnia-python-images\deconvolution\bead\bead-2.5um.tif`, which is a
Windows path, is not on this machine, and is not in the repo. In order of
preference: generate a synthetic bead in the notebook, which is self-contained and
deterministic -- and edge handling is precisely what a synthetic half-bead
demonstrates; or fetch the real bead with pooch; or add it to `test_images`, which
is already untracked and 67 MB.

Two other things the adaptation has to shed: `tnia.plotting.projections` (a
fifteen-line XY/ZY max-projection helper in the notebook avoids depending on
tnia-python) and `clij2fft` / `RedLionfishDeconv`, replaced by our
`richardson_lucy` and `richardson_lucy_cupy`. The comparison the notebook is
actually about -- circulant versus non-circulant edge handling -- is now just
`noncirc=False` against `noncirc=True` on one op.

### Sequencing

Paraxial, recentring and the enum need no new environment and land first, with
their tests. `gibson_lanni` and `envs/sdeconv` are second. The notebook is last,
once there is an API to write it against.

## The environment, as built

The question this had to answer: if sdeconv gets a pixi of its own, can it be
kept as close to the canonical `envs/pytorch` as possible, so that it (a) reuses
the same packages from cache rather than costing another multi-gigabyte torch,
and (b) is easy to fold into the canonical environment later if it earns its
place there?

**Yes, and the way to do it is to state every conda dependency exactly as
`envs/pytorch` states it** — same channel, same python, same torch packages,
same CUDA — and let sdeconv be the only thing from PyPI. Conda package identity
is `name-version-buildstring`, and pixi hard-links out of a shared package cache
keyed on exactly that, so two environments share a package only when they solve
to the *same build*, not merely the same version.

Solving both and comparing, on linux-64:

| | packages | size |
| --- | --- | --- |
| `envs/pytorch` | 219 | 2854 MB |
| `envs/sdeconv` | 106 | 2543 MB |
| — identical builds, hard-linked | 83 | 2470 MB |
| — unique to sdeconv | 23 | **73 MB** |

So the second environment costs 73 MB on a machine that already has the first:
scikit-image, scipy and some imaging codecs. Everything expensive — `libtorch`,
`libcudnn`, `nccl`, `libcublas` — is a byte-identical build and paid for once.

That is the reversal of the CPU-torch plan above. A CPU torch is a *different
build string* from `pytorch-gpu`, so it shares nothing with the canonical
environment: it would have been several hundred MB of packages that exist
nowhere else on the machine, to avoid a GPU build already sitting in the cache.
The intuition that "a PSF needs no GPU, so pin CPU" optimizes the wrong number.

For (b), folding it in later is deleting `envs/sdeconv/pixi.toml` and adding one
line to `envs/pytorch`'s `[pypi-dependencies]`. Nothing about the conda half has
to be reconciled first, because there is nothing to reconcile.

**Two things to watch.**

`envs/pytorch` leaves torch unpinned (`pytorch-gpu = "*"`), so the alignment
holds only as long as both environments are solved against a similar index.
Solved months apart they can land on different versions and share nothing. If
that starts to bite, pin both to one version rather than pinning only sdeconv.

Separately, and not caused by this work: `cuda-version = "12.2"` in
`envs/pytorch` currently pins linux-64 back to **torch 2.4.1 / cuda120**, from
September 2024, because newer conda-forge `pytorch-gpu` builds are cuda126 and
cuda129. win-64, which carries no such pin, resolves to 2.12.1. `envs/sdeconv`
inherits the same constraint and therefore matches — the sharing above is real
— but the canonical environment being two years behind on Linux and current on
Windows is worth a look on its own.

## What the port turned up

**The paraxial PSF has negative side lobes and Richardson-Lucy diverges on it.**
An ideal paraxial PSF is the Airy intensity pattern, which is non-negative.
Sampling the OTF on a grid and inverting it with a DFT is not: the result rings,
and about 1% of the PSF's mass sits below zero, with individual negatives around
1e-4 of the peak. Small enough to be invisible in simulation, and fatal for
deconvolution, which is built on ratios — fifty iterations of `richardson_lucy`
on an unclipped paraxial PSF reaches 1e53.

The op does not clip. At 1% of the mass the correction is too large to apply
silently, and it would move every number the tnia-python original produced.
Instead the docstring says so and shows the two-line fix, and there are two
tests: one asserting the negative lobes are there, one showing that a clipped
PSF deconvolves normally. Worth knowing before the notebook is written, since
the notebook is *about* deconvolution.

**The recentring headroom has to scale with `pz`, and the plan's flat
`z_size + z_size // 2` does not work.** This spec said that computing at
`z_size + z_size // 2` and cropping back "is what makes a non-zero `pz`
usable". Measured against sdeconv 1.0.4, it does not: the PSF's peak marches
toward plane zero as the point goes deeper, about 6.5 planes per micron of `pz`
at a 0.2 um axial voxel — 1.3 um of apparent focal shift per micron of depth —
and the flat headroom is exhausted around `pz` = 1 um. Beyond that the crop
runs off the front of the volume.

`pz` = 2 um is an entirely ordinary depth, so this is not an edge case; it is
most of the reason to reach for this model. The op now computes
`z_size + z_size // 2 + ceil(3 * pz / voxel_size_z)`. The factor of 3 is
empirical and deliberately generous — the measured shift is ~1.3, doubled to
~2.6 because the peak must clear `z_size // 2` rather than merely stay inside —
and at 3 the margin grows with depth instead of shrinking: 7 spare planes at
`pz` = 0, 31 at `pz` = 16 um. The cost is a taller FFT and nothing else. A test
covers 0, 1, 4 and 8 um.

**Also: sdeconv 1.0.4 does centre its PSF at `pz` = 0.** The original's comment
that it "does NOT center the result from sdeconv" is either about 0.x or about
a case not reproduced here — at `pz` = 0 the peak lands on the middle plane and
the axial profile is symmetric to the digit. Recentring earns its place at
depth, not at zero.

**`recenter_psf_axial` now raises rather than silently returning the wrong
shape.** The original computed `start = cz - newz//2` and sliced without
checking, and Python's slicing is forgiving in exactly the wrong way: a negative
start counts from the far end, an overlong stop truncates. A PSF whose peak sat
near an edge — which is what a large `pz` produces, and `pz` is the whole reason
this function exists — came back with the wrong number of planes and no
complaint.

**Two deviations from the plan above**, both small. `NA` is spelled
`numerical_aperture`, matching what the paraxial functions already called it, so
the three PSF ops name the same optical quantity the same way; a test enforces
it. And `Fluorophore` lives in `_fluorophore.py` rather than inside
`gibson_lanni.py`, since the paraxial ops take a wavelength too and neither
module should have to import the other. It subclasses `float`, so
`wavelength=Fluorophore.DAPI` passes to a parameter annotated as a plain float
and an arbitrary wavelength still works — the enum is a convenience, not the
only way to say it.

### Not in scope

psfmodels in any form; `psf_from_beads`, which becomes a command under workflow-ops;
`load_psf` and `load_and_resize_psf`; sdeconv's own deconvolution algorithms,
since we have our own; and the OpenCL backend still outstanding from the first
spec.
