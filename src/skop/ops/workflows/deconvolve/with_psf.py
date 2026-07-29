"""Generate a PSF and deconvolve with it, in one call.

The first workflow: an op with no environment, which runs on the host and
calls other ops through the runner. See docs/spec/workflow-ops.md.

Deconvolution needs a PSF and nobody has one lying about, so in practice every
deconvolution is already these two steps. Doing them by hand means running a
kernel op, finding its output layer, and feeding it back in -- and doing that
again for every change of numerical aperture. What the workflow adds is not
the composition, which is trivial, but the *pairing*: the two choosers sit in
one panel, so trying Gibson-Lanni on the GPU is two combo boxes rather than
two runs.

There is no shared-parameter problem here to speak of. The PSF ops take no
image -- they synthesize a kernel from optics alone -- so the only bound
parameters are the deconvolver's ``image`` and ``psf``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Annotated, NamedTuple

import numpy as np

from skop import Choices, ParamsFor, op, progress, run
from skop.ops.deconvolve.richardson_lucy import richardson_lucy
from skop.ops.deconvolve.richardson_lucy_cupy import richardson_lucy_cupy
from skop.ops.kernels.gibson_lanni import gibson_lanni
from skop.ops.kernels.psf import gaussian_psf
from skop.types import ImageData


class Deconvolved(NamedTuple):
    """What each stage produced, in the order the stages ran.

    Stage order rather than importance: a front end adds these to a viewer in
    declaration order, so the pipeline reads down the layer list, and the last
    stage -- the answer -- ends up on top where a new layer belongs.
    """

    #: The kernel the deconvolution used. Returned rather than discarded
    #: because it is the parameter people actually iterate on, and looking at
    #: it is the fastest way to see that the optics were entered wrong.
    psf: ImageData
    #: The restored image.
    image: ImageData


@op()
def deconvolve_with_psf(
    image: ImageData,
    psf_op: Annotated[
        Callable,
        Choices(gaussian=gaussian_psf, gibson_lanni=gibson_lanni),
    ] = gaussian_psf,
    psf_args: Annotated[dict | None, ParamsFor("psf_op")] = None,
    decon_op: Annotated[
        Callable,
        Choices(cpu=richardson_lucy, gpu=richardson_lucy_cupy),
    ] = richardson_lucy,
    decon_args: Annotated[
        dict | None, ParamsFor("decon_op", binds=("image", "psf"))
    ] = None,
) -> Deconvolved:
    """Deconvolve an image with a freshly generated PSF.

    Args:
        image: The image to restore.
        psf_op: Which kernel op to build the PSF with. ``gaussian`` is an
            approximation that is fast and always behaves; ``gibson_lanni``
            models the optics, and wants real numbers for them.
        psf_args: Settings for the chosen kernel op.
        decon_op: Which deconvolver to use. ``gpu`` needs CUDA and cupy, and
            picking it triggers an environment build the first time.
        decon_args: Settings for the chosen deconvolver, minus the image and
            the PSF, which this workflow supplies.

    Returns:
        psf: The PSF it was deconvolved with.
        image: The deconvolved image.

    Note: the two halves are not checked against each other. A PSF of a
    different rank than the image is a real technique rather than a mistake --
    a 2-D PSF deconvolves a stack slice by slice, and a 3-D one restores
    astigmatic single-molecule data -- so this passes whatever it is given
    straight through and lets the deconvolver say what it thinks.
    """
    progress("Generating PSF")
    psf = run(psf_op, **(psf_args or {}))

    progress("Deconvolving")
    restored = run(decon_op, image=image, psf=psf, **(decon_args or {}))

    return Deconvolved(np.asarray(psf), np.asarray(restored))
