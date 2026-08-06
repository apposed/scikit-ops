"""Cellpose 3, the model zoo that came before CellposeSAM.

Cellpose 4 replaced a shelf of specialised models -- cyto3, nuclei, and the
ones trained on bacteria, yeast and tissue -- with one CellposeSAM model that
is better on most things and not on everything. ``skop.ops.segment.cellpose``
is that one; this is the shelf, kept because a model trained on your organism
still beats a general one that never saw it, and because published results
were produced with these.

A separate op in a separate environment rather than a version parameter on
one op. The two are not API-compatible: v3's ``models.Cellpose`` became v4's
``models.CellposeModel``, and ``eval`` lost a return value on the way. The
old ``CellposeSegmenter`` in napari-ai-lab carried both, branching on
``cellpose.version`` at every call site -- which is exactly the runtime
version sniffing an environment per op is meant to replace. Pinning the
environment to ``cellpose <4`` means the code below has one path.

``models.Cellpose``, not ``models.CellposeModel``, is the important detail
for the built-in zoo. In v3 they are different classes: ``Cellpose`` bundles
a SizeModel, so ``diameter=None`` makes it estimate the diameter before
segmenting. ``CellposeModel`` has no such thing and would silently use its
default. That is also why v3's ``eval`` returns four values where v4 returns
three -- the fourth is the diameter it estimated.

A model you trained yourself has to use ``CellposeModel`` anyway, since a
SizeModel is trained separately and a finetuned checkpoint does not carry
one. So the op has both classes, picked by whether ``pretrained_model`` is
set, and what changes with it is what "estimate the diameter" can mean --
see ``diameter`` below.
"""

from __future__ import annotations

from enum import Enum
from functools import partial
from pathlib import Path
from typing import Annotated

import numpy as np

from skop import Axes, op, progress
from skop.types import ImageData, LabelsData

from .._util import to_gray


class PretrainedModel(Enum):
    """Cellpose 3's built-in models, by what they were trained on.

    The generalist pair first. The rest are the specialists, each finetuned
    on one imaging domain and worth reaching for when the generalist misses
    -- which is the reason this op exists at all.
    """

    cyto3 = "cyto3"
    nuclei = "nuclei"
    bacteria_phase = "bact_phase_cp3"
    bacteria_fluorescence = "bact_fluor_cp3"
    yeast_phase = "yeast_PhC_cp3"
    yeast_brightfield = "yeast_BF_cp3"
    tissue = "tissuenet_cp3"
    live_cell = "livecell_cp3"
    deepbacs = "deepbacs_cp3"


class CytoplasmChannel(Enum):
    """Which colour the cells to segment are in.

    Cellpose 3 states this as a number -- 0 for grey, then 1, 2, 3 for red,
    green and blue -- which is one-based, is not an axis index, and is easy
    to transpose by accident. Named instead.

    ``grayscale`` is the default and collapses the colours into one plane.
    """

    grayscale = "grayscale"
    red = "red"
    green = "green"
    blue = "blue"


class NucleusChannel(Enum):
    """Which colour the nuclei are in, if a second channel has them.

    The optional half of Cellpose 3's channel pair: a nuclear stain helps it
    separate cells that touch. ``none`` is the default and means there isn't
    one.
    """

    none = "none"
    red = "red"
    green = "green"
    blue = "blue"


#: Cellpose's own numbering for the two channel parameters. One-based over
#: red/green/blue, with 0 meaning grey in the first slot and absent in the
#: second.
_CHANNEL_NUMBER = {"grayscale": 0, "none": 0, "red": 1, "green": 2, "blue": 3}


@op(env="cellpose3")
def cellpose3(
    image: Annotated[ImageData, Axes("y", "x", "c?")],
    model: PretrainedModel = PretrainedModel.cyto3,
    pretrained_model: Path | None = None,
    cytoplasm_channel: CytoplasmChannel = CytoplasmChannel.grayscale,
    nucleus_channel: NucleusChannel = NucleusChannel.none,
    diameter: Annotated[
        float,
        {"widget_type": "FloatSpinBox", "min": 0.0, "max": 1000.0, "step": 1.0},
    ] = 30.0,
    flow_threshold: Annotated[
        float,
        {"widget_type": "FloatSlider", "min": 0.0, "max": 3.0, "step": 0.05},
    ] = 0.4,
    cellprob_threshold: Annotated[
        float,
        {"widget_type": "FloatSlider", "min": -6.0, "max": 6.0, "step": 0.1},
    ] = 0.0,
    niter: int = 200,
    min_size: int = 15,
    normalize: bool = True,
    use_gpu: bool = True,
) -> LabelsData:
    """Segment cells with one of Cellpose 3's pretrained models.

    Args:
        image: Plane to segment. A trailing RGB(A) axis is collapsed. A
            caller naming its axes may hand this a stack instead, and it is
            segmented plane by plane.
        model: Which pretrained model to run. cyto3 is the generalist; the
            specialists are worth trying when it misses on your organism.
            Ignored when ``pretrained_model`` is set.
        pretrained_model: A model you trained yourself, which takes
            precedence over ``model``. It must be a Cellpose 1-3 model, not
            a CPSAM one -- those load in ``cellpose``, and
            ``skop.models.cellpose_flavor`` tells the two apart from the
            file.
        cytoplasm_channel: Which colour the cells are in. Left at
            ``grayscale``, the colours are collapsed into one plane before
            Cellpose sees them -- which is right for a single-channel image
            and wrong for a stain that only shows in one colour.
        nucleus_channel: Which colour the nuclei are in, when a nuclear
            stain is present to help separate touching cells. Ignored unless
            ``cytoplasm_channel`` names a colour too: Cellpose reads the
            pair, and a grey first channel makes it drop the second.
        diameter: Expected cell diameter in pixels, where 0 means "work it
            out". What that costs depends on the model. For the built-in
            zoo it runs a size model over the image first, which is the
            right choice when object size varies between images. A model of
            your own has no size model, so 0 falls back to the diameter it
            was trained at -- right when your images resemble your training
            set, and worth overriding when they do not.
        flow_threshold: Maximum allowed flow error per mask. Higher accepts
            more irregular shapes; lower discards them.
        cellprob_threshold: Cell probability cutoff. Lower finds more, and
            grows the masks it finds.
        niter: Dynamics iterations. More is slower and helps long or
            branched cells, whose pixels need further to travel.
        min_size: Discard masks smaller than this many pixels. -1 keeps
            everything.
        normalize: Whether to percentile-normalize the plane first. Cellpose
            does this per call, so a caller running it slice by slice over a
            stack gets each plane stretched to its own range -- which turns a
            faint plane at the top of a volume into a bright one full of
            detections. Turn it off and normalize the volume beforehand when
            that matters; see ``skop.ops.workflows.segment``.
        use_gpu: Whether to use the GPU when one is available. Falls back to
            CPU rather than failing if the GPU model cannot be built.

    Returns:
        A label image, one integer per detected cell.

    Note: weights are downloaded into Cellpose's own cache on first use.
    """
    from cellpose import models

    # Naming a colour means Cellpose picks the channels itself, so it has to
    # be given them: collapsing to grey first leaves nothing to pick from,
    # and Cellpose does not complain -- its reshape short-circuits on a
    # single-channel image and never reads `channels` at all.
    select = {}
    if cytoplasm_channel is not CytoplasmChannel.grayscale:
        select["channels"] = [
            _CHANNEL_NUMBER[cytoplasm_channel.value],
            _CHANNEL_NUMBER[nucleus_channel.value],
        ]
        plane = np.asarray(image)
        progress(
            f"Segmenting the {cytoplasm_channel.value} channel"
            + (
                f", nuclei in {nucleus_channel.value}"
                if nucleus_channel is not NucleusChannel.none
                else ""
            )
        )
    else:
        plane = to_gray(image)

    # int32 reaches eval from label-image arithmetic and upstream ops, and
    # Cellpose's normalization does not handle it. Carried over from
    # napari-ai-lab's CellposeSegmenter, which hit it.
    if plane.dtype == np.int32:
        plane = plane.astype(np.float32)

    if pretrained_model is not None:
        # CellposeModel, since a finetuned checkpoint carries no SizeModel.
        name = Path(pretrained_model).name
        build = partial(models.CellposeModel, pretrained_model=str(pretrained_model))
    else:
        name = model.value
        build = partial(models.Cellpose, model_type=model.value)

    progress(f"Loading Cellpose 3 model {name}")
    try:
        net = build(gpu=use_gpu)
    except (AttributeError, ValueError, TypeError) as exc:
        # A GPU that torch can see but Cellpose cannot use fails here rather
        # than at eval. Segmenting slowly beats not segmenting.
        progress(f"GPU model unavailable ({exc}); falling back to CPU")
        net = build(gpu=False)

    if diameter <= 0 and pretrained_model is not None:
        # No size model to ask, so the diameter the model was trained at is
        # the best available answer. Cellpose would otherwise quietly use
        # CellposeModel's default of 30.
        diameter = float(getattr(net, "diam_labels", 0.0) or 0.0)
        progress(f"Using the model's trained diameter, {diameter:.1f} px")

    progress("Running Cellpose 3")
    # Four returns in v3 -- masks, flows, styles, diams -- where v4 gives
    # three. Indexed rather than unpacked so the difference cannot bite.
    result = net.eval(
        plane,
        diameter=diameter if diameter > 0 else None,
        flow_threshold=flow_threshold,
        cellprob_threshold=cellprob_threshold,
        niter=niter,
        min_size=min_size,
        normalize=normalize,
        **select,
    )
    return np.asarray(result[0]).astype(np.uint16)
