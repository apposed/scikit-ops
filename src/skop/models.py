"""Which Cellpose a model is for, answered without loading it.

A trained Cellpose model is a bare ``torch.save`` of a state dict under a name
its author chose, so the file says nothing about which Cellpose it belongs to.
Handing a Cellpose 3 model to the ``cellpose`` op, or the reverse, fails deep
inside torch with an error about unexpected keys.

The architecture is legible in the key names, though, and a torch checkpoint
is a zip whose pickled header lists them. Reading that header is a ~50 KB read
even when the weights beside it are 600 MB, and it needs only the standard
library -- so the *host* can route a model to the right op before dispatching
it, which is the point. ``cellpose`` and ``cellpose3`` live in environments
that cannot both exist, and the host has neither.

Built-in models are answered from a name instead, so that a caller with a
mixed list -- some built-ins, some files of its own -- can ask the same
question of every entry.

Standard library only, like ``skop.boxes`` and ``skop.masks``, so the host, a
worker and a front end can all import it.
"""

from __future__ import annotations

import zipfile
from enum import Enum
from pathlib import Path

__all__ = ["BUILTIN_MODELS", "CellposeFlavor", "cellpose_flavor"]


class CellposeFlavor(Enum):
    """Which Cellpose architecture a model file was trained on.

    ``sam`` is the ViT-backed CPSAM of Cellpose 4, and runs under the
    ``cellpose`` op. ``cpnet`` is the U-Net that Cellpose 1 through 3 shared,
    and runs under ``cellpose3``. The two are not interchangeable.
    """

    sam = "sam"
    cpnet = "cpnet"


#: A key prefix that appears in one architecture and not the other. CPSAM's
#: backbone is a ViT, so every block is under ``encoder.``; CPnet is a U-Net
#: with a matching ``upsample.up.res_up_*``. ``diam_mean`` and ``diam_labels``
#: are in both and tell you nothing.
_SIGNATURES = (
    (b"encoder.blocks.", CellposeFlavor.sam),
    (b"downsample.down.res_down_", CellposeFlavor.cpnet),
)


#: The models each Cellpose ships with, by the name you pass it. Cellpose 4
#: replaced the whole zoo with one model, so no name is in both versions and a
#: name alone settles the architecture. The Cellpose 3 list is its full model
#: zoo, older ``cyto``/``cyto2`` included -- they still load there, and a
#: published result may name one.
BUILTIN_MODELS: dict[str, CellposeFlavor] = {
    "cpsam": CellposeFlavor.sam,
    **dict.fromkeys(
        (
            "cyto",
            "cyto2",
            "cyto3",
            "nuclei",
            "cyto2_cp3",
            "tissuenet_cp3",
            "livecell_cp3",
            "yeast_PhC_cp3",
            "yeast_BF_cp3",
            "bact_phase_cp3",
            "bact_fluor_cp3",
            "deepbacs_cp3",
            "transformer_cp3",
            "neurips_cellpose_default",
            "neurips_cellpose_transformer",
            "neurips_grayscale_cyto2",
            "CP",
            "CPx",
            "TN1",
            "TN2",
            "TN3",
            "LC1",
            "LC2",
            "LC3",
            "LC4",
        ),
        CellposeFlavor.cpnet,
    ),
}


def cellpose_flavor(model: str | Path) -> CellposeFlavor:
    """Identify which Cellpose a model is for.

    Args:
        model: Either the name of a built-in model, such as ``"cyto3"`` or
            ``"cpsam"``, or the path of one saved by ``torch.save``. A name
            is looked up; a path is read.

    Returns:
        Which architecture it is, and so which op can run it.

    Raises:
        ValueError: If the name is not a built-in, or the file is not a torch
            checkpoint, or is one whose keys match neither architecture.
            Guessing here would send the model to an environment that dies on
            it, so this reports rather than picks.
    """
    # A plain string is a name first and a path second, so that "cyto3" means
    # the built-in even when run beside a file of that name. Spell it as a
    # Path to mean the file.
    if isinstance(model, str) and model in BUILTIN_MODELS:
        return BUILTIN_MODELS[model]

    path = Path(model)
    if not path.exists():
        raise ValueError(
            f"{model} is neither a built-in Cellpose model nor an existing file"
        )
    if not zipfile.is_zipfile(path):
        # Either not a checkpoint at all, or one from torch before 1.6. No
        # Cellpose that skop can run wrote the latter.
        raise ValueError(f"{path} is not a torch checkpoint")

    header = _header(path)
    for signature, flavor in _SIGNATURES:
        if signature in header:
            return flavor
    raise ValueError(f"{path} is not a recognized Cellpose model")


def _header(path: Path) -> bytes:
    """The pickled state dict header, without the tensors it points at.

    A torch checkpoint is a zip holding ``<name>/data.pkl`` -- the state dict
    with its storages replaced by references -- beside one entry per tensor.
    The root name is whatever the model was called when it was saved, so the
    entry is found by suffix.
    """
    with zipfile.ZipFile(path) as archive:
        names = [n for n in archive.namelist() if n.endswith("data.pkl")]
        if not names:
            raise ValueError(f"{path} is a zip, but not a torch checkpoint")
        return archive.read(names[0])
