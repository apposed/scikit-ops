"""Which Cellpose a model is for, from its name or its file.

Host-only: no environment, no torch, no download. The checkpoints here are
built by hand -- a torch save file is a zip holding a pickled state dict, and
``cellpose_flavor`` reads only the key names out of it, so a few hundred bytes
stand in for the real 600 MB one.

The bug worth catching cheaply is a model routed to the wrong op, which fails
deep inside torch with an error about unexpected keys and says nothing about
the actual mistake.
"""

from __future__ import annotations

import pickle
import zipfile

import pytest

from skop.models import BUILTIN_MODELS, CellposeFlavor, cellpose_flavor

# Real key names, trimmed. CPSAM's backbone is a ViT and lives under
# `encoder.`; CPnet is a U-Net of `downsample`/`upsample` blocks.
CPSAM_KEYS = [
    "encoder.pos_embed",
    "encoder.patch_embed.proj.weight",
    "encoder.blocks.0.attn.qkv.weight",
    "diam_mean",
    "diam_labels",
]
CPNET_KEYS = [
    "downsample.down.res_down_0.conv.conv_0.0.weight",
    "upsample.up.res_up_0.conv.conv_0.0.weight",
    "output.2.weight",
    "diam_mean",
    "diam_labels",
]


def checkpoint(tmp_path, filename, keys, root="my_model"):
    """A torch-shaped zip: one pickled state dict under an arbitrary root.

    ``root`` is whatever the model was called when it was saved, which is why
    the entry is found by suffix rather than by name.
    """
    path = tmp_path / filename
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(f"{root}/data.pkl", pickle.dumps(dict.fromkeys(keys, 0)))
    return path


@pytest.mark.parametrize(
    ("name", "flavor"),
    [
        ("cpsam", CellposeFlavor.sam),
        ("cyto3", CellposeFlavor.cpnet),
        ("nuclei", CellposeFlavor.cpnet),
        ("cyto2", CellposeFlavor.cpnet),
        ("CP", CellposeFlavor.cpnet),
    ],
)
def test_builtin_names(name, flavor):
    assert cellpose_flavor(name) is flavor


def test_cpsam_is_the_only_version_4_builtin():
    # Cellpose 4 replaced the whole zoo with one model, which is what makes a
    # name enough to settle the architecture.
    sam = [n for n, f in BUILTIN_MODELS.items() if f is CellposeFlavor.sam]
    assert sam == ["cpsam"]


def test_unknown_name_is_reported_as_such():
    with pytest.raises(ValueError, match="neither a built-in"):
        cellpose_flavor("cyto99")


def test_sam_checkpoint(tmp_path):
    path = checkpoint(tmp_path, "mine", CPSAM_KEYS)
    assert cellpose_flavor(path) is CellposeFlavor.sam


def test_cpnet_checkpoint(tmp_path):
    path = checkpoint(tmp_path, "mine", CPNET_KEYS, root="trained_2024_10_11")
    assert cellpose_flavor(path) is CellposeFlavor.cpnet


def test_diameter_keys_alone_decide_nothing(tmp_path):
    # diam_mean and diam_labels are in both architectures. They are the keys
    # you reach for first and the ones that cannot answer the question.
    path = checkpoint(tmp_path, "mine", ["diam_mean", "diam_labels"])
    with pytest.raises(ValueError, match="not a recognized"):
        cellpose_flavor(path)


def test_missing_file(tmp_path):
    with pytest.raises(ValueError, match="neither a built-in"):
        cellpose_flavor(tmp_path / "absent")


def test_not_a_checkpoint(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("this is not a model")
    with pytest.raises(ValueError, match="not a torch checkpoint"):
        cellpose_flavor(path)


def test_zip_without_a_state_dict(tmp_path):
    path = tmp_path / "archive.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("readme.txt", "no state dict here")
    with pytest.raises(ValueError, match="not a torch checkpoint"):
        cellpose_flavor(path)


def test_a_name_beats_a_file_of_that_name(tmp_path, monkeypatch):
    # Documented precedence: a plain string is a built-in first. Someone whose
    # working directory happens to hold a file called `cyto3` still gets the
    # built-in, and spells it as a Path when they mean the file.
    path = checkpoint(tmp_path, "cyto3", CPSAM_KEYS)
    monkeypatch.chdir(tmp_path)

    assert cellpose_flavor("cyto3") is CellposeFlavor.cpnet  # the built-in
    assert cellpose_flavor(path) is CellposeFlavor.sam  # the file
