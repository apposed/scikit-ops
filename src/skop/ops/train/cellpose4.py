"""Finetune CellposeSAM on your own patches.

Runs in the shared 'pytorch' environment, beside
``skop.ops.segment.cellpose4`` -- so a caller that trains and then segments
builds one environment, not two.

Cellpose 3's trainer is the exception, in ``cellpose3.py`` and its own
environment, for the same reason the inference ops are split.

Takes paths and returns a path, like every training op (design 0011). What
comes back is a **file**, not a directory: that is how Cellpose saves a
model, and it is why a caller listing trained models lists files here where
StarDist would have it list directories.
"""

from __future__ import annotations

from typing import Annotated

from skop import op, progress
from skop.ops.segment.cellpose4 import PretrainedModel

from .._util import channel_axis, to_gray
from ._util import read_pairs, relay_epochs, write_history


@op(env="pytorch")
def train_cellpose4(
    images: list[str],
    labels: list[str],
    model_dir: str,
    name: str = "cellpose4_model",
    model: PretrainedModel = PretrainedModel.cpsam_v2,
    initial_model: str = "",
    epochs: Annotated[int, {"widget_type": "SpinBox", "min": 1, "max": 5000}] = 100,
    learning_rate: float = 1e-5,
    weight_decay: float = 0.1,
    batch_size: Annotated[int, {"widget_type": "SpinBox", "min": 1, "max": 64}] = 1,
    patch_size: Annotated[
        int, {"widget_type": "SpinBox", "min": 64, "max": 1024, "step": 32}
    ] = 256,
    min_train_masks: Annotated[int, {"widget_type": "SpinBox", "min": 0, "max": 100}] = 5,
    normalize: bool = True,
    rescale: bool = False,
    nimg_per_epoch: Annotated[int, {"widget_type": "SpinBox", "min": 0, "max": 100000}] = 0,
    collapse_channels: bool = False,
    val_size: Annotated[int, {"widget_type": "SpinBox", "min": 1, "max": 100}] = 2,
    dataset_id: str = "",
) -> str:
    """Finetune CPSAM and write the model to disk.

    Args:
        images: Paths to the input patches, in order.
        labels: Paths to the label patches. ``labels[k]`` is the truth for
            ``images[k]`` -- the pairing is positional, and the caller owns it.
        model_dir: Directory to train into. Cellpose puts the model in a
            ``models/`` subdirectory of it, which is its convention and is
            left alone here.
        name: Filename for the trained model.
        model: Which built-in to finetune. ``cpdino_vitb`` is the small
            backbone and the one to reach for when the others run the GPU
            out of memory. Ignored when ``initial_model`` is set.
        initial_model: A model of your own to continue from. Empty starts
            from the built-in named by ``model``, which is the usual thing
            to do -- these are large models and a handful of patches
            finetunes one, where they would not train it from nothing. It
            must be a Cellpose 4 model, not a Cellpose 3 one, and it brings
            its own backbone.
        epochs: Passes over the training set.
        learning_rate: Cellpose's default for finetuning CPSAM, and low on
            purpose: the weights being adjusted are already good.
        weight_decay: L2 penalty on the weights.
        batch_size: Patches per batch. 1 by default because CPSAM is large
            and the memory is the constraint; raise it if the card allows.
        patch_size: Size of the crops training samples from the patches.
        min_train_masks: Skip patches with fewer masks than this. 0 keeps
            everything -- which is what you want when empty patches are
            themselves the lesson.
        normalize: Whether to percentile-normalize each patch first.
        rescale: Resize each patch so its objects match the diameter the
            network expects. Off by default here, unlike Cellpose 3: CPSAM
            was trained across scales and does not need it.
        nimg_per_epoch: How many patches make up one epoch. 0 means all of
            them, which is Cellpose's own behaviour. Set it lower when the
            training set is large enough that a pass over all of it is too
            coarse a unit to watch.
        collapse_channels: Average a trailing RGB(A) axis to grey before
            training. Must match what you pass ``skop.ops.segment.cellpose4``
            afterwards: a model finetuned on grey and then run on colour is
            being shown something it never saw.
        val_size: How many pairs, taken from the end, to hold out for
            validation.
        dataset_id: Opaque label for the training data, recorded in the
            history file so a plot can mark where it changed.

    Returns:
        The path of the model file. A ``<model file>_history.csv`` is written
        beside it -- epoch, loss, val_loss, run, model, dataset -- appended to
        when training continues, so the curve spans every run the model has
        had.
    """
    import numpy as np
    from cellpose import models, train

    X_train, Y_train, X_val, Y_val = read_pairs(images, labels, val_size)

    if collapse_channels:
        X_train = [to_gray(x) for x in X_train]
        X_val = [to_gray(x) for x in X_val]
        axis = None
    else:
        axis = channel_axis(X_train[0])

    # Labels reach here as whatever the patch writer used; the flow
    # computation wants a plain integer image.
    Y_train = [np.asarray(y).astype(np.uint16) for y in Y_train]
    Y_val = [np.asarray(y).astype(np.uint16) for y in Y_val]

    # A finetuned checkpoint carries its own backbone.
    source = initial_model or model.value
    progress(f"Loading Cellpose model {source}")
    net = models.CellposeModel(gpu=True, pretrained_model=source).net

    progress(
        f"Training {name}: {len(X_train)} train, {len(X_val)} val, "
        f"{epochs} epochs",
        0,
        epochs,
    )
    with relay_epochs(epochs) as relay:
        model_path, _, _ = train.train_seg(
            net,
            train_data=X_train,
            train_labels=Y_train,
            test_data=X_val,
            test_labels=Y_val,
            channel_axis=axis,
            n_epochs=epochs,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            batch_size=batch_size,
            bsize=patch_size,
            min_train_masks=min_train_masks,
            normalize=normalize,
            rescale=rescale,
            # None, not 0: Cellpose reads None as "every image".
            nimg_per_epoch=nimg_per_epoch or None,
            save_path=model_dir,
            model_name=name,
        )

    write_history(model_path, relay.history, name, dataset_id)
    return str(model_path)
