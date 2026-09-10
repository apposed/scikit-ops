"""Finetune one of Cellpose 3's models on your own patches.

Runs in the 'cellpose3' environment beside ``skop.ops.segment.cellpose3``.
Separate from ``train_cellpose`` for the same reason the inference ops are
separate: version 4 replaced the model zoo and moved the API, so no pin makes
the two coexist.

The channel settings are the reason to read this rather than assume it
matches CPSAM. Cellpose 3 takes two input planes, cytoplasm and optional
nuclei, and it takes them at training time as well as at inference. The two
must agree -- see ``cytoplasm_channel`` below.
"""

from __future__ import annotations

from typing import Annotated

from skop import op, progress
from skop.ops.segment.cellpose3 import (
    _CHANNEL_NUMBER,
    CytoplasmChannel,
    NucleusChannel,
    PretrainedModel,
)

from ._util import read_pairs, relay_epochs, write_history


@op(env="cellpose3")
def train_cellpose3(
    images: list[str],
    labels: list[str],
    model_dir: str,
    name: str = "cellpose3_model",
    model: PretrainedModel = PretrainedModel.cyto3,
    initial_model: str = "",
    cytoplasm_channel: CytoplasmChannel = CytoplasmChannel.grayscale,
    nucleus_channel: NucleusChannel = NucleusChannel.none,
    epochs: Annotated[int, {"widget_type": "SpinBox", "min": 1, "max": 5000}] = 100,
    learning_rate: float = 0.005,
    weight_decay: float = 1e-5,
    batch_size: Annotated[int, {"widget_type": "SpinBox", "min": 1, "max": 64}] = 8,
    patch_size: Annotated[
        int, {"widget_type": "SpinBox", "min": 64, "max": 1024, "step": 32}
    ] = 224,
    min_train_masks: Annotated[int, {"widget_type": "SpinBox", "min": 0, "max": 100}] = 5,
    normalize: bool = True,
    rescale: bool = True,
    nimg_per_epoch: Annotated[int, {"widget_type": "SpinBox", "min": 0, "max": 100000}] = 0,
    val_size: Annotated[int, {"widget_type": "SpinBox", "min": 1, "max": 100}] = 2,
    dataset_id: str = "",
) -> str:
    """Finetune a Cellpose 3 model and write it to disk.

    Args:
        images: Paths to the input patches, in order.
        labels: Paths to the label patches. ``labels[k]`` is the truth for
            ``images[k]`` -- the pairing is positional, and the caller owns it.
        model_dir: Directory to train into. Cellpose puts the model in a
            ``models/`` subdirectory of it, which is its convention and is
            left alone here.
        name: Filename for the trained model.
        model: Which built-in to start from. cyto3 is the generalist; a
            specialist trained on your imaging domain starts closer and needs
            fewer patches. Ignored when ``initial_model`` is set.
        initial_model: A model of your own to continue from, which takes
            precedence over ``model``. It must be a Cellpose 1-3 model, not a
            CPSAM one; ``skop.models.cellpose_flavor`` tells them apart from
            the file.
        cytoplasm_channel: Which colour the cells are in. Left at
            ``grayscale``, the colours are averaged into one plane before
            Cellpose sees them. **Whatever you choose here you must pass
            again to** ``skop.ops.segment.cellpose3``: the two input planes
            are what the network learned, so a model trained on green
            cytoplasm and run on a grey average is being shown something it
            never saw.
        nucleus_channel: Which colour the nuclei are in. Ignored unless
            ``cytoplasm_channel`` names a colour too -- Cellpose reads the
            pair, and a grey first slot drops the second.
        epochs: Passes over the training set. Cellpose's own default is
            2000, which is a from-scratch number; finetuning a zoo model
            wants far fewer.
        learning_rate: Cellpose 3's default. Three orders larger than
            CPSAM's, because this is a far smaller network.
        weight_decay: L2 penalty on the weights.
        batch_size: Patches per batch.
        patch_size: Size of the crops training samples from the patches.
        min_train_masks: Skip patches with fewer masks than this. 0 keeps
            everything -- which is what you want when empty patches are
            themselves the lesson.
        normalize: Whether to percentile-normalize each patch first.
        rescale: Resize each patch so its objects match the diameter the
            network expects. On by default, and worth keeping: it is what
            lets a model trained at one magnification work at another. Turn
            it off when your objects are all one size and you would rather
            the network learn that size.
        nimg_per_epoch: How many patches make up one epoch. 0 means all of
            them, which is Cellpose's own behaviour. Set it lower when the
            training set is large enough that a pass over all of it is too
            coarse a unit to watch -- 10000 patches is one very long epoch,
            where 50 an epoch gives a loss curve that moves.
        val_size: How many pairs, taken from the end, to hold out for
            validation.
        dataset_id: Opaque label for the training data, recorded in the
            history file so a plot can mark where it changed.

    Returns:
        The path of the model file. A ``<model file>_history.csv`` is written
        beside it -- epoch, loss, val_loss, run, model, dataset -- appended to
        when training continues, so the curve spans every run the model has
        had.

        The mean diameter of the training labels is saved into the model as
        ``diam_labels``. That is what ``skop.ops.segment.cellpose3`` falls
        back to when it is given ``diameter=0`` and a model of yours: a
        finetuned checkpoint carries no size model, so the size it was
        trained at is the best answer available.
    """
    from pathlib import Path

    import numpy as np
    from cellpose import models, train

    # Cellpose makes `<model_dir>/models` without parents.
    Path(model_dir).mkdir(parents=True, exist_ok=True)

    X_train, Y_train, X_val, Y_val = read_pairs(images, labels, val_size)

    # Always passed: without `channels` the training reshape is skipped and
    # a 2-D image reaches normalize_img, which wants a plane axis.
    select = {
        "channels": [
            _CHANNEL_NUMBER[cytoplasm_channel.value],
            _CHANNEL_NUMBER[nucleus_channel.value],
        ]
    }

    # int32 arrives from label arithmetic upstream; normalization chokes on it.
    X_train = [np.asarray(x).astype(np.float32) for x in X_train]
    X_val = [np.asarray(x).astype(np.float32) for x in X_val]
    Y_train = [np.asarray(y).astype(np.uint16) for y in Y_train]
    Y_val = [np.asarray(y).astype(np.uint16) for y in Y_val]

    # CellposeModel, not Cellpose: the SizeModel it bundles trains separately.
    if initial_model:
        source = initial_model
        build = dict(pretrained_model=initial_model)
    else:
        source = model.value
        build = dict(model_type=model.value)

    progress(f"Loading Cellpose 3 model {source}")
    net = models.CellposeModel(gpu=True, **build).net

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
            **select,
        )

    write_history(model_path, relay.history, name, dataset_id)
    return str(model_path)
