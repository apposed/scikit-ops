"""Train a StarDist 2D model from a list of image/label file pairs.

Ported from napari-ai-lab's ``StardistSegmenter.train``, with the directory
convention left behind: that segmenter globbed ``input0/`` and
``ground_truth0/`` and read an ``info.json`` for the axes. Here the caller has
already done that, and passes the result. See design 0011 for why.

Shares the 'stardist-tf' environment with the inference ops next door, so a
caller that trains and then predicts builds one environment, not two.
"""

from __future__ import annotations

from typing import Annotated

import numpy as np

from skop import op, progress


@op(env="stardist-tf")
def train_stardist2d(
    images: list[str],
    labels: list[str],
    model_dir: str,
    name: str = "stardist_model",
    image_axes: str = "YX",
    epochs: Annotated[int, {"widget_type": "SpinBox", "min": 1, "max": 5000}] = 100,
    steps_per_epoch: Annotated[
        int, {"widget_type": "SpinBox", "min": 1, "max": 1000}
    ] = 100,
    train_patch_size: Annotated[
        int, {"widget_type": "SpinBox", "min": 32, "max": 2048, "step": 32}
    ] = 128,
    train_batch_size: Annotated[
        int, {"widget_type": "SpinBox", "min": 1, "max": 64}
    ] = 4,
    unet_n_depth: Annotated[int, {"widget_type": "SpinBox", "min": 1, "max": 6}] = 3,
    n_rays: Annotated[int, {"widget_type": "SpinBox", "min": 4, "max": 128}] = 32,
    val_size: Annotated[int, {"widget_type": "SpinBox", "min": 1, "max": 100}] = 2,
    sparse: bool = False,
) -> str:
    """Train a StarDist 2D model and write it to disk.

    Args:
        images: Paths to the input patches, in order.
        labels: Paths to the label patches. ``labels[k]`` is the truth for
            ``images[k]`` -- the pairing is positional, and the caller owns it.
        model_dir: Directory to write the model into. StarDist creates
            ``<model_dir>/<name>`` and puts its config and weights there.
        name: Name for the trained model, and the subdirectory it lands in.
        image_axes: Axes of the input patches: "YX" for single channel,
            "YXC" when they carry a channel axis. A trivial channel axis is
            appended for "YX", since StarDist wants one. Named
            ``image_axes`` and not ``axes`` because ``Runner.run`` has an
            ``axes`` parameter of its own -- a bare ``axes=`` kwarg binds
            to that one, and the op never sees it.
        epochs: Number of training epochs.
        steps_per_epoch: Batches per epoch.
        train_patch_size: Size of the patches StarDist samples while training.
            Must not exceed the size of the patches on disk.
        train_batch_size: Patches per batch.
        unet_n_depth: Depth of the U-Net backbone.
        n_rays: Number of radial directions the star-convex polygons use.
        val_size: How many pairs, taken from the end, to hold out for
            validation.
        sparse: Set when the labels use StarDist's sparse convention, where
            unlabeled pixels are marked rather than assumed to be background.
            Shifts labels down by one so those pixels go negative and their
            loss is switched off.

    Returns:
        The path the model was written to: ``<model_dir>/<name>``.
    """
    import os

    import keras
    from stardist.models import Config2D, StarDist2D
    from tifffile import imread

    class _ProgressCallback(keras.callbacks.Callback):
        """Relay Keras epoch events to whoever is running the op.

        Defined here rather than at module level: keras is importable only
        inside the worker, and this module is imported on the host to read
        the op's signature.
        """

        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            loss = logs.get("loss", float("nan"))
            val_loss = logs.get("val_loss", float("nan"))
            progress(
                f"Epoch {epoch + 1}/{epochs} — "
                f"loss {loss:.4f}, val_loss {val_loss:.4f}",
                epoch + 1,
                epochs,
            )

    if len(images) != len(labels):
        raise ValueError(
            f"{len(images)} images and {len(labels)} labels: a training set "
            f"is pairs, so the two lists must be the same length."
        )
    if len(images) <= val_size:
        raise ValueError(
            f"{len(images)} pairs is not enough to hold {val_size} back for "
            f"validation and still train on the rest."
        )

    # The channel count comes from the patches, not from a guess: "YXC" says
    # there is a channel axis, not how wide it is.
    add_trivial_channel = not image_axes.endswith("C")

    progress(f"Reading {len(images)} pairs", 0, len(images))
    X, Y = [], []
    for i, (image_path, label_path) in enumerate(zip(images, labels)):
        x = imread(image_path)
        if add_trivial_channel:
            x = x[..., np.newaxis]
        X.append(x)
        Y.append(imread(label_path))
        progress(current=i + 1)

    # Held out from the end, matching what the pairs were ordered by. Not a
    # random split: a caller who wants one shuffles its own lists, where it
    # can record the seed alongside the patches.
    X_train, Y_train = X[:-val_size], Y[:-val_size]
    X_val, Y_val = X[-val_size:], Y[-val_size:]

    if sparse:
        Y_train = [y.astype(np.int32) - 1 for y in Y_train]
        Y_val = [y.astype(np.int32) - 1 for y in Y_val]

    n_channel_in = X[0].shape[-1]

    X_train = np.asarray(X_train, dtype=np.float32)
    Y_train = np.asarray(Y_train, dtype=np.int32)
    X_val = np.asarray(X_val, dtype=np.float32)
    Y_val = np.asarray(Y_val, dtype=np.int32)

    config = Config2D(
        n_rays=n_rays,
        axes=image_axes if image_axes.endswith("C") else image_axes + "C",
        n_channel_in=n_channel_in,
        train_patch_size=(train_patch_size, train_patch_size),
        train_batch_size=train_batch_size,
        unet_n_depth=unet_n_depth,
    )
    net = StarDist2D(config=config, name=name, basedir=model_dir)
    net.prepare_for_training()
    net.callbacks.append(_ProgressCallback())

    # Keras' own progress bar has to go, and StarDist hardcodes verbose=1
    # (model2d.py), so force it off at the one call it reaches.
    #
    # The bar writes to stdout with \r and no newline -- and stdout is the
    # pipe Appose carries its protocol on. Appose then prints an UPDATE onto
    # that unterminated line, the host cannot parse the result, and drops it
    # (service.py, _stdout_loop). The symptom is every epoch event vanishing
    # while the ones before and after training arrive fine.
    _fit = net.keras_model.fit
    net.keras_model.fit = lambda *a, **kw: _fit(*a, **{**kw, "verbose": 0})

    progress(
        f"Training {name}: {len(X_train)} train, {len(X_val)} val, "
        f"{epochs} epochs",
        0,
        epochs,
    )
    net.train(
        X_train,
        Y_train,
        validation_data=(X_val, Y_val),
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
    )

    return os.path.join(model_dir, name)

