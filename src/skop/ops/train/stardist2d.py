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


def _take_only_what_is_needed(tf) -> None:
    """Ask TensorFlow for the memory it uses, not for the whole card.

    By default TF claims almost all free VRAM the first time it touches the
    GPU. That is fine for a process that owns the machine and wrong for one
    of several -- a napari session holding torch models, a second worker, or
    just this op called twice. The symptom is a RESOURCE_EXHAUSTED failure
    while nvidia-smi shows most of the memory held by a process that is not
    using it.

    Must run before any GPU work; TensorFlow refuses to change this once the
    device is initialised, which is why the failure is ignored rather than
    raised.
    """
    for gpu in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:  # already initialised; nothing to be done here
            pass


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
    grid_size_xy: Annotated[int, {"widget_type": "SpinBox", "min": 1, "max": 8}] = 1,
    n_rays: Annotated[int, {"widget_type": "SpinBox", "min": 4, "max": 128}] = 32,
    val_size: Annotated[int, {"widget_type": "SpinBox", "min": 1, "max": 100}] = 2,
    initial_model: str = "",
    dataset_id: str = "",
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
        grid_size_xy: Predict on a subsampled grid, widening the field of
            view. ``train_patch_size`` must stay divisible by
            ``grid_size_xy * 2 ** unet_n_depth``.
        n_rays: Number of radial directions the star-convex polygons use.
        initial_model: Continue training from this model rather than from
            random weights. Either a directory holding a StarDist model, or
            the name of a pretrained one ("2D_versatile_fluo"), downloaded
            on first use -- resolved here rather than by the caller, since
            only this environment has StarDist to ask. Whichever it is, it is copied to the destination
            first, so the model trained from is left alone unless it *is* the
            destination. Its saved architecture and patch/batch settings are
            used, and the corresponding arguments above are ignored -- they
            are baked into a trained model and cannot change.
        dataset_id: Opaque label for the training data, recorded in
            history.csv so a plot can mark where it changed. Nothing here
            interprets it -- what identifies a dataset is the caller's
            business.
        val_size: How many pairs, taken from the end, to hold out for
            validation.
        sparse: Set when the labels use StarDist's sparse convention, where
            unlabeled pixels are marked rather than assumed to be background.
            Shifts labels down by one so those pixels go negative and their
            loss is switched off.

    Returns:
        The path the model was written to: ``<model_dir>/<name>``. A
        ``history.csv`` is written there too -- epoch, loss, val_loss, plus
        the run number, the model name and the dataset id -- and appended to
        when training continues. So the curve spans every run the model and
        its ancestors have had, and a change in any of the last three columns
        marks where something about the training changed.
    """
    import csv
    import json
    import os
    import shutil

    import keras
    import tensorflow as tf
    from stardist.models import Config2D, StarDist2D
    from tifffile import imread

    _take_only_what_is_needed(tf)

    history = []

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
            history.append((epoch + 1, loss, val_loss))
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

    # Lists of arrays, not one stacked array. StarDist takes either, and
    # stacking demands every patch be the same size -- which a patch
    # directory written across several sessions is not. It crops
    # train_patch_size out of each one anyway, so only being at least that
    # big matters.
    X_train = [x.astype(np.float32) for x in X_train]
    Y_train = [y.astype(np.int32) for y in Y_train]
    X_val = [x.astype(np.float32) for x in X_val]
    Y_val = [y.astype(np.int32) for y in Y_val]

    out_dir = os.path.join(model_dir, name)

    if initial_model:
        # A builtin is a directory too -- from_pretrained downloads it and
        # loads it exactly as this does, so resolving one is just finding
        # where it landed.
        source = initial_model
        if not os.path.isdir(source):
            from csbdeep.models.pretrained import get_model_folder

            # Not from_pretrained: that builds the model and then nulls its
            # basedir, so the folder is only reachable by accident. This
            # downloads and returns the folder, which is all that is wanted.
            source = str(get_model_folder(StarDist2D, initial_model))

        n_channel_pretrained = None
        if os.path.isfile(os.path.join(source, "config.json")):
            with open(os.path.join(source, "config.json")) as f:
                n_channel_pretrained = json.load(f).get("n_channel_in")
        if (
            n_channel_pretrained is not None
            and n_channel_pretrained != n_channel_in
        ):
            raise ValueError(
                f"{initial_model} takes {n_channel_pretrained}-channel input "
                f"and the patches have {n_channel_in}. A model's first layer "
                f"is fixed at training time, so this one cannot continue from "
                f"these patches."
            )

        # Copied rather than trained in place, so continuing produces a new
        # model and leaves its parent intact -- unless the caller aimed at the
        # parent on purpose, which is how "keep training this one" is said.
        if os.path.abspath(source) != os.path.abspath(out_dir):
            shutil.copytree(source, out_dir, dirs_exist_ok=True)
        # config=None loads the saved config and weights from out_dir.
        net = StarDist2D(config=None, name=name, basedir=model_dir)
        progress(f"Continuing from {source}")
    else:
        config = Config2D(
            n_rays=n_rays,
            axes=image_axes if image_axes.endswith("C") else image_axes + "C",
            n_channel_in=n_channel_in,
            train_patch_size=(train_patch_size, train_patch_size),
            train_batch_size=train_batch_size,
            unet_n_depth=unet_n_depth,
            grid=(grid_size_xy, grid_size_xy),
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

    # Appended, not overwritten: a model that has been continued has one
    # curve across every run, which is the thing worth plotting.
    # Rewritten rather than appended to: a history from before a column
    # existed has a shorter header, and appending wider rows to it produces a
    # file that reads back wrong rather than failing.
    columns = ["epoch", "loss", "val_loss", "run", "model", "dataset"]
    history_path = os.path.join(out_dir, "history.csv")
    previous = []
    if os.path.exists(history_path):
        with open(history_path) as f:
            previous = list(csv.DictReader(f))

    last_run = int(previous[-1].get("run") or 0) if previous else 0
    done = len(previous)

    with open(history_path, "w", newline="") as f:
        writer = csv.DictWriter(f, columns, restval="")
        writer.writeheader()
        for row in previous:
            writer.writerow({k: row.get(k, "") for k in columns})
        for epoch, loss, val_loss in history:
            writer.writerow(
                {
                    "epoch": done + epoch,
                    "loss": loss,
                    "val_loss": val_loss,
                    "run": last_run + 1,
                    "model": name,
                    "dataset": dataset_id,
                }
            )

    return out_dir

