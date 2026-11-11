"""Train a YOLO detector on your own boxes.

Runs in the shared 'pytorch' environment, beside ``skop.ops.detect.yolo``,
which runs what this writes.

Shaped like the Cellpose and StarDist trainers (design 0011): paths in, a path
out, the last ``val_size`` pairs held back, one framework callback relaying
each epoch to ``progress()``. What it learned from JDLL's YOLO training
(``docs/spec/jdll-yolo-train.py``) is which arguments the op owns rather than
the caller, and that ultralytics' console output has to be kept off the
runner's stdout. Design 0023 has the reasoning.

The one thing the other trainers do not have to do: ultralytics reads its
training set from disk, in its own layout, so the op writes one -- links to
the images and cleaned copies of the labels -- before it trains.
"""

from __future__ import annotations

import os
import shutil
from enum import Enum
from typing import Annotated

from opspec.op import op
from opspec.runtime import cancel_requested, progress

from ._util import write_history

#: Ultralytics' validation numbers, under the names the history file uses.
_METRICS = {
    "mAP50": "metrics/mAP50(B)",
    "mAP50_95": "metrics/mAP50-95(B)",
    "precision": "metrics/precision(B)",
    "recall": "metrics/recall(B)",
}


class Optimizer(Enum):
    """Ultralytics' optimisers. ``auto`` picks one, and a learning rate, from
    the length of the run, and ignores ``learning_rate``."""

    auto = "auto"
    SGD = "SGD"
    AdamW = "AdamW"


class Cache(Enum):
    """Where ultralytics keeps decoded images between epochs. ``off`` reads
    and decodes every image every epoch; ``ram`` decodes each once and keeps
    it (falling back to ``off`` if it will not fit); ``disk`` saves decoded
    copies as ``.npy`` beside the dataset."""

    off = "off"
    ram = "ram"
    disk = "disk"


#: Every inline augmentation ultralytics applies to detection, at the value
#: that turns it off.
_NO_AUGMENT = {
    "hsv_h": 0.0,
    "hsv_s": 0.0,
    "hsv_v": 0.0,
    "degrees": 0.0,
    "translate": 0.0,
    "scale": 0.0,
    "shear": 0.0,
    "perspective": 0.0,
    "flipud": 0.0,
    "fliplr": 0.0,
    "bgr": 0.0,
    "mosaic": 0.0,
    "mixup": 0.0,
    "cutmix": 0.0,
    "close_mosaic": 0,
}


@op(env="pytorch")
def train_yolo(
    images: list[str],
    labels: list[str],
    model_dir: str,
    name: str = "yolo_model",
    class_names: list[str] | None = None,
    initial_model: str = "yolo11n.pt",
    epochs: Annotated[int, {"widget_type": "SpinBox", "min": 1, "max": 5000}] = 100,
    imgsz: Annotated[
        int, {"widget_type": "SpinBox", "min": 32, "max": 4096, "step": 32}
    ] = 640,
    batch_size: Annotated[int, {"widget_type": "SpinBox", "min": -1, "max": 256}] = 16,
    optimizer: Optimizer = Optimizer.auto,
    learning_rate: float = 0.01,
    patience: Annotated[int, {"widget_type": "SpinBox", "min": 0, "max": 5000}] = 100,
    fliplr: float = 0.5,
    flipud: float = 0.0,
    degrees: float = 0.0,
    scale: float = 0.5,
    shear: float = 0.0,
    perspective: float = 0.0,
    mosaic: float = 1.0,
    augment: bool = True,
    cache: Cache = Cache.off,
    val_size: Annotated[int, {"widget_type": "SpinBox", "min": 0, "max": 100000}] = 2,
    seed: int = 0,
    dataset_id: str = "",
) -> str:
    """Train, or finetune, a YOLO detector and write it to disk.

    Args:
        images: Paths to the training images, in order. Any format
            ultralytics reads (jpg, png, tif, ...); not HEIC.
        labels: Paths to YOLO ``.txt`` box files. ``labels[k]`` is the truth
            for ``images[k]`` -- the pairing is positional, and the caller
            owns it. One line per box, ``class x_center y_center width
            height``, normalised to the image. A sixth column, a prediction's
            confidence, is dropped, so predicted boxes can be trained on
            directly. An empty file is an image with nothing in it, which
            teaches as much as one with boxes.
        model_dir: Directory the model is written to, as ``<name>.pt``.
            Training's working files go in ``<name>_training/`` beside it.
        name: Filename for the trained model, without ``.pt``.
        class_names: Name of each class id, in order. Empty takes the
            classes from the labels and calls them ``class0``, ``class1``...,
            or ``object`` when there is only class 0.
        initial_model: What to start from. A stock ``.pt`` (``yolo11n.pt``,
            ``yolo11m.pt``, ``yolo26n.pt``, ... -- ultralytics downloads it)
            finetunes, which is the usual thing and needs far fewer boxes. A
            ``.yaml`` (``yolo11n.yaml``) starts from nothing. The path this op
            returned carries on from that model, and its history continues.
        epochs: Passes over the training set.
        imgsz: Size training resizes each image to. Objects should end up the
            size they will be when the model is run, so train at the
            ``imgsz`` you will detect at.
        batch_size: Images per batch. -1 lets ultralytics choose from the
            GPU's memory. If the first epoch runs out of memory, ultralytics
            halves it and starts the epoch again, by itself. Ultralytics
            steps the optimiser every 64 images whatever the batch, so a
            training set of a handful of images learns very slowly: give it
            more images (copies, or augmented crops), not more epochs.
        optimizer: ``auto`` lets ultralytics choose, and is what to leave it
            at unless you are matching another run.
        learning_rate: Initial learning rate (ultralytics' ``lr0``). Ignored
            when ``optimizer`` is ``auto``.
        patience: Stop when validation has not improved for this many epochs.
            0 never stops early.
        fliplr: Chance of a left-right flip. Ultralytics' default; set it to
            0 for anything with a handedness -- text, numbers, bibs.
        flipud: Chance of an upside-down flip. 0 for photos; microscopy has no
            up and wants 0.5.
        degrees: Random rotation range, +/- degrees.
        scale: Random zoom range, +/- this fraction. Ultralytics' default of
            0.5 is half to one and a half times.
        shear: Random shear range, +/- degrees.
        perspective: Random perspective warp, 0 to 0.001.
        mosaic: Chance of a four-image mosaic. Ultralytics turns it off for
            the last 10 epochs.
        augment: False turns off all of ultralytics' inline augmentation --
            the six above, and the colour jitter and shifts that are not
            parameters here -- so each image is only resized to ``imgsz``.
            For a training set augmented beforehand, as napari-ai-lab and
            photo-enhance's ``bib-patches`` make one: what is left per batch
            is reading the image, and with ``cache`` nothing at all, so the
            GPU, not one CPU thread, sets the pace. The cost is that every
            epoch sees the same images; make more of them, or make a fresh
            set now and then and continue from the last model.
        cache: Keep decoded images between epochs. ``ram`` is the fast one,
            and needs about ``imgsz * imgsz * 3`` bytes an image (1024 px:
            3 MB, so 3000 crops is 9 GB). Worth it with or without
            ``augment``; it saves the reading, not the augmenting.
        val_size: How many pairs, taken from the end, to hold out for
            validation. Put whole photos there, not crops of photos that are
            also in training, or the validation numbers flatter the model.
            0 validates on the training set -- for checking the wiring on one
            image, where a near-perfect score is the right answer.
        seed: Seed for ultralytics' shuffling and augmentation.
        dataset_id: Opaque label for the training data, recorded in the
            history file so a plot can mark where it changed.

    Returns:
        The path of the model file, ``<model_dir>/<name>.pt``. A
        ``<model file>_history.csv`` is written beside it -- epoch, loss,
        val_loss, mAP50, mAP50_95, precision, recall, run, model, dataset --
        continued when ``initial_model`` is this same file, as the Cellpose
        trainers do.
    """
    import contextlib
    import logging
    import sys

    import torch
    from ultralytics import YOLO
    from ultralytics.utils import LOGGER

    # Absolute, because training runs in the workspace (below).
    model_dir = os.path.abspath(model_dir)
    if os.path.exists(initial_model):
        initial_model = os.path.abspath(initial_model)
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"{name}.pt")
    workspace = os.path.join(model_dir, f"{name}_training")
    os.makedirs(workspace, exist_ok=True)

    data_yaml, n_train, n_val = write_dataset(
        images, labels, os.path.join(workspace, "dataset"), val_size, class_names
    )

    if torch.cuda.is_available():
        device = 0
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    # Ultralytics writes to stdout -- its logger, and its progress bars --
    # which is the Appose protocol's channel. Both go to a log file while it
    # trains, and progress() is given the real stdout back each time it
    # reports.
    real_stdout = sys.stdout

    def report(*args, **kwargs):
        current, sys.stdout = sys.stdout, real_stdout
        try:
            progress(*args, **kwargs)
        finally:
            sys.stdout = current

    history = []
    trained = []  # epochs that trained, as on_train_epoch_end saw them

    def on_train_epoch_end(trainer):
        trained.append(int(trainer.epoch) + 1)

    def on_train_batch_end(trainer):
        # Ultralytics checks trainer.stop after every batch; it validates and
        # saves what it has, then stops.
        if cancel_requested():
            trainer.stop = True

    def on_fit_epoch_end(trainer):
        epoch = int(trainer.epoch) + 1
        # Called once more after training, to score best.pt -- and after an
        # early stop the epoch counter has moved on by then. Only an epoch
        # that trained, once.
        if not trained or epoch != trained[-1] or (history and history[-1][0] == epoch):
            return
        losses = trainer.label_loss_items(trainer.tloss, prefix="train")
        loss = _total(losses)
        metrics = dict(trainer.metrics or {})
        val_loss = _total({k: v for k, v in metrics.items() if k.startswith("val/")})
        scores = {k: _number(metrics.get(v)) for k, v in _METRICS.items()}
        history.append((epoch, loss, val_loss, *scores.values()))
        report(
            f"Epoch {epoch}/{epochs} — loss {loss:.4f}, "
            f"mAP50 {scores['mAP50']:.3f}, mAP50-95 {scores['mAP50_95']:.3f}",
            epoch,
            epochs,
        )

    log_path = os.path.join(workspace, "training.log")
    handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    saved = LOGGER.handlers[:], LOGGER.propagate
    LOGGER.handlers, LOGGER.propagate = [handler], False
    model = None
    cwd = os.getcwd()
    try:
        # Ultralytics drops downloads in the current directory -- its AMP
        # check fetches yolo26n.pt -- so let that be the workspace.
        os.chdir(workspace)
        with (
            open(log_path, "a", encoding="utf-8") as log,
            contextlib.redirect_stdout(log),
        ):
            # Inside the redirect: loading a stock model downloads it, with
            # a progress bar on stdout.
            report(f"Loading YOLO model {initial_model}")
            model = YOLO(_stock_weights(initial_model))
            model.add_callback("on_train_epoch_end", on_train_epoch_end)
            model.add_callback("on_train_batch_end", on_train_batch_end)
            model.add_callback("on_fit_epoch_end", on_fit_epoch_end)
            report(
                f"Training {name}: {n_train} train, {n_val or n_train} val"
                f"{'' if n_val else ' (the training set)'}, {epochs} epochs",
                0,
                epochs,
            )
            augmentation = {
                "fliplr": fliplr,
                "flipud": flipud,
                "degrees": degrees,
                "scale": scale,
                "shear": shear,
                "perspective": perspective,
                "mosaic": mosaic,
            }
            if not augment:
                augmentation = _NO_AUGMENT
            model.train(
                data=data_yaml,
                epochs=epochs,
                imgsz=imgsz,
                batch=batch_size,
                optimizer=optimizer.value,
                lr0=learning_rate,
                patience=patience,
                **augmentation,
                cache=False if cache is Cache.off else cache.value,
                seed=seed,
                # The op's, not the caller's: where things go, and how.
                project=workspace,
                name="run",
                exist_ok=True,
                device=device,
                # Worker processes re-import the op's host; JDLL uses 0 too.
                workers=0,
                plots=False,
                verbose=False,
                deterministic=True,
            )
        trainer = model.trainer
        best, last = str(trainer.best), str(trainer.last)
        source = best if os.path.isfile(best) else last
        if not os.path.isfile(source):
            raise RuntimeError(f"Training wrote no checkpoint in {workspace}")
        shutil.copy2(source, model_path)
        # The copy is the model. Left in place, the run's best.pt and last.pt
        # are two more models for anyone listing the folder to choose from.
        shutil.rmtree(os.path.dirname(source), ignore_errors=True)
        # So is the yolo26n.pt the AMP check downloaded into the workspace.
        for stray in os.listdir(workspace):
            if stray.endswith(".pt"):
                os.remove(os.path.join(workspace, stray))
    finally:
        os.chdir(cwd)
        LOGGER.handlers, LOGGER.propagate = saved
        handler.close()
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    write_history(
        model_path,
        history,
        name,
        dataset_id,
        initial_model,
        extra_columns=list(_METRICS),
    )
    report(f"Saved {model_path}")
    return model_path


def write_dataset(
    images: list[str],
    labels: list[str],
    root: str,
    val_size: int,
    class_names: list[str] | None = None,
) -> tuple[str, int, int]:
    """Lay the pairs out the way ultralytics reads them, under ``root``.

    ``images/{train,val}`` and ``labels/{train,val}``, with each pair's
    files sharing a stem -- ultralytics finds a label by swapping ``images``
    for ``labels`` in the image's path -- and a ``data.yaml``. The stems are
    numbered, so two images called ``0001.png`` in different folders stay
    apart. Images are hard links where the disk allows, copies where not.
    Labels are rewritten: five columns, checked.

    Pure python, so it runs and is tested wherever skop is. Returns the
    yaml's path and the train and val counts.
    """
    if len(images) != len(labels):
        raise ValueError(
            f"{len(images)} images and {len(labels)} labels: a training set "
            f"is pairs, so the two lists must be the same length."
        )
    if val_size < 0 or len(images) <= val_size:
        raise ValueError(
            f"{len(images)} pairs is not enough to hold {val_size} back for "
            f"validation and still train on the rest."
        )

    # Last time's files would be trained on too.
    shutil.rmtree(root, ignore_errors=True)
    split = len(images) - val_size
    top_class = -1
    for k, (image, label) in enumerate(zip(images, labels)):
        part = "train" if k < split else "val"
        stem = f"{k:05d}_{os.path.splitext(os.path.basename(image))[0]}"
        ext = os.path.splitext(image)[1].lower()
        image_dir = os.path.join(root, "images", part)
        label_dir = os.path.join(root, "labels", part)
        os.makedirs(image_dir, exist_ok=True)
        os.makedirs(label_dir, exist_ok=True)
        _link_or_copy(image, os.path.join(image_dir, stem + ext))
        rows = read_yolo_label(label)
        if rows:
            top_class = max(top_class, max(r[0] for r in rows))
        with open(os.path.join(label_dir, stem + ".txt"), "w") as f:
            f.writelines(
                f"{c} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n" for c, xc, yc, w, h in rows
            )

    if class_names:
        if top_class >= len(class_names):
            raise ValueError(
                f"The labels use class {top_class}, but only "
                f"{len(class_names)} class names were given."
            )
        names = list(class_names)
    elif top_class <= 0:
        names = ["object"]
    else:
        names = [f"class{i}" for i in range(top_class + 1)]

    data_yaml = os.path.join(root, "data.yaml")
    with open(data_yaml, "w", encoding="utf-8") as f:
        f.write(f"path: {_yaml_str(os.path.abspath(root))}\n")
        f.write("train: images/train\n")
        # No val pairs: validate on the training set.
        f.write(f"val: images/{'val' if val_size else 'train'}\n")
        f.write("names:\n")
        f.writelines(f"  {i}: {_yaml_str(n)}\n" for i, n in enumerate(names))
    return data_yaml, split, val_size


def read_yolo_label(path: str) -> list[tuple[int, float, float, float, float]]:
    """A YOLO box file's rows, as ``(class, xc, yc, w, h)``.

    Five columns or six -- the sixth, a prediction's confidence, is dropped.
    Anything else, or coordinates outside 0..1, is an error naming the line:
    ultralytics would skip the whole image, quietly.
    """
    rows = []
    with open(path) as f:
        for n, line in enumerate(f, 1):
            parts = line.split()
            if not parts:
                continue
            if len(parts) not in (5, 6):
                raise ValueError(
                    f"{path}, line {n}: {len(parts)} columns; a YOLO box is "
                    f"'class x_center y_center width height [confidence]'"
                )
            c = int(float(parts[0]))
            xc, yc, w, h = (float(p) for p in parts[1:5])
            if c < 0 or not all(0.0 <= v <= 1.0 for v in (xc, yc, w, h)):
                raise ValueError(
                    f"{path}, line {n}: a class below 0, or a coordinate "
                    f"outside 0..1: {line.strip()!r}"
                )
            rows.append((c, xc, yc, w, h))
    return rows


def _stock_weights(initial_model: str) -> str:
    """A bare stock name, ``yolo11n.pt``, as a file in ophelper's cache.

    Ultralytics downloads a stock model into the current directory, which in
    a worker is wherever the worker happened to start. Given a path in the
    cache instead, it downloads there, once. Paths, and ``.yaml``
    architectures, are left alone.
    """
    if (
        not initial_model.endswith(".pt")
        or os.path.dirname(initial_model)
        or os.path.exists(initial_model)
    ):
        return initial_model
    from ultralytics.utils.downloads import attempt_download_asset

    from ophelper.assets import cache_dir

    target = cache_dir() / "yolo" / initial_model
    target.parent.mkdir(parents=True, exist_ok=True)
    return attempt_download_asset(str(target))


def _link_or_copy(source: str, target: str) -> None:
    try:
        os.link(source, target)
    except OSError:  # another drive, or a filesystem without hard links
        shutil.copy2(source, target)


def _yaml_str(value: str) -> str:
    """Quoted for YAML: a Windows path or a class name can hold ':' or '#'."""
    return "'" + str(value).replace("'", "''") + "'"


def _number(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _total(values) -> float:
    # label_loss_items returns the bare key list when there is no loss yet.
    if not isinstance(values, dict):
        return float("nan")
    numbers = [_number(v) for v in values.values()]
    return float(sum(numbers)) if numbers else float("nan")
