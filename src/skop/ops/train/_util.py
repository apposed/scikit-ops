"""Shared by the two Cellpose trainers.

Both wrap the same ``cellpose.train.train_seg`` in different versions of the
package. Pure python, so either environment can import it.
"""

from __future__ import annotations

import csv
import logging
import os
import re

from skop import progress

#: The line train_seg logs at the end of an epoch, in both versions:
#: ``12, train_loss=0.4213, test_loss=0.5012, LR=0.000010, time 3.21s``
_EPOCH_LINE = re.compile(r"^(\d+), train_loss=(\S+), test_loss=(\S+),")


class relay_epochs:
    """Turn Cellpose's epoch log line into ``progress()`` calls.

    ``train_seg`` takes no callback, so its log is the only live signal. A
    handler rather than a stream capture: Appose's protocol is on stdout.
    Removed on exit, or the next run reports this one's epochs too.
    """

    def __init__(self, epochs: int):
        self.epochs = epochs
        self.history: list[tuple[int, float, float]] = []

    def __enter__(self):
        relay = self

        class Handler(logging.Handler):
            def emit(self, record):
                match = _EPOCH_LINE.match(record.getMessage())
                if match is None:
                    return
                epoch = int(match.group(1)) + 1
                loss, val_loss = float(match.group(2)), float(match.group(3))
                relay.history.append((epoch, loss, val_loss))
                progress(
                    f"Epoch {epoch}/{relay.epochs} — "
                    f"loss {loss:.4f}, val_loss {val_loss:.4f}",
                    epoch,
                    relay.epochs,
                )

        self._handler = Handler()
        self._log = logging.getLogger("cellpose.train")
        self._level = self._log.level
        self._propagate = self._log.propagate
        # INFO, or the record is dropped before any handler sees it. No
        # propagation, or it reaches a stdout handler and breaks the protocol.
        self._log.setLevel(logging.INFO)
        self._log.propagate = False
        self._log.addHandler(self._handler)
        return self

    def __exit__(self, *exc):
        self._log.removeHandler(self._handler)
        self._log.setLevel(self._level)
        self._log.propagate = self._propagate


def write_history(model_path, history, name: str, dataset_id: str) -> str:
    """Append this run's epochs to ``<model file>_history.csv``. Columns
    match ``train_stardist2d``. Rewritten rather than appended: an older
    file has a shorter header, and wider rows read back wrong."""
    columns = ["epoch", "loss", "val_loss", "run", "model", "dataset"]
    path = f"{model_path}_history.csv"

    previous = []
    if os.path.exists(path):
        with open(path) as f:
            previous = list(csv.DictReader(f))
    last_run = int(previous[-1].get("run") or 0) if previous else 0
    # Where the last run got to, not how many rows it wrote: Cellpose logs
    # every tenth epoch, so rows and epochs are not the same count.
    done = int(previous[-1].get("epoch") or 0) if previous else 0

    with open(path, "w", newline="") as f:
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
    return path


def read_pairs(images: list[str], labels: list[str], val_size: int):
    """Read the patches, validation split off the end. Not random -- a caller
    wanting that shuffles its own lists and records the seed."""
    from tifffile import imread

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

    progress(f"Reading {len(images)} pairs", 0, len(images))
    X, Y = [], []
    for i, (image_path, label_path) in enumerate(zip(images, labels)):
        X.append(imread(image_path))
        Y.append(imread(label_path))
        progress(current=i + 1)

    return X[:-val_size], Y[:-val_size], X[-val_size:], Y[-val_size:]
