"""train_yolo's host-side half: the dataset it lays out for ultralytics.

Pure python, so these run without torch. Training itself is in
test_ops_e2e.py, in the pytorch environment.
"""

from __future__ import annotations

import csv
import os

import pytest

from skop.ops.train._util import write_history
from skop.ops.train.yolo import read_yolo_label, write_dataset


@pytest.fixture
def pairs(tmp_path):
    """Three images with labels; two share a file name, in different folders."""
    images, labels = [], []
    for k, (folder, stem) in enumerate([("a", "img"), ("b", "img"), ("a", "other")]):
        d = tmp_path / "src" / folder
        d.mkdir(parents=True, exist_ok=True)
        image = d / f"{stem}.jpg"
        image.write_bytes(b"not really a jpeg")
        label = d / f"{stem}_{k}.txt"
        label.write_text(f"0 0.5 0.5 0.2 0.1 0.9\n{k % 2} 0.25 0.25 0.1 0.1\n")
        images.append(str(image))
        labels.append(str(label))
    return images, labels


def _read(path):
    with open(path) as f:
        return f.read()


def test_layout_is_what_ultralytics_reads(pairs, tmp_path):
    images, labels = pairs
    root = tmp_path / "dataset"
    data_yaml, n_train, n_val = write_dataset(images, labels, str(root), val_size=1)

    assert (n_train, n_val) == (2, 1)
    train = sorted(os.listdir(root / "images" / "train"))
    # Numbered, so the two img.jpg stay apart.
    assert train == ["00000_img.jpg", "00001_img.jpg"]
    assert os.listdir(root / "images" / "val") == ["00002_other.jpg"]
    # Each label sits where ultralytics looks: images -> labels, .txt.
    for name in train:
        assert (root / "labels" / "train" / (name[:-4] + ".txt")).exists()

    text = _read(data_yaml)
    assert "train: images/train" in text and "val: images/val" in text
    # Classes 0 and 1 in the labels, no names given.
    assert "0: 'class0'" in text and "1: 'class1'" in text


def test_confidence_column_is_dropped(pairs, tmp_path):
    images, labels = pairs
    write_dataset(images, labels, str(tmp_path / "d"), val_size=0)
    rows = _read(tmp_path / "d" / "labels" / "train" / "00000_img.txt").splitlines()
    assert [len(r.split()) for r in rows] == [5, 5]


def test_no_val_validates_on_training(pairs, tmp_path):
    images, labels = pairs
    data_yaml, n_train, n_val = write_dataset(images, labels, str(tmp_path / "d"), 0)
    assert (n_train, n_val) == (3, 0)
    assert "val: images/train" in _read(data_yaml)


def test_class_names_are_written_and_checked(pairs, tmp_path):
    images, labels = pairs
    data_yaml, _, _ = write_dataset(
        images, labels, str(tmp_path / "d"), 1, class_names=["bib", "partial bib"]
    )
    assert "0: 'bib'" in _read(data_yaml)
    with pytest.raises(ValueError, match="class 1"):
        write_dataset(images, labels, str(tmp_path / "e"), 1, class_names=["bib"])


def test_one_class_is_object(tmp_path):
    image = tmp_path / "x.png"
    image.write_bytes(b"")
    label = tmp_path / "x.txt"
    label.write_text("")  # nothing in this image: still a training example
    data_yaml, _, _ = write_dataset([str(image)], [str(label)], str(tmp_path / "d"), 0)
    assert "0: 'object'" in _read(data_yaml)


def test_rewriting_clears_the_last_dataset(pairs, tmp_path):
    images, labels = pairs
    root = str(tmp_path / "d")
    write_dataset(images, labels, root, val_size=0)
    write_dataset(images[:1], labels[:1], root, val_size=0)
    assert os.listdir(os.path.join(root, "images", "train")) == ["00000_img.jpg"]


def test_mismatched_lists_are_refused(pairs, tmp_path):
    images, labels = pairs
    with pytest.raises(ValueError, match="same length"):
        write_dataset(images, labels[:2], str(tmp_path / "d"), 0)
    with pytest.raises(ValueError, match="not enough"):
        write_dataset(images, labels, str(tmp_path / "d"), 3)


@pytest.mark.parametrize(
    "line", ["0 0.5 0.5 0.2", "0 0.5 0.5 1.2 0.1", "-1 0.5 0.5 0.2 0.1"]
)
def test_bad_label_lines_name_the_line(tmp_path, line):
    label = tmp_path / "bad.txt"
    label.write_text("0 0.5 0.5 0.2 0.1\n" + line + "\n")
    with pytest.raises(ValueError, match="line 2"):
        read_yolo_label(str(label))


def test_history_takes_extra_columns(tmp_path):
    model = tmp_path / "m.pt"
    path = write_history(
        str(model),
        [(1, 2.0, 1.5, 0.3, 0.1)],
        "m",
        "bibs-v1",
        extra_columns=["mAP50", "mAP50_95"],
    )
    with open(path) as f:
        rows = list(csv.DictReader(f))
    assert list(rows[0]) == [
        "epoch",
        "loss",
        "val_loss",
        "mAP50",
        "mAP50_95",
        "run",
        "model",
        "dataset",
    ]
    assert rows[0]["mAP50"] == "0.3" and rows[0]["dataset"] == "bibs-v1"


def test_history_without_extras_is_unchanged(tmp_path):
    path = write_history(str(tmp_path / "c"), [(1, 2.0, 1.5)], "c", "")
    with open(path) as f:
        assert next(csv.reader(f)) == [
            "epoch",
            "loss",
            "val_loss",
            "run",
            "model",
            "dataset",
        ]
