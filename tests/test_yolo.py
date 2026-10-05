"""Tiled inference with known boxes, plus measured-memory batching.

Torch and OpenCV are optional on the host; these run wherever they exist.
The detector is replaced so coordinates, duplicates and retries are exact.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import pytest

from skop.ops.detect import yolo

torch = pytest.importorskip("torch")
pytest.importorskip("cv2")


@pytest.fixture
def detector(monkeypatch, tmp_path):
    state = SimpleNamespace(calls=[], device=None, limit=None, boxes=None, peak=0.5e9)

    def predict(inputs, **kwargs):
        coords = np.rint(inputs[:, :2, 0, 0].numpy() * 255).astype(int)
        state.calls.append(coords.tolist())
        state.peak = 1.5e9
        if state.limit is not None and len(coords) > state.limit:
            raise torch.OutOfMemoryError("CUDA out of memory")
        rows = (
            state.boxes(coords, inputs)
            if state.boxes is not None
            else [[[8, 8, 24, 24, 0.9, 0]] for _ in coords]
        )
        return [
            SimpleNamespace(
                boxes=SimpleNamespace(data=torch.tensor(row).float().reshape(-1, 6))
            )
            for row in rows
        ]

    def move(device):
        state.device = device

    model = SimpleNamespace(
        task="detect",
        model=SimpleNamespace(args={"imgsz": 64}, stride=torch.tensor([32])),
        to=move,
        predict=predict,
        predictor=SimpleNamespace(results=None),
    )
    monkeypatch.setitem(
        sys.modules, "ultralytics", SimpleNamespace(YOLO=lambda _: model)
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    state.weights = tmp_path / "model.pt"
    state.weights.touch()
    return state


def image(height=110, width=150):
    y, x = np.indices((height, width), dtype=np.uint8)
    return np.stack((y, x, np.zeros_like(y)), axis=-1)


def test_tiles_cover_edges_and_restore_coordinates(detector):
    result = yolo(
        image(), detector.weights, object_size=40**2 * 0.001 / 1.5, overlap=0.25
    )
    origins = [(y, x) for y in (0, 30, 60, 70) for x in (0, 30, 60, 90, 110)]
    assert [tuple(c) for batch in detector.calls for c in batch] == origins
    np.testing.assert_allclose(
        result.boxes, [[y + 5, x + 5, y + 15, x + 15] for y, x in origins]
    )
    np.testing.assert_allclose(result.confidences, [0.9] * len(origins))
    assert result.classes == [0] * len(origins)


@pytest.mark.parametrize("object_size", [None, 150, 8250])
def test_whole_image_when_size_is_absent_or_in_range(detector, object_size):
    yolo(image(), detector.weights, object_size=object_size)
    assert len(detector.calls) == len(detector.calls[0]) == 1


def test_short_axis_padding_preserves_scale(detector):
    def boxes(coords, inputs):
        # Only the top half of a 40 px tile contains this 20 px image.
        assert inputs.shape[2:] == (64, 64)
        assert torch.allclose(
            inputs[:, :, 32:, :], torch.full_like(inputs[:, :, 32:, :], 114 / 255)
        )
        return [[[8, 8, 24, 24, 0.9, 0]] for _ in coords]

    detector.boxes = boxes
    result = yolo(
        image(20, 110), detector.weights, object_size=40**2 * 0.001 / 1.5, overlap=0.25
    )
    np.testing.assert_allclose(
        result.boxes, [[5, x + 5, 15, x + 15] for x in (0, 30, 60, 70)]
    )


def test_partial_duplicates_lose_to_full_boxes_and_classes_stay_separate(detector):
    def boxes(coords, inputs):
        rows = []
        for y, x in coords:
            if y != 0 or x not in (0, 30):
                rows.append([])
                continue
            # One box crosses x=40, so tile zero sees a high-confidence fragment.
            row = [(34 - x) * 1.6, 12 * 1.6, (min(46, x + 40) - x) * 1.6, 28 * 1.6]
            rows.append([row + [0.95 if x == 0 else 0.8, 0], row + [0.7, 1]])
        return rows

    detector.boxes = boxes
    result = yolo(
        image(), detector.weights, object_size=40**2 * 0.001 / 1.5, overlap=0.25
    )
    np.testing.assert_allclose(result.boxes, [[12, 34, 28, 46]] * 2, atol=1e-5)
    np.testing.assert_allclose(result.confidences, [0.8, 0.7])
    assert result.classes == [0, 1]


@pytest.mark.parametrize("device", ["cuda", "mps"])
def test_batch_budget_excludes_resident_model(detector, monkeypatch, device):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: device == "cuda")
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: device == "mps")
    backend = SimpleNamespace(
        is_available=lambda: device == "cuda",
        synchronize=lambda: None,
        empty_cache=lambda: None,
        mem_get_info=lambda: (7.5e9, 8e9),
        memory_allocated=lambda: 0.5e9,
        reset_peak_memory_stats=lambda: None,
        max_memory_allocated=lambda: detector.peak,
        driver_allocated_memory=lambda: detector.peak,
        recommended_max_memory=lambda: 8e9,
    )
    monkeypatch.setattr(torch, device, backend)
    monkeypatch.setitem(
        sys.modules,
        "psutil",
        SimpleNamespace(virtual_memory=lambda: SimpleNamespace(available=8e9)),
    )
    yolo(image(), detector.weights, object_size=40**2 * 0.001 / 1.5, overlap=0.25)
    # 90% * 7.5 GB / (1.5 GB peak - 0.5 GB model) = six tiles.
    assert [len(batch) for batch in detector.calls] == [1, 6, 6, 6, 1]
    assert detector.device == ("cuda:0" if device == "cuda" else "mps")


def test_oom_retries_the_same_tiles(detector, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (7.5e9, 8e9))
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda: 0.5e9)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 1.5e9)
    detector.limit = 3
    result = yolo(
        image(), detector.weights, object_size=40**2 * 0.001 / 1.5, overlap=0.25
    )
    assert [len(batch) for batch in detector.calls[:3]] == [1, 6, 3]
    assert detector.calls[1][:3] == detector.calls[2]
    assert len(result.boxes) == 20


def test_no_detections_returns_empty_aligned_outputs(detector):
    detector.boxes = lambda coords, inputs: [[] for _ in coords]
    result = yolo(image(), detector.weights)
    assert result.boxes.shape == (0, 4)
    assert result.confidences == result.classes == []


def test_single_tile_keeps_the_models_own_suppression(detector):
    detector.boxes = lambda coords, inputs: [
        [[8, 8, 24, 24, 0.9, 0], [8, 8, 12, 12, 0.8, 0]] for _ in coords
    ]
    result = yolo(image(), detector.weights)
    assert len(result.boxes) == 2
