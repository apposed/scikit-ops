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
    state = SimpleNamespace(
        calls=[],
        device=None,
        limit=None,
        boxes=None,
        peak=0.5e9,
        resident=0.5e9,
        setup=0,
        cost=lambda count: count * 1e9,
        hook=None,
    )

    def forward(inputs):
        coords = np.rint(inputs[:, :2, 0, 0].numpy() * 255).astype(int)
        state.calls.append(coords.tolist())
        if len(state.calls) == 1:
            state.resident += state.setup
        state.peak = state.resident + state.cost(len(coords))
        if state.hook is not None:
            state.hook()
        if state.limit is not None and len(coords) > state.limit:
            raise torch.OutOfMemoryError("CUDA out of memory")
        rows = (
            state.boxes(coords, inputs)
            if state.boxes is not None
            else [[[8, 8, 24, 24, 0.9, 0]] for _ in coords]
        )
        return [torch.tensor(row).float().reshape(-1, 6) for row in rows]

    forward.args = {"imgsz": 64}
    forward.stride = torch.tensor([32])
    forward.fuse = lambda **kwargs: forward
    forward.eval = lambda: forward
    forward.float = lambda: forward
    model = SimpleNamespace(
        task="detect",
        model=forward,
        names={0: "a", 1: "b"},
        to=lambda device: setattr(state, "device", device),
    )
    monkeypatch.setitem(
        sys.modules, "ultralytics", SimpleNamespace(YOLO=lambda _: model)
    )
    monkeypatch.setitem(
        sys.modules,
        "ultralytics.utils.nms",
        SimpleNamespace(
            non_max_suppression=lambda predictions, **kwargs: predictions,
        ),
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    # Memory tests exercise real CPU tensors with simulated accelerator APIs.
    empty, transfer = torch.empty, torch.Tensor.to

    def cpu_empty(*args, **kwargs):
        kwargs.pop("pin_memory", None)
        return empty(*args, **kwargs)

    def cpu_transfer(self, device, *args, **kwargs):
        if str(device).startswith(("cuda", "mps")):
            return self
        return transfer(self, device, *args, **kwargs)

    monkeypatch.setattr(torch, "empty", cpu_empty)
    monkeypatch.setattr(torch.Tensor, "to", cpu_transfer)
    state.weights = tmp_path / "model.pt"
    state.weights.touch()
    return state


@pytest.fixture
def accelerator(detector, monkeypatch):
    def configure(device="cuda"):
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: device == "mps")

        def empty_cache():
            detector.peak = detector.resident

        backend = SimpleNamespace(
            is_available=lambda: device == "cuda",
            synchronize=lambda: None,
            empty_cache=empty_cache,
            mem_get_info=lambda: (8e9 - detector.resident, 8e9),
            memory_reserved=lambda: detector.resident,
            reset_peak_memory_stats=empty_cache,
            max_memory_reserved=lambda: detector.peak,
            driver_allocated_memory=lambda: detector.peak,
            recommended_max_memory=lambda: 8e9,
        )
        monkeypatch.setattr(torch, device, backend)
        if device == "mps":
            monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        monkeypatch.setitem(
            sys.modules,
            "psutil",
            SimpleNamespace(
                virtual_memory=lambda: SimpleNamespace(available=8e9),
            ),
        )
        return backend

    return configure


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
def test_batch_budget_excludes_resident_model(detector, accelerator, device):
    accelerator(device)
    yolo(image(), detector.weights, object_size=40**2 * 0.001 / 1.5, overlap=0.25)
    # After warm-up: 90% * 7.5 GB / 1 GB per tile = six tiles.
    assert [len(batch) for batch in detector.calls] == [1, 1, 1, 6, 6, 5]
    assert detector.device == ("cuda:0" if device == "cuda" else "mps")


def test_setup_memory_is_not_multiplied_per_tile(detector, accelerator):
    detector.setup = 1e9
    detector.cost = lambda count: count * 0.5e9
    accelerator()
    yolo(
        image(240, 240), detector.weights, object_size=40**2 * 0.001 / 1.5, overlap=0.25
    )
    # Setup remains resident: 90% * 6.5 GB / 0.5 GB = eleven tiles.
    assert max(map(len, detector.calls)) == 11


def test_batch_size_grows_with_measured_efficiency(detector, accelerator):
    detector.cost = lambda count: 0.8e9 + count * 0.2e9
    accelerator()
    yolo(
        image(240, 240), detector.weights, object_size=40**2 * 0.001 / 1.5, overlap=0.25
    )
    sizes = [len(batch) for batch in detector.calls]
    assert 6 in sizes
    assert max(sizes) > 6
    assert sum(sizes) == 64


def test_small_batch_estimate_is_checked_at_intermediate_sizes(detector, accelerator):
    detector.cost = lambda count: count * 0.01e9
    accelerator()
    yolo(
        image(240, 240), detector.weights, object_size=40**2 * 0.001 / 1.5, overlap=0.25
    )
    assert [len(batch) for batch in detector.calls[:4]] == [1, 1, 1, 8]
    assert sum(map(len, detector.calls)) == 64


def test_oom_retries_same_tiles_and_recovers_a_larger_batch(detector, accelerator):
    accelerator()
    detector.limit = 5
    result = yolo(
        image(240, 240), detector.weights, object_size=40**2 * 0.001 / 1.5, overlap=0.25
    )
    assert [len(batch) for batch in detector.calls[:5]] == [1, 1, 1, 6, 3]
    assert detector.calls[3][:3] == detector.calls[4]
    assert any(len(batch) == 5 for batch in detector.calls[5:])
    successful = [
        tuple(c) for batch in detector.calls if len(batch) <= 5 for c in batch
    ]
    assert len(successful) == len(set(successful)) == 64
    assert len(result.boxes) == 64


def test_batch_over_budget_is_reduced_without_losing_tiles(detector, accelerator):
    accelerator()
    detector.cost = lambda count: count * (1e9 if count <= 2 else 1.2e9)
    yolo(
        image(240, 240), detector.weights, object_size=40**2 * 0.001 / 1.5, overlap=0.25
    )
    sizes = [len(batch) for batch in detector.calls]
    assert 6 in sizes and 5 in sizes
    assert sizes.count(6) == 1
    assert sum(sizes) == 64


def test_preparation_runs_during_inference_with_task_context(
    detector, accelerator, monkeypatch
):
    import importlib
    import threading

    from skop import _progress

    accelerator()
    module = importlib.import_module(yolo.__module__)
    original = module.to_rgb
    prepared = threading.Event()
    task = SimpleNamespace(cancel_requested=False, update=lambda **event: None)

    def to_rgb(crop):
        if threading.current_thread() is not threading.main_thread():
            assert _progress._current.get() is task
        if tuple(crop[0, 0, :2]) == (0, 60):
            prepared.set()
        return original(crop)

    def hook():
        if len(detector.calls) == 2:
            assert prepared.wait(2), "next tile must be prepared during inference"

    monkeypatch.setattr(module, "to_rgb", to_rgb)
    detector.hook = hook
    token = _progress._bind(task)
    try:
        yolo(image(), detector.weights, object_size=40**2 * 0.001 / 1.5, overlap=0.25)
    finally:
        _progress._unbind(token)


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
