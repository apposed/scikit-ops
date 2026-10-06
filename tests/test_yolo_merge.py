"""GreedyNMM semantics, and optional comparisons against pinned SAHI.

Reference checks: pip install sahi==0.12.8, then pytest tests/test_yolo_merge.py.
SAHI is not needed by the implementation or by the explicit example tests.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from skop.ops.detect._merge import greedy_nmm

CASES = {
    "empty": [],
    "singleton": [[0, 0, 10, 10, 0.9, 0]],
    "fragments_at_threshold": [[0, 0, 10, 10, 0.9, 0], [5, 0, 15, 10, 0.8, 0]],
    "contained": [[0, 0, 10, 10, 0.8, 0], [2, 2, 4, 4, 0.9, 0]],
    "classes": [[0, 0, 10, 10, 0.9, 2], [0, 0, 10, 10, 0.7, 1]],
    "chain": [[0, 0, 10, 10, 0.9, 0], [5, 0, 15, 10, 0.8, 0], [10, 0, 20, 10, 0.7, 0]],
    "growing_keeper": [
        [0, 0, 10, 10, 0.9, 0],
        [0, 0, 10, 100, 0.8, 0],
        [5, 0, 25, 10, 0.7, 0],
    ],
    "coordinate_tie": [
        [10, 0, 20, 10, 0.9, 0],
        [5, 0, 15, 10, 0.9, 0],
        [0, 0, 10, 10, 0.9, 0],
    ],
    "score_rounding": [[10, 0, 20, 10, 0.9 + 1e-10, 0], [5, 0, 15, 10, 0.9, 0]],
    "duplicate_tie": [[0, 0, 10, 10, 0.9, 0], [0, 0, 10, 10, 0.9, 0]],
    "zero_area": [[0, 0, 0, 10, 0.9, 0], [0, 0, 10, 10, 0.8, 0]],
    "touching": [[0, 0, 10, 10, 0.9, 0], [10, 0, 20, 10, 0.8, 0]],
}


def rows(values):
    return np.asarray(values, dtype=np.float64).reshape(-1, 6)


def test_threshold_equality_merges_fragments():
    np.testing.assert_array_equal(
        greedy_nmm(rows(CASES["fragments_at_threshold"])),
        rows([[0, 0, 15, 10, 0.9, 0]]),
    )


def test_contained_fragment_keeps_maximum_score():
    np.testing.assert_array_equal(
        greedy_nmm(rows(CASES["contained"])),
        rows([[0, 0, 10, 10, 0.9, 0]]),
    )


def test_classes_are_separate_and_in_sahi_order():
    np.testing.assert_array_equal(
        greedy_nmm(rows(CASES["classes"])),
        rows([[0, 0, 10, 10, 0.7, 1], [0, 0, 10, 10, 0.9, 2]]),
    )


def test_greedy_groups_do_not_follow_transitive_overlap():
    np.testing.assert_array_equal(
        greedy_nmm(rows(CASES["chain"])),
        rows([[0, 0, 15, 10, 0.9, 0], [10, 0, 20, 10, 0.7, 0]]),
    )


def test_claimed_candidate_is_rechecked_after_union_grows():
    # C matches the original A, but its IoS drops below .5 after A unions B.
    # SAHI has already claimed C, so it neither merges nor emits it again.
    np.testing.assert_array_equal(
        greedy_nmm(rows(CASES["growing_keeper"])),
        rows([[0, 0, 10, 100, 0.9, 0]]),
    )


def test_coordinate_ties_are_independent_of_input_order():
    np.testing.assert_array_equal(
        greedy_nmm(rows(CASES["coordinate_tie"])),
        rows([[0, 0, 15, 10, 0.9, 0], [10, 0, 20, 10, 0.9, 0]]),
    )


def test_zero_threshold_includes_disjoint_boxes():
    np.testing.assert_array_equal(
        greedy_nmm(rows(CASES["touching"]), 0),
        rows([[0, 0, 20, 10, 0.9, 0]]),
    )


def test_zero_area_box_is_not_merged_at_positive_threshold():
    np.testing.assert_array_equal(
        greedy_nmm(rows(CASES["zero_area"])), rows(CASES["zero_area"])
    )


def test_empty_collection_keeps_its_columns():
    assert greedy_nmm(rows([])).shape == (0, 6)


def test_grid_handles_a_large_box_without_expanding_its_area_into_cells():
    small = [[i * 32, 0, i * 32 + 1, 1, 0.8, 0] for i in range(512)]
    data = rows([[0, 0, 1_000_000, 1_000_000, 0.9, 0], *small])
    np.testing.assert_array_equal(greedy_nmm(data), data[:1])


@pytest.fixture
def reference():
    sahi = pytest.importorskip("sahi", reason="reference checks need sahi==0.12.8")
    assert sahi.__version__ == "0.12.8", "reference checks require sahi==0.12.8"
    from sahi.postprocess.backends import set_postprocess_backend
    from sahi.postprocess.combine import GreedyNMMPostprocess
    from sahi.prediction import ObjectPrediction

    set_postprocess_backend("numpy")

    def run(data, threshold):
        predictions = [
            ObjectPrediction(
                bbox=row[:4].tolist(),
                score=float(row[4]),
                category_id=int(row[5]),
            )
            for row in data
        ]
        processor = GreedyNMMPostprocess(
            match_metric="IOS",
            match_threshold=threshold,
            class_agnostic=False,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            output = processor(predictions)
        return rows([[*p.bbox.to_xyxy(), p.score.value, p.category.id] for p in output])

    return run


@pytest.mark.parametrize("values", CASES.values(), ids=CASES.keys())
@pytest.mark.parametrize("threshold", [0, 0.1, 0.5, 1])
def test_examples_match_sahi(reference, values, threshold):
    data = rows(values)
    np.testing.assert_array_equal(
        greedy_nmm(data, threshold), reference(data, threshold)
    )


@pytest.mark.parametrize("seed", range(12))
@pytest.mark.parametrize("threshold", [0, 0.1, 0.5, 1])
def test_random_box_sets_match_sahi(reference, seed, threshold):
    rng = np.random.default_rng(seed)
    xy = rng.uniform(0, 500, (200, 2))
    sizes = 2 ** rng.uniform(-3, 10, (200, 2))
    data = np.column_stack(
        (xy, xy + sizes, rng.choice([0.7, 0.8, 0.9], 200), rng.integers(0, 3, 200))
    )
    np.testing.assert_array_equal(
        greedy_nmm(data, threshold), reference(data, threshold)
    )


@pytest.mark.parametrize("threshold", [0.5 - 1e-10, 0.5 + 1e-10])
def test_matching_and_union_recheck_use_sahi_precision(reference, threshold):
    data = rows(CASES["fragments_at_threshold"])
    np.testing.assert_array_equal(
        greedy_nmm(data, threshold), reference(data, threshold)
    )


@pytest.mark.parametrize("layout", ["spread", "duplicates", "mixed_sizes"])
def test_large_sets_match_the_sahi_spatial_path(reference, layout):
    i = np.arange(2100)
    xy = np.column_stack((16 + (i % 70) * 64, 16 + (i // 70) * 64))
    data = np.column_stack((xy, xy + 32, np.linspace(0.99, 0.5, 2100), np.zeros(2100)))
    if layout == "duplicates":
        data[1::2, :4] = data[::2, :4] + 1
    elif layout == "mixed_sizes":
        data[0, :4] = [0, 0, 1_000_000, 1_000_000]
    np.testing.assert_array_equal(greedy_nmm(data), reference(data, 0.5))


def test_merge_obeys_cancellation(monkeypatch):
    import importlib

    module = importlib.import_module(greedy_nmm.__module__)
    monkeypatch.setattr(module, "cancel_requested", lambda: True)
    with pytest.raises(RuntimeError, match="YOLO cancelled"):
        greedy_nmm(rows(CASES["singleton"]))
