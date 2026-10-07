"""Carrying out a plan: slicing, calling, and stacking the results.

Planning itself is opspec's, and tested there (opspec/tests/test_plan.py).
These run in-process, through ``_adapt.execute`` directly, so nothing here
needs a worker. ``test_runner.py`` covers the trip over the wire.
"""

from __future__ import annotations

from typing import Annotated

import numpy as np
import pytest

import skop
from skop import _adapt
from skop.ops import toy


def plan_for(fn, param, array, axes, **kwargs):
    return skop.plan(fn, param, array, axes, **kwargs)


# -- execution -----------------------------------------------------------


def run_adapted(fn, param, array, axes, **kwargs):
    """Plan and execute in-process, the way a worker does."""
    plan = plan_for(fn, param, array, axes, **kwargs)
    spec = skop.OpSpec.from_op(fn)
    return _adapt.execute(spec, fn, {param: array}, {param: plan})


def test_iteration_stacks_results():
    stack = np.zeros((3, 4, 6))
    labels = run_adapted(toy.quadrants, "image", stack, list("zyx"))
    assert labels.shape == (3, 4, 6)


def test_iteration_renumbers_labels_across_slices():
    # Each plane labels its quadrants 1..4 on its own; stacking as-is would
    # claim object 1 in plane 0 and object 1 in plane 1 are the same thing.
    labels = run_adapted(toy.quadrants, "image", np.zeros((3, 4, 6)), list("zyx"))
    assert sorted(np.unique(labels)) == list(range(1, 13))
    assert sorted(np.unique(labels[0])) == [1, 2, 3, 4]
    assert sorted(np.unique(labels[2])) == [9, 10, 11, 12]


def test_slice_plan_runs_once():
    stack = np.zeros((3, 4, 6))
    labels = run_adapted(
        toy.quadrants, "image", stack, list("zyx"), dispositions={0: skop.SELECT}
    )
    assert labels.shape == (4, 6)


def test_iteration_over_two_axes():
    labels = run_adapted(toy.quadrants, "image", np.zeros((2, 3, 4, 6)), list("tzyx"))
    assert labels.shape == (2, 3, 4, 6)
    assert labels.max() == 4 * 6


def test_transposed_stack_reaches_the_op_in_declared_order():
    # x, y transposed *and* an extra axis to iterate: both at once.
    labels = run_adapted(toy.quadrants, "image", np.zeros((6, 3, 4)), list("xzy"))
    assert labels.shape == (3, 4, 6)


def test_a_remapped_run_processes_cross_sections():
    # Feeding the op ZY planes instead of YX ones, iterating over x.
    labels = run_adapted(
        toy.quadrants,
        "image",
        np.zeros((5, 4, 6)),
        list("zyx"),
        mapping=(0, 1),
        dispositions={2: skop.ITERATE},
    )
    assert labels.shape == (6, 5, 4)


def test_scalar_outputs_stack_into_an_array():
    totals = _adapt._reassemble("op", "total", [1.0, 2.0, 3.0], (3,), None)
    assert totals.tolist() == [1.0, 2.0, 3.0]


def test_unstackable_output_says_so():
    varying = [np.zeros((2, 3)), np.zeros((5, 3))]
    with pytest.raises(TypeError, match="shape varies"):
        _adapt._reassemble("op", "points", varying, (2,), skop.Role.points)
    with pytest.raises(TypeError, match="cannot be stacked"):
        _adapt._reassemble("op", "notes", ["a", "b"], (2,), None)


@skop.op
def masked_sum(
    image: Annotated[np.ndarray, skop.Axes("y", "x")],
    mask: Annotated[np.ndarray, skop.Axes("y", "x")],
) -> float:
    return float((image * mask).sum())


def test_two_stacks_are_looped_over_together():
    image = np.arange(3)[:, None, None] * np.ones((3, 4, 6))
    mask = np.ones((3, 4, 6))
    plans = {
        name: plan_for(masked_sum, name, array, list("zyx"))
        for name, array in (("image", image), ("mask", mask))
    }
    spec = skop.OpSpec.from_op(masked_sum)
    sums = _adapt.execute(spec, masked_sum, {"image": image, "mask": mask}, plans)
    # One call per z, each with that z of both: 0, 24, 48.
    assert list(sums) == [0.0, 24.0, 48.0]


def test_stacks_looped_over_together_must_match():
    image, mask = np.ones((3, 4, 6)), np.ones((2, 4, 6))
    plans = {
        name: plan_for(masked_sum, name, array, list("zyx"))
        for name, array in (("image", image), ("mask", mask))
    }
    spec = skop.OpSpec.from_op(masked_sum)
    with pytest.raises(ValueError, match="as many slices"):
        _adapt.execute(spec, masked_sum, {"image": image, "mask": mask}, plans)


def test_no_plan_means_no_adaptation():
    spec = skop.OpSpec.from_op(toy.quadrants)
    plain = _adapt.execute(spec, toy.quadrants, {"image": np.zeros((4, 6))}, {})
    assert plain.shape == (4, 6)
