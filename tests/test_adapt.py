"""Fitting a caller's array to the pattern an op declared.

These run in-process: planning is pure arithmetic over axis names, and
execution is exercised through ``_adapt.execute`` directly, so nothing here
needs a worker. ``test_runner.py`` covers the trip over the wire.
"""

from __future__ import annotations

import numpy as np
import pytest

import skop
from skop import _adapt, _spec
from skop.ops import toy


def plan_for(fn, param, array, axes, **kwargs):
    return skop.plans(fn, param, array, axes, **kwargs)


# -- the declaration -----------------------------------------------------


def test_pattern_parsing():
    axes = skop.Axes("yxc?")
    assert axes.declared == ("y", "x", "c")
    assert axes.core == ("y", "x")
    assert axes.optional == ("c",)


def test_pattern_rejects_nonsense():
    with pytest.raises(ValueError, match="Unknown axis"):
        skop.Axes("yxq")
    with pytest.raises(ValueError, match="Repeated axis"):
        skop.Axes("yxy")


def test_axes_reach_the_spec():
    spec = skop.spec(toy.quadrants)
    image = next(p for p in spec.params if p.name == "image")
    assert image.axes == skop.Axes("yx", extra=skop.Extra.iterate)
    # An Axes composes with the role already on the parameter.
    assert image.role is skop.Role.image
    assert image.type is np.ndarray


def test_unannotated_parameter_has_no_axes():
    spec = skop.spec(toy.scale)
    assert next(p for p in spec.params if p.name == "image").axes is None


# -- planning ------------------------------------------------------------


def test_exact_match_is_a_no_op():
    plan = plan_for(toy.quadrants, "image", np.zeros((4, 6)), "yx")[0]
    assert plan.lossless
    assert plan.calls == 1
    assert plan.iterate == ()
    assert plan.transpose == (0, 1)


def test_transposed_input_is_reordered():
    plan = plan_for(toy.quadrants, "image", np.zeros((6, 4)), "xy")[0]
    assert plan.transpose == (1, 0)
    assert _adapt.apply(plan, np.zeros((6, 4))).shape == (4, 6)


def test_stack_iterates_when_the_op_allows_it():
    plans = plan_for(toy.quadrants, "image", np.zeros((5, 4, 6)), "zyx")
    assert plans[0].lossless
    assert plans[0].iterate == ("z",)
    assert plans[0].calls == 5
    assert plans[0].output_axes == ("z", "y", "x")
    # The lossy alternative is offered, never first.
    assert not plans[1].lossless
    assert plans[1].select == (("z", 0),)


def test_slice_candidate_honors_the_viewer_position():
    plans = plan_for(
        toy.quadrants, "image", np.zeros((5, 4, 6)), "zyx", position={"z": 3}
    )
    lossy = plans[-1]
    assert lossy.select == (("z", 3),)
    assert "z=3" in lossy.summary


def test_reject_leaves_only_the_lossy_candidate():
    spec = skop.spec(toy.quadrants)
    image = next(p for p in spec.params if p.name == "image")
    strict = _spec.ParamSpec(
        name=image.name,
        type=image.type,
        default=image.default,
        direction=None,
        axes=skop.Axes("yx"),
    )
    plans = _adapt._candidates(strict, "zyx", (5, 4, 6))
    assert len(plans) == 1
    assert not plans[0].lossless


def test_passthrough_hands_extra_axes_to_the_op():
    from skop.ops import threshold

    plan = plan_for(threshold.otsu, "image", np.zeros((5, 4, 6)), "zyx")[0]
    assert plan.lossless
    assert plan.iterate == ()
    assert plan.calls == 1


def test_missing_core_axis_is_refused():
    with pytest.raises(ValueError, match="no y axis"):
        plan_for(toy.quadrants, "image", np.zeros((5, 6)), "zx")


def test_axis_labels_must_match_the_array():
    with pytest.raises(ValueError, match="axis label"):
        plan_for(toy.quadrants, "image", np.zeros((4, 6)), "zyx")
    with pytest.raises(ValueError, match="unknown axis label"):
        plan_for(toy.quadrants, "image", np.zeros((4, 6)), "yq")


def test_choose_refuses_to_discard_data_on_its_own():
    plans = plan_for(toy.quadrants, "image", np.zeros((5, 4, 6)), "zyx")
    assert _adapt.choose(plans) is plans[0]
    with pytest.raises(ValueError, match="discard data"):
        _adapt.choose(plans[1:])


def test_plan_survives_the_wire():
    plan = plan_for(toy.quadrants, "image", np.zeros((5, 4, 6)), "zyx")[0]
    assert _adapt.AdaptationPlan.from_dict(plan.to_dict()) == plan


# -- execution -----------------------------------------------------------


def run_adapted(fn, param, array, axes, index=0, **kwargs):
    """Plan and execute in-process, the way a worker does."""
    plan = plan_for(fn, param, array, axes, **kwargs)[index]
    spec = skop.spec(fn)
    return _adapt.execute(spec, fn, {param: array}, {param: plan})


def test_iteration_stacks_results():
    stack = np.zeros((3, 4, 6))
    labels = run_adapted(toy.quadrants, "image", stack, "zyx")
    assert labels.shape == (3, 4, 6)


def test_iteration_renumbers_labels_across_slices():
    # Each plane labels its quadrants 1..4 on its own; stacking as-is would
    # claim object 1 in plane 0 and object 1 in plane 1 are the same thing.
    labels = run_adapted(toy.quadrants, "image", np.zeros((3, 4, 6)), "zyx")
    assert sorted(np.unique(labels)) == list(range(1, 13))
    assert sorted(np.unique(labels[0])) == [1, 2, 3, 4]
    assert sorted(np.unique(labels[2])) == [9, 10, 11, 12]


def test_slice_plan_runs_once():
    stack = np.zeros((3, 4, 6))
    labels = run_adapted(toy.quadrants, "image", stack, "zyx", index=1)
    assert labels.shape == (4, 6)


def test_iteration_over_two_axes():
    labels = run_adapted(toy.quadrants, "image", np.zeros((2, 3, 4, 6)), "tzyx")
    assert labels.shape == (2, 3, 4, 6)
    assert labels.max() == 4 * 6


def test_transposed_stack_reaches_the_op_in_declared_order():
    # x, y transposed *and* an extra axis to iterate: both at once.
    labels = run_adapted(toy.quadrants, "image", np.zeros((6, 3, 4)), "xzy")
    assert labels.shape == (3, 4, 6)


def test_scalar_outputs_stack_into_an_array():
    totals = _adapt._reassemble("op", "total", [1.0, 2.0, 3.0], (3,), None)
    assert totals.tolist() == [1.0, 2.0, 3.0]


def test_unstackable_output_says_so():
    varying = [np.zeros((2, 3)), np.zeros((5, 3))]
    with pytest.raises(TypeError, match="shape varies"):
        _adapt._reassemble("op", "points", varying, (2,), skop.Role.points)
    with pytest.raises(TypeError, match="cannot be stacked"):
        _adapt._reassemble("op", "notes", ["a", "b"], (2,), None)


def test_no_plan_means_no_adaptation():
    spec = skop.spec(toy.quadrants)
    plain = _adapt.execute(spec, toy.quadrants, {"image": np.zeros((4, 6))}, {})
    assert plain.shape == (4, 6)
