"""Fitting a caller's array to the axes an op consumes: the plan.

Planning is pure arithmetic over axis names and shapes. Carrying a plan out
-- slicing, calling, stacking -- touches pixels, and belongs to a runner.
"""

from __future__ import annotations

import numpy as np
import pytest
from sample_ops import otsu, quadrants, scale

from opspec.op import Axes, OpSpec, ParamSpec, Role
from opspec.plan import (
    ITERATE,
    PASS,
    SELECT,
    AdaptationPlan,
    build,
    normalize_axes,
    plan,
)


def plan_for(fn, param, array, axes, **kwargs):
    return plan(fn, param, array, axes, **kwargs)


def param_with(axes):
    """A ParamSpec like quadrants' image, but declaring *axes*."""
    image = next(p for p in OpSpec.from_op(quadrants).params if p.name == "image")
    return ParamSpec(
        name=image.name,
        type=image.type,
        default=image.default,
        axes=axes,
    )


# -- the declaration -----------------------------------------------------


def test_one_argument_is_one_slot():
    axes = Axes("y", "x", "c?")
    assert axes.names == ("y", "x", "c")
    assert axes.core == ("y", "x")
    assert axes.optional == frozenset({"c"})
    assert not axes.variadic


def test_labels_may_be_any_string():
    # There is no agreed letter for a lifetime bin, and n-D means n-D.
    axes = Axes("lifetime", "y", "x")
    assert axes.names == ("lifetime", "y", "x")
    assert axes.core == ("lifetime", "y", "x")


def test_a_lone_iterable_is_the_whole_sequence():
    assert Axes(list("zyx")) == Axes("z", "y", "x")
    assert Axes(["lifetime", "y", "x"]) == Axes("lifetime", "y", "x")


def test_a_lone_stray_question_mark_is_refused():
    # list() splits the '?' loose from the axis it belongs to.
    with pytest.raises(ValueError, match="lone '\\?' is not an axis"):
        Axes(list("yxc?"))


def test_a_wildcard_slot_has_no_name_and_may_repeat():
    # A 2-D triangulation does not care whether it gets y x, z x or z y.
    axes = Axes("*", "*")
    assert axes.names == ("*", "*")
    assert [slot.name for slot in axes.slots] == [None, None]


def test_an_optional_wildcard_is_refused():
    # It could never be filled: wildcards match no name, and an optional slot
    # is filled by name and nothing else.
    with pytest.raises(ValueError, match="not a usable slot"):
        Axes("y", "x", "*?")


def test_variadic_takes_any_number_of_axes():
    axes = Axes(variadic=True)
    assert axes.slots == ()
    assert axes.variadic


def test_axes_rejects_nonsense():
    with pytest.raises(ValueError, match="Repeated axis"):
        Axes("y", "x", "y")
    with pytest.raises(ValueError, match="separator"):
        Axes("z y x")
    with pytest.raises(ValueError, match="not a non-empty string"):
        Axes("y", "")


def test_synonyms_resolve_to_the_canonical_name():
    # scikit-image's spelling, ImageJ's, and Bio-Formats' upper case, all of
    # which a front end might report, all mean the same three axes.
    assert Axes("pln", "row", "col") == Axes("z", "y", "x")
    assert Axes("slice", "channel") == Axes("z", "c")
    assert Axes(list("ZYX")) == Axes("z", "y", "x")
    assert Axes("Frame", "Ch") == Axes("t", "c")


def test_labels_without_a_canonical_equivalent_survive_untouched():
    # bioimage.io's batch axis and SCIFIO's lifetime are not synonyms of
    # anything; folding them into something canonical would be a guess.
    assert Axes("b", "lifetime", "polarization").names == (
        "b",
        "lifetime",
        "polarization",
    )
    # Nor is bioio's 's' -- it distinguishes RGB samples from channels.
    assert Axes("s").names == ("s",)


def test_a_synonym_collision_is_a_repeated_axis():
    with pytest.raises(ValueError, match="Repeated axis 'y'"):
        Axes("y", "row")


def test_axes_reach_the_spec():
    spec = OpSpec.from_op(quadrants)
    image = next(p for p in spec.params if p.name == "image")
    assert image.axes == Axes("y", "x")
    # An Axes composes with the role already on the parameter.
    assert image.role is Role.image
    assert image.type is np.ndarray


def test_unannotated_parameter_has_no_axes():
    spec = OpSpec.from_op(scale)
    assert next(p for p in spec.params if p.name == "image").axes is None


# -- mapping input axes onto slots ---------------------------------------


def test_a_name_match_wins_over_position():
    plan = plan_for(quadrants, "image", np.zeros((6, 4)), list("xy"))
    assert plan.mapping == (1, 0)  # y <- axis 1, x <- axis 0
    assert plan.warnings == ()


def test_an_op_declaring_y_accepts_an_array_labelled_row():
    plan = plan_for(quadrants, "image", np.zeros((5, 4, 6)), ("pln", "row", "col"))
    assert plan.iterate == (0,)
    assert plan.output_axes == ("z", "y", "x")
    assert plan.warnings == ()


def test_unnamed_axes_are_mapped_by_position_right_aligned():
    # A plain ndarray with no labels at all: the innermost axes are the ones
    # a 2-D imaging op means, so they fill the slots and the rest is spare.
    plan = plan_for(quadrants, "image", np.zeros((5, 4, 6)), [None, None, None])
    assert plan.mapping == (1, 2)
    assert plan.iterate == (0,)
    # Nothing was named, so nothing can be said to be misaligned.
    assert plan.warnings == ()


def test_a_name_mismatch_warns_rather_than_refuses():
    # The headline of the hints-not-requirements rule. A 2-D op fed z and x
    # runs; it just says which slot got fed something it did not ask for.
    plan = plan_for(quadrants, "image", np.zeros((5, 6)), list("zx"))
    assert plan.mapping == (0, 1)
    assert plan.warnings == ("y is being fed the z axis",)
    assert plan.uses_all_data


def test_a_wildcard_slot_never_warns():
    plan = build(param_with(Axes("*", "*")), list("zy"), (5, 4))
    assert plan.mapping == (0, 1)
    assert plan.warnings == ()


def test_an_optional_slot_is_filled_by_name_and_never_by_position():
    # Load-bearing: dropping z into the channel slot would have to_gray
    # average across the stack instead of iterating over it.
    axes = Axes("y", "x", "c?")
    plan = build(param_with(axes), list("zyx"), (5, 4, 6))
    assert plan.mapping == (1, 2, None)
    assert plan.iterate == (0,)
    # Given a real channel axis, it does fill.
    plan = build(param_with(axes), list("yxc"), (4, 6, 3))
    assert plan.mapping == (0, 1, 2)
    assert plan.iterate == ()


def test_too_few_axes_is_the_one_thing_that_cannot_be_adapted():
    with pytest.raises(ValueError, match="consumes 2 axes but was given 1"):
        plan_for(quadrants, "image", np.zeros((4,)), ["x"])


def test_axis_labels_must_match_the_array():
    with pytest.raises(ValueError, match="axis label"):
        plan_for(quadrants, "image", np.zeros((4, 6)), list("zyx"))


def test_a_bare_string_is_one_label_on_the_caller_side_too():
    assert normalize_axes("lifetime") == ("lifetime",)
    assert normalize_axes(("lifetime", "y", "x")) == ("lifetime", "y", "x")
    with pytest.raises(ValueError, match="is one axis label"):
        normalize_axes("zyx")


def test_a_caller_may_map_the_slots_itself():
    # The whole point of the redesign: which axis feeds which slot belongs to
    # whoever owns the data. Here, ZY cross-sections instead of YX planes.
    plan = plan_for(
        quadrants,
        "image",
        np.zeros((5, 4, 6)),
        list("zyx"),
        mapping=(0, 1),
        dispositions={2: ITERATE},
    )
    assert plan.iterate == (2,)
    assert plan.calls == 6
    assert plan.output_axes == ("x", "z", "y")
    assert plan.warnings == ("y is being fed the z axis", "x is being fed the y axis")


def test_a_mapping_that_cannot_work_is_refused():
    with pytest.raises(ValueError, match="does not have"):
        plan_for(quadrants, "image", np.zeros((4, 6)), list("yx"), mapping=(0, 9))
    with pytest.raises(ValueError, match="more than one slot"):
        plan_for(quadrants, "image", np.zeros((4, 6)), list("yx"), mapping=(1, 1))
    with pytest.raises(ValueError, match="required"):
        plan_for(quadrants, "image", np.zeros((4, 6)), list("yx"), mapping=(0, None))


# -- what becomes of the leftovers ---------------------------------------


def test_exact_match_is_a_no_op():
    plan = plan_for(quadrants, "image", np.zeros((4, 6)), list("yx"))
    assert plan.uses_all_data
    assert plan.calls == 1
    assert plan.iterate == ()
    assert plan.transpose == (0, 1)
    assert plan.summary == "as is"


def test_transposed_input_is_reordered():
    plan = plan_for(quadrants, "image", np.zeros((6, 4)), list("xy"))
    assert plan.transpose == (1, 0)


def test_a_leftover_axis_is_iterated_by_default():
    plan = plan_for(quadrants, "image", np.zeros((5, 4, 6)), list("zyx"))
    assert plan.uses_all_data
    assert plan.iterate == (0,)
    assert plan.calls == 5
    assert plan.output_axes == ("z", "y", "x")


def test_a_leftover_axis_can_be_selected_instead():
    plan = plan_for(
        quadrants,
        "image",
        np.zeros((5, 4, 6)),
        list("zyx"),
        position={"z": 3},
        dispositions={0: SELECT},
    )
    assert not plan.uses_all_data
    assert plan.select == ((0, 3),)
    assert "z=3" in plan.summary


def test_a_position_may_be_keyed_by_axis_index():
    # The only way to place an *unnamed* axis -- and what a front end that has
    # not worked out any names still knows about its own sliders.
    plan = plan_for(
        quadrants,
        "image",
        np.zeros((5, 4, 6)),
        [None, "y", "x"],
        position={0: 3},
        dispositions={0: SELECT},
    )
    assert plan.select == ((0, 3),)


def test_an_index_key_outranks_a_name_key():
    plan = plan_for(
        quadrants,
        "image",
        np.zeros((5, 4, 6)),
        list("zyx"),
        position={"z": 1, 0: 3},
        dispositions={0: SELECT},
    )
    assert plan.select == ((0, 3),)


def test_a_variadic_op_is_handed_its_leftovers_whole():
    plan = plan_for(otsu, "image", np.zeros((5, 4, 6)), list("zyx"))
    assert plan.uses_all_data
    assert plan.iterate == ()
    assert plan.passed == (0, 1, 2)
    assert plan.calls == 1


def test_a_variadic_op_can_be_iterated_when_the_user_wants_that():
    # One threshold for the volume by default; one per plane on request.
    plan = plan_for(
        otsu,
        "image",
        np.zeros((5, 4, 6)),
        list("zyx"),
        dispositions={0: ITERATE},
    )
    assert plan.iterate == (0,)
    assert plan.calls == 5


def test_only_a_variadic_op_may_be_handed_extra_axes():
    with pytest.raises(ValueError, match="not variadic"):
        plan_for(
            quadrants,
            "image",
            np.zeros((5, 4, 6)),
            list("zyx"),
            dispositions={0: PASS},
        )


def test_a_disposition_for_a_consumed_axis_is_refused():
    with pytest.raises(ValueError, match="consumed by a slot"):
        plan_for(
            quadrants,
            "image",
            np.zeros((5, 4, 6)),
            list("zyx"),
            dispositions={1: ITERATE},
        )


def test_the_default_plan_never_discards_data():
    # What used to be choose()'s runtime refusal is now structural: nothing
    # the planner settles on by itself drops a thing.
    for labels, shape in ((list("zyx"), (5, 4, 6)), (list("tzyx"), (2, 5, 4, 6))):
        assert plan_for(quadrants, "image", np.zeros(shape), labels).uses_all_data


def test_a_non_canonical_axis_is_iterated_like_any_other():
    # The op has no idea what a lifetime bin is, and does not need one.
    plan = plan_for(quadrants, "image", np.zeros((7, 4, 6)), ("lifetime", "y", "x"))
    assert plan.iterate == (0,)
    assert plan.calls == 7
    assert plan.output_axes == ("lifetime", "y", "x")


def test_a_non_canonical_axis_can_be_consumed():
    plan = build(param_with(Axes("lifetime", "y", "x")), list("zyx"), (5, 4, 6))
    assert plan.mapping == (0, 1, 2)
    assert plan.warnings == ("lifetime is being fed the z axis",)


def test_plan_survives_the_wire():
    plan = plan_for(quadrants, "image", np.zeros((5, 4, 6)), list("zyx"))
    assert AdaptationPlan.from_dict(plan.to_dict()) == plan
