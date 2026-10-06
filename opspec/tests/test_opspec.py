"""Reading an op's signature into an OpSpec."""

from __future__ import annotations

from typing import Annotated

import numpy as np
import pytest
from sample_ops import (
    blur_careful,
    detect,
    find_nothing,
    otsu,
    scale,
    scale_into,
    workflow,
)

from opspec.op import (
    COMPUTER,
    FUNCTION,
    Mut,
    OpSpec,
    Out,
    OutputSpec,
    ParamsFor,
    Role,
    is_op,
    op,
)


def test_decorator_is_transparent():
    # An op stays an ordinary function for direct callers.
    assert scale(np.ones(3), factor=2.0).total == 6.0


def test_scalar_op_spec():
    spec = OpSpec.from_op(scale)
    assert spec.module == "sample_ops"
    assert spec.function == "scale"
    assert spec.env == "minimal"
    assert spec.form == FUNCTION
    assert [p.name for p in spec.params] == ["image", "factor"]


def test_outputs_are_output_specs():
    # .outputs describes each output in full, as .inputs does each input.
    spec = OpSpec.from_op(scale)
    assert [o.name for o in spec.outputs] == ["scaled", "total"]
    assert [o.type for o in spec.outputs] == [np.ndarray, float]


def test_spec_is_read_once():
    assert OpSpec.from_op(scale) is OpSpec.from_op(scale)


def test_bare_decorator():
    @op
    def plain(x: int) -> int:
        return x

    assert is_op(plain)
    assert OpSpec.from_op(plain).env is None


def test_no_environment_is_a_workflow():
    # An op with no environment runs where the caller is; a workflow is one.
    @op
    def anywhere(x: int) -> int:
        return x

    assert OpSpec.from_op(anywhere).is_workflow
    assert not OpSpec.from_op(scale).is_workflow


def test_wrapping_an_existing_function():
    def plain(x: int) -> int:
        return x + 1

    wrapped = op()(plain)
    assert is_op(wrapped)
    assert wrapped(1) == 2


def test_main_thread_and_exclusive():
    @op(env="gui", main_thread=True, exclusive=True)
    def pinned(x: int) -> int:
        return x

    spec = OpSpec.from_op(pinned)
    assert spec.main_thread
    assert spec.exclusive
    assert not OpSpec.from_op(scale).main_thread


def test_not_an_op_is_refused():
    def plain(x: int) -> int:
        return x

    with pytest.raises(TypeError, match="Not an op"):
        OpSpec.from_op(plain)


def test_ui_hints_survive_annotation():
    factor = next(p for p in OpSpec.from_op(scale).params if p.name == "factor")
    assert factor.type is float
    assert factor.default == 2.0
    assert factor.ui["widget_type"] == "FloatSlider"
    assert factor.ui["max"] == 10.0


def test_ui_hints_survive_a_none_default():
    # Before 3.11, get_type_hints rewrote a None-defaulted parameter's
    # annotation as Optional[...], which hashes the Annotated alias -- and so
    # its dict of UI hints -- and blew up. The annotation must come back
    # exactly as written, on every version an op environment might pin.
    @op(env="minimal")
    def dimmer(
        level: Annotated[float | None, {"widget_type": "FloatSlider"}] = None,
    ) -> float:
        return level or 0.0

    level = OpSpec.from_op(dimmer).params[0]
    assert level.ui["widget_type"] == "FloatSlider"
    assert level.type == (float | None)


def test_computer_form_detected():
    spec = OpSpec.from_op(scale_into)
    assert spec.form == COMPUTER
    assert [o.name for o in spec.outputs] == ["result"]
    # Output buffers are not inputs, so a GUI never asks for them.
    assert [p.name for p in spec.inputs] == ["image", "factor"]


def test_required_versus_optional():
    required = {p.name for p in OpSpec.from_op(scale).params if p.required}
    assert required == {"image"}


def test_rejects_var_args():
    @op(env="minimal")
    def bad(*args):
        return args

    with pytest.raises(TypeError, match="may not declare"):
        OpSpec.from_op(bad)


def test_rejects_mixed_forms():
    @op(env="minimal")
    def bad(a: Out[np.ndarray], b: Mut[np.ndarray]) -> None:
        pass

    with pytest.raises(TypeError, match="mixes Out and Mut"):
        OpSpec.from_op(bad)


def test_roles_are_read_off_inputs_and_outputs():
    spec = OpSpec.from_op(find_nothing)
    image = next(p for p in spec.params if p.name == "image")
    assert image.role is Role.image
    # The declared type is unchanged: a role annotates, it does not replace.
    assert image.type is np.ndarray
    assert [(o.name, o.role) for o in spec.outputs] == [
        ("labels", Role.labels),
        ("points", Role.points),
    ]


def test_role_composes_with_out():
    spec = OpSpec.from_op(scale_into)
    result = next(p for p in spec.params if p.name == "result")
    assert result.direction is not None
    assert result.role is Role.image
    assert result.type is np.ndarray
    # A computer op's outputs are its Out params, roles and all.
    assert spec.outputs == (OutputSpec("result", np.ndarray, Role.image),)


def test_role_on_a_plain_return():
    spec = OpSpec.from_op(otsu)
    assert spec.return_type is np.ndarray
    assert spec.return_role is Role.labels
    assert spec.outputs == (OutputSpec("result", np.ndarray, Role.labels),)


def test_missing_roles_are_none_not_guesses():
    # scale is deliberately unannotated: opspec reports no role rather than
    # assuming an array is an image. Guessing is a front end's job.
    spec = OpSpec.from_op(scale)
    assert next(p for p in spec.params if p.name == "image").role is None
    assert [o.role for o in spec.outputs] == [None, None]


def test_choices_and_params_for():
    spec = OpSpec.from_op(workflow)
    assert spec.is_workflow
    params = {p.name: p for p in spec.params}
    assert params["step"].choices.labels == ("fast", "careful")
    assert params["step"].choices.op("careful") is blur_careful
    assert params["step_args"].params_for == ParamsFor("step", binds=("image",))
    # The list constrains a GUI, not the function.
    assert workflow(np.ones(2), step=lambda image: image * 3)[0] == 3


def test_output_roles_resolve_in_the_result_types_own_module():
    # Boxes lives in sample_results, and sample_ops never imports BoxesOf:
    # the field annotations are strings that only resolve over there.
    assert [(o.name, o.role) for o in OpSpec.from_op(detect).outputs] == [
        ("boxes", Role.boxes)
    ]
