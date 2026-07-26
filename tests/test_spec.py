"""Signature introspection: what the host and the GUI layer read off an op."""

from __future__ import annotations

from typing import Annotated

import numpy as np
import pytest

import skop
from skop.ops import toy


def test_decorator_is_transparent():
    # An op stays an ordinary function for direct callers.
    assert toy.add(2, 3) == 5


def test_scalar_op_spec():
    spec = skop.spec(toy.add)
    assert spec.module == "skop.ops.toy"
    assert spec.function == "add"
    assert spec.env == "minimal"
    assert spec.form is skop.FUNCTION
    assert [p.name for p in spec.params] == ["a", "b"]
    assert spec.outputs == ("result",)


def test_named_tuple_outputs():
    spec = skop.spec(toy.scale)
    assert spec.outputs == ("scaled", "total")
    assert spec.form is skop.FUNCTION


def test_ui_hints_survive_annotation():
    spec = skop.spec(toy.scale)
    factor = next(p for p in spec.params if p.name == "factor")
    assert factor.type is float
    assert factor.default == 2.0
    assert factor.ui["widget_type"] == "FloatSlider"
    assert factor.ui["max"] == 10.0


def test_ui_hints_survive_a_none_default():
    # Before 3.11, get_type_hints rewrote a None-defaulted parameter's
    # annotation as Optional[...], which hashes the Annotated alias -- and so
    # its dict of UI hints -- and blew up. The annotation must come back
    # exactly as written, on every version an op environment might pin.
    @skop.op(env="minimal")
    def dimmer(
        level: Annotated[float | None, {"widget_type": "FloatSlider"}] = None,
    ) -> float:
        return level or 0.0

    level = skop.spec(dimmer).params[0]
    assert level.ui["widget_type"] == "FloatSlider"
    assert level.type == (float | None)


def test_computer_form_detected():
    spec = skop.spec(toy.scale_into)
    assert spec.form is skop.COMPUTER
    assert spec.outputs == ("result",)
    # Output buffers are not inputs, so a GUI never asks for them.
    assert [p.name for p in spec.inputs] == ["image", "factor"]


def test_required_versus_optional():
    spec = skop.spec(toy.scale)
    required = {p.name for p in spec.params if p.required}
    assert required == {"image"}


def test_rejects_var_args():
    @skop.op(env="minimal")
    def bad(*args):
        return args

    with pytest.raises(TypeError, match="may not declare"):
        skop.spec(bad)


def test_rejects_mixed_forms():
    @skop.op(env="minimal")
    def bad(a: skop.Out[np.ndarray], b: skop.Mut[np.ndarray]) -> None:
        pass

    with pytest.raises(TypeError, match="mixes Out and Mut"):
        skop.spec(bad)


def test_roles_are_read_off_inputs_and_outputs():
    spec = skop.spec(toy.find_nothing)
    image = next(p for p in spec.params if p.name == "image")
    assert image.role is skop.Role.image
    # The declared type is unchanged: a role annotates, it does not replace.
    assert image.type is np.ndarray
    assert [(o.name, o.role) for o in spec.output_specs] == [
        ("labels", skop.Role.labels),
        ("points", skop.Role.points),
    ]


def test_role_composes_with_out():
    spec = skop.spec(toy.scale_into)
    result = next(p for p in spec.params if p.name == "result")
    assert result.direction is not None
    assert result.role is skop.Role.image
    assert result.type is np.ndarray
    # A computer op's outputs are its Out params, roles and all.
    assert spec.output_specs == (
        skop.OutputSpec("result", np.ndarray, skop.Role.image),
    )


def test_role_on_a_plain_return():
    from skop.ops.threshold import otsu

    spec = skop.spec(otsu)
    assert spec.return_type is np.ndarray
    assert spec.output_specs == (
        skop.OutputSpec("result", np.ndarray, skop.Role.labels),
    )


def test_missing_roles_are_none_not_guesses():
    # toy.scale is deliberately unannotated: skop reports 'unknown' rather
    # than assuming an array is an image. Guessing is a front end's job.
    spec = skop.spec(toy.scale)
    assert next(p for p in spec.params if p.name == "image").role is None
    assert [o.role for o in spec.output_specs] == [None, None]
    assert [o.name for o in spec.output_specs] == ["scaled", "total"]
    assert [o.type for o in spec.output_specs] == [np.ndarray, float]


def test_progress_outside_worker_is_a_noop():
    # Mode B: called directly, with nobody listening.
    skop.progress("hello", 1, 2)
    assert skop.cancel_requested() is False
