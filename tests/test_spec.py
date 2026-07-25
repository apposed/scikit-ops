"""Signature introspection: what the host and the GUI layer read off an op."""

from __future__ import annotations

import numpy as np
import pytest

import opkit
from ops import toy


def test_decorator_is_transparent():
    # An op stays an ordinary function for direct callers.
    assert toy.add(2, 3) == 5


def test_scalar_op_spec():
    spec = opkit.spec(toy.add)
    assert spec.module == "ops.toy"
    assert spec.function == "add"
    assert spec.env == "minimal"
    assert spec.form is opkit.FUNCTION
    assert [p.name for p in spec.params] == ["a", "b"]
    assert spec.outputs == ("result",)


def test_named_tuple_outputs():
    spec = opkit.spec(toy.scale)
    assert spec.outputs == ("scaled", "total")
    assert spec.form is opkit.FUNCTION


def test_ui_hints_survive_annotation():
    spec = opkit.spec(toy.scale)
    factor = next(p for p in spec.params if p.name == "factor")
    assert factor.type is float
    assert factor.default == 2.0
    assert factor.ui["widget_type"] == "FloatSlider"
    assert factor.ui["max"] == 10.0


def test_computer_form_detected():
    spec = opkit.spec(toy.scale_into)
    assert spec.form is opkit.COMPUTER
    assert spec.outputs == ("result",)
    # Output buffers are not inputs, so a GUI never asks for them.
    assert [p.name for p in spec.inputs] == ["image", "factor"]


def test_required_versus_optional():
    spec = opkit.spec(toy.scale)
    required = {p.name for p in spec.params if p.required}
    assert required == {"image"}


def test_rejects_var_args():
    @opkit.op(env="minimal")
    def bad(*args):
        return args

    with pytest.raises(TypeError, match="may not declare"):
        opkit.spec(bad)


def test_rejects_mixed_forms():
    @opkit.op(env="minimal")
    def bad(a: opkit.Out[np.ndarray], b: opkit.Mut[np.ndarray]) -> None:
        pass

    with pytest.raises(TypeError, match="mixes Out and Mut"):
        opkit.spec(bad)


def test_progress_outside_worker_is_a_noop():
    # Mode B: called directly, with nobody listening.
    opkit.progress("hello", 1, 2)
    assert opkit.cancel_requested() is False
