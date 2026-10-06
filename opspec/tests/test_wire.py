"""The wire form: an OpSpec as JSON, for another process or language.

An in-process host reads ``OpSpec`` as a Python object. A Java front end
reads this dict instead, and has no way to complain if it changes -- it will
simply read the wrong thing. So these tests fix it.
"""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np
import pytest
from sample_ops import (
    dilate,
    find_nothing,
    otsu,
    quadrants,
    scale,
    scale_into,
    workflow,
)

from opspec.op import (
    BOOL,
    ENUM,
    FLOAT,
    INT,
    NDARRAY,
    OUT,
    PATH,
    STR,
    UNKNOWN,
    OpSpec,
    ParamsFor,
    Role,
    op,
    type_spec,
)
from opspec.types import LabelsOf


class Flavor(Enum):
    sweet = "SW"
    savory = "SV"


def roundtrip(spec: OpSpec) -> OpSpec:
    """A spec through JSON and back, as a front end would receive it."""
    return OpSpec.from_dict(json.loads(json.dumps(spec.to_dict())))


# -- the wire vocabulary ------------------------------------------------


@pytest.mark.parametrize(
    ("annotation", "expected"),
    [
        (int, INT),
        (float, FLOAT),
        (str, STR),
        (bool, BOOL),
        (np.ndarray, NDARRAY),
        (Path, PATH),
        (Flavor, ENUM),
        (tuple[float, float], UNKNOWN),
        (None, UNKNOWN),
    ],
)
def test_wire_type_names(annotation, expected):
    assert type_spec(annotation).name == expected


def test_bool_is_not_an_int():
    # bool subclasses int, and a checkbox is not a number field.
    assert type_spec(bool).name == BOOL


def test_annotated_types_classify_as_what_they_wrap():
    assert type_spec(LabelsOf[np.ndarray]).name == NDARRAY


def test_enum_carries_its_choices():
    spec = type_spec(Flavor)
    assert [(c.name, c.value) for c in spec.choices] == [
        ("sweet", "SW"),
        ("savory", "SV"),
    ]


def test_optional_is_the_inner_type_marked_nullable():
    # Both spellings of a union have to classify the same way; the older
    # one is what an op pinned to an older Python is free to use.
    spec = type_spec(Optional[float])  # noqa: UP045
    assert spec.name == FLOAT
    assert spec.nullable

    modern = type_spec(float | None)
    assert (modern.name, modern.nullable) == (FLOAT, True)


def test_multi_member_union_is_unknown():
    assert type_spec(int | str).name == UNKNOWN


def test_unknown_says_what_it_could_not_render():
    # A front end that cannot render a parameter has to explain which one and
    # why, so the spelling of the original annotation has to survive.
    assert "tuple" in type_spec(tuple[int, ...]).detail


# -- OpSpec as JSON -----------------------------------------------------


def test_spec_survives_json():
    spec = OpSpec.from_op(scale)
    back = roundtrip(spec)
    assert back.name == spec.name
    assert back.module == spec.module
    assert back.function == spec.function
    assert back.env == spec.env
    assert back.form == spec.form
    assert back.doc == spec.doc
    assert [p.name for p in back.params] == [p.name for p in spec.params]


@pytest.mark.parametrize(
    "fn", [scale, scale_into, quadrants, otsu, find_nothing, dilate]
)
def test_round_trip_is_stable(fn):
    spec = OpSpec.from_op(fn)
    assert roundtrip(spec).to_dict() == spec.to_dict()


def test_outputs_are_written_out_not_derived():
    # A NamedTuple return does not cross the boundary, so the outputs have
    # to travel as data rather than be recomputed from a type.
    back = roundtrip(OpSpec.from_op(scale))
    assert [o.name for o in back.outputs] == ["scaled", "total"]


def test_output_roles_survive():
    back = roundtrip(OpSpec.from_op(find_nothing))
    assert [(o.name, o.role) for o in back.outputs] == [
        ("labels", Role.labels),
        ("points", Role.points),
    ]


def test_return_type_and_role_survive():
    data = OpSpec.from_op(otsu).to_dict()
    assert data["return_type"]["name"] == NDARRAY
    assert data["return_role"] == "labels"
    back = OpSpec.from_dict(data)
    assert back.return_type.name == NDARRAY
    assert back.return_role is Role.labels


def test_main_thread_and_exclusive_survive():
    @op(env="gui", main_thread=True, exclusive=True)
    def pinned(x: int) -> int:
        return x

    back = roundtrip(OpSpec.from_op(pinned))
    assert back.main_thread
    assert back.exclusive


def test_ui_hints_survive():
    back = roundtrip(OpSpec.from_op(scale))
    factor = next(p for p in back.params if p.name == "factor")
    assert factor.ui == {
        "widget_type": "FloatSlider",
        "min": 0.0,
        "max": 10.0,
        "step": 0.1,
    }


def test_required_and_default_survive():
    image, factor = roundtrip(OpSpec.from_op(scale)).params
    assert image.required
    assert not factor.required
    assert factor.default == 2.0


def test_out_params_are_marked():
    back = roundtrip(OpSpec.from_op(scale_into))
    result = next(p for p in back.params if p.name == "result")
    assert result.direction is OUT
    assert result not in back.inputs


def test_enum_default_travels_as_its_value():
    # The worker rebuilds an Enum from its value, so that is what a front end
    # must send back -- and so what the default has to be spelled as.
    shape = next(
        p.to_dict() for p in OpSpec.from_op(dilate).params if p.name == "shape"
    )
    assert shape["type"]["name"] == ENUM
    assert shape["default"] in [c["value"] for c in shape["type"]["choices"]]

    # And the names are what a dialog shows, alongside the values it sends.
    back = roundtrip(OpSpec.from_op(dilate))
    footprint = next(p for p in back.params if p.name == "shape")
    assert [c.name for c in footprint.type.choices] == ["ball", "box", "diamond"]


def test_axes_survive():
    image = roundtrip(OpSpec.from_op(quadrants)).params[0]
    assert image.axes is not None
    assert image.axes.names == ("y", "x")
    assert not image.axes.variadic


def test_variadic_axes_survive():
    image = roundtrip(OpSpec.from_op(otsu)).params[0]
    assert image.axes is not None
    assert image.axes.variadic
    assert image.axes.slots == ()


def test_role_survives():
    assert roundtrip(OpSpec.from_op(quadrants)).params[0].role is Role.image


def test_choices_and_params_for_survive_as_ids():
    # A front end in another language gets the menu, not the functions.
    step, step_args = roundtrip(OpSpec.from_op(workflow)).params[1:]
    assert step.choices.ids == (
        ("fast", "sample_ops:blur_fast"),
        ("careful", "sample_ops:blur_careful"),
    )
    assert step_args.params_for == ParamsFor("step", binds=("image",))


def test_unrenderable_param_costs_only_itself():
    @op(env="minimal")
    def awkward(
        image: np.ndarray,
        window: tuple[int, int] = (3, 3),
        sigma: float = 1.0,
    ) -> np.ndarray: ...

    back = roundtrip(OpSpec.from_op(awkward))
    kinds = {p.name: p.type.name for p in back.params}
    assert kinds == {"image": NDARRAY, "window": UNKNOWN, "sigma": FLOAT}
    # And it is optional, so a front end can leave it alone and still run.
    assert not next(p for p in back.params if p.name == "window").required
