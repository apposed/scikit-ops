"""The serialized half of the contract: what a front end reads over Appose.

An in-process front end reads ``OpSpec`` as a Python object, so nothing here
was needed until a second front end turned up in another language. These
tests fix the wire form, because a Java reader has no way to complain that it
changed -- it will simply read the wrong thing.
"""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np
import pytest

import skop
import skop.types
from skop import host
from skop.ops import toy


class Flavor(Enum):
    sweet = "SW"
    savory = "SV"


def roundtrip(spec: skop.OpSpec) -> skop.OpSpec:
    """A spec through JSON and back, as a front end would receive it."""
    return skop.OpSpec.from_dict(json.loads(json.dumps(spec.to_dict())))


# -- the wire vocabulary ------------------------------------------------


@pytest.mark.parametrize(
    ("annotation", "expected"),
    [
        (int, skop.INT),
        (float, skop.FLOAT),
        (str, skop.STR),
        (bool, skop.BOOL),
        (np.ndarray, skop.NDARRAY),
        (Path, skop.PATH),
        (Flavor, skop.ENUM),
        (tuple[float, float], skop.UNKNOWN),
        (None, skop.UNKNOWN),
    ],
)
def test_wire_type_names(annotation, expected):
    assert skop.type_spec(annotation).name == expected


def test_bool_is_not_an_int():
    # bool subclasses int, and a checkbox is not a number field.
    assert skop.type_spec(bool).name == skop.BOOL


def test_annotated_types_classify_as_what_they_wrap():
    assert skop.type_spec(skop.types.LabelsData).name == skop.NDARRAY


def test_enum_carries_its_choices():
    spec = skop.type_spec(Flavor)
    assert [(c.name, c.value) for c in spec.choices] == [
        ("sweet", "SW"),
        ("savory", "SV"),
    ]


def test_optional_is_the_inner_type_marked_nullable():
    # Both spellings of a union have to classify the same way; the older
    # one is what an op pinned to an older Python is free to use.
    spec = skop.type_spec(Optional[float])  # noqa: UP045
    assert spec.name == skop.FLOAT
    assert spec.nullable

    modern = skop.type_spec(float | None)
    assert (modern.name, modern.nullable) == (skop.FLOAT, True)


def test_multi_member_union_is_unknown():
    assert skop.type_spec(int | str).name == skop.UNKNOWN


def test_unknown_says_what_it_could_not_render():
    # A front end that cannot render a parameter has to explain which one and
    # why, so the spelling of the original annotation has to survive.
    assert "tuple" in skop.type_spec(tuple[int, ...]).detail


def test_every_op_classifies():
    # UNKNOWN is allowed; a crash while classifying is not.
    specs, failures = skop.discover()
    assert not failures
    for spec in specs:
        for param in spec.params:
            assert skop.type_spec(param.type).name in skop.WIRE_TYPES


# -- OpSpec as JSON -----------------------------------------------------


def test_spec_survives_json():
    spec = skop.spec(toy.scale)
    back = roundtrip(spec)
    assert back.name == spec.name
    assert back.module == spec.module
    assert back.function == spec.function
    assert back.env == spec.env
    assert back.form == spec.form
    assert back.doc == spec.doc
    assert [p.name for p in back.params] == [p.name for p in spec.params]


def test_every_op_round_trips():
    specs, _ = skop.discover()
    assert specs
    for spec in specs:
        assert roundtrip(spec).to_dict() == spec.to_dict()


def test_derived_outputs_are_written_out_not_derived():
    # A NamedTuple return does not cross the boundary, so the output names
    # have to travel as data rather than be recomputed from a type.
    spec = skop.spec(toy.scale)
    assert spec.outputs == ("scaled", "total")
    assert roundtrip(spec).outputs == ("scaled", "total")


def test_output_roles_survive():
    back = roundtrip(skop.spec(toy.find_nothing))
    assert [(o.name, o.role) for o in back.output_specs] == [
        ("labels", skop.Role.labels),
        ("points", skop.Role.points),
    ]


def test_ui_hints_survive():
    back = roundtrip(skop.spec(toy.scale))
    factor = next(p for p in back.params if p.name == "factor")
    assert factor.ui == {
        "widget_type": "FloatSlider",
        "min": 0.0,
        "max": 10.0,
        "step": 0.1,
    }


def test_required_and_default_survive():
    back = roundtrip(skop.spec(toy.scale))
    image, factor = back.params
    assert image.required
    assert not factor.required
    assert factor.default == 2.0


def test_out_params_are_marked():
    back = roundtrip(skop.spec(toy.scale_into))
    result = next(p for p in back.params if p.name == "result")
    assert result.direction is skop._spec.OUT
    assert result not in back.inputs


def test_enum_default_travels_as_its_value():
    # The worker rebuilds an Enum from its value, so that is what a front end
    # must send back -- and so what the default has to be spelled as.
    from skop.ops import morphology

    spec = skop.spec(morphology.dilation)
    shape = next(p.to_dict() for p in spec.params if p.name == "shape")
    assert shape["type"]["name"] == skop.ENUM
    assert shape["default"] in [c["value"] for c in shape["type"]["choices"]]

    # And the names are what a dialog shows, alongside the values it sends.
    back = roundtrip(spec)
    footprint = next(p for p in back.params if p.name == "shape")
    assert [c.name for c in footprint.type.choices] == ["ball", "box", "diamond"]


def test_axes_survive():
    back = roundtrip(skop.spec(toy.quadrants))
    image = back.params[0]
    assert image.axes is not None
    assert image.axes.names == ("y", "x")
    assert not image.axes.variadic


def test_variadic_axes_survive():
    from skop.ops import threshold

    back = roundtrip(skop.spec(threshold.otsu))
    image = back.params[0]
    assert image.axes is not None
    assert image.axes.variadic
    assert image.axes.slots == ()


def test_role_survives():
    back = roundtrip(skop.spec(toy.quadrants))
    assert back.params[0].role is skop.Role.image


def test_unrenderable_param_costs_only_itself():
    @skop.op(env="minimal")
    def awkward(
        image: np.ndarray,
        window: tuple[int, int] = (3, 3),
        sigma: float = 1.0,
    ) -> np.ndarray: ...

    back = roundtrip(skop.spec(awkward))
    kinds = {p.name: p.type.name for p in back.params}
    assert kinds == {
        "image": skop.NDARRAY,
        "window": skop.UNKNOWN,
        "sigma": skop.FLOAT,
    }
    # And it is optional, so a front end can leave it alone and still run.
    assert not next(p for p in back.params if p.name == "window").required


# -- describe -----------------------------------------------------------


def test_describe_is_json():
    described = host.describe()
    json.dumps(described)  # raises if anything in it is not JSON-safe
    assert described["package"] == "skop.ops"
    assert described["ops"]
    assert described["failures"] == []


def test_describe_matches_discover():
    specs, _ = skop.discover()
    described = host.describe()
    assert [op["name"] for op in described["ops"]] == [s.name for s in specs]


def test_describe_reports_failures_rather_than_raising(tmp_path, monkeypatch):
    package = tmp_path / "brokenops"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "bad.py").write_text("import definitely_not_installed\n")
    monkeypatch.syspath_prepend(str(tmp_path))

    described = host.describe("brokenops")
    assert described["ops"] == []
    (failure,) = described["failures"]
    assert failure["module"] == "brokenops.bad"
    assert "definitely_not_installed" in failure["error"]
    assert any("definitely_not_installed" in i for i in failure["heavy_imports"])


# -- plan ---------------------------------------------------------------


def test_plan_by_op_id():
    plan = host.plan("skop.ops.toy:quadrants", "image", [3, 64, 32], ["z", "y", "x"])
    json.dumps(plan)
    assert plan["mapping"] == [1, 2]
    assert plan["iterate"] == [0]
    assert plan["calls"] == 3
    assert plan["warnings"] == []


def test_plan_matches_the_in_process_call():
    over_wire = host.plan(
        "skop.ops.toy:quadrants", "image", [3, 64, 32], ["z", "y", "x"]
    )
    in_process = skop.plan(toy.quadrants, "image", (3, 64, 32), list("zyx"))
    assert over_wire == in_process.to_dict()


def test_plan_warns_rather_than_refusing():
    plan = host.plan("skop.ops.toy:quadrants", "image", [4, 8], ["z", "x"])
    assert plan["warnings"] == ["y is being fed the z axis"]


def test_plan_accepts_json_string_keys():
    # JSON has no integer object keys, so an axis index arrives as a string.
    plan = host.plan(
        "skop.ops.toy:quadrants",
        "image",
        [3, 64, 32],
        ["z", "y", "x"],
        position={"0": 2},
        dispositions={"0": skop.SELECT},
    )
    assert plan["select"] == [[0, 2]]
    assert not plan["lossless"]


def test_plan_accepts_an_explicit_mapping():
    plan = host.plan(
        "skop.ops.toy:quadrants",
        "image",
        [3, 64, 32],
        ["z", "y", "x"],
        mapping=[0, 2],
    )
    assert plan["mapping"] == [0, 2]
    assert plan["iterate"] == [1]


def test_plan_accepts_unnamed_axes():
    plan = host.plan("skop.ops.toy:quadrants", "image", [64, 32], [None, None])
    assert plan["mapping"] == [0, 1]
    assert plan["warnings"] == []


def test_plan_rejects_an_unknown_op():
    with pytest.raises(ValueError, match="No op named"):
        host.plan("skop.ops.toy:nonesuch", "image", [4, 4], ["y", "x"])


def test_plan_rejects_a_malformed_op_id():
    with pytest.raises(ValueError, match="Not an op ID"):
        host.plan("skop.ops.toy.quadrants", "image", [4, 4], ["y", "x"])


# -- the axis-order trap ------------------------------------------------


def test_axis_order_is_numpy_order():
    """The reversal an ImgLib2-shaped host has to do before calling plan.

    An ImgPlus with axes (X, Y, Z) is a numpy array of shape (z, y, x). Get
    this wrong and nothing raises: the op runs on transposed data and returns
    a plausible, wrong answer. So the contract is stated as a test.
    """
    imglib2_axes = ["x", "y", "z"]  # x-fastest, as ImgLib2 reports them
    imglib2_dims = [32, 64, 3]

    numpy_axes = list(reversed(imglib2_axes))
    numpy_shape = list(reversed(imglib2_dims))

    plan = host.plan("skop.ops.toy:quadrants", "image", numpy_shape, numpy_axes)
    assert plan["input_axes"] == ["z", "y", "x"]
    assert plan["warnings"] == []
    # y and x fill the slots; z is what is iterated over.
    assert plan["mapping"] == [1, 2]
    assert plan["iterate"] == [0]

    # And the same call without reversing is wrong in a way that does not
    # raise -- which is exactly why the reversal has to be tested, not trusted.
    unreversed = host.plan("skop.ops.toy:quadrants", "image", numpy_shape, imglib2_axes)
    assert unreversed["mapping"] == [1, 0]
    assert unreversed["iterate"] == [2]
    assert unreversed["calls"] == 32
    # Every name matched a slot, so there is nothing at all to warn about.
    # The op runs 32 times over transposed planes and produces a label image
    # of the right shape. Only this test can catch that.
    assert unreversed["warnings"] == []


# -- constants ----------------------------------------------------------


def test_constants_are_what_the_runner_actually_uses():
    from skop import runner

    constants = host.constants()
    assert constants["call"] == runner._CALL
    assert constants["init"] == runner._INIT
    json.dumps(constants)


def test_constants_describe_the_vocabularies():
    constants = host.constants()
    assert set(constants["wire_types"]) == set(skop.WIRE_TYPES)
    assert set(constants["roles"]) == {r.value for r in skop.Role}
    assert set(constants["dispositions"]) == {skop.ITERATE, skop.SELECT, skop.PASS}
