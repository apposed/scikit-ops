"""The serialized half of the contract: what a front end reads over Appose.

An in-process front end reads ``OpSpec`` as a Python object, so nothing here
was needed until a second front end turned up in another language. These
tests fix the wire form, because a Java reader has no way to complain that it
changed -- it will simply read the wrong thing.
"""

from __future__ import annotations

import json
from enum import Enum

import pytest

import skop
import skop.host
import skop.types
from skop import host
from skop.ops import toy


class Flavor(Enum):
    sweet = "SW"
    savory = "SV"


def roundtrip(spec: skop.OpSpec) -> skop.OpSpec:
    """A spec through JSON and back, as a front end would receive it."""
    return skop.OpSpec.from_dict(json.loads(json.dumps(spec.to_dict())))


# -- every op through the wire ----------------------------------------
#
# What the wire form is, and that it round-trips, is opspec's and tested there
# (opspec/tests/test_wire.py). These check skop's own ops against it.


def test_every_op_classifies():
    # UNKNOWN is allowed; a crash while classifying is not.
    specs, failures = skop.discover()
    assert not failures
    for spec in specs:
        for param in spec.params:
            assert skop.type_spec(param.type).name in skop.WIRE_TYPES


def test_every_op_round_trips():
    specs, _ = skop.discover()
    assert specs
    for spec in specs:
        assert roundtrip(spec).to_dict() == spec.to_dict()


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
    assert not plan["uses_all_data"]


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


# --- Where the worker looks for skop --------------------------------------
#
# A checkout's root holds only skop, so it goes first and shadows the copy
# an environment installs. An install's root is the host's site-packages,
# which holds everything, and putting that first hands a worker the host's
# whole environment -- fatal when the two run different Pythons. See the
# note on skop.host.INIT_APPEND.


def test_a_source_tree_is_a_checkout(tmp_path):
    assert skop.host.is_checkout(tmp_path)


def test_site_packages_is_not_a_checkout():
    import sysconfig

    purelib = sysconfig.get_paths()["purelib"]
    assert not skop.host.is_checkout(purelib)


def test_checkout_root_goes_first(tmp_path):
    script = skop.host.init_script(tmp_path)
    assert "sys.path.insert(0, " in script
    assert "sys.path.append(" not in script


def test_installed_root_goes_last():
    import sysconfig

    script = skop.host.init_script(sysconfig.get_paths()["purelib"])
    assert "sys.path.append(" in script
    assert "sys.path.insert(0, " not in script


def test_metadata_variant_tracks_the_same_decision(tmp_path):
    import sysconfig

    assert "skop_describe" in skop.host.init_script(tmp_path, metadata=True)
    installed = skop.host.init_script(
        sysconfig.get_paths()["purelib"], metadata=True
    )
    assert "sys.path.append(" in installed


def test_constants_still_carry_the_original_scripts():
    """Additive: a host that knows only these two behaves as it did."""
    constants = skop.host.constants()
    assert constants["init"] == skop.host.INIT
    assert constants["metadata_init"] == skop.host.METADATA_INIT
    assert "init_append" in constants
    assert "metadata_init_append" in constants
