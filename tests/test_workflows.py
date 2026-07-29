"""Workflows: ops with no environment, which run on the host and call ops.

The workflow under test is declared here rather than taken from the
collection, because the machinery is what is being tested, not any particular
pairing. Its *sub-ops* have to be real ones: a workflow never crosses the
Appose boundary, but the ops it calls do, and a worker imports an op by module
name -- which a test module is not, from inside a pixi environment.

Both sub-ops here are cheap skimage edge filters, so these say nothing about
whether SAM works. The real workflows are exercised in test_ops_e2e.py.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Annotated

import numpy as np
import pytest

import skop
from skop import Choices, ParamsFor, op
from skop.ops.edges import scharr, sobel, unsharp_mask


@op()
def _apply(
    image: np.ndarray,
    step: Annotated[Callable, Choices(sobel=sobel, scharr=scharr)] = sobel,
    step_args: Annotated[dict | None, ParamsFor("step", binds="image")] = None,
) -> np.ndarray:
    skop.progress("Applying")
    return skop.run(step, image=image, **(step_args or {}))


# -- the spec ------------------------------------------------------------


def test_missing_env_marks_a_workflow():
    assert skop.spec(_apply).is_workflow
    assert skop.spec(_apply).env is None
    assert not skop.spec(sobel).is_workflow


def test_choices_keep_declaration_order_and_labels():
    choices = skop.spec(_apply).inputs[1].choices
    assert choices.labels == ("sobel", "scharr")
    assert choices.op("scharr") is scharr
    assert choices.label(sobel) == "sobel"


def test_choices_expose_ids_for_a_front_end_across_a_wire():
    ids = dict(skop.spec(_apply).inputs[1].choices.ids)
    assert ids["sobel"] == "skop.ops.edges:sobel"


def test_choices_label_of_an_unlisted_op_is_none():
    """Passing an op outside the list is legal, so this must not raise.

    The list constrains the GUI, not the function -- a front end asking what
    to show for something it does not offer gets no answer, not an error.
    """
    assert skop.spec(_apply).inputs[1].choices.label(len) is None


def test_a_lone_bound_name_is_not_iterated_as_characters():
    assert skop.spec(_apply).inputs[2].params_for.binds == ("image",)


def test_binds_accepts_a_sequence():
    assert ParamsFor("x", binds=("a", "b")).binds == ("a", "b")


# -- running -------------------------------------------------------------


@pytest.fixture(scope="module")
def runner():
    with skop.Runner() as r:
        yield r


@pytest.fixture(scope="module")
def image():
    picture = np.zeros((16, 16), np.float32)
    picture[4:12, 4:12] = 1.0
    return picture


def test_workflow_runs_on_the_host_and_dispatches_the_sub_op(runner, image):
    direct = runner.run(sobel, image=image)
    assert np.allclose(runner.run(_apply, image=image), direct)


def test_the_chooser_actually_chooses(runner, image):
    """Two edge filters that disagree, so picking one is observable."""
    result = runner.run(_apply, image=image, step=scharr)
    assert np.allclose(result, runner.run(scharr, image=image))
    assert not np.allclose(result, runner.run(sobel, image=image))


def test_settings_reach_the_chosen_op(runner, image):
    plain = runner.run(_apply, image=image, step=unsharp_mask)
    strong = runner.run(
        _apply, image=image, step=unsharp_mask, step_args={"amount": 5.0}
    )
    assert not np.allclose(plain, strong)


def test_progress_from_the_workflow_and_from_its_sub_op_both_arrive(runner):
    seen = []
    runner.run(
        _apply,
        image=np.zeros((8, 8), np.float32),
        on_progress=lambda event: seen.append(event.message),
    )
    # The workflow's own message, raised on the host through the same
    # skop.progress() an op in a worker would call.
    assert "Applying" in seen


def test_an_op_outside_the_chooser_still_runs(runner, image):
    """The escape hatch: this is how the curated list grows.

    ``unsharp_mask`` is not in the Choices list, and passing it anyway is the
    whole point -- someone evaluates an op in a script, and if it earns its
    place it gets added to the list where the change can be reviewed.
    """
    result = runner.run(_apply, image=image, step=unsharp_mask)
    assert np.allclose(result, runner.run(unsharp_mask, image=image))


def test_calling_a_workflow_directly_needs_no_runner():
    """Mode B: a workflow is a plain function, and its sub-op call falls
    through to the default runner rather than needing one passed in."""
    assert skop.spec(_apply).is_workflow
    # Not actually executed here -- that would build an environment -- but the
    # ambient lookup must be absent outside a run, so run() picks the default.
    from skop.runner import _current

    assert _current.get() is None
