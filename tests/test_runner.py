"""End-to-end: ops running in a real Appose worker, in a real environment.

The first run builds the 'minimal' environment, which takes a while. Later
runs reuse it, since Appose keys environments by name.
"""

from __future__ import annotations

import numpy as np
import pytest

import skop
from skop.ops import toy


@pytest.fixture(scope="module")
def runner():
    with skop.Runner() as r:
        yield r


def test_scalar_round_trip(runner):
    assert runner.run(toy.add, a=17, b=25) == 42


def test_op_runs_in_its_own_environment(runner):
    # The point of the whole exercise: the op executes on the environment's
    # interpreter, not the host's, so their dependency sets are independent.
    import sys

    worker = runner.run(toy.probe)
    assert sys.executable not in worker
    assert "skop-minimal" in worker


def test_array_round_trip_and_multiple_outputs(runner):
    image = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    result = runner.run(toy.scale, image=image, factor=3.0)

    assert isinstance(result, toy.ScaleResult)
    np.testing.assert_allclose(result.scaled, image * 3.0)
    assert result.total == pytest.approx(float((image * 3.0).sum()))
    # The result is a real local array, not a view of released shared memory.
    assert result.scaled.base is None


def test_ops_sharing_an_env_share_a_worker(runner):
    runner.run(toy.add, a=1, b=1)
    runner.run(toy.scale, image=np.zeros((2, 2), dtype=np.float32))
    assert len(runner._services) == 1


def test_computer_form_fills_callers_buffer(runner):
    image = np.arange(12, dtype=np.float32).reshape(3, 4)
    out = np.zeros_like(image)

    returned = runner.run(toy.scale_into, image=image, result=out, factor=5.0)

    np.testing.assert_allclose(out, image * 5.0)
    assert returned is out


def test_progress_events_reach_the_host(runner):
    messages = []
    total = runner.run(
        toy.slow_sum,
        image=np.ones((10, 10), dtype=np.float32),
        steps=4,
        on_progress=lambda event: messages.append(event.message),
    )
    assert total == pytest.approx(100.0)
    assert any(m and "Summing chunk" in m for m in messages)


def test_unknown_argument_is_rejected_before_dispatch(runner):
    with pytest.raises(TypeError, match="has no parameter"):
        runner.run(toy.add, a=1, b=2, c=3)


def test_missing_argument_is_rejected_before_dispatch(runner):
    with pytest.raises(TypeError, match="missing required argument"):
        runner.run(toy.add, a=1)


def test_op_failure_surfaces_as_an_exception(runner):
    from appose import TaskException

    with pytest.raises(TaskException):
        runner.run(toy.scale, image="not an array")
