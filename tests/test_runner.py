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


def test_on_start_hands_back_a_cancellable_task(runner):
    import threading

    # run() blocks, so a GUI's Cancel button only has something to press if
    # it gets the task from another thread. slow_sum polls cancel_requested()
    # and returns what it summed before being stopped.
    def cancel_soon(task):
        threading.Timer(0.3, task.cancel).start()

    total = runner.run(
        toy.slow_sum,
        image=np.ones((10, 10), dtype=np.float32),
        steps=40,
        on_start=cancel_soon,
    )
    # A canceled op still delivers its outputs -- they are simply partial.
    assert 0.0 < total < 100.0


def test_build_subscribers_are_handed_to_the_builder(monkeypatch):
    # Building is the slow part of a first run, and it happens inside run(),
    # so anything with a progress bar needs these to reach Appose. Checked
    # against a fake builder rather than by building for real.
    import appose

    recorded: dict[str, list] = {"progress": [], "output": [], "error": []}

    class FakeBuilder:
        def name(self, _env_name):
            return self

        def subscribe_progress(self, sub):
            recorded["progress"].append(sub)
            return self

        def subscribe_output(self, sub):
            recorded["output"].append(sub)
            return self

        def subscribe_error(self, sub):
            recorded["error"].append(sub)
            return self

        def build(self):
            return "an environment"

    monkeypatch.setattr(appose, "pixi", lambda _config: FakeBuilder())

    r = skop.Runner()
    r.subscribe_build_progress(lambda title, current, maximum: None)
    r.subscribe_build_output(lambda text: None)
    r.subscribe_build_error(lambda text: None)

    assert r.environment("minimal") == "an environment"
    assert len(recorded["progress"]) == 1
    assert len(recorded["output"]) == 1
    assert len(recorded["error"]) == 1


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


def test_a_2d_op_iterates_over_a_stack_in_the_worker(runner):
    # Naming the axes is what unlocks this: quadrants is strictly 2-D, and
    # the whole stack crosses the boundary once, not once per plane.
    stack = np.zeros((3, 8, 6), dtype=np.float32)
    labels = runner.run(toy.quadrants, image=stack, axes={"image": list("zyx")})

    assert labels.shape == (3, 8, 6)
    assert labels.max() == 12  # Four quadrants per plane, renumbered.


def test_iteration_reports_progress_per_slice(runner):
    seen = []
    runner.run(
        toy.quadrants,
        image=np.zeros((4, 8, 6), dtype=np.float32),
        axes={"image": list("zyx")},
        on_progress=lambda event: seen.append(event),
    )
    assert any("Slice 4 of 4" in str(getattr(e, "message", "")) for e in seen)


def test_an_unusable_input_is_refused_before_dispatch(runner):
    # Planning happens on the host, so a hopeless call never reaches a worker.
    with pytest.raises(ValueError, match="no y axis"):
        runner.run(toy.quadrants, image=np.zeros((3, 6)), axes={"image": list("zx")})


def test_a_chosen_plan_is_obeyed(runner):
    stack = np.arange(3 * 8 * 6, dtype=np.float32).reshape(3, 8, 6)
    plan = skop.plans(toy.quadrants, "image", stack, list("zyx"), position={"z": 2})[-1]
    assert not plan.lossless

    labels = runner.run(toy.quadrants, image=stack, plans={"image": plan})
    assert labels.shape == (8, 6)


def test_unadapted_arrays_are_untouched(runner):
    # Saying nothing about axes means the op sees exactly what was passed.
    image = np.ones((2, 3, 4), dtype=np.float32)
    assert runner.run(toy.scale, image=image, factor=2.0).scaled.shape == (2, 3, 4)
