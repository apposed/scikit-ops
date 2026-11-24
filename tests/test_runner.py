"""End-to-end: ops running in a real Appose worker, in a real environment.

The first run builds the 'minimal' environment, which takes a while. Later
runs reuse it, since Appose keys environments by name.
"""

from __future__ import annotations

import numpy as np
import pytest

import oprunner
import opspec.plan
from skop.ops import toy


@pytest.fixture(scope="module")
def runner():
    with oprunner.Runner() as r:
        yield r


def test_scalar_round_trip(runner):
    assert runner.run(toy.add, a=17, b=25) == 42


def test_op_runs_in_its_own_environment(runner):
    # The point of the whole exercise: the op executes on the environment's
    # interpreter, not the host's, so their dependency sets are independent.
    import sys

    worker = runner.run(toy.probe)
    assert sys.executable not in worker
    assert "scikit-ops--minimal" in worker


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

    r = oprunner.Runner()
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
    # Too few axes is the only thing left that cannot be adapted: a name that
    # does not match is a warning now, not a refusal.
    with pytest.raises(ValueError, match="consumes 2 axes but was given 1"):
        runner.run(toy.quadrants, image=np.zeros((6,)), axes={"image": ["x"]})


def test_a_mismatched_axis_name_runs_anyway_and_says_so(runner):
    # The op wanted y and x; it is being given z and x. That is the user's
    # call to make, so it runs -- with the misalignment recorded on the plan.
    plan = opspec.plan.plan(toy.quadrants, "image", np.zeros((3, 6)), list("zx"))
    assert plan.warnings == ("y is being fed the z axis",)

    labels = runner.run(toy.quadrants, image=np.zeros((3, 6)), plans={"image": plan})
    assert labels.shape == (3, 6)


def test_a_chosen_plan_is_obeyed(runner):
    stack = np.arange(3 * 8 * 6, dtype=np.float32).reshape(3, 8, 6)
    plan = opspec.plan.plan(
        toy.quadrants,
        "image",
        stack,
        list("zyx"),
        position={"z": 2},
        dispositions={0: opspec.plan.SELECT},
    )
    assert not plan.uses_all_data

    labels = runner.run(toy.quadrants, image=stack, plans={"image": plan})
    assert labels.shape == (8, 6)


def test_unadapted_arrays_are_untouched(runner):
    # Saying nothing about axes means the op sees exactly what was passed.
    image = np.ones((2, 3, 4), dtype=np.float32)
    assert runner.run(toy.scale, image=image, factor=2.0).scaled.shape == (2, 3, 4)


# -- ops outside scikit-ops ------------------------------------------------

WILDCAT = """
import numpy as np
from opspec.op import op

@op(env="skimage")
def plus_one(image: np.ndarray) -> np.ndarray:
    return image + 1
"""


def _import_from(folder, name, source):
    import importlib
    import sys

    (folder / f"{name}.py").write_text(source)
    sys.path.insert(0, str(folder))
    try:
        return importlib.import_module(name)
    finally:
        sys.path.remove(str(folder))


def test_a_module_s_origin_is_the_folder_it_imports_from(tmp_path):
    from oprunner.runner import _origin

    import skop.ops.smooth  # noqa: F401  (imported, so it has an origin)

    package = tmp_path / "pkg"
    package.mkdir()
    (package / "__init__.py").write_text("")
    _import_from(package, "mod_in_pkg_test", "")  # a plain module
    assert _origin("mod_in_pkg_test") == {"root": str(package.resolve())}
    assert _origin("skop.ops.smooth")["root"].endswith("src")
    assert _origin("no_such_module") is None


def test_an_environment_beside_the_op_comes_first(tmp_path):
    (tmp_path / "envs" / "skimage").mkdir(parents=True)
    (tmp_path / "envs" / "skimage" / "pixi.toml").write_text("[workspace]\n")
    _import_from(tmp_path, "op_beside_envs_test", WILDCAT)
    with oprunner.Runner() as runner:
        mine = runner._env_for("skimage", "op_beside_envs_test")
        ours = runner._env_for("skimage", "skop.ops.smooth")
    assert mine.config == tmp_path.resolve() / "envs" / "skimage" / "pixi.toml"
    # A plain folder: its name, a hash of where it is, and the ID.
    assert mine.name.startswith(tmp_path.name + "-")
    assert mine.name.endswith("--skimage")
    assert mine.folder.name == mine.name
    assert ours.name == "scikit-ops--skimage"


def test_an_environment_is_named_after_its_package(tmp_path):
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "My_Ops"\n')
    (tmp_path / "envs" / "skimage").mkdir(parents=True)
    (tmp_path / "envs" / "skimage" / "pixi.toml").write_text("[workspace]\n")
    _import_from(tmp_path, "op_in_a_package_test", WILDCAT)
    with oprunner.Runner() as runner:
        env = runner._env_for("skimage", "op_in_a_package_test")
    assert env.name == "my-ops--skimage"


def test_a_runner_says_which_environment_an_op_runs_in():
    from skop.ops.smooth import gaussian

    with oprunner.Runner() as runner:
        env = runner.env_for(gaussian)
    assert (env.id, env.name) == ("skimage", "scikit-ops--skimage")
    assert env.config.name == "pixi.toml"


@pytest.mark.env("skimage")
def test_an_op_outside_scikit_ops_runs_in_an_environment(tmp_path):
    module = _import_from(tmp_path, "wildcat_op_test", WILDCAT)
    with oprunner.Runner() as runner:
        result = runner.run(module.plus_one, image=np.zeros(3))
    assert list(result) == [1, 1, 1]


# -- opspec.runtime: progress, cancel and run, through this runner ------------

FROM_OPSPEC = """
import numpy as np
from opspec.op import op
from opspec.runtime import cancel_requested, progress, run

@op(env="skimage")
def counts(image: np.ndarray) -> np.ndarray:
    for i in range(3):
        progress(f"Step {i + 1} of 3", i, 3)
    return image + int(cancel_requested())

@op
def two_steps(image: np.ndarray) -> np.ndarray:
    progress("Calling counts")
    return run(counts, image=image)

@op
def announce(image: np.ndarray) -> np.ndarray:
    progress("On the host")
    return image
"""


def _messages(events):
    return [getattr(e, "message", None) for e in events if getattr(e, "message", None)]


def test_an_op_on_the_host_reports_through_opspec_runtime(tmp_path):
    module = _import_from(tmp_path, "opspec_runtime_host_test", FROM_OPSPEC)
    events = []
    with oprunner.Runner() as runner:
        runner.run(module.announce, image=np.zeros(2), on_progress=events.append)
    assert _messages(events) == ["On the host"]


@pytest.mark.env("skimage")
def test_an_op_in_a_worker_reports_through_opspec_runtime(tmp_path):
    module = _import_from(tmp_path, "opspec_runtime_worker_test", FROM_OPSPEC)
    events = []
    with oprunner.Runner() as runner:
        result = runner.run(module.counts, image=np.zeros(2), on_progress=events.append)
    assert list(result) == [0, 0]
    assert _messages(events)[-3:] == ["Step 1 of 3", "Step 2 of 3", "Step 3 of 3"]


@pytest.mark.env("skimage")
def test_a_workflow_s_run_goes_through_the_runner(tmp_path):
    module = _import_from(tmp_path, "opspec_runtime_workflow_test", FROM_OPSPEC)
    events, started = [], []
    with oprunner.Runner() as runner:
        runner.run(
            module.two_steps,
            image=np.zeros(2),
            on_progress=events.append,
            on_start=started.append,
        )
        # counts ran in its environment, so the runner started a worker.
        assert any(key[0] == "skimage" for key in runner._services)
    assert "Step 3 of 3" in _messages(events)


class _ScriptedBuilder:
    """A stand-in for appose's builder that 'prints' and then succeeds or
    fails, as pixi would."""

    def __init__(self, out: str, err: str, fail: bool):
        self.out, self.err, self.fail = out, err, fail
        self.outs, self.errs = [], []

    def name(self, _env_name):
        return self

    def subscribe_progress(self, _sub):
        return self

    def subscribe_output(self, sub):
        self.outs.append(sub)
        return self

    def subscribe_error(self, sub):
        self.errs.append(sub)
        return self

    def build(self):
        for sub in self.outs:
            sub(self.out)
        for sub in self.errs:
            sub(self.err)
        if self.fail:
            raise RuntimeError("pixi build failed")
        return "an environment"


@pytest.fixture
def scratch_envs(tmp_path, monkeypatch):
    """Builds land in an empty appose directory, and pixi is left alone."""
    import oprunner._pixi

    monkeypatch.setenv("APPOSE_ENVS_DIR", str(tmp_path))
    monkeypatch.setattr(oprunner._pixi, "ensure_recent_pixi", lambda report: None)
    return tmp_path


def test_a_build_writes_its_log(scratch_envs, monkeypatch):
    import appose

    builder = _ScriptedBuilder("Solving\n", "✔ The default environment has been installed.\n", False)
    monkeypatch.setattr(appose, "pixi", lambda _config: builder)
    r = oprunner.Runner()
    env = r._own_env("minimal")
    assert env.status == "missing"

    assert r.build(env) == "an environment"
    text = env.log.read_text(encoding="utf-8")
    assert env.log.parent == scratch_envs
    assert "Solving" in text and "has been installed" in text
    assert text.rstrip().endswith("# OK")


def test_a_failed_build_says_what_pixi_said(scratch_envs, monkeypatch):
    import appose

    from oprunner.runner import BuildFailed

    err = (
        "DEBUG resolving 1\nDEBUG resolving 2\n"
        "Error:   × failed to solve the pypi requirements of environment 'default'\n"
        "  ╰─▶ no version of scikit-ops==9.9\n"
    )
    monkeypatch.setattr(appose, "pixi", lambda _config: _ScriptedBuilder("", err, True))
    r = oprunner.Runner()
    env = r._own_env("minimal")

    with pytest.raises(BuildFailed) as caught:
        r.build(env)
    message = str(caught.value)
    assert "failed to solve" in message and "no version of scikit-ops" in message
    assert "DEBUG resolving 1" not in message  # the error, not the debug log
    assert str(env.log) in message
    assert "# FAILED" in env.log.read_text(encoding="utf-8")


def test_collection_envs_lists_skops_recipes_as_its_ops_name_them():
    r = oprunner.Runner()
    envs = {e.id: e for e in r.collection_envs("skop")}
    assert {"minimal", "pytorch", "stardist-tf"} <= envs.keys()
    from skop.ops.segment import cellpose4

    assert envs["pytorch"] == r.env_for(cellpose4)
    assert r.collection_envs("no_such_package_here") == []
