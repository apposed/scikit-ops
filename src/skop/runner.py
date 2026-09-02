"""The host-side half of skop: build environments and run ops in them.

The host never imports an op's dependencies. It imports the op module only to
read its signature -- which is cheap, because ops keep heavy imports inside
their function bodies -- and dispatches the actual call to a worker process
living in the op's declared environment.
"""

from __future__ import annotations

import json
import contextlib
import contextvars
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import appose
import numpy as np

from . import _adapt, _codec, _progress, _spec
from .host import CALL as _CALL
from .host import INIT as _INIT  # noqa: F401  (kept for out-of-process hosts)
from .host import init_script as _init_script_for

if TYPE_CHECKING:
    from typing import Self

# Note: _CALL and _INIT live in skop.host rather than here, because an
# out-of-process host needs them too and there must be exactly one copy of
# them. See docs/spec/fiji-front-end.md.


@dataclass(frozen=True)
class _Event:
    """The shape of an Appose TaskEvent, for progress raised on the host."""

    message: str | None = None
    current: int | None = None
    maximum: int | None = None


class _HostTask:
    """Stands in for an Appose task while a workflow runs on the host.

    A workflow has no worker and therefore no task, but everything watching an
    op expects one: ``skop.progress()`` looks for something to update, and a
    GUI's Cancel button needs a handle to act on. This is that handle.
    """

    def __init__(self, on_progress: Callable[[Any], None] | None) -> None:
        self._on_progress = on_progress
        self.cancel_requested = False
        self._child: Any = None

    def update(
        self,
        message: str | None = None,
        current: int | None = None,
        maximum: int | None = None,
    ) -> None:
        self.relay(_Event(message, current, maximum))

    def relay(self, event: Any) -> None:
        """Pass an event on, whether it came from here or from a sub-op."""
        if self._on_progress is not None:
            self._on_progress(event)

    def adopt(self, task: Any) -> None:
        """Note the sub-op task now running, so a cancel can reach it."""
        self._child = task
        if self.cancel_requested:
            task.cancel()

    def cancel(self) -> None:
        self.cancel_requested = True
        # Nearly all of a workflow's time is spent inside a sub-op, so a
        # cancel aimed at the workflow is useless unless it reaches that.
        if self._child is not None:
            self._child.cancel()


class _Ambient:
    """The runner a workflow's own ``skop.run`` calls should go through.

    A workflow does not take a runner as a parameter, for the same reason an
    op does not take a progress reporter as one (design 0001): it would make
    calling the function directly awkward, and mode B has to stay a plain
    Python call. So the runner is ambient, exactly as ``progress()`` is.
    """

    def __init__(self, runner: Runner, task: _HostTask) -> None:
        self._runner = runner
        self._task = task

    def run(self, fn: Callable, args: dict | None = None, **kwargs: Any) -> Any:
        # A sub-op's progress is the workflow's progress: without this the
        # panel sits at "running" for the four minutes SAM takes.
        kwargs.setdefault("on_progress", self._task.relay)
        kwargs.setdefault("on_start", self._task.adopt)
        return self._runner.run(fn, args, **kwargs)


_current: contextvars.ContextVar[_Ambient | None] = contextvars.ContextVar(
    "skop_runner", default=None
)


def _default_root() -> Path:
    """The directory holding the ``skop`` package."""
    return Path(__file__).resolve().parent.parent


def _default_envs_dir(root: Path) -> Path:
    """Where the ``envs/<env-id>/pixi.toml`` recipes live.

    Two places, because there are two ways to have skop. A checkout keeps them
    beside ``src``, where they are edited. An install has no tree, so they ride
    inside the package, put there by ``force-include`` in pyproject.toml.

    The checkout wins when both exist -- the same rule ``host.INIT`` applies to
    the code, so an edited recipe takes effect without a reinstall. Falls back
    to the packaged path, so a failure names where an install actually looked.
    """
    checkout = root.parent / "envs"
    if checkout.is_dir():
        return checkout
    return Path(__file__).resolve().parent / "envs"


class Runner:
    """Builds environments, keeps workers warm, and dispatches op calls."""

    def __init__(
        self,
        root: Path | str | None = None,
        envs_dir: Path | str | None = None,
        debug: bool = False,
    ) -> None:
        self.root = Path(root).resolve() if root else _default_root()
        self.envs_dir = (
            Path(envs_dir).resolve() if envs_dir else _default_envs_dir(self.root)
        )
        self.debug = debug
        self._services: dict[tuple, Any] = {}
        self._envs: dict[tuple, Any] = {}
        self._build_progress: list[Callable[[str, int, int], None]] = []
        self._build_output: list[Callable[[str], None]] = []
        self._build_error: list[Callable[[str], None]] = []

    # -- build reporting -------------------------------------------------

    def subscribe_build_progress(
        self, subscriber: Callable[[str, int, int], None]
    ) -> None:
        """Hear about environment build progress, as (title, current, maximum).

        Building an environment is by far the slowest thing a first run does
        -- minutes, for a TensorFlow or PyTorch stack -- and it happens inside
        ``run``, with nothing else to show for it. Anything with a progress
        bar wants these.

        Titles come from Appose's PixiInstallMonitor, and name the phase:
        "Solving", "Installing conda packages", "Downloading PyPI packages",
        "Installing PyPI packages", "Done". Note that subscribing here is
        what switches the monitor on -- it runs pixi under ``-vv`` to read
        its phase transitions, so without a progress subscriber a build
        reports only its final summary line.
        """
        self._build_progress.append(subscriber)

    def subscribe_build_output(self, subscriber: Callable[[str], None]) -> None:
        """Hear the build tool's standard output, in raw chunks."""
        self._build_output.append(subscriber)

    def subscribe_build_error(self, subscriber: Callable[[str], None]) -> None:
        """Hear the build tool's standard error, in raw chunks.

        Note: this is the stderr *stream*, not a failure report. Pixi writes
        its ordinary status there, success message included, and once a
        progress subscriber has turned on ``-vv`` it writes its whole debug
        log there too. A build that actually fails raises from ``run``.
        """
        self._build_error.append(subscriber)

    # -- environments ----------------------------------------------------

    def env_config(self, env_id: str) -> Path:
        config = self.envs_dir / env_id / "pixi.toml"
        if not config.exists():
            known = ", ".join(sorted(self.env_ids())) or "none"
            raise FileNotFoundError(
                f"No environment '{env_id}' at {config}. Known environments: {known}"
            )
        return config

    def env_ids(self) -> Iterator[str]:
        if not self.envs_dir.is_dir():
            return
        for child in sorted(self.envs_dir.iterdir()):
            if (child / "pixi.toml").exists():
                yield child.name

    def env_dir(self, env_id: str) -> Path:
        """Where Appose keeps the built environment for *env_id*.

        Matching ``environment``'s ``appose.pixi(config).name(f"skop-{env_id}")``.
        The directory need not exist; that is what "missing" means below.
        """
        try:
            from appose.util.filepath import appose_envs_dir
        except ImportError:  # older appose kept it next door
            from appose.util.environment import appose_envs_dir

        return Path(appose_envs_dir()) / f"skop-{env_id}"

    def environment_status(self, env_id: str) -> str:
        """Whether the built environment matches its definition.

        One of ``"missing"``, ``"stale"`` or ``"up to date"``. Answered from
        the two files, without building anything, so it costs nothing to ask.

        Appose records the full text of the pixi.toml it built from, in
        ``appose.json`` beside the environment. Comparing that against the
        definition in ``envs/<env_id>/pixi.toml`` says whether the environment
        anyone is about to run in is the one currently described.

        Worth having because the failure it catches is silent. An environment
        built before a fix stays built, keeps running, and reports nothing --
        which is how a stack can be missing CUDA for months while every run
        appears to succeed.
        """
        config = self.env_config(env_id)
        record = self.env_dir(env_id) / "appose.json"
        if not record.exists():
            return "missing"
        try:
            built = json.loads(record.read_text()).get("content", "")
        except (OSError, ValueError):
            return "stale"
        return (
            "up to date"
            if built.strip() == config.read_text().strip()
            else "stale"
        )

    def ensure_environment(
        self,
        env_id: str,
        variant: str | None = None,
        report: Callable[[str], None] = print,
    ) -> Any:
        """Build *env_id* if it is missing or out of date, saying which.

        ``environment`` builds or reuses and tells you neither, so a cell that
        sits silent for ten minutes looks the same as one that did nothing.
        This says what it is about to do first, and subscribes to the build
        output for the duration if nothing else has, so a rebuild is visible
        as it happens rather than only in the wall clock.

        Args:
            env_id: The environment to ensure, e.g. ``"stardist-tf"``.
            variant: A named pixi sub-environment, or None for the default.
            report: Where the status lines go. ``print`` suits a notebook.

        Returns:
            The Appose environment, as ``environment`` does.
        """
        status = self.environment_status(env_id)
        report(f"{env_id}: {status}")
        if status == "up to date":
            return self.environment(env_id, variant)

        report(
            f"{env_id}: building from {self.env_config(env_id)} -- "
            f"this takes minutes and gigabytes on a first build"
        )
        added = []
        if not self._build_output:
            self._build_output.append(report)
            added.append(self._build_output)
        if not self._build_error:
            self._build_error.append(report)
            added.append(self._build_error)
        try:
            env = self.environment(env_id, variant)
        finally:
            for subscribers in added:
                subscribers.pop()
        report(f"{env_id}: ready")
        return env

    def environment(self, env_id: str, variant: str | None = None) -> Any:
        """Build (or reuse) the Appose environment for an env ID.

        Appose keys environments by name in its shared environments
        directory, so several ops declaring the same env ID land on the same
        installation, and editing the pixi.toml triggers a rebuild.
        """
        key = (env_id, variant)
        if key in self._envs:
            return self._envs[key]

        config = self.env_config(env_id)
        builder = appose.pixi(config).name(f"skop-{env_id}")
        for progress_subscriber in self._build_progress:
            builder = builder.subscribe_progress(progress_subscriber)
        for output_subscriber in self._build_output:
            builder = builder.subscribe_output(output_subscriber)
        for error_subscriber in self._build_error:
            builder = builder.subscribe_error(error_subscriber)
        if self.debug:
            builder = builder.log_debug()
        env = builder.build()

        if variant is not None:
            # Named pixi sub-environments (features) require appose >= 0.12.
            if not hasattr(env, "activate"):
                raise RuntimeError(
                    f"Environment variant '{variant}' requires appose >= 0.12; "
                    f"installed appose has no Environment.activate()."
                )
            env = env.activate(variant)

        self._envs[key] = env
        return env

    def _init_script(self, env_id: str) -> str:
        script = _init_script_for(self.root)
        extra = self.envs_dir / env_id / "init.py"
        if extra.exists():
            script = f"{script}\n{extra.read_text(encoding='utf-8')}"
        return script

    def service(self, spec: _spec.OpSpec, variant: str | None = None) -> Any:
        """Get the worker serving this op, starting one if needed.

        Ops sharing an environment share a worker, unless one of them asks
        to be exclusive.
        """
        key = (spec.env, variant, spec.name if spec.exclusive else None)
        service = self._services.get(key)
        if service is not None:
            return service

        env = self.environment(spec.env, variant)
        service = env.python().init(self._init_script(spec.env))
        if self.debug:
            service.debug(lambda line: print(f"[{spec.env}] {line}"))
        service.start()
        self._services[key] = service
        return service

    # -- running ---------------------------------------------------------

    def run(
        self,
        fn: Callable,
        args: dict | None = None,
        *,
        variant: str | None = None,
        axes: dict[str, str] | None = None,
        plans: dict[str, _adapt.AdaptationPlan] | None = None,
        position: dict[str, int] | None = None,
        on_progress: Callable[[Any], None] | None = None,
        on_start: Callable[[Any], None] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Run an op in its environment and return its result.

        Args:
            fn: The op function, as decorated with ``@op``.
            args: Op arguments, as a dict. Merged with ``**kwargs``, which is
                the more convenient form when no argument name collides with
                this method's own parameters.
            variant: Optional named pixi sub-environment (e.g. ``"cuda"``).
            axes: What each array argument actually is, as ``{"image":
                list("zyx")}`` or ``{"image": ("pln", "row", "col")}``. Naming
                them lets skop fit them to what the op consumes -- transposing,
                or iterating a 2-D op over a stack. An axis label is any
                string, so ``("lifetime", "y", "x")`` works as well as the
                canonical letters, and known synonyms resolve to them.
                Unnamed arrays are passed through untouched.
            plans: Explicit ``AdaptationPlan``s, from ``skop.plan``, for
                callers making the per-axis decisions themselves. Overrides
                ``axes`` for the parameters it names. The plan skop builds on
                its own never discards data, so supply one whenever that is
                what you actually want.
            position: Current position along each axis, used only when a
                supplied plan indexes down to a single plane.
            on_progress: Called with each Appose TaskEvent as it arrives.
            on_start: Called with the Appose Task once it has been submitted.
                This call blocks until the op finishes, so a caller wanting to
                cancel one -- a GUI, typically -- needs a handle on it from
                another thread. Waiting for the first progress event instead
                would leave silent ops uncancellable.
        """
        spec = _spec.spec(fn)
        call_args = dict(args or {})
        call_args.update(kwargs)
        _validate(spec, call_args)

        if spec.is_workflow:
            # No environment to dispatch to, and nothing to encode: a workflow
            # runs here, and the ops it calls each cross the boundary
            # themselves. Axis adaptation is skipped for the same reason --
            # the sub-ops adapt their own arrays.
            return self._run_here(fn, call_args, on_progress, on_start)

        adaptations = _adaptations(fn, call_args, axes, plans, position)

        service = self.service(spec, variant)

        refs: list = []
        # Buffers the caller owns and expects to see written to: for computer
        # and inplace ops we must copy results back out of shared memory.
        buffers: dict[str, tuple[np.ndarray, Any]] = {}
        try:
            encoded = {}
            by_name = {p.name: p for p in spec.params}
            for name, value in call_args.items():
                param = by_name[name]
                if param.direction is not None and isinstance(value, np.ndarray):
                    nda = _codec._to_ndarray(value, refs)
                    buffers[name] = (value, nda)
                    encoded[name] = nda
                else:
                    encoded[name] = _codec.encode(value, refs)

            task = service.task(
                _CALL,
                {
                    "module": spec.module,
                    "function": spec.function,
                    "kwargs": encoded,
                    "plans": [plan.to_dict() for plan in adaptations.values()],
                },
                queue="main" if spec.main_thread else None,
            )
            if on_progress is not None:
                task.listen(on_progress)
            if on_start is not None:
                on_start(task)
            task.wait_for()

            for caller_array, nda in buffers.values():
                caller_array[...] = nda.ndarray()

            out_refs: list = []
            try:
                outputs = _codec.decode(dict(task.outputs), out_refs)
                outputs = _codec.copy_out(outputs)
            finally:
                # The host owns cleanup of shared memory, including blocks
                # the worker allocated for results.
                _codec.release(out_refs, unlink=True)

            return _unpack(spec, outputs, buffers)
        finally:
            _codec.release(refs, unlink=True)

    def _run_here(
        self,
        fn: Callable,
        call_args: dict,
        on_progress: Callable[[Any], None] | None,
        on_start: Callable[[Any], None] | None,
    ) -> Any:
        """Call a workflow in this process, with this runner made ambient."""
        task = _HostTask(on_progress)
        if on_start is not None:
            on_start(task)
        reporting = _progress._bind(task)
        ambient = _current.set(_Ambient(self, task))
        try:
            return fn(**call_args)
        finally:
            _current.reset(ambient)
            _progress._unbind(reporting)

    def close(self) -> None:
        """Shut down every worker this runner started."""
        for service in self._services.values():
            # A worker that already died is still closed, as far as we care.
            with contextlib.suppress(Exception):
                service.close()
        self._services.clear()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc_value, exc_tb) -> None:
        self.close()


def _validate(spec: _spec.OpSpec, args: dict) -> None:
    known = {p.name for p in spec.params}
    unknown = set(args) - known
    if unknown:
        raise TypeError(
            f"Op {spec.name} has no parameter(s): {', '.join(sorted(unknown))}"
        )
    missing = [p.name for p in spec.params if p.required and p.name not in args]
    if missing:
        raise TypeError(
            f"Op {spec.name} is missing required argument(s): {', '.join(missing)}"
        )


def _adaptations(
    fn: Callable,
    args: dict,
    axes: dict[str, str] | None,
    plans: dict[str, _adapt.AdaptationPlan] | None,
    position: dict[str, int] | None,
) -> dict[str, _adapt.AdaptationPlan]:
    """Settle on one plan per parameter whose array was given axis labels.

    A parameter goes unadapted unless someone said something about it. That
    keeps naming axes opt-in: an existing caller passing a bare array gets
    exactly the behavior it got before.
    """
    chosen = dict(plans or {})
    for name, labels in (axes or {}).items():
        if name in chosen:
            continue
        chosen[name] = _adapt.plan(fn, name, args[name], labels, position)
    return chosen


def _unpack(spec: _spec.OpSpec, outputs: dict, buffers: dict) -> Any:
    names = spec.outputs
    if not names:
        return None
    if spec.form is not _spec.FUNCTION:
        # Results landed in the caller's own buffers.
        owned = [buffers[name][0] for name in names if name in buffers]
        if not owned:
            return None
        return owned[0] if len(owned) == 1 else tuple(owned)
    if names == ("result",):
        return outputs.get("result")
    values = [outputs.get(name) for name in names]
    factory = spec.return_type
    if getattr(factory, "_fields", None) == tuple(names):
        return factory(*values)
    return tuple(values)


_default_runner: Runner | None = None


def default_runner() -> Runner:
    """A process-wide Runner, for callers that do not want to manage one."""
    global _default_runner
    if _default_runner is None:
        _default_runner = Runner()
    return _default_runner


def run(fn: Callable, args: dict | None = None, **kwargs: Any) -> Any:
    """Run an op via the default Runner.

    Inside a workflow this goes through *that* runner instead, so a sub-op
    reuses the warm workers the workflow was started with -- and so a panel's
    progress bar and Cancel button reach it.
    """
    ambient = _current.get()
    if ambient is not None:
        return ambient.run(fn, args, **kwargs)
    return default_runner().run(fn, args, **kwargs)
