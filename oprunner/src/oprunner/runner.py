"""The host-side half of oprunner: build environments and run ops in them.

The host never imports an op's dependencies. It imports the op module only to
read its signature -- which is cheap, because ops keep heavy imports inside
their function bodies -- and dispatches the actual call to a worker process
living in the op's declared environment.
"""

from __future__ import annotations

import contextlib
import contextvars
import functools
import hashlib
import itertools
import json
import re
import sys
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from types import UnionType
from typing import TYPE_CHECKING, Any, Union, get_args, get_origin

import appose
import numpy as np

from opspec.op import FUNCTION, OpSpec, _strip
from opspec.runtime import bind_runner, bind_task, unbind_runner, unbind_task
from opspec.tiling import TilePlan
from opspec.types import Array

from . import _adapt, _codec, _tiling
from .host import CALL as _CALL
from .host import INIT as _INIT  # noqa: F401  (kept for out-of-process hosts)
from .host import init_script as _init_script_for
from .host import is_checkout

if TYPE_CHECKING:
    from typing import Self

# Note: _CALL and _INIT live in oprunner.host rather than here, because an
# out-of-process host needs them too and there must be exactly one copy of
# them. See docs/spec/fiji-front-end.md.


@dataclass(frozen=True)
class _Event:
    """The shape of an Appose TaskEvent, for progress raised on the host."""

    message: str | None = None
    current: int | None = None
    maximum: int | None = None
    #: (m, n) when a tiled run starts its tile m of n. The op's own progress
    #: for that tile follows, as ordinary events.
    tile: tuple[int, int] | None = None


class _HostTask:
    """Stands in for an Appose task while a workflow runs on the host.

    A workflow has no worker and therefore no task, but everything watching an
    op expects one: ``opspec.runtime.progress()`` looks for something to update, and a
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
    """The runner a workflow's own ``opspec.runtime.run`` calls should go through.

    A workflow does not take a runner as a parameter, for the same reason an
    op does not take a progress reporter as one (design 0001): it would make
    calling the function directly awkward, and mode B has to stay a plain
    Python call. So the runner is ambient, exactly as ``progress()`` is.
    """

    def __init__(
        self, runner: Runner, task: _HostTask, tiler: _tiling.Tiler | None = None
    ) -> None:
        self._runner = runner
        self._task = task
        self._tiler = tiler

    def run(self, fn: Callable, args: dict | None = None, **kwargs: Any) -> Any:
        # A sub-op's progress is the workflow's progress: without this the
        # panel sits at "running" for the four minutes SAM takes.
        kwargs.setdefault("on_progress", self._task.relay)
        kwargs.setdefault("on_start", self._task.adopt)
        # The tiling the workflow's caller chose applies to the ops it calls,
        # unless the workflow chose for one itself.
        if self._tiler is not None and not {"tiler", "memory"} & kwargs.keys():
            kwargs["tiler"] = self._tiler
        return self._runner.run(fn, args, **kwargs)


_current: contextvars.ContextVar[_Ambient | None] = contextvars.ContextVar(
    "oprunner", default=None
)


def _default_root() -> Path:
    """The directory holding the ``oprunner`` package: a checkout's
    ``oprunner/src``, or site-packages."""
    return Path(__file__).resolve().parent.parent


def _default_envs_dir(root: Path) -> Path | None:
    """Environments asked for by ID alone, with no op to find them beside.

    An op's environment is found beside the op's own files (``_env_for``), so
    this is only for calls such as ``ensure_environment("cupy")``. In a
    scikit-ops checkout it is the repository's ``envs/``: oprunner sits in
    ``oprunner/src``, two folders below. Elsewhere there is none.

    A stopgap: once collections say where their environments are, a runner
    needs no default.
    """
    checkout = root.parent.parent / "envs"
    return checkout if checkout.is_dir() else None


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
        if self.envs_dir is None:
            raise FileNotFoundError(
                f"No environment '{env_id}': this runner has no envs_dir. "
                "Pass Runner(envs_dir=...), or ask about an op with env_for(op)."
            )
        config = self.envs_dir / env_id / "pixi.toml"
        if not config.exists():
            known = ", ".join(sorted(self.env_ids())) or "none"
            raise FileNotFoundError(
                f"No environment '{env_id}' at {config}. Known environments: {known}"
            )
        return config

    def env_ids(self) -> Iterator[str]:
        if self.envs_dir is None or not self.envs_dir.is_dir():
            return
        for child in sorted(self.envs_dir.iterdir()):
            if (child / "pixi.toml").exists():
                yield child.name

    def env_dir(self, env_id: str) -> Path:
        """Where Appose keeps the built environment for *env_id*.

        ``<package>--<env_id>``, as ``environment`` builds it. The
        directory need not exist; that is what "missing" means below.
        """
        return self._own_env(env_id).folder

    def environment_status(self, env_id: str) -> str:
        """Whether the built environment matches its definition.

        One of ``"ready"``, ``"stale"`` or ``"missing"``. Answered from
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
        return "ready" if built.strip() == config.read_text().strip() else "stale"

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
        if status == "ready":
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
        return self._build(self._own_env(env_id), variant)

    def env_for(self, op: Callable) -> Env:
        """The environment *op* runs in: the file that defines it, and where
        it is built. Nothing is built by asking.
        """
        spec = OpSpec.from_op(op)
        if spec.env is None:
            raise ValueError(f"Op {spec.name} has no environment: it runs here")
        return self._env_for(spec.env, spec.module)

    def _env_for(self, env_id: str, module: str) -> Env:
        """The environment an op in *module* means by *env_id*.

        An ``envs/<env_id>/pixi.toml`` beside the op's own files comes first:
        in its folder, or any folder above it. So a collection of ops outside
        scikit-ops can bring its own environments, even ones named like
        scikit-ops' -- each is built, and run, as itself. Failing that, this
        runner's ``envs_dir``.
        """
        found = getattr(sys.modules.get(module), "__file__", None)
        if found is not None:
            for folder in Path(found).resolve().parents:
                config = folder / "envs" / env_id / "pixi.toml"
                if config.exists():
                    return Env(env_id, config, _package_name(module, folder))
        return self._own_env(env_id)

    def _own_env(self, env_id: str) -> Env:
        """One of this runner's own environments, from its ``envs_dir``."""
        config = self.env_config(env_id)
        return Env(env_id, config, _package_name(__name__, self.envs_dir.parent))

    def _build(self, env: Env, variant: str | None = None) -> Any:
        """Build (or reuse) one environment."""
        key = (env.config, variant)
        if key in self._envs:
            return self._envs[key]

        builder = appose.pixi(env.config).name(env.name)
        for progress_subscriber in self._build_progress:
            builder = builder.subscribe_progress(progress_subscriber)
        for output_subscriber in self._build_output:
            builder = builder.subscribe_output(output_subscriber)
        for error_subscriber in self._build_error:
            builder = builder.subscribe_error(error_subscriber)
        if self.debug:
            builder = builder.log_debug()
        built = builder.build()

        if variant is not None:
            # Named pixi sub-environments (features) require appose >= 0.12.
            if not hasattr(built, "activate"):
                raise RuntimeError(
                    f"Environment variant '{variant}' requires appose >= 0.12; "
                    f"installed appose has no Environment.activate()."
                )
            built = built.activate(variant)

        self._envs[key] = built
        return built

    def _init_script(self, env: Env) -> str:
        script = _init_script_for(self.root)
        extra = env.config.parent / "init.py"
        if extra.exists():
            script = f"{script}\n{extra.read_text(encoding='utf-8')}"
        return script

    def service(self, spec: OpSpec, variant: str | None = None) -> Any:
        """Get the worker serving this op, starting one if needed.

        Ops sharing an environment share a worker, unless one of them asks
        to be exclusive.
        """
        env = self._env_for(spec.env, spec.module)
        key = (spec.env, env.config, variant, spec.name if spec.exclusive else None)
        service = self._services.get(key)
        if service is not None:
            return service

        built = self._build(env, variant)
        service = built.python().init(self._init_script(env))
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
        memory: int | str | None = None,
        tiler: _tiling.Tiler | None = None,
        out: Any = None,
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
                them lets oprunner fit them to what the op consumes -- transposing,
                or iterating a 2-D op over a stack. An axis label is any
                string, so ``("lifetime", "y", "x")`` works as well as the
                canonical letters, and known synonyms resolve to them.
                Unnamed arrays are passed through untouched.
            plans: Explicit ``AdaptationPlan``s, from ``opspec.plan.plan``, for
                callers making the per-axis decisions themselves. Overrides
                ``axes`` for the parameters it names. The plan oprunner builds on
                its own never discards data, so supply one whenever that is
                what you actually want.
            position: Current position along each axis, used only when a
                supplied plan indexes down to a single plane.
            on_progress: Called with each Appose TaskEvent as it arrives.
            on_start: Called with the Appose Task when it is submitted. This
                can be used to get a reference to the running task, then the
                reference can be used to monitor or cancel the task.
            memory: A memory budget for the op, as bytes or a size such as
                ``"1G"``. An op declaring tiling hints (``@op(tile=...,
                peak_memory=...)``) whose input would need more is run tile by
                tile, each within the budget; see ``oprunner._tiling``. Left out,
                the budget is 85% of the memory available right now, cgroup
                limits included (``oprunner._tiling.default_budget``); ``"off"``
                never tiles. An op declaring no hints always runs whole. A
                shortcut for ``tiler=Tiler(memory)``.
            tiler: The caller's tiling choices -- budget, overlap, tile size
                -- as a ``oprunner.Tiler``; it takes precedence over ``memory``.
                ``Tiler.off()`` never tiles. For a workflow, it applies to
                the ops the workflow runs.
            out: Where an op's single array result is written, tiled or not:
                an array that takes ``out[region] = values`` (numpy, zarr,
                HDF5), or a function ``(shape, dtype) -> array`` that makes
                one once the result's dtype is known. Returned in place of a
                new array. Left out, a tiled result too big for memory goes
                to a temporary ``.npy`` memmap.
        """
        spec = OpSpec.from_op(fn)
        call_args = dict(args or {})
        call_args.update(kwargs)
        _validate(spec, call_args)

        if tiler is None:
            tiler = _tiling.Tiler.off() if memory == "off" else _tiling.Tiler(memory)
        plan = tiler.plan(spec, call_args)
        if plan is not None and plan.calls > 1:
            if axes or any(_adapts(p) for p in (plans or {}).values()):
                raise NotImplementedError(
                    f"Op {spec.name}: tiling and axis adaptation in one call "
                    "are not supported yet; give one or the other"
                )
            # All the inputs in spec.tile are tiled the same way; one left out
            # of the call -- an optional mask -- stays None.
            inputs = {name: call_args.get(name) for name in spec.tile}
            started = itertools.count(1)

            def one_tile(pieces: dict[str, Any]) -> Any:
                number = next(started)
                if on_progress is not None:
                    on_progress(
                        _Event(
                            f"Tile {number} of {plan.calls}",
                            number - 1,
                            plan.calls,
                            tile=(number, plan.calls),
                        )
                    )
                return self.run(
                    fn,
                    {**call_args, **pieces},
                    tiler=_tiling.Tiler.off(),  # Already cut to fit.
                    variant=variant,
                    on_progress=on_progress,
                    on_start=on_start,
                )

            # The whole result is in RAM, whatever device the op is on:
            # a GPU op's budget says nothing about how much RAM is free.
            room = (
                _tiling.default_budget() if spec.peak_memory.device == "gpu" else None
            )
            return _tiling.run_tiles(one_tile, inputs, plan, out, room, spec.merge)

        call_args = _to_declared(spec, call_args)

        if spec.is_workflow:
            # No environment to dispatch to, and nothing to encode: a workflow
            # runs here, and the ops it calls each cross the boundary
            # themselves. Axis adaptation is skipped for the same reason --
            # the sub-ops adapt their own arrays.
            return _into(
                out, self._run_here(fn, call_args, on_progress, on_start, tiler)
            )

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
                    "origin": _origin(spec.module),
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

            return _into(out, _unpack(spec, outputs, buffers))
        finally:
            _codec.release(refs, unlink=True)

    def _run_here(
        self,
        fn: Callable,
        call_args: dict,
        on_progress: Callable[[Any], None] | None,
        on_start: Callable[[Any], None] | None,
        tiler: _tiling.Tiler | None = None,
    ) -> Any:
        """Call a workflow in this process, with this runner made ambient."""
        task = _HostTask(on_progress)
        if on_start is not None:
            on_start(task)
        reporting = bind_task(task)
        here = _Ambient(self, task, tiler)
        ambient = _current.set(here)
        # opspec.runtime.run goes through this runner too.
        sub_ops = bind_runner(here)
        try:
            return fn(**call_args)
        finally:
            unbind_runner(sub_ops)
            _current.reset(ambient)
            unbind_task(reporting)

    #: Run in a worker to hand GPU memory back. Torch frees tensors into its
    #: own cache, not to the driver; cupy does the same. TensorFlow keeps
    #: what it took either way -- see docs/design/0019.
    _RELEASE = """
import gc
gc.collect()
try:
    import torch
    torch.cuda.empty_cache()
except Exception:
    pass
try:
    import cupy
    cupy.get_default_memory_pool().free_all_blocks()
except Exception:
    pass
task.outputs['released'] = True
"""

    def release(self, env: str | None = None, close: bool = False) -> None:
        """Give back GPU memory in live workers.

        *env* names one environment, or None for all of them. By default the
        worker stays up and only returns what it had cached; what an op still
        references stays held. ``close=True`` shuts the worker down instead,
        which is the only thing that frees a TensorFlow environment.
        """
        for key, service in list(self._services.items()):
            if env is not None and key[0] != env:
                continue
            with contextlib.suppress(Exception):
                if close:
                    service.close()
                    del self._services[key]
                else:
                    service.task(self._RELEASE).wait_for()

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


def _validate(spec: OpSpec, args: dict) -> None:
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


def _adapts(plan: _adapt.AdaptationPlan) -> bool:
    """Whether a plan changes anything: slices, iterates or reorders.

    A front end hands over a plan for every input whose op declares axes,
    including the many that come out "as is"; only the rest stand in the way
    of tiling.
    """
    identity = tuple(range(len(plan.transpose)))
    return bool(plan.iterate or plan.select) or tuple(plan.transpose) != identity


def _into(out: Any, result: Any) -> Any:
    """*result*, written into *out* when the caller gave one."""
    if out is None:
        return result
    if isinstance(result, tuple):
        raise TypeError(
            f"out takes a single array, and this op returned {len(result)} values"
        )
    array = np.asarray(result)
    target = _tiling.prepare_out(out, array.shape, array.dtype)
    target[...] = array
    return target


def tile_plan(
    spec: OpSpec, args: dict, memory: int | str | None = None
) -> TilePlan | None:
    """How ``Runner.run(..., memory=memory)`` would cut this call.

    Kept for front ends that predate the Tiler: ``Tiler(memory).plan``.
    """
    return _tiling.Tiler(memory).plan(spec, args)


def _to_declared(spec: OpSpec, args: dict) -> dict:
    """If a param isn't a numpy array but the op wants one, convert it.

    The whole array is read into memory (see docs/design/0017).
    """
    converted = dict(args)
    for param in spec.params:
        value = args.get(param.name)
        if isinstance(value, np.ndarray) or not isinstance(value, Array):
            continue
        if not _wants_numpy(param.type, value):
            continue
        if param.direction is not None:
            raise TypeError(
                f"Op {spec.name}: {param.name} is written to, so it must be "
                f"numpy, not {type(value).__name__}"
            )
        converted[param.name] = np.asarray(value)
    return converted


def _wants_numpy(annotation: Any, value: Any) -> bool:
    """Does the annotation need numpy? Not if it also accepts *value* as is."""
    if get_origin(annotation) in (Union, UnionType):
        members = [_strip(a) for a in get_args(annotation)]
    else:
        members = [annotation]
    members = [get_origin(m) or m for m in members]  # NDArray[...] -> ndarray
    if np.ndarray not in members:
        return False
    return not any(isinstance(m, type) and isinstance(value, m) for m in members)


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


def _unpack(spec: OpSpec, outputs: dict, buffers: dict) -> Any:
    names = tuple(o.name for o in spec.outputs)
    if not names:
        return None
    if spec.form != FUNCTION:
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


@dataclass(frozen=True)
class Env:
    """An environment an op asked for: its ID, and the file that defines it.

    ``Runner.env_for(op)`` gives one; ``name`` and ``folder`` say what it is
    built as, and where.
    """

    id: str
    config: Path
    #: Whose environment it is: the op's package, e.g. ``tnia-ops``.
    package: str

    @property
    def name(self) -> str:
        """What Appose builds it as: ``<package>--<id>``.

        ``scikit-ops--decon`` and ``tnia-ops--decon`` are two environments,
        though both are ``decon``: two collections' authors need not know of
        each other for their environments to stay apart.
        """
        return f"{self.package}--{self.id}"

    @property
    def folder(self) -> Path:
        """Where Appose builds it. It need not exist yet."""
        try:
            from appose.util.filepath import appose_envs_dir
        except ImportError:  # older appose kept it next door
            from appose.util.environment import appose_envs_dir

        return Path(appose_envs_dir()) / self.name


def _package_name(module: str, root: Path) -> str:
    """Whose environments these are, for naming them: a package's name.

    From the ``pyproject.toml`` beside ``envs/``, for a checkout; else from
    the installed distribution *module* belongs to; else, for a plain folder,
    its name and a short hash of where it is.
    """
    project = root / "pyproject.toml"
    if project.exists():
        found = _project_name(project.read_text(encoding="utf-8"))
        if found:
            return _normalized(found)
    for name in _distributions().get(module.split(".")[0], []):
        return _normalized(name)
    where = hashlib.sha1(str(root.resolve()).encode()).hexdigest()[:6]
    return f"{root.name}-{where}"


def _project_name(text: str) -> str | None:
    """``[project] name`` from a pyproject.toml's text."""
    try:
        import tomllib
    except ImportError:  # Python 3.10
        section = text.split("[project]", 1)[-1]
        found = re.search(r'^name\s*=\s*"([^"]+)"', section, re.MULTILINE)
        return found.group(1) if found else None
    return tomllib.loads(text).get("project", {}).get("name")


def _normalized(name: str) -> str:
    """A package name as PyPI compares them: tnia_ops and Tnia.Ops are tnia-ops."""
    return re.sub(r"[-_.]+", "-", name).lower()


@functools.cache
def _distributions() -> dict:
    """Top-level import name to distribution names. Slow to work out, once."""
    import importlib.metadata

    return importlib.metadata.packages_distributions()


def _origin(module: str) -> dict | None:
    """Where a worker can import *module* from, as the host found it.

    ``{"root": folder}``, the folder to put on the path; None when the host
    has no file for it, or the op is in a script run directly (``__main__``),
    which a worker cannot import.
    """
    found = getattr(sys.modules.get(module), "__file__", None)
    if found is None:
        return None
    if module == "__main__":
        return None
    path = Path(found).resolve()
    # pkg/sub/mod.py is pkg.sub.mod: up one folder per dotted part, and one
    # more for a package's __init__.py.
    depth = module.count(".") + (path.name == "__init__.py")
    root = path.parents[depth]
    origin: dict = {"root": str(root)}
    if not is_checkout(root):
        # Installed: the worker's environment should have its own copy, and
        # the host's site-packages holds packages built for the host's Python,
        # so the worker only falls back to it, last on its path.
        origin["append"] = True
    return origin


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
