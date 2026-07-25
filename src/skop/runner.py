"""The host-side half of skop: build environments and run ops in them.

The host never imports an op's dependencies. It imports the op module only to
read its signature -- which is cheap, because ops keep heavy imports inside
their function bodies -- and dispatches the actual call to a worker process
living in the op's declared environment.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

import appose
import numpy as np

from . import _codec, _spec

if TYPE_CHECKING:
    from typing import Self

# The script sent for every op call. Its last statement is an expression
# yielding a dict, which Appose turns into the task's outputs.
_CALL = "skop_invoke(task, module, function, kwargs)"

# Installed into every worker via the service init script. Names defined here
# become worker exports, and so are in scope for every task.
#
# NB: The search path is extended here rather than via PYTHONPATH. Appose
# gained per-service env vars in 0.12; setting them on the builder instead
# would fold a machine-specific path into the environment's identity, causing
# a rebuild whenever the checkout moves.
_INIT = """
import sys
sys.path.insert(0, {root!r})
import numpy  # NB: must precede the worker's I/O loop on Windows.
import skop.worker
skop_invoke = skop.worker.invoke
"""


def _default_root() -> Path:
    """The directory holding the ``skop`` package."""
    return Path(__file__).resolve().parent.parent


def _default_envs_dir(root: Path) -> Path:
    """The ``envs`` directory of the checkout whose ``src`` is *root*.

    Note: environment definitions are data alongside the source tree, not
    inside it, so they sit one level up from the packages themselves.
    """
    return root.parent / "envs"


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
        script = _INIT.format(root=str(self.root))
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
        on_progress: Callable[[Any], None] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Run an op in its environment and return its result.

        Args:
            fn: The op function, as decorated with ``@op``.
            args: Op arguments, as a dict. Merged with ``**kwargs``, which is
                the more convenient form when no argument name collides with
                this method's own parameters.
            variant: Optional named pixi sub-environment (e.g. ``"cuda"``).
            on_progress: Called with each Appose TaskEvent as it arrives.
        """
        spec = _spec.spec(fn)
        call_args = dict(args or {})
        call_args.update(kwargs)
        _validate(spec, call_args)

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
                },
                queue="main" if spec.main_thread else None,
            )
            if on_progress is not None:
                task.listen(on_progress)
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
    """Run an op via the default Runner."""
    return default_runner().run(fn, args, **kwargs)
