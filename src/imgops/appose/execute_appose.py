"""
Appose Execution Utility.

Home for everything "run a segmenter in another Python environment":

* :class:`RemoteEnvironment` -- a named handle for an external interpreter/env.
* :class:`RemoteEnvironmentRegistry` -- session + on-disk store of known envs,
  including a per-segmenter pin so callers can look up which env to use.
* :func:`execute_appose` -- hand a segmenter + image to an env and get a
  numpy mask back (raises on failure).
* :func:`run_segmenter_remotely` -- convenience wrapper that resolves the
  env from the registry.

Kept headless (no Qt imports) so it can be used from scripts and tests.
The Qt-side "pick an env" dialog lives in ``apps/remote_env_dialog.py``.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Environment model + registry
# ---------------------------------------------------------------------------


@dataclass
class RemoteEnvironment:
    """A named handle for an external Python environment."""

    name: str
    path: str
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> RemoteEnvironment:
        return cls(
            name=d.get("name", ""),
            path=d.get("path", ""),
            notes=d.get("notes", ""),
        )


def _default_registry_path() -> Path:
    """Location of the persisted registry file."""
    return Path.home() / ".napari_ai_lab" / "environments.json"


@dataclass
class RemoteEnvironmentRegistry:
    """Session store of RemoteEnvironment + per-segmenter pins.

    Persists to ~/.napari_ai_lab/environments.json so users don't have
    to re-pick every launch. Access the process-wide instance via
    :func:`get_registry`.
    """

    environments: dict = field(default_factory=dict)
    pins: dict = field(default_factory=dict)
    storage_path: Path = field(default_factory=_default_registry_path)

    def add(self, env: RemoteEnvironment) -> None:
        self.environments[env.name] = env
        self.save()

    def remove(self, name: str) -> None:
        self.environments.pop(name, None)
        self.pins = {k: v for k, v in self.pins.items() if v != name}
        self.save()

    def get(self, name: str) -> RemoteEnvironment | None:
        return self.environments.get(name)

    def list(self) -> list:
        return list(self.environments.values())

    def pin(self, segmenter_class_name: str, env_name: str) -> None:
        if env_name not in self.environments:
            raise KeyError(f"No environment registered as {env_name!r}")
        self.pins[segmenter_class_name] = env_name
        self.save()

    def unpin(self, segmenter_class_name: str) -> None:
        self.pins.pop(segmenter_class_name, None)
        self.save()

    def resolve(self, segmenter_class_name: str) -> RemoteEnvironment | None:
        env_name = self.pins.get(segmenter_class_name)
        if env_name is None:
            return None
        return self.environments.get(env_name)

    def save(self) -> None:
        try:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "environments": [
                    e.to_dict() for e in self.environments.values()
                ],
                "pins": self.pins,
            }
            with open(self.storage_path, "w") as f:
                json.dump(payload, f, indent=2)
        except OSError as e:
            print(f"Warning: could not save remote env registry: {e}")

    def load(self) -> None:
        if not self.storage_path.exists():
            return
        try:
            with open(self.storage_path) as f:
                payload = json.load(f)
            self.environments = {
                d["name"]: RemoteEnvironment.from_dict(d)
                for d in payload.get("environments", [])
                if d.get("name")
            }
            self.pins = dict(payload.get("pins", {}))
        except (OSError, json.JSONDecodeError, KeyError) as e:
            print(f"Warning: could not load remote env registry: {e}")


_REGISTRY: RemoteEnvironmentRegistry | None = None


def get_registry() -> RemoteEnvironmentRegistry:
    """Return the process-wide RemoteEnvironmentRegistry."""
    global _REGISTRY
    if _REGISTRY is None:
        reg = RemoteEnvironmentRegistry()
        reg.load()
        _REGISTRY = reg
    return _REGISTRY


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


def execute_appose(
    image,
    segmenter,
    environment_path,
    additional_inputs=None,
    points=None,
    shapes=None,
):
    """Execute a segmenter's execution string in a remote env via appose.

    Args:
        image: Input numpy image.
        segmenter: Instance with get_execution_string(image, **kwargs).
        environment_path: Filesystem path or a RemoteEnvironment.
        additional_inputs: Extra {name: appose.NDArray} inputs.
        points: Optional numpy array forwarded as "points".
        shapes: Optional array forwarded as "shapes".

    Returns:
        numpy.ndarray mask.

    Raises:
        ImportError: appose not installed locally.
        RuntimeError: remote task failed or produced no "mask" output.
    """
    try:
        import appose
    except ImportError as e:
        raise ImportError(
            "appose is required for remote segmenter execution -- "
            "install it in this environment (`pip install appose`)."
        ) from e

    execution_string = segmenter.get_execution_string()

    ndarr_img = appose.NDArray(dtype=str(image.dtype), shape=image.shape)
    ndarr_img.ndarray()[:] = image
    inputs = {"image": ndarr_img}

    # Flattened member variables (primitives) so the worker script can
    # reconstruct what it needs without access to the segmenter instance.
    if hasattr(segmenter, "get_execution_inputs"):
        inputs.update(segmenter.get_execution_inputs())

    def _wrap_array(arr, name):
        arr_np = np.asarray(arr)
        wrapped = appose.NDArray(dtype=str(arr_np.dtype), shape=arr_np.shape)
        wrapped.ndarray()[:] = arr_np
        inputs[name] = wrapped

    if points is not None:
        _wrap_array(points, "points")
    if shapes is not None:
        _wrap_array(shapes, "shapes")
    if additional_inputs:
        inputs.update(additional_inputs)
    
    # env_path is typically just the env name (e.g. "default"). Build
    # the full pixi env path under the repo: pixi/<repo_name>/.pixi/envs/<env>
    env = appose.pixi().wrap(environment_path)
    
    with env.python() as python:
        task = python.task(execution_string, inputs=inputs, queue="main")
        task.wait_for()

        if task.error:
            raise RuntimeError(f"Remote task failed: {task.error}")

        result = task.outputs.get("mask")
        if result is None:
            raise RuntimeError(
                "Remote task produced no 'mask' output. "
                f"Available outputs: {list(task.outputs.keys())}"
            )

        if hasattr(result, "ndarray"):
            return np.asarray(result.ndarray()).copy()
        return np.asarray(result)


def run_segmenter_remotely(
    segmenter,
    image,
    points=None,
    shapes=None,
    env: RemoteEnvironment | None = None,
):
    """Resolve env from registry if not given, then execute_appose."""
    if env is None:
        env = get_registry().resolve(type(segmenter).__name__)
    if env is None:
        raise RuntimeError(
            f"No remote environment configured for "
            f"{type(segmenter).__name__}. Use the 'Choose Environment...' "
            f"button under the dependency banner to pick one."
        )
    return execute_appose(image, segmenter, env, points=points, shapes=shapes)
