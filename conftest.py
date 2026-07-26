"""Test configuration: source imports, and opting in to expensive environments.

Putting src/ on sys.path mirrors what the Runner does for worker processes,
where the same directory is injected through the service init script.
"""

import functools
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from appose.util.filepath import appose_envs_dir

src = str(Path(__file__).resolve().parent / "src")
if src not in sys.path:
    sys.path.insert(0, src)


def pytest_addoption(parser):
    parser.addoption(
        "--build-envs",
        action="store_true",
        default=False,
        help=(
            "Run every op test, building any environment that is missing. "
            "Off by default: a build can mean a multi-gigabyte download. "
            "Equivalent to setting PYTEST_ADDOPTS=--build-envs."
        ),
    )


def _is_built(env_id: str) -> bool:
    return (Path(appose_envs_dir()) / f"skop-{env_id}").is_dir()


@functools.lru_cache(maxsize=1)
def _has_nvidia_gpu() -> bool:
    """Whether this machine has an NVIDIA GPU to run CUDA ops on.

    Separate from whether an environment is built: a machine can install the
    cupy environment perfectly well and still have no device to use it on, and
    that should skip rather than fail. Asked of the driver rather than of cupy,
    which the host environment does not have.
    """
    if shutil.which("nvidia-smi") is None:
        return False
    try:
        return (
            subprocess.run(
                ["nvidia-smi", "-L"], capture_output=True, timeout=30, check=False
            ).returncode
            == 0
        )
    except (OSError, subprocess.SubprocessError):
        return False


def pytest_collection_modifyitems(config, items):
    """Skip env-marked tests whose environment is not built yet.

    An op test is only expensive the first time, when its environment has to
    be installed -- so the default is to run whatever is already on the
    machine and skip the rest, and ``--build-envs`` is how CI (or a thorough
    local run) asks for the installs to happen.
    """
    for item in items:
        # A missing GPU is hardware, not an install: --build-envs cannot fix
        # it, so this skip applies whether or not the flag was passed.
        if item.get_closest_marker("gpu") is not None and not _has_nvidia_gpu():
            item.add_marker(
                pytest.mark.skip(reason="no NVIDIA GPU found (nvidia-smi -L)")
            )

        if config.getoption("--build-envs"):
            continue
        # A test may declare more than one environment -- comparing two
        # backends needs both -- so every marker has to be satisfied.
        missing = [
            marker.args[0]
            for marker in item.iter_markers("env")
            if not _is_built(marker.args[0])
        ]
        if missing:
            item.add_marker(
                pytest.mark.skip(
                    reason=f"env {', '.join(repr(e) for e in missing)} "
                    "is not built; pass --build-envs"
                )
            )
