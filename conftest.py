"""Test configuration: source imports, and opting in to expensive environments.

Putting src/ on sys.path mirrors what the Runner does for worker processes,
where the same directory is injected through the service init script.
"""

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


def pytest_collection_modifyitems(config, items):
    """Skip env-marked tests whose environment is not built yet.

    An op test is only expensive the first time, when its environment has to
    be installed -- so the default is to run whatever is already on the
    machine and skip the rest, and ``--build-envs`` is how CI (or a thorough
    local run) asks for the installs to happen.
    """
    if config.getoption("--build-envs"):
        return
    for item in items:
        marker = item.get_closest_marker("env")
        if marker is None:
            continue
        env_id = marker.args[0]
        if not _is_built(env_id):
            item.add_marker(
                pytest.mark.skip(
                    reason=f"env '{env_id}' is not built; pass --build-envs"
                )
            )
