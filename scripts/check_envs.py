#!/usr/bin/env python3
"""Check that every environment in envs/ still solves.

Solving is not building. This resolves each `envs/<id>/pixi.toml` against the
channels and reports whether a consistent set of packages exists -- seconds to
a minute each, no gigabytes downloaded. Building is what `pytest --build-envs`
does, and it is far too slow to run for its own sake.

That distinction matters because the failure this exists to catch is a solve
failure. `envs/pytorch` declared a conda `appose` alongside a pinned
scikit-ops that sourced appose from git; the two could not be satisfied
together, and the manifest was unsolvable from the day it was written. Nothing
noticed, because an environment built before that edit goes on working
indefinitely -- a stale lock file is a working lock file. It surfaced only when
something finally forced a re-solve, weeks later and far from the cause.

So this is a check on the *manifests*, which the test suite cannot make: the
op tests run inside an environment and therefore cannot tell you whether that
environment could still be created today.

    python scripts/check_envs.py             # all of them
    python scripts/check_envs.py pytorch     # just one or two
    python scripts/check_envs.py --quiet     # only failures

Exits non-zero if any environment fails to solve.

A caveat worth keeping in mind: this also goes red when a channel moves under
a lock that was fine yesterday. It answers "are these manifests satisfiable
right now", which is a health check rather than a regression gate.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ENVS = Path(__file__).resolve().parent.parent / "envs"


def check(env_dir: Path, quiet: bool) -> tuple[bool, str]:
    """Solve one environment. Returns (ok, detail)."""
    # `pixi lock` writes envs/<id>/pixi.lock if it is missing or stale, which
    # is the point -- a lock that has to change is a manifest whose solution
    # moved. --check reports that without writing, so a red run leaves the
    # tree exactly as it found it.
    result = subprocess.run(
        ["pixi", "lock", "--check"],
        cwd=env_dir,
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        return True, "lock is current"

    output = f"{result.stdout}\n{result.stderr}".strip()

    # Distinguish the two non-zero cases, which mean very different things.
    # An unsatisfiable solve is a broken manifest; a merely outdated lock is
    # normal drift and is not something to fail a check over on its own.
    unsatisfiable = any(
        phrase in output
        for phrase in ("unsatisfiable", "failed to solve", "failed to resolve")
    )
    if unsatisfiable:
        return False, output if not quiet else _last_error(output)
    return True, "lock is out of date, but solves"


def _last_error(output: str) -> str:
    """The tail of pixi's complaint, for a quiet run."""
    lines = [line for line in output.splitlines() if line.strip()]
    return "\n".join(lines[-8:])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("envs", nargs="*", help="env ids; default is all of them")
    parser.add_argument(
        "--quiet", action="store_true", help="print only failures, and briefly"
    )
    args = parser.parse_args()

    names = args.envs or sorted(d.name for d in ENVS.iterdir() if (d / "pixi.toml").is_file())

    failures = []
    for name in names:
        env_dir = ENVS / name
        if not (env_dir / "pixi.toml").is_file():
            print(f"{name}: no pixi.toml", file=sys.stderr)
            failures.append(name)
            continue

        ok, detail = check(env_dir, args.quiet)
        if ok:
            if not args.quiet:
                print(f"  ok    {name}  ({detail})")
        else:
            failures.append(name)
            print(f"  FAIL  {name}")
            for line in detail.splitlines():
                print(f"        {line}")

    print()
    if failures:
        print(f"{len(failures)} of {len(names)} environments do not solve: "
              f"{', '.join(failures)}")
        return 1
    print(f"all {len(names)} environments solve")
    return 0


if __name__ == "__main__":
    sys.exit(main())
