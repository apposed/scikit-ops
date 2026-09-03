# 0020 — Reproducible environments

**Status:** proposed. Nothing built.

An environment is isolated and precisely described. It is not reproducible:
build the same recipe twice, a month apart, and get two different sets of
packages.

## Why

`.gitignore` line 6 is `pixi.lock`. Every environment under `envs/` has a lock
on disk and none of them is committed, so `pixi.toml` ships in the package and
the lock does not. Each user solves from scratch at build time, against
whatever conda-forge and PyPI hold that day.

So the promise the project makes — very repeatable, very precise, isolated
environments — is kept for the recipe and not for the result. Two people
running the same op get the same *description* of an environment and possibly
different TensorFlow builds inside it.

This surfaced twice in one afternoon. The `rev` pinning scikit-ops in
`envs/stardist-tf/pixi.toml` was a month stale, so a pip-installed host got a
worker with no training ops; and `ensure_environment` rebuilds in place, so the
environment a result came from is gone once the recipe changes.

## What would fix it

**Ship the locks.** Un-ignore `envs/*/pixi.lock`, commit them, and
`force-include` them the way `pixi.toml` already is. A build becomes a download
rather than a solve, and every machine gets the same bytes. This is most of the
value and nearly none of the work.

**Put a version in the environment id.** `skop-stardist-tf` becomes
`skop-stardist-tf-<hash of pixi.lock>`. Old environments stay on disk, a paper
can cite one, and rebuilding stops being destructive: a changed recipe builds
alongside rather than over the environment a result came from.

**Separate following from pinning.** `ensure_environment` follows the recipe
and is right for development. Anything whose results matter wants the opposite
— an environment that never moves under it. Two named calls, so the choice is
visible at the call site rather than implied.

**Pin scikit-ops by version, not by rev.** Once it is released, the pin is
`scikit-ops == 0.2.0`. A stale version number is legible; a stale 40-character
SHA is not, and nothing reminds anyone to bump it.

## Open

Whether locks should be per-platform-complete (pixi solves all four platforms
into one lock, which is large) or whether the shipped lock covers only the
platform being built for. The first is reproducible everywhere and adds
megabytes to the package; the second is smaller and only reproducible where it
was solved.
