# scikit-ops — notes for Claude

Image-processing **ops** (`skop`, the collection; a working name) and
**oprunner**, the machinery that runs them. An op is an ordinary Python function; the same function can be called directly, run in an
isolated environment through Appose, or wrapped in a GUI, and nothing about it
changes between those modes.

Read [docs/README.md](docs/README.md) first for the design notes. `docs/design/NNNN-*.md`
is settled and written after the fact; `docs/spec/*.md` is proposed and not
built, and graduates into a numbered design doc when it lands.

## Layout

```
src/skop/ops/<ns>[.py|/]  ops, one namespace per file or package (the collection)
oprunner/                 the runner, its own package: environments, workers, tiling
opspec/                   the spec layer, its own package: @op, OpSpec, roles, plans
ophelper/                 helpers for writing ops, its own package: boxes, masks, assets
envs/<env-id>/pixi.toml   environment definitions, shared between ops
pixi/examples/            the host for notebooks and interactive tests
notebooks/                narrative examples, outputs committed
website/                  the docs website: one Sphinx site, built by Read the Docs
interactive_tests/        runnable scripts that open a viewer
test_images/              gitignored -- large TIFFs, not in the repo
```

## Two kinds of environment, never blurred

- **Host** — built by uv from `pyproject.toml`, or by pixi from
  `pixi/examples/pixi.toml`. Holds what the *caller* imports: napari,
  matplotlib, tnia-python, ipykernel. The `examples` dependency group is this,
  and `[tool.uv] default-groups` keeps it in `.venv` so a bare `uv sync` cannot
  strip the Jupyter kernel out from under an editor.
- **Op** — built by pixi, which appose downloads itself, from
  `envs/<id>/pixi.toml`. Holds torch, cupy, an old numpy. Never a host
  dependency; adding one rebuilds the monolith this design exists to avoid.

Built op environments live in `~/.local/share/appose/<package>--<env-id>/`
(`scikit-ops--skimage`), **not** in `envs/` — that directory holds the recipe
only. Named after the package, so two collections' `skimage` stay apart.

## Facts worth not rediscovering

- The spec layer is **opspec**, a standard-library-only package kept in
  `opspec/` and published separately. Import its names from it, never through
  skop or oprunner; `oprunner._adapt` takes planning from it and keeps execution. Where the two
  differed, the decisions and reasons are in
  [opspec/docs/design/0003](opspec/docs/design/0003-reconciling-with-scikit-ops.md).
  opspec has its own [CLAUDE.md](opspec/CLAUDE.md): it never imports numpy.
- Read a spec with `OpSpec.from_op(fn)`. `spec.outputs` are `OutputSpec`s,
  not names; boxes are `Role.boxes`; a plan says `uses_all_data`.
- In a checkout, importing oprunner puts `opspec/src` and `ophelper/src` first
  on `sys.path`, the way `oprunner.host.INIT` puts `oprunner/src` first, and an
  op's own folder comes from the host (`origin`). That is how a worker finds
  today's code: the environments pin an older scikit-ops.
- `oprunner.discover()` returns `(specs, failures)`. `specs` is every op declared,
  including ones this interpreter could not run — a catalogue, because a front
  end must offer Cellpose before its environment exists. `failures` is op
  *modules that would not import*, which enforces the rule that heavy imports
  live inside function bodies. It is **not** "cannot run here".
- `spec.env` is the environment an op declares; `env is None` means a workflow
  (`spec.is_workflow`), which runs on the host and dispatches sub-ops.
- Ops must keep heavy imports inside the function body, or discovery breaks.
- Roles (`docs/design/0003`) say what an array *means*; axis mapping
  (`docs/design/0006`) belongs to the caller, and iteration runs in the worker.
- `pixi.lock` is gitignored (`.gitignore:6`), so pixi environments resolve
  fresh per machine. `uv.lock` *is* tracked.

## Commands

```sh
uv sync                      # host env, incl. the examples group
uv run pytest                # fast tests, skop's and opspec's; skips anything needing an env
uv run pytest --build-envs   # builds missing op environments; slow first time
uv run ruff check --fix && uv run ruff format
uv run jupyter lab           # notebooks; or select .venv as the kernel
website/serve.sh             # the docs website, at http://127.0.0.1:8000
```

Never hand-merge `uv.lock`. Restore it, rebase, and regenerate with `uv sync`.

## Notebooks

Outputs are committed on purpose — the figures are the product, and should
render for someone who has installed nothing. Restart-and-run-all before
committing so execution counts read 1, 2, 3, and keep figures small, since
base64 PNGs are permanent repo weight.

[notebooks/README.md](notebooks/README.md) is the setup: the host environment,
the two pixi environments for released vs local tnia-python, and registering a
kernel for each.

## Related repositories (siblings, and required to be)

- `../skop-napari` — the napari front end, `napari-skopkit` (the folder keeps
  its old name); depends on this by path source. `../skop-fiji` is the Fiji
  one, `fiji-skopkit`.
- `../tnia-python` — projection figures and the box/mask overlays the notebooks
  draw with. A host dependency only; no op imports it. Its
  [extras](../tnia-python/docs/install.md) landed in 0.2.0, so the dependency
  here is `tnia-python[plotting]>=0.2.2` — matplotlib, scipy, scikit-image, and
  none of the deep-learning weight. **The floor names a released version and
  moves when one ships, never ahead of it**: a floor on an unreleased version
  makes the environment unsolvable.

  `pixi/examples` always takes it from PyPI. `pixi/tnia-dev` is the same
  environment with an editable sibling checkout instead, for changing the
  plotting helpers and a notebook together — gitignored, personal, and not
  something a notebook should be committed from. See
  [notebooks/README.md](notebooks/README.md).
- `../napari-ai-lab` — separate napari plugin collection, same author.
