# 0004 — Reporting an environment build

## The problem

The first run of an op in an unbuilt environment takes minutes. That build
happens *inside* `Runner.run()`, because that is the only place skop knows
which environment is needed. A GUI calling `run()` therefore sits there with
nothing to show, on the single slowest operation the project performs, and no
way to distinguish it from a hang.

## The decision

Three subscription methods on `Runner`, passed straight through to Appose's
fluid builder:

```python
runner.subscribe_build_progress(lambda title, current, maximum: ...)
runner.subscribe_build_output(lambda text: ...)
runner.subscribe_build_error(lambda text: ...)
```

Subscribers accumulate in lists and are attached in `environment()`:

```python
builder = appose.pixi(config).name(f"skop-{env_id}")
for subscriber in self._build_progress:
    builder = builder.subscribe_progress(subscriber)
...
```

Deliberately a thin pass-through. skop does not parse, buffer, rate-limit or
reformat any of it. Every consumer wants something different — a progress bar,
a log, a CI transcript — and anything skop did here would be in one of their
ways.

## Two things the names get wrong

Both of these were determined empirically, by building throwaway pixi
environments and watching what actually arrived, rather than by reading the
method names. Both are worth knowing before writing a front end.

**`subscribe_error` is the stderr stream, not a failure report.** Pixi writes
all of its ordinary status there, success message included —
`✔ The default environment has been installed.` arrives on the "error" channel.
Routing this straight to a GUI's error display, which was the obvious first
implementation, turns every successful build into an error notification. A
build that genuinely fails raises out of `run()` and takes the normal exception
path; that is the only failure signal.

**Subscribing to progress is what *enables* it.** Appose's `PixiInstallMonitor`
only wires up when a progress subscriber exists, and it works by running pixi
under `-vv` and reading phase transitions out of its log. Two consequences:

- A caller that subscribes only to output/error gets a nearly silent build.
- A caller that subscribes to progress gets pixi's entire `-vv` debug log on
  the output and error channels. Filtering that down to something human-facing
  is the front end's job (skop does not know what "human-facing" means here).

Progress titles name the phase — `Solving`, `Installing conda packages`,
`Downloading PyPI packages`, `Installing PyPI packages`, `Done` — each with a
real denominator.

## Why the dependency is pinned to Appose's main branch

`PixiInstallMonitor` landed in appose-python commit `3e97f55` and has never
been released; 0.11 has no monitor at all and reports only a final summary
line. Both this project and `skop-napari` therefore source Appose from its main
branch:

```toml
[tool.uv.sources]
appose = { git = "https://github.com/apposed/appose-python", branch = "main" }
```

This is temporary and should be removed the moment a release carries the
monitor. It is called out here because the first version of this documentation
described the callbacks as "carrying less than their names suggest" — accurate
for 0.11, wrong for 0.12.0.dev0 — and anyone reading against a released Appose
will see the 0.11 behaviour and conclude the docs are lying.

## Related

`Runner.run()` also takes `on_start`, called with the Appose `Task` as soon as
it is submitted. `run()` blocks until the op finishes, so a GUI wanting to
cancel needs the handle from another thread. Waiting for the first *progress*
event instead — the tidier-looking option — would leave a silent op
uncancellable.
