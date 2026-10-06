# opspec

A small package that describes ops for plugins, runners and hosts
(napari, CellProfiler, notebooks). It has no dependencies beyond the Python
standard library.

- **Declare** an op: `@op`, `Slot`, `OpSpec`, `Role`
- **Type** its data: `Array` Protocol, `ImageOf`
- **Run** it: `Runner` and `Task` Protocols (interfaces only)
- **Fit** a caller's axes to the op: `AdaptationPlan`, `plan()`

## The one rule

opspec never imports numpy, but it can *describe* numpy
(`ImageOf[np.ndarray]`). Code that computes on arrays (loops, tilers,
codecs) belongs in a runner, not here.

To be enforced by a test: importing opspec must not load numpy.

## Layout

    src/opspec/      core, standard library only
    examples/        small numbered notebooks, one idea each; uses numpy
    tests/           core tests without numpy; example tests skip if numpy is missing
    docs/design/     one short note per decision, written before the code

## How we work

- Slow and small. One idea per step, one step per commit.
- A short design note in docs/design/ comes before each new idea.
- Every declaration is optional. Leaving one out means behaviour stays as it is today.
- Each step gets its own small notebook, one idea each, and they all run.

## Commits

- The human writes the message. If it's fast and rough, Claude fixes typos
  and grammar only, keeps the wording and voice, and shows the result
  before committing.
- Short subject, plus at most a couple of plain lines. Readable by anyone,
  but specific enough to know where to look.
- Commit only when asked. Never push without asking.

## AI use

AI-assisted, openly. Commits keep the `Co-Authored-By: Claude` trailer.
The human reads, understands and tests every change.
