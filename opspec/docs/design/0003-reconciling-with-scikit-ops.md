# 0003 — Reconciling with scikit-ops

**Status:** decided 2026-10-06, not yet carried out.

## Background

opspec started as a copy of scikit-ops' spec code (`skop/_spec.py`, and the
planning half of `skop/_adapt.py`), and the two drifted apart. opspec now
lives inside scikit-ops, in `opspec/`, and scikit-ops will import its spec
from opspec and delete its own copy. Where the two differ, one has to win.

## The rule

opspec follows scikit-ops, except where opspec changed something on purpose,
for a reason. A difference with no known reason goes scikit-ops' way.

## Where opspec's way wins

| | Difference | opspec | scikit-ops | Reason |
|---|---|---|---|---|
| K1 | Role for boxes | `Role.boxes` | `Role.shapes` | Boxes are a meaning of their own: detectors return them. A host maps the role to its own display, napari Shapes or Fiji ROIs. |
| K2 | Resolving output annotations | against the op's own globals, falling back to raw annotations | module globals | Bug fixes: Appose's worker turns every annotation into a string, and a file run as a script has no module. |
| K3 | `Builder` and `Runner` protocols | yes | — | Adds only. scikit-ops' runner already satisfies them (0002). |
| K4 | `env=None` | no environment: runs where the caller is | a workflow | Lets a plain function be an op, `op()(fn)`. A scikit-ops workflow is one such op, one that calls others. |
| K5 | `.outputs` | `OutputSpec`s | names, with details in `.output_specs` | Reads as the outputs of the function, not of the spec, and matches `.inputs`, which returns `ParamSpec`s. `.output_specs` goes. |
| K6 | Plan field | `uses_all_data` | `lossless` | Says what it means. "lossless" sounds like compression. |
| K7 | Marker set by `@op` | `__opspec__` | `__skop__` | Internal. Named after the package that sets it. |
| K8 | Reading a spec | `OpSpec.from_op(fn)` | `spec(fn)` | Reads as what it does, and is what opspec's docs, notebooks and examples use. |
| K9 | Bare `@op` | allowed | `@op(...)` only | Easier to read for an op with no environment. |

## Where scikit-ops' way wins

- **`_ui_hints` stays private.** Nothing outside `op.py` calls it; examples
  read the result from `ParamSpec.ui`. opspec's public `ui_hints_of` goes.
- **`return_type` and `return_role` stay in the wire form.** `to_dict`
  writes them, and `from_dict` reads them rather than rebuilding them from
  `outputs`. skop-fiji parses them.

## What opspec gains from scikit-ops

Never ported, rather than dropped:

- `@op(main_thread=..., exclusive=...)`, and the same fields on `OpSpec` and
  in the wire form.
- `Choices` and `ParamsFor`, with `choices_of`, `params_for_of`, and
  `ParamSpec.choices` and `.params_for`, so workflows can offer a menu of
  ops and their settings.
- `OpSpec.is_workflow`, read against K4: an op with no environment.
- scikit-ops' cache of a function's spec, so it is read once.

## The wire form after

```python
{
  "name": "threshold", "module": ..., "function": ..., "env": "skimage",
  "form": "function", "doc": ..., "params": [...],
  "outputs": [{"name": "result", "type": {"name": "ndarray"}, "role": "labels"}],
  "return_type": {"name": "ndarray"}, "return_role": "labels",
  "main_thread": False, "exclusive": False,
}
```

A parameter also carries `direction` for `Out` and `Mut`, and `choices` and
`params_for` when it has them.

## What changes with it

These land together: a scikit-ops sending the new form breaks the front
ends as they are.

- **scikit-ops.** `_spec.py` and the planning half of `_adapt.py` import from
  opspec; `apply` and `execute`, the `*Data` aliases and `discover` stay.
  `from skop import op, Axes` keeps working. Call sites change for K1, K5,
  K6 and K8, and the runner and worker for K6, which crosses to the worker.
- **skop-napari.** `.outputs` and `.output_specs` (K5), `Role.boxes` (K1),
  `uses_all_data` and `from_op` in its tests.
- **skop-fiji.** `wire/OpSpec.java` reads `outputs` as dicts and drops
  `output_specs` (K5); `wire/AdaptationPlan.java` reads `uses_all_data`
  (K6); `wire/Role.java` and `Roles.java` take `boxes` (K1); `WireTest.java`'s
  sample dicts follow.
