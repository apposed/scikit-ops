# Spec — a Fiji front end

**Status:** proposed, nothing built. The plan for `skop-fiji`, the second front
end, written against what [front-ends.md](front-ends.md) says the boundary is
and what [skop-napari](https://github.com/apposed/skop-napari) learned building
the first one.

Working name `skop-fiji`; Maven coordinate `org.apposed:skop-fiji`; package
`org.apposed.skop.fiji`; distributed as a Fiji update site.

## The headline: one command per op

skop-napari's [design 0002](https://github.com/apposed/skop-napari/blob/main/docs/design/0002-one-panel-for-all-ops.md)
settled for a single Ops panel with a picker inside it, because npe2 wants a
static manifest and skop finds its ops by importing them. That compromise is
not needed here. SciJava registers modules at runtime —
`DefaultMutableModuleInfo` plus `ModuleService.addModules()` — so **each op
becomes its own command**:

- its own entry under `Plugins ▸ scikit-ops ▸ <namespace> ▸ <op>`;
- in Fiji's search bar for free, since `ModuleSearcher` indexes `ModuleService`;
- with a dialog the input harvester generates from the module's items;
- individually scriptable and macro-recordable.

skop-napari's [dynamic-registration.md](https://github.com/apposed/skop-napari/blob/main/docs/spec/dynamic-registration.md)
describes a napari feature that has not landed. Fiji has had it for a decade,
so the Fiji front end starts where the napari one hopes to end up. There is no
op-picker panel; the search bar *is* the browse view.

Macro recordability is the part with no napari equivalent at all, and it makes
the op ID load-bearing: `skop.ops.threshold:otsu` ends up in someone's saved
macro, so it is a public string that may not be reworded casually.

## Architecture: Java is the host

Three topologies were considered.

| | Shape | |
| --- | --- | --- |
| A | Java → a Python control process holding a `Runner` → a worker per env | Rejected |
| B | Java builds the environments and drives the workers directly | **Chosen** |
| C | Pure Java, no Python at all | Impossible |

**A is rejected on copies.** `_codec._to_ndarray` copies any plain
`numpy.ndarray` into a fresh shared memory block, so an array crossing
Java → control → worker is copied on the way in and again on the way out, for
data that was in shared memory to begin with. It also puts a third process in
the chain and proxies progress and cancellation through it twice.

**C is impossible** because `discover()` works by importing op modules. Some
Python must run before anything can be listed.

**B**, then: `SkopRunner.java` is a port of `runner.py`'s dispatch. It builds
each environment with appose-java's `PixiBuilder`, keeps one `Service` per
`(env, variant, exclusive)` key exactly as `Runner.service` does, encodes
arguments, posts the same `skop_invoke(task, module, function, kwargs, plans)`
task, decodes the outputs, and copies back the buffers of computer- and
inplace-form ops. A Java `ShmImg` becomes an `NDArray` that goes straight into
the worker's task inputs: one block, no copies, and the worker sees exactly
what `runner.py` would have handed it.

The two jobs that must stay in Python are `discover()` and `skop.plan()`. Both
are metadata-only, so both run as tasks in one long-lived service in the
**`minimal`** environment. That is the "`OpSpec` over the wire" item
[front-ends.md](front-ends.md) lists as unresolved, and this is what resolves
it.

The risk of B is duplication: `_INIT`, the `_CALL` template and the encoding
rules now exist in two languages. Keep the strings in skop and let Java read
them (`skop.host` constants delivered alongside the specs) rather than
transcribing them into Java literals.

## What scikit-ops must gain first

None of this is Fiji-shaped; it is the serialized half of the contract that
`front-ends.md` already anticipated.

1. **`OpSpec` as JSON.** `to_dict`/`from_dict` covering `name`, `module`,
   `function`, `env`, `form`, `doc`, and every `ParamSpec` and `OutputSpec`
   field — including `ui` hints, `role`, `axes`, `direction` and defaults.
   Exposed as a `skop.host:describe()` task returning the whole `discover()`
   result, failures included.
2. **A wire vocabulary for `ParamSpec.type`.** It is a live Python type object
   today. Java needs names: `int`, `float`, `str`, `bool`, `ndarray`, `path`,
   `enum` (with its choices), and `unknown`. `unknown` is what lets the Fiji
   side reproduce skop-napari [design 0003](https://github.com/apposed/skop-napari/blob/main/docs/design/0003-building-input-widgets.md)'s
   sorting: a parameter it cannot render is left at its default and reported,
   or, if it is required, disables the run and says why. One awkward parameter
   must not cost the whole op.
3. **`plan()` over the wire.** `AdaptationPlan.to_dict`/`from_dict` already
   exist; what is missing is a task taking `(op, param, shape, labels,
   position)` and returning the plan and its warnings, so the axis logic is
   computed in one place rather than reimplemented in Java.
4. **skop reachable from a worker without a checkout.** See below.

## Getting skop into the environments

`Runner._INIT` inserts a checkout's `src` on `sys.path`, and no
`envs/*/pixi.toml` installs skop at all. Neither shipping Python inside a JAR
nor publishing to PyPI is wanted yet, so each `pixi.toml` gains a pinned git
dependency:

```toml
[pypi-dependencies]
scikit-ops = { git = "https://github.com/apposed/scikit-ops.git", rev = "<sha>" }
```

The sys.path injection stays and keeps working for scikit-ops' own development:
it is inserted at position 0, so a checkout shadows the pinned copy, and a
developer editing an op still sees the edit without touching the pin.

Two consequences worth knowing before the first pin lands:

- **The pin is part of every environment's identity**, so bumping it rebuilds
  all nine environments. Bump deliberately, not per commit.
- Once scikit-ops is released, this becomes an ordinary version constraint and
  the git URL disappears. The pin is a stand-in for a release, not a design.

## OpSpec → SciJava module

| From `OpSpec` | Becomes |
| --- | --- |
| `name` | the module identifier, and the recorded command string |
| `module` namespace | the menu path |
| `doc` summary, `Args:` entries | `setDescription` on the info and its items |
| a `ParamSpec` | a `DefaultMutableModuleItem` |
| `ParamSpec.type` | that item's Java type |
| `ParamSpec.ui` | `setMinimumValue`, `setMaximumValue`, `setStepSize`, and `setWidgetStyle` (`FloatSlider` → `NumberWidget.SLIDER_STYLE`) |
| `ParamSpec.role` | a `Dataset`/`ImgLabeling`/… item rather than a text field |
| `ParamSpec.axes` | the axis-mapping items, added dynamically |
| `direction is OUT` | nothing — a user is never asked for a buffer |
| an `OutputSpec` | an `ItemIO.OUTPUT` item |
| `env` | shown in the dialog; selects the service that runs it |

**Registration must not block startup.** `describe()` imports every module
under `skop.ops`, and on a first launch it must build the `minimal`
environment before it can do even that. Register on a background thread and
cache the JSON on disk, keyed by a hash of the ops tree, so the second launch
populates the menu immediately.

## Roles → Fiji

The lookup table that is the only Fiji-specific part of the front end, the
counterpart of `skop_napari/_roles.py`.

| Role | Fiji |
| --- | --- |
| `image` | `Dataset`/`ImgPlus` over a `ShmImg` |
| `labels` | `ImgLabeling` |
| `masks` | ROI Manager / `Overlay` |
| `points` | `PointRoi` / `Overlay` |
| `shapes` | `Overlay` of rectangle ROIs |
| `vectors` | `Overlay` of arrows, or a table |
| `tracks` | a SciJava `Table` at first; a TrackMate model later |
| `surface` | imagej-mesh |
| no role | a row in a `Table` |

`labels` is `ImgLabeling` because that is what a label image *is*; a
glasbey-LUT `Dataset` is a rendering of one, and picking the rendering as the
representation would throw away the structure every downstream ImgLib2 consumer
wants. Conversions between labelings and images are useful in their own right
and belong in `skop.ops.labels`, not in the front end.

`masks` is the role that comes out *better* here than in napari.
[Design 0008](../design/0008-mask-detector-ops.md) exists because a napari
Labels layer cannot show overlapping objects, so `skop.masks` must project them
first. Fiji's ROI Manager holds overlapping ROIs natively, so the projection
becomes one of several things a user may ask for rather than a precondition for
seeing anything at all.

Guessing a role for an unannotated array happens here, never in skop —
[design 0003](../design/0003-semantic-roles.md), and the rule
[front-ends.md](front-ends.md) ends on.

## Axes: Fiji knows them, and the order is reversed

skop-napari spends a module (`_axes.py`) working out what a layer's axes are,
from metadata, xarray dims, NGFF axes, `axis_labels` and `rgb`, and leaves
unnamed whatever none of those say. `ImgPlus` **carries `CalibratedAxis` labels
explicitly**, so the Fiji side reads what it is told and only falls back to
guessing for a bare `Img`. skop's `ALIASES` already speaks the vocabulary —
`slice` → `z`, `frame` → `t` — because it was assembled from ImageJ2's
`AxisType` among others.

The trap is the other half: **ImgLib2 is x-fastest, numpy is last-fastest.** An
`ImgPlus` whose axes are `(X, Y, Z)` is a numpy array of shape `(z, y, x)`, so
the label list handed to `plan()` is the ImgLib2 axis order reversed. Get this
wrong and nothing crashes — the op runs on transposed data and returns a
plausible, wrong answer. It is the first thing to write a round-trip test for,
before any UI exists.

The mapping UI itself (a combo per slot, and iterate / current position / pass
per leftover axis, per skop-napari design 0007) does not fit a statically
harvested dialog, since the items depend on the shape of an image the user has
not chosen yet. Use `DynamicCommand` with a callback on the image parameter
that adds the mapping items once a `Dataset` is selected. If that proves
unworkable, fall back to accepting skop's default plan and surfacing its
warnings, which never discards data — but that is the fallback, not the plan.

## Threading, progress, cancellation, errors

SciJava already runs modules off the EDT, so there is no equivalent of
skop-napari `_run.py`'s thread-worker plumbing. `TaskEvent`s drive
`StatusService`; the harvester's Cancel maps onto Appose task cancellation,
which `skop.cancel_requested()` reads on the far side.

Environment builds get the treatment [design 0004](../design/0004-build-feedback.md)
describes, through appose-java's `PixiInstallMonitor`, which reports the same
phases (`Solving`, `Installing conda packages`, …) as the Python one. A first
run of a TensorFlow or PyTorch environment is minutes long, and unlike a
napari user launched from a terminal, a Fiji user has nowhere else to look. So
the front end also wants a `Plugins ▸ scikit-ops ▸ Manage Environments`
command: list what is built, prebuild one deliberately, delete one that went
wrong.

Errors follow the same reasoning as skop-napari's: the interesting part of a
failure is another interpreter's traceback, so it goes to `LogService` in full
and to a dialog, rather than into a one-line field.

## Environment cache location

Appose's default shared directory, not somewhere under the Fiji installation.
That is where skop puts environments already, which means **skop-napari and
skop-fiji share built environments** — a user who has already paid for the
`stardist-tf` build in napari does not pay again in Fiji.

## Layout

```
SkopService.java     op discovery, environment lifecycle, one shared runner
SkopRunner.java      port of runner.py: build, service per env, invoke
OpModuleInfo.java    OpSpec -> ModuleInfo
OpModule.java        the Module that runs one op
Params.java          ParamSpec -> MutableModuleItem
Roles.java           Role -> Fiji type
Axes.java            ImgPlus axes -> skop labels; the order flip lives here
Results.java         outputs -> Dataset / ImgLabeling / ROIs / Table
wire/                the OpSpec JSON reading
```

## Phasing

| | |
| --- | --- |
| **P0** | In scikit-ops: `OpSpec` JSON, `describe` and `plan` tasks, the pinned git dependency in each `pixi.toml` |
| **P1** | `SkopRunner` plus a headless test running `toy:add` and `threshold:otsu` on a `ShmImg` from Java. No UI. This proves the whole boundary, and is where the axis-order bug surfaces |
| **P2** | Dynamic module registration, image in and image out, progress, cancel, errors. The first shippable thing |
| **P3** | The rest of the roles: `ImgLabeling`, ROIs, tables, masks into the ROI Manager |
| **P4** | Axis-mapping UI, environment manager, update site, macro-recording polish |

## Decided, and not to be relitigated without new evidence

- **`ImgLabeling` for `labels`**, not a LUT'd `Dataset`. See above.
- **A pinned git dependency**, not Python in a JAR and not a premature PyPI
  release.
- **Ops are not registered as SciJava Ops.** Two "ops" vocabularies in one
  application is a support burden for little gain. The interesting direction is
  the opposite one — wrapping a scikit-op *as* a SciJava Op to inherit its
  matching — and that is out of scope for the first build, not ruled out.
- **Appose's default environment directory**, for the sharing above.

## Still open

- Whether `_INIT` and `_CALL` are delivered to Java as data or transcribed.
  Data, if it can be made to read well.
- What an op with a `MasksData` output offers by default: the ROI Manager, a
  projection, or a choice.
- Whether the ImageJ1 side needs anything beyond imagej-legacy's
  `ImagePlus`/`Dataset` conversion. Probably not, but no one has tried it with
  a shared-memory-backed `ShmImg` underneath.
