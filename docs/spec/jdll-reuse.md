# What JDLL has that skop could reuse

Status: notes, running. YOLO training is built, after the skop trainers
rather than JDLL's script: [0023](../design/0023-yolo-training-op.md).

Found since, in JDLL's `main` (2026-11): `src/main/resources/python/yolo_data.py`
is real Python -- a custom ultralytics trainer and dataset that read TIFF and
16-bit images and min/max-normalise them, the normalisation saved with the
model -- and the training call now passes `trainer=` from it, which the
reading copy below predates. OOM retry is in `YoloTrainingService.java`, on
the Java side; ultralytics 8.4 now does it in the first epoch by itself.
`YoloDatasetPreparer.java` reuses a YOLO folder or builds one from masks by
cropping around objects and tiling dense areas at `imgsz`.

[JDLL](../../../JDLL) (`../JDLL`, sibling checkout) is the Java deep-learning
library behind deepImageJ. It already drives Python through Appose, so the
Python half of its integrations is reusable even though the host is Java.
This file records what we find as we read it.

## Yolo.java

`src/main/java/io/bioimage/modelrunner/model/special/yolo/Yolo.java`, ~1000
lines. `Yolo extends DLModelPytorchProtected`. Wraps ultralytics YOLO for
inference and training.

### How it talks to Appose

The Python is **built as a Java string** — every line concatenated with
`+ nl` — and sent as one `python.task(code)`. No `.py` files ship; parameters
are spliced in with `r'...'` literals, and a dict of free-form train options
travels as base64'd JSON to dodge quoting.

```java
PixiEnvironmentSpec envSpec = resolvePytorchEnv();
Environment env = Appose.pixi()
        .environment(envSpec.getSelectedEnvironment())
        .wrap(envSpec.getEnvironmentDirectory());
Service python = env.python();
Task task = python.task(buildTrainingCode(...));
task.listen(event -> handleTrainingEvent(event, progress, preview, log));
task.waitFor();
```

Two code builders:

- `LOAD_MODEL_CODE_ABSTRACT` (line ~109) — picks a device (cuda / mps / cpu),
  `YOLO(weights)`, then `task.export(...)` the model, torch, device, etc. so
  later tasks in the same service reuse them. Returns `model.names` as the
  class mapping. This is model residency the way 0019 means it.
- `buildTrainingCode(...)` (line ~596) — the training script, ~280 lines of
  Python. The interesting part.

### The training script, as Python

[jdll-yolo-train.py](jdll-yolo-train.py) is a reading copy: roughly what the
generated script looks like, with example values filled in and the
`TrainingCodeUtils` helpers inlined. Stripped of the Java, it is an ordinary ultralytics training run plus
callbacks that report through `task.update`:

| ultralytics callback | what it reports |
| --- | --- |
| `on_train_start` | total steps; runtime info (versions, resolved device, batch) |
| `on_train_epoch_start` | runtime info again |
| `on_train_batch_end` | per-step progress + losses; checks for cancel |
| `on_train_epoch_end` | decides whether this epoch captures a preview |
| `on_val_start` | wraps `validator.update_metrics` to capture predictions |
| `on_val_end` | writes preview boxes to JSON, emits a `preview` event |
| `on_fit_epoch_end` | epoch metrics; tracks best mAP50-95 |

Every `info` payload carries a `type` field — `runtime`, `progress`,
`preview`, `checkpoint` — and the Java side dispatches on it.

Other things it does that we would need too:

- **Cancel** — the Java side touches a file; the step callback checks
  `os.path.exists(cancel_signal_path)` and sets `trainer.stop`. Crude, works
  across the process boundary without anything from Appose.
- **Logging** — ultralytics' `LOGGER` and stdout/stderr redirected to
  `run_dir/training.log`, so the Appose stdout channel stays clean.
- **Validation preview** — a fixed random sample (≤100) of val images; every N
  epochs their predicted boxes (conf ≥ 0.25) go to
  `validation_preview/epoch_NNNN.json` and `latest.json`. The GUI draws them.
- **Outputs** — copies best (else last) checkpoint to the requested path;
  returns versions, resolved device/batch, completed/best epoch, metrics.
- **Cleanup** — `finally` deletes model/trainer and frees torch memory.

Helpers live in `TrainingCodeUtils` (not yet read): `apposeStdoutCapture`,
`taskUpdateFunction`, `scalarFunction`, `cleanDictFunction`,
`pytorchMemoryCleanupFunction`.

### Inference

[jdll-yolo-infer.py](jdll-yolo-infer.py) is the reading copy. Two tasks on
one long-lived service:

1. **Load**, once — `YOLO(weights)`, `task.export(model=...)`, return
   `model.names`.
2. **Infer**, per image or tile — attach to the input Java wrote into shared
   memory, convert axes (`ConvertDims`), letterbox, `model(tensor)`, pack the
   results into a dense `(B, max_boxes, 6)` float32 array
   `[x1, y1, x2, y2, conf, cls]` in fresh shared memory (unused rows zero),
   undo the letterbox, return the shm name/dtype/shape.

The Python helpers live in `model/python/methods/`, one Java class each:
`ConvertDims` (axis letters → `bcyx` torch, grey → 3 channels),
`LetterboxPreprocessing` (resize longest side to 640, pad to stride 32, 2-D
and 3-D), `UndoLetterboxProcessingBoundingBoxes`. Names get a random UUID
suffix so repeated tasks in one worker never collide.

Things worth noticing:

- **The letterbox is hand-written because the input is a torch tensor.**
  Given a tensor, ultralytics skips its own preprocessing, so JDLL redoes it.
  Given a numpy HWC image, ultralytics letterboxes and rescales boxes itself —
  which is what our `object_aware_yolo` op does, in about ten lines. Worth
  checking: ultralytics expects a tensor in 0–1, and the pad value here is
  114, which is the 0–255 convention. Either the Java side scales, or this is
  a latent bug. Not yet checked.
- **Tiling is on the Java side** — `getTileMaker` returns a `DetectionMerger`
  sized by `setObjectSize(Rectangle)`, merging boxes across tiles. That is
  the interesting part for [0017](../0017-memory-and-tiled-processing/);
  not yet read.
- **Shared memory by hand.** JDLL builds `SharedMemory` objects in the
  script and passes names back. skop gets this from Appose's `NDArray`
  without writing it; nothing to take here.
- **2-D only** — `// TODO add 3D`, inputs must be `xy` or `xyc`.

As an op, inference is small: it is `object_aware_yolo` with a user-supplied
`.pt` instead of a fixed download, plus `cls` in the result. The thing to take
from JDLL is not this script but the detection merger, if tiling is wanted.

### Training parameters — `YoloTrainingConfig`

`gui/custom/yolo/YoloTrainingConfig.java` is the GUI's training config. It
separates three kinds of parameter, and that split is the useful part:

**What the user picks** (constructor arguments, all shown in the GUI):

| JDLL | meaning | default |
| --- | --- | --- |
| `modelName` | name of the run / output | — |
| `datasetYamlPath` | ultralytics `data.yaml` | — |
| `epochs` | | 100 |
| `imageSize` | `imgsz` | 640 |
| `fineTune` + `baseModelPath` | start from a `.pt` | — |
| `scratchArchitecture` | else start from `yolo26{n,s,m,l,x}.yaml` | `yolo26n.yaml` |
| `modelsDir`, `outputWeightsPath` | where it goes | — |
| `previewEpochPeriod` | validation preview every N epochs | 1 |
| `device` | cpu / cuda / mps | cpu |

**The ultralytics pass-through** (`defaultTrainingOptions`), about 45 keys.
These are ultralytics' own defaults, written out so they get recorded, and a
per-architecture config file can override them:

- optimiser: `optimizer=auto`, `lr0=0.01`, `lrf=0.01`, `momentum=0.937`,
  `weight_decay=5e-4`, `warmup_*`, `cos_lr`, `patience=100`, `nbs=64`
- loss weights: `box=7.5`, `cls=0.5`, `dfl=1.5`
- schedule/data: `close_mosaic=10`, `cache`, `rect`, `multi_scale`,
  `fraction`, `single_cls`, `classes`, `freeze`, `amp`, `compile`
- validation: `val`, `split`, `conf`, `iou=0.7`, `max_det=300`
- `seed=0`, `deterministic=True`
- **augmentations** (flattened into the train call): `hsv_h/s/v`, `degrees`,
  `translate=0.1`, `scale=0.5`, `shear`, `perspective`, `flipud=0`,
  `fliplr=0.5`, `bgr`, `mosaic=1.0`, `mixup`, `cutmix`, `copy_paste`

**What the runtime owns**, stripped from user options even if a config file
sets them: `data, model, project, name, device, workers, exist_ok, verbose,
plots, resume, save, save_period`, plus `batch`, which is handled separately:
`"auto"` or an int, with **OOM retry** that halves it down to `minimum_batch`.

**What gets recorded.** `config.json` beside the weights: framework,
software versions (jdll, python, ultralytics, torch, torchvision), model (start
mode, source), dataset path, the full training map, and requested vs
**resolved** device and batch. It's provenance, the same thing the recorder
branch is after.

#### A first guess at the skop op

Following `train_cellpose3`'s shape: the user-facing set as named
parameters, and the rest left at ultralytics' defaults rather than spelled
out.

```python
@op(env="pytorch")                  # already has ultralytics, for fastsam
def train_yolo(
    images: list[str],
    labels: list[str],              # 0011: paths in, not a data.yaml
    model_dir: str,
    name: str = "yolo_model",
    architecture: Architecture = Architecture.yolo26n,  # from scratch
    initial_model: str = "",        # a .pt to fine-tune; wins over architecture
    epochs: int = 100,
    imgsz: int = 640,               # also the inference tile scale -- see yolo-sam-tiling.md
    batch_size: int = 16,           # or auto; OOM retry is a runner question
    learning_rate: float = 0.01,    # lr0
    patience: int = 100,
    single_cls: bool = True,        # class-agnostic, as 0007 wants
    fliplr: float = 0.5,
    flipud: float = 0.5,            # microscopy has no "up"; ultralytics says 0
    degrees: float = 0.0,
    mosaic: float = 1.0,
    val_fraction: float = 0.2,
    seed: int = 0,
) -> str:                           # path to best.pt
```

Choices made here, all open:

- **Paths, not `data.yaml`.** 0011 says training ops take lists of paths.
  That means the op writes the YOLO dataset layout itself, which is what
  JDLL's `YoloDatasetPreparer` does (not yet read). Labels would be label
  images converted to boxes, or box files.
- **`single_cls=True`** by default, since our detectors are class-agnostic.
- **`flipud` on.** Microscopy is rotation-invariant; ultralytics' default
  of 0 is for photos.
- Only a few augmentations are exposed; the rest wait for the augment ops
  in [0015](../0015-augment-ops.md).
- No `device`. skop ops pick it themselves (see `object_aware_yolo`).
- Preview, cancel, and runtime info come back through skop's progress
  channel, not as parameters.

### What this means for skop

- The Python is liftable almost as-is into an op body: the callbacks become a
  `train/yolo` op under [0011](../0011-deep-learning-training-ops.md),
  reporting through our progress channel instead of hand-rolled `task.update`.
- The `type`-tagged progress events are a schema worth comparing against ours
  before inventing more.
- File-based cancel is a fallback worth knowing about if Appose cancel does
  not reach into a running trainer.

## UNet, Cellpose, and MONAI — next

**We want UNet inference and training too.** JDLL has the same split for
UNet as for YOLO:

- `model/special/unet/Unet.java` — the model wrapper and Appose code; read
  it the way we read `Yolo.java`.
- `gui/custom/unet/UnetTrainingConfig.java` — the training parameters, the
  counterpart of `YoloTrainingConfig`.
- Also `UnetBackend`, `UnetDatasetInspector`, `UnetValidationPreview`,
  `UnetTrainingProgress`, `UnetInferenceService`.

**Cellpose has no training config.** `model/special/cellpose/Cellpose.java`
and `gui/custom/cellpose/*` are inference only, as far as the file names show.
We already train Cellpose (`ops/train/cellpose3.py`, `cellpose4.py`), so
there's nothing to take there.

**MONAI UNet is already done in napari-ai-lab**, and it's the Python
reference to compare against JDLL's UNet:

- `../napari-ai-lab/src/napari_ai_lab/Segmenters/GlobalSegmenters/MonaiUNetSegmenter.py`
  and `MonaiUNetSegmenter3D.py`
- `datasets/pytorch_semantic_3d_dataset.py`, `utilities/resize_util.py`
- `tests/test_monai_unet.py`
- `design/proposed/0003-optional-dependencies.md` mentions it

The question for both: which parameters a `segment/unet` and a `train/unet`
op need, and whether MONAI's sliding-window inference covers the tiling in
[yolo-sam-tiling.md](../0017-memory-and-tiled-processing/yolo-sam-tiling.md) for the dense case (a UNet stitches,
it does not merge boxes).

## Still to read

- `TrainingCodeUtils` — the shared Python fragments.
- `resolvePytorchEnv` / `PixiEnvironmentSpec` — how JDLL defines and locates
  its pixi environments; compare with `envs/<id>/pixi.toml`.
- `DetectionMerger.toGlobalDetection` — whether edge-touching boxes are
  dropped before NMS. The rest of the merger is summarised in
  [yolo-sam-tiling.md](../0017-memory-and-tiled-processing/yolo-sam-tiling.md).
- `YoloDatasetPreparer` — how it builds the YOLO dataset layout from
  images and annotations.
- `TrainingConfigFiles` — the per-architecture config files that override
  the defaults.
- Whether the Java side scales input to 0–1 before inference (the 114 pad).
- `JDLL/python/` — whatever Python already lives as real files.
