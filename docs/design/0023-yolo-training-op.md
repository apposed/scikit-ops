# 0023 — Training YOLO

**Status:** built — `skop.ops.train.yolo.train_yolo`. Follows
[0011](0011-deep-learning-training-ops.md); the reading of JDLL it started
from is [jdll-reuse.md](../spec/jdll-reuse.md) and
[jdll-yolo-train.py](../spec/jdll-yolo-train.py).

## The template is ours, not JDLL's

JDLL trains YOLO well. Its Python is built as Java strings and reports
through seven ultralytics callbacks and a `type`-tagged event schema, with
cancel by touching a file, stdout swapped around every report, and a
`config.json` of provenance. Most of that machinery exists because Java is
the host. skop already has the parts it stands in for, and three training
ops that use them, so the YOLO op is shaped like those and takes from JDLL
only what they lack.

| | Cellpose 4 / StarDist | `train_yolo` |
| --- | --- | --- |
| Input | `images`, `labels`: paths, paired by position | same; labels are YOLO `.txt` |
| Validation | the last `val_size` pairs | same; `0` validates on the training set |
| Progress | `progress()` per epoch, from a Keras callback or a log line | `progress()` per epoch, from `on_fit_epoch_end` |
| Cancel | not yet | `cancel_requested()` checked in `on_train_batch_end`; sets `trainer.stop` |
| History | `<model>_history.csv`, continued when `initial_model` is the output | same, plus `mAP50`, `mAP50_95`, `precision`, `recall` |
| Returns | the model's path | `<model_dir>/<name>.pt` |

`write_history` gained `extra_columns` for the metrics; without them it
writes exactly what it did.

## What came from JDLL

- **Which arguments are the op's.** `project`, `name`, `exist_ok`,
  `device`, `workers=0`, `plots=False`, `verbose=False`, `deterministic`.
  The caller does not choose where training writes or how it runs.
- **Stdout belongs to the runner.** Ultralytics' logger is bound to stdout
  at import, and its progress bars write there too; Appose's protocol is on
  stdout. The op points `LOGGER` at `<name>_training/training.log`,
  redirects stdout to the same file while it trains, and puts the real
  stdout back for each `progress()` call. It is JDLL's `_task_update` trick,
  in five lines.

## What did not come from JDLL

- **The event schema, the validation preview, `config.json`.** None of the
  other trainers has them. If they are wanted, they are wanted for all the
  trainers at once. Ultralytics' own `args.yaml` and `results.csv` are left
  in the workspace, which covers most of the provenance.
- **OOM retry.** JDLL restarts Python with half the batch. Ultralytics 8.4
  does this itself in the first epoch -- halves the batch and restarts the
  epoch -- which is where it happens.
- **`YoloDatasetPreparer`.** 1900 lines of Java that crop and tile images
  for training. Cropping is augmentation, and belongs to the caller
  (0011, 0015) -- see *Bibs* below for one.
- **`yolo_data.py`'s normalisation.** JDLL now trains on 16-bit and TIFF
  images through a custom dataset that min/max-normalises. Not needed for
  photos, and taken when microscopy needs it -- it is already Python.

## Running what it trains: `yolo`

`skop.ops.detect.yolo` is the main YOLO detector, and the pair to this op:
any checkpoint, the whole image or overlapping tiles, at the size the model
was trained at, which it reads from the `.pt`. `fastsam` and
`object_aware_yolo` keep their names because they are special, each with its
own fixed model. `jdll_yolo`, a port of JDLL's inference (its hand-written
letterbox, one pass, no classes), was retired for `yolo`, which did all it
did. JDLL calls its own simply `Yolo`.

## The dataset is the one new thing

Cellpose and StarDist take arrays, so 0011's lists of paths go straight in.
Ultralytics only reads from disk, in its own layout: `images/train`,
`labels/train`, the same for `val`, and a `data.yaml`, with each label
found by swapping `images` for `labels` in the image's path. So the op
writes that layout under `<name>_training/dataset/` from the two lists:
images hard-linked (copied across drives), numbered so two `0001.png` from
different folders stay apart; labels rewritten, five columns, checked. A
sixth column, a prediction's confidence, is dropped, so a detector's own
output can be trained on as it is. A malformed line is an error naming it;
ultralytics would quietly skip the image.

This is ultralytics' convention inside the op, not a convention on the
caller, so 0011's decision stands.

## Small things found on the way

- **Stock weights.** `YOLO("yolo11n.pt")` downloads into the current
  directory, which in a worker is anywhere. A bare stock name is fetched
  into ophelper's cache instead. The AMP self-check downloads `yolo26n.pt`
  into the current directory too, so training runs with the workspace as
  the current directory, and stray `.pt` files are removed after -- a
  `models/` folder is listed for models, and should list only real ones.
- **One copy of the model.** `best.pt` is copied to `<name>.pt` and the
  run's `weights/` removed, for the same reason.
- **Few images learn slowly.** Ultralytics steps the optimiser every 64
  images (`nbs`), whatever the batch. One image with batch 1 steps once
  every 64 epochs. The one-image wiring check (train = val, after the 2024
  ladybug notebook) uses 64 copies; at 40 epochs it finds all 71 ladybugs
  and spots, mAP50 0.97.
- **The final check is not an epoch.** Ultralytics calls
  `on_fit_epoch_end` once more after training, to score `best.pt`, and
  after an early stop its epoch counter has moved on. The op records only
  epochs that `on_train_epoch_end` saw, once each.
- **`optimizer=auto` ignores `lr0`.** So `optimizer` is a parameter, and
  `learning_rate` says it is ignored under `auto`.

## Augmentation: the op's knobs are the inline ones

Ultralytics augments inline. The op exposes the ones a caller has a reason
to change -- `fliplr`, `flipud`, `degrees`, `scale`, `shear`,
`perspective`, `mosaic` -- at ultralytics' defaults, which are for photos of
things without a handedness. Microscopy wants `flipud=0.5`; text wants
`fliplr=0`.

Heavier augmentation is done before, by the caller, as 0011 says. The
precedents are albumentations with `format='yolo'`
(`tnia.deeplearning.augmentation.uber_augmenter_im_bb`, used by
napari-easy-augment-batch-dl's `perform_yolo_augmentation`, which also
paints out unlabelled regions with the mean).

### Bibs

The first real use: photo-enhance's `bib-patches` command writes crops of
race photos for a bib detector, and `scripts/train_bib_detector.py` there
runs this op through the runner. A bib is 40-80 px in a 4032 px photo, so
the crops zoom each bib to a random 16-160 px in a 1024 crop, with a little
perspective and stretch, light, blur and JPEG -- no flips, no rotation. Bibs
the labeller doubted are painted over rather than left as unlabelled
background, and validation is whole photos held out, not crops of photos
also trained on.

First run (2026-11-11, one race, 79 photos, 507 bibs from OWLv2 and OCR):
1175 crops, `yolo11m.pt`, `imgsz` 1024, batch 16, on an RTX 3090. Early
stopping ended it at epoch 41 of 80, best at 21: mAP50 0.63, mAP50-95 0.45
on 29 images from 8 held-out photos. Those scores are against OWLv2's boxes,
which are sometimes the card and sometimes just the digits, and miss bibs.
Looked at by eye, the model found about 28 of the 29 bibs in the held-out
photos, OWLv2 found 21 and boxed a house number. It made one false
detection. Hand-checked boxes are the next step, more than more epochs.

## Open

- **Data loading is the bottleneck.** `workers=0` (as JDLL) keeps
  ultralytics from spawning processes inside an Appose worker, but one thread
  reads, decodes and augments every image, every epoch: about 1 s a batch of
  16 at 1024 px, with the GPU mostly waiting. Two parameters answer it for a
  set augmented beforehand: `augment=False` turns off every inline
  augmentation (each image is only resized), and `cache` keeps decoded
  images between epochs (`ram`, `disk`). Together they leave nothing per
  batch on the CPU; the price is the same images every epoch. Not yet timed
  on the bibs. Whether worker processes are safe in a worker is untested.

- **The runner and an unreleased op.** The pytorch environment pins
  scikit-ops `0.1.0.dev2`, which has no `train_yolo`; a checkout's runner
  sends the op's folder (0022), so it runs from here, but an installed one
  needs a release.
- **Classes.** `class_names` names them; `single_cls` is not exposed. 0007's
  detectors are class-agnostic, but a trained detector usually is not.
- **Tiled training.** Training at `imgsz` and running tiled at the same
  scale (0017's `yolo` op) is the pairing to document once there is a model
  it matters for.
