# Approximately the script JDLL's Yolo.buildTrainingCode() sends to Appose,
# with the Java concatenation removed and example values filled in. Not run;
# a reading copy. See jdll-reuse.md. `task` is the Appose worker's global.

import base64
import contextlib
import json
import logging
import os
import platform
import random
import shutil
import sys

_appose_stdout = sys.stdout  # TrainingCodeUtils.apposeStdoutCapture(), roughly

import torch
import torchvision
import ultralytics
from ultralytics import YOLO
from ultralytics.utils import LOGGER

# ---- parameters, spliced in by Java as r'...' literals ----------------------
model_source = r"yolo26n.yaml"  # or an absolute path to a .pt to fine-tune
dataset_yaml = r"/data/cells/data.yaml"
output_weights = r"/models/cells/cells.pt"
project = r"/models"
run_name = r"cells"
epochs = 50
imgsz = 640
batch_size = 16  # batchLiteral(): an int, or 'auto' resolved per device
train_options = json.loads(base64.b64decode("e30=").decode("utf-8"))  # {} here
preview_epoch_period = 5
preview_sample_count = 100
preview_confidence = 0.25
cancel_signal_path = r"/tmp/jdll-cancel-1234"
requested_device = "cuda"
# ---------------------------------------------------------------------------

train_device = 0 if requested_device == "cuda" else requested_device
is_accelerated = requested_device != "cpu"
progress_every_n_steps = 5 if is_accelerated else 1
log_every_n_steps = 50 if is_accelerated else 10

os.makedirs(project, exist_ok=True)
run_dir = os.path.join(project, run_name)
preview_dir = os.path.join(run_dir, "validation_preview")
preview_json_path = os.path.join(preview_dir, "latest.json")
preview_manifest_path = os.path.join(preview_dir, "manifest.json")
os.makedirs(preview_dir, exist_ok=True)

# ultralytics' logger goes to a file, so Appose's stdout channel stays clean
yolo_log_path = os.path.join(run_dir, "training.log")
_handler = logging.FileHandler(yolo_log_path, mode="a", encoding="utf-8")
_handler.setFormatter(logging.Formatter("%(message)s"))
LOGGER.handlers = [_handler]
LOGGER.propagate = False
LOGGER.setLevel(logging.INFO)

state = {
    "step": 0, "total_steps": 0,
    "preview_paths": set(), "preview_order": [], "preview_results": {},
    "preview_epoch": 0, "capture_preview": False,
    "completed_epochs": 0, "best_epoch": 0, "best_score": None,
    "best_metrics": {}, "final_metrics": {},
    "resolved_batch": None, "resolved_device": None,
}


# ---- TrainingCodeUtils helpers ---------------------------------------------
def _task_update(**kwargs):
    # stdout is redirected during model.train(); Appose needs the real one
    current = sys.stdout
    try:
        sys.stdout = _appose_stdout
        task.update(**kwargs)
    finally:
        sys.stdout = current


def _scalar(value):
    try:
        if hasattr(value, "detach"):
            value = value.detach().cpu()
        if hasattr(value, "item"):
            value = value.item()
        return float(value)
    except Exception:
        return None


def _clean_dict(values):
    return {str(k): s for k, v in dict(values or {}).items() if (s := _scalar(v)) is not None}


def _cleanup_pytorch_memory():
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()


# ---- cancel: Java touches a file, we notice it between batches ---------------
def _cancel_requested():
    return bool(cancel_signal_path) and os.path.exists(cancel_signal_path)


def _request_stop(trainer):
    trainer.stop = True
    trainer.epochs = min(int(trainer.epochs), int(trainer.epoch) + 1)


# ---- progress reporting ------------------------------------------------------
def _losses(trainer):
    try:
        return _clean_dict(trainer.label_loss_items(trainer.tloss, prefix="train"))
    except Exception:
        return {}


def _metrics(trainer):
    return _clean_dict(getattr(trainer, "metrics", {}))


def _runtime_info(trainer):
    state["resolved_batch"] = int(trainer.batch_size)
    state["resolved_device"] = str(trainer.device)
    return {
        "type": "runtime",
        "python_version": platform.python_version(),
        "ultralytics_version": ultralytics.__version__,
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "resolved_device": state["resolved_device"],
        "resolved_batch": state["resolved_batch"],
    }


def _emit_runtime(trainer):
    _task_update(info=_runtime_info(trainer))


def _emit_train_start(trainer):
    state["total_steps"] = int(trainer.epochs) * len(trainer.train_loader)
    runtime = _runtime_info(trainer)
    _task_update(info=runtime)
    _task_update(
        message="YOLO training started", current=0, maximum=state["total_steps"],
        info={"type": "progress", "epoch": 0, "step": 0, "total_epochs": epochs,
              "total_steps": state["total_steps"], "losses": {}, "metrics": {}},
    )


def _emit_step_progress(trainer):
    state["step"] += 1
    if _cancel_requested():
        _request_stop(trainer)
    step, total = state["step"], state["total_steps"]
    if not (step == 1 or step % progress_every_n_steps == 0 or step == total):
        return
    epoch = int(trainer.epoch) + 1
    losses = _losses(trainer)
    kwargs = {"current": step, "maximum": total,
              "info": {"type": "progress", "epoch": epoch, "step": step,
                       "total_epochs": epochs, "total_steps": total,
                       "losses": losses, "metrics": {}}}
    if step == 1 or step % log_every_n_steps == 0 or step == total:
        kwargs["message"] = f"YOLO training step {step}/{total}"
        print(f"step {step:05d}/{total} epoch={epoch}/{epochs} losses={losses}", flush=True)
    _task_update(**kwargs)


def _emit_epoch_progress(trainer):
    if _cancel_requested():
        _request_stop(trainer)
    epoch = int(trainer.epoch) + 1
    metrics = _metrics(trainer)
    state["completed_epochs"] = epoch
    state["final_metrics"] = metrics
    score = metrics.get("metrics/mAP50-95(B)")
    if score is not None and (state["best_score"] is None or score > state["best_score"]):
        state.update(best_score=score, best_epoch=epoch, best_metrics=dict(metrics))
    _task_update(
        message=f"YOLO training epoch {epoch}/{epochs}",
        current=state["step"], maximum=state["total_steps"],
        info={"type": "progress", "epoch": epoch, "epoch_complete": True,
              "step": state["step"], "total_epochs": epochs,
              "losses": _losses(trainer), "metrics": metrics},
    )


# ---- validation preview: boxes on a fixed sample of val images ---------------
def _prepare_preview_epoch(trainer):
    epoch = int(trainer.epoch) + 1
    state["preview_epoch"] = epoch
    state["capture_preview"] = epoch % preview_epoch_period == 0 or epoch == epochs
    state["preview_results"] = {}


def _ensure_preview_sample(validator):
    if not state["preview_order"]:
        files = [str(x) for x in validator.dataloader.dataset.im_files]
        if len(files) > preview_sample_count:
            files = random.sample(files, preview_sample_count)
        state["preview_order"] = files
        state["preview_paths"] = set(files)
        with open(preview_manifest_path, "w", encoding="utf-8") as f:
            json.dump({"images": files}, f)
    return state["preview_paths"]


def _capture_preview_predictions(validator, preds, batch):
    if not state["capture_preview"]:
        return
    selected = _ensure_preview_sample(validator)
    for si, pred in enumerate(preds):
        path = str(batch["im_file"][si])
        if path not in selected:
            continue
        # ultralytics internals -- the fragile part
        pbatch = validator._prepare_batch(si, batch)
        predn = validator.scale_preds(validator._prepare_pred(pred), pbatch)
        boxes = [
            {"xyxy": [float(x) for x in box], "confidence": float(conf),
             "class_id": int(cls), "class_name": str(validator.names[int(cls)])}
            for box, conf, cls in zip(predn["bboxes"].cpu().tolist(),
                                      predn["conf"].cpu().tolist(),
                                      predn["cls"].cpu().tolist())
            if conf >= preview_confidence
        ]
        state["preview_results"][path] = {"path": path, "boxes": boxes}


def _on_val_start(validator):
    # monkey-patch update_metrics once, to see predictions as they go by
    _ensure_preview_sample(validator)
    if getattr(validator, "_deepimagej_preview_wrapped", False):
        return
    original = validator.update_metrics

    def wrapped(preds, batch):
        result = original(preds, batch)
        _capture_preview_predictions(validator, preds, batch)
        return result

    validator.update_metrics = wrapped
    validator._deepimagej_preview_wrapped = True


def _emit_preview_results(validator):
    if not state["capture_preview"]:
        return
    images = [state["preview_results"].get(p, {"path": p, "boxes": []})
              for p in state["preview_order"]]
    payload = {"epoch": state["preview_epoch"],
               "confidence_threshold": preview_confidence, "images": images}
    epoch_path = os.path.join(preview_dir, "epoch_%04d.json" % payload["epoch"])
    for path in (epoch_path, preview_json_path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f)
    _task_update(message=f"YOLO validation preview epoch {payload['epoch']}",
                 current=payload["epoch"], maximum=epochs,
                 info={"type": "preview", "epoch": payload["epoch"], "preview_path": epoch_path})
    state["capture_preview"] = False


# ---- the actual work ---------------------------------------------------------
try:
    model = YOLO(model_source)
    model.add_callback("on_train_start", _emit_train_start)
    model.add_callback("on_train_epoch_start", _emit_runtime)
    model.add_callback("on_train_batch_end", _emit_step_progress)
    model.add_callback("on_train_epoch_end", _prepare_preview_epoch)
    model.add_callback("on_val_start", _on_val_start)
    model.add_callback("on_val_end", _emit_preview_results)
    model.add_callback("on_fit_epoch_end", _emit_epoch_progress)

    train_args = dict(train_options)
    train_args.update(data=dataset_yaml, epochs=epochs, imgsz=imgsz, batch=batch_size,
                      project=project, name=run_name, exist_ok=True, verbose=False,
                      plots=False, seed=0, deterministic=True, workers=0,
                      nbs=64, device=train_device)
    with open(yolo_log_path, "a", encoding="utf-8") as log, \
            contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        results = model.train(**train_args)

    trainer = model.trainer
    best, last = str(trainer.best), str(trainer.last)
    for kind, path in (("last", last), ("best", best)):
        _task_update(message=f"YOLO {kind} checkpoint: {path}",
                     info={"type": "checkpoint", "kind": kind, "path": path})
    source = best if os.path.isfile(best) else last
    if not os.path.isfile(source):
        raise RuntimeError("Could not find YOLO training checkpoint to save.")
    os.makedirs(os.path.dirname(output_weights), exist_ok=True)
    shutil.copy2(source, output_weights)
    _task_update(message=f"Exported/final YOLO model file: {output_weights}",
                 info={"type": "checkpoint", "kind": "final", "path": output_weights})

    task.outputs["result"] = output_weights
    task.outputs["python_version"] = platform.python_version()
    task.outputs["ultralytics_version"] = ultralytics.__version__
    task.outputs["torch_version"] = torch.__version__
    task.outputs["resolved_device"] = state["resolved_device"]
    task.outputs["resolved_batch"] = state["resolved_batch"]
    task.outputs["completed_epochs"] = state["completed_epochs"]
    task.outputs["best_epoch"] = state["best_epoch"]
    task.outputs["best_metrics"] = state["best_metrics"]
    task.outputs["final_metrics"] = state["final_metrics"]
finally:
    for name in ("results", "trainer", "model"):
        globals().pop(name, None)
    _cleanup_pytorch_memory()
