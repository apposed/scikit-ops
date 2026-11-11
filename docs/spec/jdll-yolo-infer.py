# Approximately the two scripts JDLL's Yolo sends to Appose for inference,
# with the Java concatenation removed and example values filled in. Not run;
# a reading copy. See jdll-reuse.md. `task` is the Appose worker's global.
#
# JDLL suffixes every helper and variable with a random UUID (model_3f2a...,
# convert_dims_9b1c...) so repeated tasks in one worker never collide; the
# names below drop that.

# =============================================================================
# Task 1 -- load, once per service (LOAD_MODEL_CODE_ABSTRACT)
# =============================================================================
import os
from multiprocessing import shared_memory

import torch

use_cuda, use_mps = True, False  # spliced in as True/False from the device
device = "cpu"
if use_cuda and torch.cuda.is_available():
    device = "cuda"
elif use_mps:
    from torch.backends import mps
    if mps.is_built() and mps.is_available():
        device = "mps"

from ultralytics import YOLO

model = YOLO(r"/models/cells/cells.pt")

# export = keep alive in the worker for later tasks (model residency, 0019)
task.export(shared_memory=shared_memory, YOLO=YOLO, torch=torch, os=os,
            device=device, model=model)
task.outputs["mappings"] = model.names  # {0: 'cell', 1: 'debris', ...}


# =============================================================================
# Task 2 -- inference, once per image or tile (createInputsCode)
# =============================================================================
import numpy as np
import torch.nn.functional as F


def convert_dims(x, in_order, *, out_order="bcyx", n_channels=3, dtype=np.float32,
                 device=None):
    """Reorder axes by letter, drop/add axes, force n_channels, return torch."""
    x = np.asarray(x)
    for i in range(len(in_order) - 1, -1, -1):  # drop axes not wanted (take index 0)
        if in_order[i] not in out_order:
            x = np.take(x, 0, axis=i)
            in_order = in_order[:i] + in_order[i + 1:]
    for ax in out_order:  # add missing axes at the end
        if ax not in in_order:
            x = np.expand_dims(x, -1)
            in_order += ax
    x = np.transpose(x, [in_order.index(ax) for ax in out_order])
    c_idx = out_order.index("c")
    if x.shape[c_idx] == 1:  # grey -> RGB by broadcast
        shape = list(x.shape)
        shape[c_idx] = n_channels
        x = np.broadcast_to(x, shape)
    x = np.ascontiguousarray(x.astype(dtype))
    return torch.as_tensor(x).to(device=device, non_blocking=True)


def letterbox(x, *, target_max_side=640, stride=32, padding_value=114.0):
    """Resize longest side to 640, pad Y/X to a multiple of 32, centred."""
    _, _, y, xdim = x.shape
    scale = target_max_side / max(y, xdim)
    new_y, new_x = round(y * scale), round(xdim * scale)
    x = F.interpolate(x, size=(new_y, new_x), mode="bilinear", align_corners=False)
    pad_y, pad_x = (-new_y) % stride, (-new_x) % stride
    top, left = pad_y // 2, pad_x // 2
    bottom, right = pad_y - top, pad_x - left
    x = F.pad(x, (left, right, top, bottom), value=padding_value)
    meta = {"scale": scale, "pad": (top, bottom, left, right), "original_hw": (y, xdim)}
    return x.contiguous(), meta


def undo_letterbox_boxes(boxes, meta):
    """(B, N, 6) xyxy+conf+cls, letterboxed coords -> original image coords."""
    top, _, left, _ = meta["pad"]
    orig_y, orig_x = meta["original_hw"]
    s = meta["scale"]
    boxes[:, :, [0, 2]] = np.clip((boxes[:, :, [0, 2]] - left) / s, 0, orig_x)
    boxes[:, :, [1, 3]] = np.clip((boxes[:, :, [1, 3]] - top) / s, 0, orig_y)
    return boxes


# ---- input: Java wrote the image into shared memory; attach to it -----------
created_shms.clear()
input_shm = shared_memory.SharedMemory(name="/psm_8a1b2c3d", size=512 * 512 * 3 * 4)
created_shms.append(input_shm)
input = np.ndarray(512 * 512 * 3, dtype="float32", buffer=input_shm.buf).reshape([512, 512, 3])

# axes come from the Java Tensor: 'xyc' (or 'xy', padded to xyc by Java)
input_torch, meta = letterbox(convert_dims(input, "xyc", device=device))

# a torch tensor goes straight in, so ultralytics' own letterbox is skipped --
# hence the hand-written one above
out_list = model(input_torch, device=device)

# ---- output: a dense (B, max_boxes, 6) float32 array in new shared memory ---
shms, shm_names, dtypes, dims = [], [], [], []
task.export(shms=shms, shm_names=shm_names, dtypes=dtypes, dims=dims)

max_box = max(1, max(0 if r.boxes is None else len(r.boxes) for r in out_list))
shm = shared_memory.SharedMemory(create=True, size=len(out_list) * max_box * 6 * 4)
box_tensor = np.ndarray((len(out_list), max_box, 6), dtype="float32", buffer=shm.buf)
box_tensor.fill(0)  # unused rows stay zero: "no box" is conf == 0
shms.append(shm)
shm_names.append(shm.name)
dtypes.append(str(box_tensor.dtype))
dims.append(box_tensor.shape)

for i_r, r in enumerate(out_list):
    xyxy = r.boxes.xyxy.cpu().numpy()
    conf = r.boxes.conf.cpu().numpy()
    cls = r.boxes.cls.cpu().numpy()
    for i_b, (box, c, k) in enumerate(zip(xyxy, conf, cls)):
        box_tensor[i_r, i_b, :4] = box
        box_tensor[i_r, i_b, 4] = c
        box_tensor[i_r, i_b, 5] = k

box_tensor = undo_letterbox_boxes(box_tensor, meta)

# Java reads the array back out of shared memory by name; a later task closes it
task.outputs["shm_names"] = list(shm_names)
task.outputs["dtypes"] = list(dtypes)
task.outputs["dims"] = list(dims)
