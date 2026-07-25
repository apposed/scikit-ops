# stardist_worker.py

from imgops.implementations.stardist2d import run_stardist
import appose

labels = run_stardist(
    image,
    model_name,
    prob_thresh,
    nms_thresh,
    normalize,
)

mask = appose.NDArray(
    dtype=str(labels.dtype),
    shape=labels.shape,
)

mask.ndarray()[:] = labels

task.outputs["mask"] = mask