# cellpose_worker.py

from imgops.implementations.cellpose import run_cellpose
import appose

labels = run_cellpose(
    image,
    diameter,
    flow_threshold,
    cellprob_threshold,
    use_gpu,
)

mask = appose.NDArray(
    dtype=str(labels.dtype),
    shape=labels.shape,
)

mask.ndarray()[:] = labels

task.outputs["mask"] = mask
