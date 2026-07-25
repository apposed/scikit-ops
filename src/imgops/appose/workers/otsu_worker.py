# otsu_worker.py

from imgops.implementations.skimagessegmenter import run_otsu
import appose

labels = run_otsu(
    image,
    invert,
    label_objects,
)

mask = appose.NDArray(
    dtype=str(labels.dtype),
    shape=labels.shape,
)

mask.ndarray()[:] = labels

task.outputs["mask"] = mask
