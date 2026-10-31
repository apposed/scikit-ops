# Writing ops

An op is a plain Python function.

## The decorator

```python
from opspec.op import Overlap, PeakMemory, op

@op                                  # runs in host environment
@op(env="skimage")                   # runs in a worker environment
@op(env="pytorch", exclusive=True)   # runs in it's own worker environment 
@op(env="napari", main_thread=True)  # needs the main thread
```

Tiling hints are optional. The runner can use this information to tile an
input that the op cannot process in memory. Note that it is not just a matter
of the input fitting in memory, but also the other parameters and any computation
buffers (for example FFTs, conv-nets, etc.) that the algorithm needs.

Tiling uses the following:

- `tile`: the inputs that will be tiled the same way
- `split`: the axes for tiling (default: any axis)
- `overlap`: how far each tile goes past the edge, in pixels or as a
  function of a parameter
- `peak_memory`: how many times the input size the op needs for computation
  buffers
- `merge`: how the tiles are put back together, `"crop"` (keep each tile's
  core, drop the overlap) or `"blend"` (neighbours mixed)

Example op decorators with tiling. A simple one:

```python
@op(env="skimage", tile="image", overlap=16, merge="crop")
def smooth(image: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    return skimage.filters.gaussian(image, sigma)
```

And one with all the hints:

```python
@op(
    env="cupy",
    tile=("image", "mask"),         # list of inputs that will be tiled
    split=("y", "x"),               # axes for tiling
    overlap=Overlap(param="psf", of="shape", scale=0.5),  # an overlap that is a function of a parameter
    peak_memory=PeakMemory(16, "float32", device="gpu"),  # op needs 16 times input for computational buffers
    merge="blend",                  # blend using merge
)
```

## Annotations

An array parameter has two kinds of type. The **array type** is the
container the op receives: `np.ndarray` for numpy only, or `Array` for any
array (numpy, cupy, dask, zarr, xarray). The **dtype** is the kind of number
inside it: `uint8`, `uint16`, `float32` and so on.

| You write | It means |
|---|---|
| `ImageOf[np.ndarray]` | an image that uses numpy array |
| `ImageOf[Array]` | an image of an Array type; the op uses only `shape`, `dtype` and indexing |
| `ImageOf[MyArrayProtocol]` | an image that uses a hypothetical array protocol |
| `LabelsOf[np.ndarray]` | a label that uses numpy array |
| `MasksOf`, `BoxesOf`, `PointsOf` | other roles |
| `Annotated[np.ndarray, Axes("y", "x")]` | 2-D; stacks are looped |
| `Axes("y", "x", "c?")` | optional channel axis |
| `Axes(variadic=True)` | any number of axes |
| `Annotated[ImageOf[np.ndarray], Dtype("uint8", "uint16")]` | only these dtypes |
| `Annotated[float, {"min": 0, "max": 10}]` | a slider's range |
| an `Enum` | a dropdown |
| `Path` | a file picker |
| `X \| None = None` | optional input |
| `Out[ImageOf[np.ndarray]]` | caller allocates the output |
| `Mut[ImageOf[np.ndarray]]` | changed in place |
| `-> NamedTuple` | several named outputs |
| `-> LabelsOf[np.ndarray] \| None` | output may be missing |
| docstring `Args:` | tooltips |
| `Choices(cpu=..., gpu=...)` | pick an op |
| `ParamsFor("decon_op")` | the picked op's settings |

## Examples

### The smallest op

```python
import numpy as np
from opspec.op import op
from opspec.types import ImageOf

@op(env="skimage")
def smooth(image: ImageOf[np.ndarray], sigma: float = 2.0) -> ImageOf[np.ndarray]:
    """Gaussian blur.

    Args:
        image: The image to blur.
        sigma: Blur width, in pixels.
    """
    from skimage.filters import gaussian  # heavy imports go inside

    return gaussian(image, sigma)
```

### Any array type: one plane, read lazily

```python
from typing import Annotated

from opspec.op import Axes, op
from opspec.types import Array, ImageOf

@op
def middle_plane(
    image: Annotated[ImageOf[Array], Axes("z", "y", "x")],
) -> ImageOf[Array]:
    """The middle z plane. The op only indexes, so a zarr or dask volume
    is never loaded whole."""
    return image[image.shape[0] // 2]
```

### Labels out, with a dropdown and a slider

```python
from enum import Enum
from typing import Annotated

import numpy as np
from opspec.op import op
from opspec.types import ImageOf, LabelsOf

class Method(Enum):
    otsu = "otsu"
    li = "li"

@op(env="skimage")
def threshold(
    image: ImageOf[np.ndarray],
    method: Method = Method.otsu,
    blur: Annotated[float, {"min": 0.0, "max": 10.0}] = 1.0,
) -> LabelsOf[np.ndarray]:
    from skimage import filters, measure

    smoothed = filters.gaussian(image, blur)
    cutoff = getattr(filters, f"threshold_{method.value}")(smoothed)
    return measure.label(smoothed > cutoff)
```

### A 2-D op, run on stacks

```python
from typing import Annotated

import numpy as np
from opspec.op import Axes, op
from opspec.types import ImageOf, LabelsOf

@op(env="stardist")
def nuclei(
    image: Annotated[ImageOf[np.ndarray], Axes("y", "x")],
    probability: Annotated[float, {"min": 0.0, "max": 1.0}] = 0.5,
) -> LabelsOf[np.ndarray]:
    from stardist.models import StarDist2D

    model = StarDist2D.from_pretrained("2D_versatile_fluo")
    labels, _ = model.predict_instances(image, prob_thresh=probability)
    return labels
```

### Two outputs, one optional

```python
from typing import Annotated, NamedTuple

import numpy as np
from opspec.op import Axes, op
from opspec.types import ImageOf, LabelsOf

class NucleiAndCells(NamedTuple):
    nuclei: LabelsOf[np.ndarray]
    cells: LabelsOf[np.ndarray] | None  # None: nothing to show

@op(env="pytorch")
def nuclei_and_cells(
    image: Annotated[ImageOf[np.ndarray], Axes("y", "x", "c?")],
    find_cells: bool = True,
) -> NucleiAndCells:
    nuclei = np.zeros(image.shape[:2], np.int32)  # the model goes here
    cells = nuclei.copy() if find_cells else None
    return NucleiAndCells(nuclei, cells)
```

### Weights from a file

```python
from pathlib import Path

import numpy as np
from opspec.op import op
from opspec.types import ImageOf, LabelsOf

@op(env="pytorch")
def custom_model(image: ImageOf[np.ndarray], weights: Path) -> LabelsOf[np.ndarray]:
    import torch

    model = torch.load(weights, weights_only=False)
    return model(image)
```

### Writing into the caller's array

```python
from typing import Annotated

import numpy as np
from opspec.op import Dtype, Out, op
from opspec.types import ImageOf

UNSIGNED = Dtype("uint8", "uint16")  # max - image stays positive

@op(env="skimage")
def invert(
    image: Annotated[ImageOf[np.ndarray], UNSIGNED],
    result: Out[Annotated[ImageOf[np.ndarray], UNSIGNED]],
) -> None:
    result[...] = image.max() - image
```

### Tiled on the GPU

```python
from typing import Annotated

import numpy as np
from opspec.op import Axes, Dtype, Overlap, PeakMemory, op
from opspec.types import ImageOf

HALF_PSF = Overlap(param="psf", of="shape", scale=0.5)

@op(
    env="cupy",
    tile="image",
    split=("y", "x"),
    overlap=HALF_PSF,
    peak_memory=PeakMemory(16, "float32", device="gpu"),
    pad=HALF_PSF,  # the FFT pads by half the PSF too
    merge="blend",
)
def deconvolve(
    image: Annotated[ImageOf[np.ndarray], Axes("z?", "y", "x"), Dtype("float32")],
    psf: Annotated[np.ndarray, Dtype("float32")],
    iterations: int = 10,
) -> ImageOf[np.ndarray]:
    import cupy as cp

    ...  # Richardson-Lucy on the GPU
    return image
```

### A workflow: ops calling ops

```python
from collections.abc import Callable
from typing import Annotated

import numpy as np
from my_ops import deconvolve, gaussian_psf  # ops from above
from opspec.op import Choices, ParamsFor, op
from opspec.types import ImageOf

@op  # no environment: runs where it's called
def deconvolve_with_psf(
    image: ImageOf[np.ndarray],
    psf_op: Annotated[Callable, Choices(gaussian=gaussian_psf)] = gaussian_psf,
    psf_args: Annotated[dict | None, ParamsFor("psf_op")] = None,
    decon_op: Annotated[Callable, Choices(cpu=deconvolve)] = deconvolve,
    decon_args: Annotated[dict | None, ParamsFor("decon_op", binds=("image", "psf"))] = None,
) -> ImageOf[np.ndarray]:
    from opspec.runtime import run

    psf = run(psf_op, **(psf_args or {}))
    return run(decon_op, image=image, psf=psf, **(decon_args or {}))
```
