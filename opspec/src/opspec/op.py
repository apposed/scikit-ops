"""Declare ops: the @op decorator, and the parameters that pick other ops.

Reading an op into an OpSpec is in describe.py. Standard library only,
because every worker environment imports it.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

# These live in axes.py, resources.py, tile.py and types.py, the vocabulary a
# host can use without ops; they are imported here so `from opspec.op import
# Role` and the like keep working.
from .axes import Axes, Slot  # noqa: F401 - for `from opspec.op import Axes`
from .resources import PeakMemory
from .tile import Overlap
from .types import (  # noqa: F401 - for `from opspec.op import Out`
    Direction,
    Dtype,
    Mut,
    Out,
    Role,
)

# Short names for the two directions, as op.py and its users compare them.
OUT = Direction.OUT
MUT = Direction.MUT


# Op computation forms, in the SciJava Ops sense.
FUNCTION = "function"  # returns a new output
COMPUTER = "computer"  # caller supplies the output buffer, op fills it
INPLACE = "inplace"  # op mutates one of its inputs


@dataclass(frozen=True, init=False)
class Choices:
    """A list of the ops a workflow parameter may be filled with.

    Attached to a ``Callable`` parameter, so a GUI can offer a combo box
    instead of asking for an import path::

        psf_op: Annotated[Callable, Choices(gaussian=gaussian_psf,
                                            gibson_lanni=gibson_lanni)]

    The keywords are the menu labels: "gpu" is easier to read than
    ``richardson_lucy_cupy``.

    **The list limits the GUI, not the function**: a script may pass any op.
    That is how the list grows: someone tries an untested op in a script, it
    works, and it is added here, where the change is reviewed. The list is
    written by hand, not discovered, because it means "these are tested", not
    "these are installed".
    """

    options: tuple[tuple[str, Callable], ...]

    def __init__(self, **options: Callable) -> None:
        object.__setattr__(self, "options", tuple(options.items()))

    @property
    def labels(self) -> tuple[str, ...]:
        """The menu labels."""
        return tuple(label for label, _ in self.options)

    def op(self, label: str) -> Callable:
        """The op a label names."""
        return dict(self.options)[label]

    def label(self, fn: Callable) -> str | None:
        """The label for *fn*, or None if it isn't listed."""
        return next((label for label, op in self.options if op is fn), None)

    @property
    def ids(self) -> tuple[tuple[str, str], ...]:
        """``(label, "module:function")`` pairs: the form sent over the wire.

        A front end in another process, Fiji say, needs the menu but can't
        import the functions. A ``Choices`` rebuilt from the wire holds these
        IDs instead of functions, and ``ids`` returns them unchanged;
        ``op()`` and ``label()`` don't work on it.
        """
        return tuple(
            (label, op if isinstance(op, str) else f"{op.__module__}:{op.__name__}")
            for label, op in self.options
        )


@dataclass(frozen=True, init=False)
class ParamsFor:
    """Marks a dict parameter as holding the arguments of a chosen op::

        decon_op: Annotated[Callable, Choices(cpu=..., gpu=...)] = richardson_lucy
        decon_args: Annotated[dict, ParamsFor("decon_op",
                                              binds=("image", "psf"))] = None

    ``binds`` names the chosen op's parameters that the workflow fills
    itself, from its own inputs or an earlier step's output. A GUI shows only
    the chosen op's *other* parameters, so two steps that both take an image
    don't both ask for it.

    ``binds`` is declared, not inferred from names. Matching names would
    catch the image, but not that a mask op's ``boxes`` come from the
    detector, and half a rule is harder to explain than none.
    """

    chooser: str
    binds: tuple[str, ...]

    def __init__(self, chooser: str, *, binds: Any = ()) -> None:
        # One name as a string is common; don't bind it letter by letter.
        if isinstance(binds, str):
            binds = (binds,)
        object.__setattr__(self, "chooser", chooser)
        object.__setattr__(self, "binds", tuple(binds))


@dataclass(frozen=True)
class _OpConfig:
    env: str | None
    main_thread: bool = False
    exclusive: bool = False
    tile: tuple[str, ...] = ()
    overlap: Overlap | None = None
    peak_memory: PeakMemory | None = None
    pad: Overlap | None = None
    merge: str | None = None
    split: tuple[str, ...] = ()


def op(
    fn: Callable | None = None,
    *,
    env: str | None = None,
    main_thread: bool = False,
    exclusive: bool = False,
    tile: str | tuple[str, ...] | None = None,
    overlap: Overlap | int | None = None,
    peak_memory: PeakMemory | None = None,
    pad: Overlap | int | None = None,
    merge: str | None = None,
    split: tuple[str, ...] | None = None,
) -> Callable:
    """Declare a function as an op.

    Sets an attribute on the function and returns it unchanged, so calling
    it directly still works. Use it bare or with arguments, like
    ``@dataclass``::

        @op
        def smooth(image: ImageOf[np.ndarray]) -> ImageOf[np.ndarray]: ...

        @op(env="cupy")
        def deconvolve(image: ImageOf[cp.ndarray]) -> ImageOf[cp.ndarray]: ...

    Args:
        env: The environment the op runs in. Without one, the op runs
            wherever the caller is; a workflow (an op that calls other ops)
            is like this.
        main_thread: Whether the op must run on its worker's main thread.
        exclusive: Whether the op needs a worker to itself, instead of
            sharing one with other ops in the same environment.
        tile: The parameter or parameters a runner may cut into tiles to
            fit memory. All are tiled the same way.
        overlap: How far a tile reads past its core: an ``Overlap``, or a
            number of pixels.
        peak_memory: The op's peak memory, as a ``PeakMemory``. A runner
            sizes tiles from it.
        pad: How far the op pads its input on each side, as an ``Overlap``
            or a number of pixels. ``peak_memory`` then counts the padded
            size. An FFT op pads by part of its kernel.
        merge: How tiles are put back together. ``"crop"`` keeps each
            tile's core and drops the overlap. ``"blend"`` fades each tile
            out across the overlap, so neighbours mix where they meet.
        split: The axes a tile may be cut along, named as in the tiled
            input's ``Axes``. ``("y", "x")`` keeps z whole, as decon wants
            with a PSF long in z. Left out, any axis may be cut.
    """

    def decorate(f: Callable) -> Callable:
        f.__opspec__ = _OpConfig(
            env=env,
            main_thread=main_thread,
            exclusive=exclusive,
            tile=(tile,) if isinstance(tile, str) else tuple(tile or ()),
            overlap=Overlap(overlap) if isinstance(overlap, int) else overlap,
            peak_memory=peak_memory,
            pad=Overlap(pad) if isinstance(pad, int) else pad,
            merge=merge,
            split=tuple(split or ()),
        )
        return f

    return decorate(fn) if fn is not None else decorate


def is_op(obj: Any) -> bool:
    """Whether ``obj`` carries the ``@op`` decorator."""
    return callable(obj) and isinstance(getattr(obj, "__opspec__", None), _OpConfig)


# OpSpec and the rest of reading an op moved to describe.py. They are looked
# up there only when asked for, so `from opspec.op import OpSpec` keeps
# working without op.py and describe.py importing each other at load time.
def __getattr__(name: str) -> Any:
    from . import describe

    try:
        return getattr(describe, name)
    except AttributeError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
