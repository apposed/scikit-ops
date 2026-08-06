"""The worker side of the wire: rebuilding the types JSON flattened.

An Enum travels as its value and a Path as a string, so an op run through a
Runner would otherwise receive something different from what the same op
receives when called directly. ``test_host.py`` fixes what goes onto the
wire; this fixes what comes back off it.

Host-only: no environment, no worker process. ``_coerce`` is the pure
function in the middle, so it is tested as one.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path

import skop
from skop import worker


class Colour(Enum):
    red = "red"
    green = "green"


@skop.op(env="minimal")
def paths(
    required: Path,
    optional: Path | None = None,
    colour: Colour = Colour.red,
    optional_colour: Colour | None = None,
    untouched: int = 3,
) -> float:
    return 0.0


def coerce(**args):
    """Run ``_coerce`` over the op above, as a worker does on arrival."""
    return worker._coerce(skop.spec(paths), args)


def test_path_is_rebuilt():
    assert coerce(required="/tmp/model")["required"] == Path("/tmp/model")


def test_optional_path_is_rebuilt():
    # The regression this file exists for. `Path | None` is a union, not
    # `Path`, so an identity check against `Path` misses it -- and the
    # parameter arrives as the string it travelled as.
    assert coerce(optional="/tmp/model")["optional"] == Path("/tmp/model")


def test_optional_left_empty_stays_none():
    # Path(None) raises, so the None has to be recognised before the rebuild.
    assert coerce(optional=None)["optional"] is None


def test_enum_is_rebuilt():
    assert coerce(colour="green")["colour"] is Colour.green


def test_optional_enum_is_rebuilt():
    assert coerce(optional_colour="red")["optional_colour"] is Colour.red


def test_already_rich_values_pass_through():
    # A direct call hands over the real types. Coercing them again must be a
    # no-op rather than a second conversion.
    given = {"required": Path("/tmp/model"), "colour": Colour.green}
    back = coerce(**given)
    assert back["required"] == Path("/tmp/model")
    assert back["colour"] is Colour.green


def test_other_types_are_untouched():
    assert coerce(untouched=7)["untouched"] == 7


def test_absent_parameters_are_not_invented():
    # Only what was sent is coerced; a defaulted parameter stays absent so
    # the function's own default applies.
    assert coerce(required="/tmp/model").keys() == {"required"}


def test_unwrap_optional_leaves_real_unions_alone():
    # Two live types is not an optional, and there is no single type to
    # rebuild it as, so it has to be left as it arrived.
    assert worker._unwrap_optional(int | str) == (int | str)
    assert worker._unwrap_optional(Path | None) is Path
    assert worker._unwrap_optional(Path) is Path
