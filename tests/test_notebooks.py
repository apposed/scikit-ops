"""A weight limit on committed notebooks.

Notebooks carry their outputs on purpose -- the figures are the product, and
should render for someone who has installed nothing (notebooks/README.md). The
cost is that every figure is a base64 PNG inside the file, base64 of already
compressed data does not delta-compress, and so *every re-run stores a fresh
full copy in git history*. A notebook revised five times is five complete
copies of its figures, forever.

Four notebooks were already 58% of this repository's history when this was
written. That scales badly and it scales invisibly, because nothing about
adding a notebook feels expensive at the time.

Hence a tripwire rather than a judgment call. Nothing here runs a notebook --
these are static file sizes.
"""

from __future__ import annotations

from pathlib import Path

import pytest

NOTEBOOKS = Path(__file__).resolve().parent.parent / "notebooks"

#: What one notebook may weigh. cellpose_mixed is 1.5 MB for three figures at
#: 72 dpi, which is what a disciplined notebook costs, so the limit sits above
#: that with room for a re-run to land differently. Small enough that a stray
#: full-resolution grid trips it -- the case worth catching, since one
#: oversized figure is usually most of a notebook's weight.
LIMIT_MB = 2.0

#: Notebooks that predate the limit. Shrink when next re-run rather than
#: re-running them just for this; an entry here is a debt, not a dispensation.
GRANDFATHERED = {
    "detection/detect-then-mask.ipynb": 3.0,
}


def notebooks():
    return sorted(NOTEBOOKS.rglob("*.ipynb"))


def test_there_are_notebooks_to_check():
    # Guards against the glob silently finding nothing and every size test
    # below passing vacuously.
    assert notebooks()


@pytest.mark.parametrize("path", notebooks(), ids=lambda p: p.name)
def test_notebook_is_not_oversized(path):
    relative = path.relative_to(NOTEBOOKS).as_posix()
    limit = GRANDFATHERED.get(relative, LIMIT_MB)
    size = path.stat().st_size / 1e6

    assert size <= limit, (
        f"{relative} is {size:.2f} MB, over its {limit:.2f} MB limit. "
        "Figures are stored as base64 PNGs and every re-run adds a full copy "
        "to history. Lower plt.rcParams['figure.dpi'], shrink figsize, or "
        "drop a panel -- one large grid is usually most of the weight."
    )


def test_grandfathered_entries_still_exist():
    # An exemption for a notebook that has been renamed or deleted is dead
    # weight that quietly raises the limit for nothing.
    for relative in GRANDFATHERED:
        assert (NOTEBOOKS / relative).exists(), (
            f"{relative} is exempted but no longer exists; drop it from GRANDFATHERED."
        )


def test_grandfathered_notebooks_are_not_growing():
    # The exemptions are ceilings for shrinking toward LIMIT_MB, so a
    # grandfathered notebook that grows past its recorded size fails rather
    # than quietly taking up the slack.
    for relative, limit in GRANDFATHERED.items():
        assert limit > LIMIT_MB, (
            f"{relative} is exempted at {limit} MB, which is not above the "
            f"{LIMIT_MB} MB limit -- the entry does nothing and should go."
        )
