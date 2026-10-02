"""Record runs into an Icechunk repo, which works like git for images.

Needs the ``icechunk`` extra: ``pip install scikit-ops[icechunk]``.

Images are ``zarr.Array``s opened from an Icechunk repo.
"""

from __future__ import annotations

from pathlib import Path

import icechunk
import zarr

from .record import Record

#: The tag on the commit that loaded the original image.
ORIGINAL = "original"


def _original_name(repo: icechunk.Repository) -> str:
    """The name the original image was loaded under."""
    snapshot = repo.lookup_snapshot(repo.lookup_tag(ORIGINAL))
    return snapshot.metadata["name"]


def icechunk_open(
    repo: str | Path,
    name: str | None = None,
    *,
    branch: str = "main",
    tag: str | None = None,
    snapshot: str | None = None,
) -> zarr.Array:
    """Open an image from the Icechunk repo at ``repo``, read-only.

    With no ``name``, the original image, exactly as it was loaded. With a
    ``name``, something a run created: the newest, unless you ask for an older
    one by ``tag`` or ``snapshot`` (a record id from ``runner.log``).
    """
    storage = icechunk.local_filesystem_storage(str(repo))
    repo_ = icechunk.Repository.open(storage)
    if name is None:
        tag, snapshot = ORIGINAL, None
        name = _original_name(repo_)
    pinned = tag or snapshot
    session = repo_.readonly_session(
        branch=None if pinned else branch, tag=tag, snapshot_id=snapshot
    )
    return zarr.open_array(session.store, path=name, mode="r")


def icechunk_imread(file: str | Path, repo: str | Path) -> zarr.Array:
    """Copy an image file into ``repo`` the first time, then open it.

    The image is named after the file: ``coins.tif`` becomes ``coins``. The
    commit that loads it is tagged ``original``, which never moves.
    """
    from skimage import io

    file = Path(file)
    name = file.stem

    # Icechunk: open the repo, or make it
    storage = icechunk.local_filesystem_storage(str(repo))
    repo_ = icechunk.Repository.open_or_create(storage)

    # First time only: copy the pixels in as one commit, tagged original
    if ORIGINAL not in repo_.list_tags():
        session = repo_.writable_session("main")
        zarr.create_array(session.store, name=name, data=io.imread(file))
        snapshot = session.commit(f"load {file.name}", metadata={"name": name})
        repo_.create_tag(ORIGINAL, snapshot)

    return icechunk_open(repo)


class IcechunkRecorder:
    """The recipe plus the result, as one Icechunk commit per run."""

    def __init__(self, repo: str | Path):
        storage = icechunk.local_filesystem_storage(str(repo))
        self.repo = icechunk.Repository.open_or_create(storage)

    def identify(self, value) -> str | None:
        # Icechunk: a zarr.Array from a session knows its snapshot
        session = getattr(getattr(value, "store", None), "session", None)
        return f"{value.path}@{session.snapshot_id}" if session else None

    def save(self, name, op, params, inputs, result):
        if ORIGINAL in self.repo.list_tags() and name == _original_name(self.repo):
            raise ValueError(
                f"{name!r} is the original image; save the result under another name"
            )

        # Icechunk: write the result and commit it with the recipe
        session = self.repo.writable_session("main")
        zarr.create_array(session.store, name=name, data=result, overwrite=True)
        snapshot = session.commit(
            f"{op} {params}",
            metadata={"name": name, "op": op, "params": params, "inputs": inputs},
        )

        # The saved result, as a zarr.Array that knows its snapshot
        session = self.repo.readonly_session(snapshot_id=snapshot)
        return zarr.open_array(session.store, path=name, mode="r")

    def log(self, name):
        return [
            Record(c.id, c.message, c.metadata)
            for c in self.repo.ancestry(branch="main")
            if c.metadata.get("name") == name
        ]
