# Open items

Things known to be wrong or undecided, too small or too unresolved to deserve
their own numbered design. One heading each, newest at the top.

**Status** is one of:

- `open` — known, not yet decided what to do.
- `decided: <what>` — the call has been made, not yet built.
- `done` — built. Move the item to the *Resolved* section at the bottom with
  a one-line note on what happened, rather than deleting it.

Ask "what else to do" in any session and this file is the answer.

---

## Environments are isolated but not reproducible

**Status:** open — see [design 0020](design/0020-reproducible-environments.md).

`pixi.lock` is gitignored, so every user solves from scratch and two builds of
the same recipe a month apart differ. Committing the locks is most of the fix.

---

## The scikit-ops pin in each env goes stale silently

**Status:** open — bitten once, 2026-09-03.

`envs/*/pixi.toml` pins scikit-ops by git rev. Nothing bumps it, so a worker
runs whatever was current when someone last remembered. It was a month behind
and predated `src/skop/ops/train` entirely.

A checkout hides this: `host.py` prepends the checkout and shadows the pin. A
pip install appends instead, so the worker's own copy wins and the ops are
simply missing. Every developer sees it working and every user does not.

Fixed for now by bumping the rev. The real fix is pinning a released version,
which is [0020](design/0020-reproducible-environments.md)'s last point.
