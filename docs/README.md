# scikit-ops documentation

Numbered documents live in `docs/design/`, and a **`Status:` line** says
whether the thing exists yet — several are proposals. `docs/spec/` holds
unnumbered proposals. Both are in use.

```
docs/design/NNNN-*.md   Numbered, permanent. Status says implemented,
                        proposed, or placeholder. A proposal is amended in
                        place as it is built, not moved.
docs/spec/*.md          Unnumbered proposals.
```

The original rule was that a `spec/` file **graduates** into a numbered
`design/` document when the work lands. In practice nothing graduated and
unbuilt documents ended up numbered anyway (0011, 0012), so the folder is not
a reliable answer to "does this exist" — the status line is. Numbering a
document from the moment it is written also means links to it never break.

The [README](../README.md) is the short form — what the project is and how to
use it. These documents are the long form, aimed at someone (human or model)
who needs to change the machinery and wants to know which of its oddities are
load-bearing.

## Design

| # | Topic | Done |
| --- | --- | --- |
| [0001](design/0001-ops-are-plain-functions.md) | An op is an ordinary function | yes |
| [0002](design/0002-named-environments.md) | Environments are named and shared | yes |
| [0003](design/0003-semantic-roles.md) | Roles: what an array means | yes |
| [0004](design/0004-build-feedback.md) | Reporting an environment build | yes |
| [0005](design/0005-dimensional-adaptation.md) | Fitting an n-D array to an m-D op | yes, part superseded by 0006 |
| [0006](design/0006-axis-mapping.md) | Axis mapping belongs to the user | yes |
| [0007](design/0007-box-detector-ops.md) | Box detectors as a kind | yes |
| [0008](design/0008-mask-detector-ops.md) | Mask detectors: boxes in, masks out | yes |
| 0009 | — never written | — |
| [0010](design/0010-theoretical-psf-ops.md) | Gibson-Lanni and paraxial PSFs | yes |
| [0011](design/0011-deep-learning-training-ops.md) | Training ops take a list of paths | part: `ops/train/stardist2d.py` |
| [0012](design/0012-notebook-rendering.md) | Rendering op results in notebooks | no, placeholder |
| [0013](design/0013-make-box-detector-ops.md) | Build plan behind 0007 and 0008 | yes |
| [0014](design/0014-make-decon-ops.md) | Richardson-Lucy and the Gaussian PSF | part: no OpenCL |
| [0015](design/0015-augment-ops.md) | Augmentation as its own kind of op | no, behind 0011 |
| [0016](design/0016-choosing-an-environment.md) | Which environment a new op declares | yes, a rule in force |
| [0017](design/0017-memory-and-tiled-processing.md) | Memory declarations and tiling | no |
| [0018](design/0018-explicit-array-carriers.md) | Every op states its array carrier | no |
| [0019](design/0019-model-residency.md) | How long a loaded model stays resident | no |
| [0020](design/0020-reproducible-environments.md) | Committed locks, so a rebuild reproduces | no |

## Open items

[OPEN.md](OPEN.md) — known-broken and undecided things without a design yet.

## Spec

| | |
| --- | --- |
| [form-adaptation.md](spec/form-adaptation.md) | Calling a computer op as a function, and back |
| [front-ends.md](spec/front-ends.md) | What a second front end (Fiji) needs from here |
| [fiji-front-end.md](spec/fiji-front-end.md) | What that front end looks like: one command per op, Java as the host |
| [workflow-ops.md](spec/workflow-ops.md) | Ops built out of other ops, and what two real workflows need |
| [per-object-features.md](spec/per-object-features.md) | How a value per detected object should travel |
