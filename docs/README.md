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

| | |
| --- | --- |
| [0001](design/0001-ops-are-plain-functions.md) | An op is an ordinary function; everything else is read off it |
| [0002](design/0002-named-environments.md) | Environments are named and shared, not derived per op |
| [0003](design/0003-semantic-roles.md) | What an array *means*, and why skop never guesses |
| [0004](design/0004-build-feedback.md) | Reporting an environment build to whoever is running the op |
| [0005](design/0005-dimensional-adaptation.md) | Fitting an n-D array to an m-D op (superseded in part by 0006) |
| [0006](design/0006-axis-mapping.md) | Ops declare arity and hints; the axis mapping belongs to the user |
| [0007](design/0007-box-detector-ops.md) | Box detectors as a substitutable kind, and the first two |
| [0008](design/0008-mask-detector-ops.md) | Boxes in, one mask per box out, and why masks are not a label image |
| [0010](design/0010-theoretical-psf-ops.md) | Gibson-Lanni and the paraxial models, and what the license allows |
| [0013](design/0013-make-box-detector-ops.md) | The build plan behind 0007 and 0008 |
| [0014](design/0014-make-decon-ops.md) | Richardson-Lucy on numpy and cupy, and the Gaussian PSF |
| [0015](design/0015-augment-ops.md) | Augmentation as its own kind of op, separate from training — *proposed* |
| [0016](design/0016-choosing-an-environment.md) | Which environment a new op declares: anchors, and why a small new one beats a bigger old one |

## Spec

| | |
| --- | --- |
| [form-adaptation.md](spec/form-adaptation.md) | Calling a computer op as a function, and back |
| [front-ends.md](spec/front-ends.md) | What a second front end (Fiji) needs from here |
| [fiji-front-end.md](spec/fiji-front-end.md) | What that front end looks like: one command per op, Java as the host |
| [workflow-ops.md](spec/workflow-ops.md) | Ops built out of other ops, and what two real workflows need |
| [per-object-features.md](spec/per-object-features.md) | How a value per detected object should travel |
