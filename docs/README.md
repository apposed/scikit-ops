# scikit-ops documentation

Two kinds of document live here, and the difference is whether the thing exists.

```
docs/design/NNNN-*.md   Settled. Why the code is shaped this way, and what
                        else was considered. Written after the fact.
docs/spec/*.md          Proposed. Not built. Deleted or graduated into a
                        design doc once it is.
```

A `spec/` file **graduates**: when the work lands, it becomes the next numbered
`design/` document, rewritten in the past tense and carrying whatever the
implementation taught us that the plan did not know. So `spec/` shrinks over
time and `design/` grows, which is the correct direction.

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

## Spec

| | |
| --- | --- |
| [form-adaptation.md](spec/form-adaptation.md) | Calling a computer op as a function, and back |
| [front-ends.md](spec/front-ends.md) | What a second front end (Fiji) needs from here |
