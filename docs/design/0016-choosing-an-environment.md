# 0016 — Choosing an environment for a new op

**Status:** proposed as a rule, describing environments that already exist.
Nothing to build: `envs/` today already follows what is written here, and this
document exists to say *why*, so the next ten ops do not each re-litigate it.

[0002](0002-named-environments.md) decided that environments are named, defined
as data, and shared between ops. It did not say **which** environment a new op
should declare, and that is the question every new op actually raises.

## The question, as it is usually put

> Ops keep arriving. If each one brings its dependency to an existing
> environment, that environment grows without bound. If each one makes its own,
> we end up with dozens. Which mistake are we making?

Put that way it looks like a dial with a bad setting at both ends. It is not.
The two failure modes have different *kinds* of cost, and once that is seen the
rule falls out.

## The two costs are not the same kind

**Many environments costs resources.** Disk, and build time on first use. It is
paid per environment actually used, and every environment stays independently
correct — a bad solve in one cannot reach another.

**One environment growing costs coupling.** Every op in it shares a single
solve. Add albumentations to `stardist-tf` and albumentations' numpy floor now
negotiates with `numpy <2`; if it loses, StarDist breaks for reasons that have
nothing to do with StarDist, and the op that caused it is not the op that fails.
The cost is paid by ops that never asked for the new dependency, and it is paid
in correctness rather than in disk.

Disk is recoverable. A wrong solve is a bug hunt. So the default when unsure is
**a new environment**, not a bigger one.

## Environments are keyed by their anchor

What actually distinguishes the environments in `envs/` is not what their ops
*do*. It is one heavy, opinionated dependency that dictates everything else:

| env | anchor | what the anchor forces |
| --- | --- | --- |
| `stardist-tf` | tensorflow 2.10.1 | python `==3.10`, numpy `<2` |
| `unseg-cv` | unseg | python 3.9, numpy `==1.24.3`, scipy `==1.9.1` |
| `pytorch` | torch + CUDA 12.2 | cudnn, a GPU build |
| `cellpose3` | cellpose `<4` | torch |
| `cupy` | CUDA | cuda-version pin |
| `segment-everything` | segment-anything + onnx | conda opencv, timm |
| `sdeconv` | sdeconv | torch |
| `skimage` | *none* | nothing; 40 ops share it |
| `minimal` | *none* | nothing; 8 ops share it |

An environment's identity is its anchor. That is what 0002's sharing argument
is really about: TensorFlow takes minutes to install and seconds to import, and
neither cost should be paid twice. The thing worth sharing is the expensive,
constraining thing.

So the membership test is not "does this environment already have something
close to what I need". It is:

**Does this op need that anchor, and can it live with what the anchor forces?**

Two yeses: join. Otherwise: don't.

## The inversion

The part that is counter-intuitive, and the reason this document exists:

> A new environment is cheapest exactly when its dependencies are light — which
> is precisely the case where the instinct says "don't bother, just add it to
> one that exists."

The environments worth being reluctant to multiply are the torch / TensorFlow /
CUDA ones, and those already exist and are already shared. The reluctance is
aimed at the wrong target.

Measured on one developer machine, each environment sized on its own (listing
them together makes `du` attribute shared inodes to whichever sorts first,
which is how this gets misread):

| env | standalone |
| --- | --- |
| `minimal` | 302 MB |
| `skimage` | 533 MB |
| `cupy` | 2.0 GB |
| `stardist-tf` | 2.2 GB |
| `sdeconv` | 8.1 GB |
| `cellpose3` | 9.8 GB |
| `pytorch` | 11 GB |
| `segment-everything` | 12 GB |

Two orders of magnitude between the ends of that table, and the standalone
figure overstates the light end anyway. The eight together sum to 45 GB
standalone but occupy **25 GB** — hardlinking absorbs the rest. `skimage` is
533 MB alone but costs **300 MB** on top of an existing `minimal`, because
python, numpy and the shared libraries are already on disk.

So the marginal cost of an anchor-less environment is a few hundred megabytes,
against 8–12 GB for an anchored one. Build time is near-nothing too: the
package caches on that same machine (`~/.cache/rattler` 51 GB, `~/.cache/uv`
3 GB) are larger than every built environment combined, so a light environment
mostly resolves out of cache rather than off the network. Those caches, not the
environments, are what actually consumes disk — which is worth knowing before
economising in the wrong place.

Its converse is the rule that actually prevents bloat:

**Never add a light dependency to a heavy environment because it is already
there.** It buys nothing — the light thing was cheap to install on its own —
and it costs the coupling described above.

## Duplication of light dependencies is fine, and already happening

The worry that two environments must not contain the same package is worth
naming and dismissing. `opencv` already appears twice, in two packagings, at
two versions:

```
envs/unseg-cv/pixi.toml           opencv-python-headless ==4.7.0.72   (PyPI)
envs/segment-everything/pixi.toml py-opencv                            (conda)
```

They do not share a solve, do not constrain each other, and neither is wrong.
Duplication across environments is not debt. Duplication *within* one — two
packagings of opencv in a single environment — is the thing that breaks, and
napari-ai-lab's `pixi/pytorch_napari/pixi.toml` carries a long comment about
exactly that failure.

## The rule

1. **Identify the op's anchor** — the heaviest, most version-opinionated thing
   it imports. Often there isn't one.
2. **If it has an anchor that an existing environment already has**, declare
   that environment. This is the sharing 0002 exists for.
3. **If it has an anchor nothing else has**, it gets a new environment. Obvious,
   and not the hard case.
4. **If it has no anchor**, give it a small environment of its own, or `minimal`
   / `skimage` if it genuinely needs only what those already carry. Do not
   economise here — this is the cheap case.
5. **Never** add to a heavy environment for convenience.

Step 4 is where judgement remains: `skimage` has 40 ops and no anchor, so it is
a real home for anything needing scikit-ops plus scikit-image and nothing more.
The test for joining it is the same as for any other — *nothing more*. An op
that would add a dependency to `skimage` is an op that should have its own
environment, because the 40 ops there would otherwise start carrying it.

## Worked example: albumentations

The case this document was written from. [0015](0015-augment-ops.md) proposes
`@op(env="albumentations")` and does not argue for the name.

Its anchor: **none.** No CUDA, no python pin, no hard numpy ceiling. The
evidence is direct rather than inferred — napari-ai-lab's host environment runs
python 3.12 with numpy 2.4, and cellpose's PyPI stack there already supplies
opencv-python-headless, scipy and pydantic, which is albumentations' entire
dependency set bar `albucore` and two small compiled wheels.

By the rule:

- Not `pytorch` — that would couple a CPU image transform to CUDA.
- Not `skimage` — it would put opencv behind 40 ops that do not want it (step
  4's *nothing more* test).
- Not `stardist-tf` — the numpy `<2` negotiation described above, for an op
  that has no reason to be near TensorFlow.
- **Its own**: python, numpy, albumentations, and the scikit-ops pin every
  environment carries. A few lines. opencv arrives as albumentations' own
  dependency rather than something declared.

Size, by the figures above: roughly **500 MB standalone and 200 MB marginal** —
`minimal`'s 302 MB plus opencv 72 MB, scipy 92 MB, pydantic 9 MB and about
10 MB of albucore, simsimd and stringzilla. Around 2% of `pytorch`.

## What the rule does not settle

**An op needing two anchors.** Augmentation inside a torch training loop needs
albumentations *and* torch, and no assignment of one environment is right. 0015
avoids the one instance of this deliberately — per-epoch augmentation is a
parameter on the training op, expressed in that framework's own vocabulary,
rather than an op. That dodge will not work every time, and when it fails the
answer is probably a second environment that duplicates the light half, not a
merge of the heavy halves.

**When an anchor-less environment should absorb a second op.** Two ops with no
anchor and overlapping light dependencies could share, and the rule above is
mildly biased against it. That bias is deliberate while the collection is
small — merging two small environments later is easy, and unpicking one shared
environment that grew is not.

**Whether many warm workers is a problem.** Each environment in use is a
process. Nothing today runs enough environments at once for this to have been
measured, and the question should be reopened with numbers rather than
predicted here.
