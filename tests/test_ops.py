"""Every op in the collection, checked without any of their dependencies.

These tests are the discovery-as-enforcement rule in action: they run in the
host environment, which has numpy and nothing else an op needs. An op that
imports its heavy dependencies at module scope fails here, loudly, rather
than mysteriously at run time.
"""

from __future__ import annotations

import pytest

import skop

SPECS, FAILURES = skop.discover()
BY_NAME = {s.name: s for s in SPECS}


def test_every_op_module_imports():
    assert FAILURES == [], "\n".join(str(f) for f in FAILURES)


def test_collection_is_not_empty():
    assert len(SPECS) >= 15


@pytest.mark.parametrize("spec", SPECS, ids=lambda s: s.name)
def test_op_declares_a_known_environment(spec):
    runner = skop.Runner()
    # Raises FileNotFoundError, listing what does exist, if the env is absent.
    assert runner.env_config(spec.env).exists()


@pytest.mark.parametrize("spec", SPECS, ids=lambda s: s.name)
def test_op_params_are_fully_annotated(spec):
    import inspect

    unannotated = [p.name for p in spec.params if p.type is inspect.Parameter.empty]
    assert unannotated == [], f"{spec.name} has unannotated params: {unannotated}"


@pytest.mark.parametrize("spec", SPECS, ids=lambda s: s.name)
def test_op_declares_outputs(spec):
    assert spec.outputs, f"{spec.name} declares no outputs"


def test_stardist_ops_share_one_environment():
    # The payoff of named environments: one TensorFlow build, several ops.
    assert BY_NAME["skop.ops.segment.stardist2d:stardist2d_fluo"].env == "stardist-tf"
    assert BY_NAME["skop.ops.segment.stardist2d:stardist2d_he"].env == "stardist-tf"
    assert BY_NAME["skop.ops.segment.starfun3d:segment_nuclei"].env == "stardist-tf"


def test_the_two_stardist_models_differ_in_what_they_consume():
    # Why they are two ops: H&E is trained on stain colour and always needs a
    # channel axis, where the fluorescence model collapses one if given it.
    fluo = BY_NAME["skop.ops.segment.stardist2d:stardist2d_fluo"]
    he = BY_NAME["skop.ops.segment.stardist2d:stardist2d_he"]
    assert next(p for p in fluo.params if p.name == "image").axes.optional == {"c"}
    assert next(p for p in he.params if p.name == "image").axes.optional == frozenset()


def test_enum_params_carry_their_choices():
    spec = BY_NAME["skop.ops.segment.starfun3d:segment_nuclei"]
    model = next(p for p in spec.params if p.name == "model")
    assert [m.value for m in model.type] == ["confocal", "sospim", "spinning"]


def test_unseg_reports_counts_alongside_masks():
    spec = BY_NAME["skop.ops.segment.unseg:unseg"]
    assert spec.outputs == ("nuclei", "cells", "n_nuclei", "n_cells")


def test_starfun3d_returns_labels_and_points():
    spec = BY_NAME["skop.ops.segment.starfun3d:segment_nuclei"]
    assert spec.outputs == ("labels", "points")


def test_deconvolution_backends_differ_only_in_environment():
    # The reason there are two ops rather than one with a backend argument:
    # @op(env=...) is fixed per function. Their signatures must stay identical,
    # so that choosing a backend is choosing on speed alone.
    cpu = BY_NAME["skop.ops.deconvolve.richardson_lucy:richardson_lucy"]
    gpu = BY_NAME["skop.ops.deconvolve.richardson_lucy_cupy:richardson_lucy_cupy"]

    assert cpu.env == "skimage"
    assert gpu.env == "cupy"
    assert [(p.name, p.type, p.default) for p in cpu.params] == [
        (p.name, p.type, p.default) for p in gpu.params
    ]
    assert cpu.return_type is gpu.return_type


def test_gaussian_psf_needs_no_environment_of_its_own():
    # A PSF is just numpy, so it shares the environment already there.
    assert BY_NAME["skop.ops.kernels.psf:gaussian_psf"].env == "skimage"
