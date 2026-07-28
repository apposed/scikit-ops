"""Every op in the collection, checked without any of their dependencies.

These tests are the discovery-as-enforcement rule in action: they run in the
host environment, which has numpy and nothing else an op needs. An op that
imports its heavy dependencies at module scope fails here, loudly, rather
than mysteriously at run time.
"""

from __future__ import annotations

import inspect

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


def test_box_detectors_are_substitutable():
    # The property a workflow depends on when it offers a choice of detector:
    # same outputs, same shared parameters, same return type. Only the
    # environment and the model-specific extras differ.
    fastsam = BY_NAME["skop.ops.detect.fastsam:fastsam"]
    object_aware = BY_NAME["skop.ops.detect.object_aware_yolo:object_aware_yolo"]

    assert fastsam.env == "pytorch"
    assert object_aware.env == "segment-everything"
    assert fastsam.outputs == object_aware.outputs == ("boxes",)
    assert fastsam.return_type is object_aware.return_type

    shared = ("image", "conf", "iou", "max_det", "imgsz")
    for spec in (fastsam, object_aware):
        assert shared == tuple(p.name for p in spec.params if p.name in shared)


def test_detected_boxes_are_shapes():
    # Without the role a front end has no way to know these are rectangles
    # to draw rather than an array to display.
    from skop import Role

    for name in (
        "skop.ops.detect.fastsam:fastsam",
        "skop.ops.detect.object_aware_yolo:object_aware_yolo",
    ):
        boxes = next(o for o in BY_NAME[name].output_specs if o.name == "boxes")
        assert boxes.role is Role.shapes


MASK_DETECTORS = (
    "skop.ops.mask.microsam:microsam_masks",
    "skop.ops.mask.mobilesam:mobilesam_masks",
)


@pytest.mark.parametrize("name", MASK_DETECTORS)
def test_mask_detector_takes_boxes_and_returns_masks_and_boxes(name):
    # The two-stage contract: what a box detector returns is what this
    # accepts, and what it returns says which prompts survived.
    from skop import Role

    spec = BY_NAME[name]
    assert spec.outputs == ("masks", "boxes")

    inputs = {p.name: p.role for p in spec.params}
    assert inputs["image"] is Role.image
    assert inputs["boxes"] is Role.shapes

    outputs = {o.name: o.role for o in spec.output_specs}
    # Not Role.labels: these overlap, and a front end has to project them
    # before it can show them at all.
    assert outputs["masks"] is Role.masks
    assert outputs["boxes"] is Role.shapes


def test_mask_detectors_are_substitutable():
    # The same property the box detectors have, one stage downstream: a
    # workflow offering a choice between them gets the same call and the same
    # outputs either way, differing only in environment and in extras.
    microsam, mobilesam = (BY_NAME[name] for name in MASK_DETECTORS)

    assert microsam.env == "pytorch"
    assert mobilesam.env == "segment-everything"
    assert microsam.return_type is mobilesam.return_type

    shared = ("image", "boxes")
    for spec in (microsam, mobilesam):
        assert shared == tuple(p.name for p in spec.params if p.name in shared)
        # The shared core comes first; what differs is optional and trailing.
        for param in spec.params:
            if param.name not in shared:
                assert param.default is not inspect.Parameter.empty


def test_mask_detector_returns_nothing_without_loading_a_model():
    # Reachable on the host precisely because the empty case short-circuits
    # ahead of the micro_sam import -- which is also what keeps a workflow
    # from loading 375 MB of weights to answer no prompts.
    import numpy as np

    from skop import boxes, masks
    from skop.ops.mask import microsam_masks

    result = microsam_masks(np.zeros((8, 9), dtype=np.uint8), boxes.EMPTY)

    assert result.masks.shape == (0, 8, 9)
    assert result.boxes.shape == (0, 4)
    # An empty collection still projects to a blank image of the right size.
    assert masks.to_labels_2d(result.masks).shape == (8, 9)


def test_gaussian_psf_needs_no_environment_of_its_own():
    # A PSF is just numpy, so it shares the environment already there.
    assert BY_NAME["skop.ops.kernels.psf:gaussian_psf"].env == "skimage"


def test_only_the_psf_model_that_needs_torch_pays_for_it():
    # The kernels namespace spans two environments on purpose: sdeconv drags
    # in torch, and the models that do not need it must not be made to wait
    # on a multi-gigabyte install.
    assert BY_NAME["skop.ops.kernels.gibson_lanni:gibson_lanni"].env == "sdeconv"
    assert BY_NAME["skop.ops.kernels.paraxial:paraxial_psf"].env == "skimage"
    assert BY_NAME["skop.ops.kernels.paraxial:paraxial_otf"].env == "skimage"


def test_psf_models_agree_on_what_the_optics_are_called():
    # Not a shared signature -- the models genuinely know different amounts
    # about the microscope -- but where they do ask the same question they
    # have to spell it the same way, or a front end cannot carry a value from
    # one to the other and a caller has to look it up every time.
    shared = ("wavelength", "numerical_aperture")
    for name in (
        "skop.ops.kernels.paraxial:paraxial_psf",
        "skop.ops.kernels.paraxial:paraxial_otf",
        "skop.ops.kernels.gibson_lanni:gibson_lanni",
    ):
        names = {p.name for p in BY_NAME[name].params}
        assert set(shared) <= names, f"{name} is missing {set(shared) - names}"
