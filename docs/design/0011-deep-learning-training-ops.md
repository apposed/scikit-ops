# Deep learning annotation and training ops

Deep learning training ops have some special considerations. Segmentation is the first case we will tackle, so the examples below lean that way, but the same considerations apply to restoration (denoising, deconvolution, super-resolution), classification, and registration.

1.  They require input/truth pairs. For segmentation these are images and the ground truth label. For restoration they are typically a low quality image and a high quality counterpart, for example a noisy acquisition paired with a long exposure or averaged version of the same field. For classification the truth is a label per image or per object.

2.  The truth can take many forms. For segmentation it can be a semantic label image, an instance label image, or shapes (for example bounding boxes or points representing the location of objects). For restoration it is another image of the same shape as the input. For classification it is a scalar or a table of values. The op needs to know which form it is getting.  For super-resolution the input is a low resolution image and the truth a high resolution image.

3.  This type of data could be represented in Napari as viewer layers, but is more often curated and stored on disk.

4.  If the data is in Napari, it is often convenient to use bounding boxes to mark good areas. For example, it may be tedious to label all the data, so you label some of it and mark which areas are labeled to satisfaction. The same trick is useful for restoration, where you may only have well registered input/truth pairs in part of the field.

5.  After step 4 there can be an intermediate step where the truth in Napari is augmented to create a set of patches with more variation. From a single label, hundreds of patches could be created, and this is often the data we want to use for training. Note that augmentation has to be applied consistently to the input and the truth whenever the truth is spatial, which covers both label images and restoration targets.

6.  Alternatively, the augmentation is often done inline with the training. The advantage of doing it pre-training is that you can sometimes use slower but more powerful augmentations, like warping and color augmentation. The disadvantage is that you may have a fixed number of augmentations. The advantage of inline is an infinite number of variations.

7.  So the training op often has to take a file location as input(s), where the truth will be put in a specific format compatible with the formats the training op can read. The format varies by task and by framework, so this is a per-op concern rather than something we can settle once.
