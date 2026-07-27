Features
========

There are several pipelines in this repository. There is an overall flow
for the pipeline that makes noisy RAW images that has these steps:

 #. Read an RGB Dataset, for instance CIFAR10 from `torchvision.datasets`.
 #. Convert each RGB into a clean RAW image.
 #. Add noise.
 #. Batch the images for training.
 #. Train a neural net on the images.

The different pipelines store intermediates in different ways.

Write Noisy Images to Disk
--------------------------

If you want to generate a directory full of noisy images to use in any way you
want, then run:

```bash
uv run scangen generate-raw <config-file>
```

This will read RGB files, use the RGB-to-RAW conversion, and write images
to an output directory.


Custom Dataset for PyTorch Training
-----------------------------------

There are two steps to this. One is to generate clean RAW images. It saves
a lot of time. The clean RAW images are distributed in the data directory
as `cifar10_raw.h5` and `cifar100_raw.h5`. They were created from CIFAR-10/100
RGB images using:

```bash
uv run scangen pregenerate-raw "cifar10"
uv run scangen pregenerate-raw "cifar100" --output-path=cifar100_raw.h5
```

Then the `scanbase` package shows how to use the `NoiseCIFARDataset`.
This custom `torch.Dataset` reads the pregenerated RAW, adds noise, and returns
it for use in training.
