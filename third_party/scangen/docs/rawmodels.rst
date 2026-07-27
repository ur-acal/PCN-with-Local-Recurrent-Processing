RGB-to-RAW Models
=================

CycleISP
--------

This is the first model implemented. We used the pretrained waits from the
CycleISP paper to convert CIFAR-10 and CIFAR-100 into a clean RAW dataset
before applying noise.

The steps are:

 #. Read CIFAR-10/100 images at 3x32x32 in RGB.
 #. Upscale them to 3x256x256 to match the input for the CycleISP model.
 #. Run the CycleISP model which produces 4x256x256.
 #. Downscale the output to 4x16x16 where the 4 are RGGB channels so this
    represents a 32x32 sensor with Bayer filtering.
 #. Then the noise model adds noise to the downsampled clean RAW data.
