Noise models
============

There are two noise models.

pointwise Noise Model
---------------------

This uses `photon transfer`_ analysis of sensor behavior to assign noise values
to pixels. The shot noise is Gaussian-distributed at a constant level.
The read noise depends on single pixel values, where a brighter pixel has more
noise. The relationship between pixel value and noise comes from the linked
document.

Shot noise is defined in the configuration file as `max_shot_noise`. The
`min_shot_noise` isn't used, and the rest of the noise parameters are unused.
Read noise is determined from a fit curve found in the file called `noise_d300.py`.


cycleisp Noise Model
---------------------

This model is reflected in the file `src/scangen/pipeline/noise.py`.
It comes from the CycleISP paper which bases its noise model on a paper
by Brooks.

 #. There is one draw per batch for noise parameters for shot and read.
 #. There is a second draw per pixel that uses pixel values to determine variance on a Gaussian.

For the second draw, the noise is drawn from a Gaussian with variance equal to
`sqrt(shot_parameter * max(pixel_value, 0) + read_parameter)`.

.. _photon transfer: https://photonstophotos.net/GeneralTopics/Sensors_&_Raw/Sensor_Analysis_Primer/Photon_Transfer_Curve.htm
