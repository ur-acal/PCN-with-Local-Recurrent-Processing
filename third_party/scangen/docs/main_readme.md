# Scangen

This directory is for making RAW data that simulates noise from a sensor.

There are three subdirectories:

 * `scangen` - A package that creates the RAW data with noise.
   Look in these docs for more details about the RAW format and noise.
 * `scanbase` - A package that demonstrates how to use a scangen noise dataset.
   This is just a demo with a neural net that has poor accuracy.
 * `data` - Contains input data from CIFAR10/100, RAW versions, and model weights.

Before running any packages, define an environment variable to point to the
data directory:
```bash
export SCANGENDATA=/path/to/data
```

For these packages, all input data goes in `SCANGENDATA` so inputs are specified
relative to that directory. You can put configuration files where you want, and
you can put output files where you want.
