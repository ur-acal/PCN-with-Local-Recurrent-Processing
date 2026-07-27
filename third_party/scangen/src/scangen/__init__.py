"""ScAN Gen: Generate noisy RAW data from RGB inputs for neural network training.

This package transforms clean RGB images (like CIFAR-10/100) into realistic noisy RAW
sensor data using learned camera models and physics-based noise simulation.
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("scangen")
except PackageNotFoundError:
    # package is not installed
    __version__ = "unknown"
