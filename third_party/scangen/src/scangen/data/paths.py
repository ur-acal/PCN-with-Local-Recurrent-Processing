from functools import cache
import importlib
from os import getenv
from pathlib import Path


@cache
def data_directory():
    """
    Look for data in the SCANGENDATA environment variable or a "data" directory
    inside the package or beside the main package directory.
    """
    environment = getenv("SCANGENDATA")
    if environment is not None:
        environment = Path(environment)
        if not environment.exists():
            raise FileNotFoundError(f"Data directory not found: {environment}")
        return environment.resolve()
    
    package_dir = Path(importlib.import_module("scangen").__file__).parent.parent.parent

    for trial_path in [package_dir / "data", package_dir.parent / "data"]:
        if trial_path.exists():
           return trial_path.resolve()

    return None
