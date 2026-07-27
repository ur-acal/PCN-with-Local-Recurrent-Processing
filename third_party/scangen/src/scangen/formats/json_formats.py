import json
from pathlib import Path

class PathEncoder(json.JSONEncoder):
    """
    Teach json.dumps to turn pathlib.Path into strings.
    """
    def default(self, obj):
        if isinstance(obj, Path):
            return str(obj)
        return super().default(obj)
