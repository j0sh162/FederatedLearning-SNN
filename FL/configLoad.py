from typing import Any

import yaml

def load_config(path: object = "config.yaml") -> Any:
    with open(path, "r") as f:
        return yaml.safe_load(f)
