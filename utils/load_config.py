import json
from pathlib import Path
from typing import Any, Dict


def load_config(config_path: str) -> Dict[str, Any]:
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as config_file:
        return json.load(config_file)