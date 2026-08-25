import yaml
from pathlib import Path

def load_config(path: str | Path) -> dict:
    path = Path(path)
    with path.open("r") as f:
        cfg = yaml.safe_load(f)
    return cfg
