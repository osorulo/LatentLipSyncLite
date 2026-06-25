import os
from pathlib import Path

BASE_DIR = Path(os.environ.get("LLS_BASE_DIR", ".")).resolve()
CHECKPOINTS_DIR = Path(os.environ.get("CHECKPOINTS_DIR", BASE_DIR / "checkpoints"))
CONFIGS_DIR = Path(os.environ.get("CONFIGS_DIR", BASE_DIR / "configs"))
VOCES_DIR = Path(os.environ.get("VOCES_DIR", BASE_DIR / "voces"))
HF_HOME = os.environ.get("HF_HOME", str(BASE_DIR / ".hf_cache"))