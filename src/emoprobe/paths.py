import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
CACHE = Path(os.environ.get("EMOPROBE_CACHE", ROOT / "cache"))
RESULTS = Path(os.environ.get("EMOPROBE_RESULTS", ROOT / "results"))
FIGS = ROOT / "figures"

SHORT = {"Qwen/Qwen2.5-3B-Instruct": "qwen3b", "unsloth/Meta-Llama-3.1-8B-Instruct": "llama8b"}


def slug(model_name: str) -> str:
    return SHORT.get(model_name, model_name.replace("/", "_"))
