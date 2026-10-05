import hashlib
import json
import time
from pathlib import Path

import joblib

MODEL_DIR = Path(__file__).resolve().parent / "models"


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_model(task, models, meta):
    MODEL_DIR.mkdir(exist_ok=True)
    version = f"{task}-{time.strftime('%Y%m%d-%H%M%S')}"
    d = MODEL_DIR / version
    d.mkdir()
    joblib.dump(models, d / "model.joblib")
    meta = dict(meta, version=version, task=task, created=time.strftime("%Y-%m-%dT%H:%M:%S"))
    (d / "meta.json").write_text(json.dumps(meta, indent=2, default=float))
    return version


def list_versions(task=None):
    if not MODEL_DIR.exists():
        return []
    out = [json.loads(p.read_text()) for p in sorted(MODEL_DIR.glob("*/meta.json"))]
    return [m for m in out if task is None or m["task"] == task]


def load_model(version=None, task=None):
    vs = list_versions(task)
    if not vs:
        raise FileNotFoundError("no trained model found - run `python train.py` first")
    meta = next((m for m in vs if m["version"] == version), None) if version else vs[-1]
    if meta is None:
        raise FileNotFoundError(f"unknown model version {version}")
    return joblib.load(MODEL_DIR / meta["version"] / "model.joblib"), meta
