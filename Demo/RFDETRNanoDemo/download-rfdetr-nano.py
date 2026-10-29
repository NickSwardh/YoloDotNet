"""Download RF-DETR Nano demo assets.

This script uses the official Roboflow Inference SDK to cache `rfdetr-nano`
locally, then copies the ONNX weights and class names into test/assets/Models.

Usage:
    python Demo/RFDETRNanoDemo/download-rfdetr-nano.py

If your Roboflow environment requires authentication, set ROBOFLOW_API_KEY first.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path


MODEL_ID = "rfdetr-nano"
ROBOFLOW_CACHE_MODEL_ID = Path("coco") / "38"


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    models_dir = repo_root / "test" / "assets" / "Models"
    cache_dir = models_dir / ".roboflow-cache"
    target_weights = models_dir / "rfdetr-nano.onnx"
    target_classes = models_dir / "rfdetr-nano-class_names.txt"

    models_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MODEL_CACHE_DIR"] = str(cache_dir)

    try:
        import urllib3.contrib.pyopenssl as pyopenssl

        pyopenssl.inject_into_urllib3()
    except Exception:
        pass

    try:
        from inference import get_model
    except ModuleNotFoundError:
        print("Missing dependency: inference")
        print("Install it with: python -m pip install inference")
        return 1

    print(f"MODEL_CACHE_DIR={cache_dir}")
    print(f"Downloading {MODEL_ID} with Roboflow Inference SDK...")

    api_key = os.environ.get("ROBOFLOW_API_KEY")
    kwargs = {"model_id": MODEL_ID}

    if api_key:
        kwargs["api_key"] = api_key

    get_model(**kwargs)

    cached_model_dir = cache_dir / ROBOFLOW_CACHE_MODEL_ID
    cached_weights = cached_model_dir / "weights.onnx"
    cached_classes = cached_model_dir / "class_names.txt"

    if not cached_weights.exists():
        print(f"Could not find downloaded weights: {cached_weights}")
        return 1

    shutil.copy2(cached_weights, target_weights)

    if cached_classes.exists():
        shutil.copy2(cached_classes, target_classes)

    print(f"Saved weights: {target_weights}")
    print(f"Saved classes: {target_classes}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
