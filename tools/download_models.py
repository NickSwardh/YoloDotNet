# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Niklas Swärd
# https://github.com/NickSwardh/YoloDotNet
"""
Downloads the Ultralytics models used by the YoloDotNet tests and demos and
exports them to ONNX with the correct opset version for best performance.

The models are saved to test/assets/Models, where the tests find them automatically.
Models that already exist are skipped; delete a file to export it again.

Requires Ultralytics 8.4.0 or later:  pip install "ultralytics>=8.4.0"
Usage:                                python tools/download_models.py
"""

import contextlib
import os
import shutil
import sys
import tempfile
import urllib.request
from pathlib import Path

OUTPUT_FOLDER = Path(__file__).resolve().parent.parent / "test" / "assets" / "Models"
WEIGHTS_URL = "https://github.com/ultralytics/assets/releases/download/v8.4.0/"
BAR_WIDTH = 25

YELLOW, GREEN, RED, RESET = "\033[93m", "\033[92m", "\033[91m", "\033[0m"

LICENSE_NOTICE = f"""{YELLOW}
================================================================================
  MODEL LICENSE NOTICE - PLEASE READ
================================================================================
  This script downloads the models from Ultralytics:
  {WEIGHTS_URL}

  The models are NOT part of YoloDotNet.
  They are owned by Ultralytics and licensed under AGPL-3.0.

  YoloDotNet is MIT licensed, but that license does NOT cover these models.
  The models have their own license, controlled by Ultralytics.

  - Do NOT use these models in commercial or closed-source software unless
    you either comply with AGPL-3.0, which requires releasing your
    application's source code under the same license, or have a commercial
    license from Ultralytics.
  - AGPL-3.0 has strict requirements for distribution and network deployment.
  - You alone are responsible for complying with the model licenses.

  Details: https://www.ultralytics.com/license/
================================================================================
{RESET}"""

# (Ultralytics weights, opset). The ONNX file keeps the Ultralytics name, e.g. yolo12s.pt -> yolo12s.onnx
# YOLOv5u-YOLOv12 and RT-DETR use opset 17. YOLO26 uses opset 18 with nms=False (see README).
MODELS = [
    ("yolov5su.pt", 17),
    ("yolov8s.pt", 17),
    ("yolov8s-cls.pt", 17),
    ("yolov8s-obb.pt", 17),
    ("yolov8s-seg.pt", 17),
    ("yolov8s-pose.pt", 17),
    ("yolov9s.pt", 17),
    ("yolov10s.pt", 17),
    ("yolo11s.pt", 17),
    ("yolo11s-cls.pt", 17),
    ("yolo11s-obb.pt", 17),
    ("yolo11s-seg.pt", 17),
    ("yolo11s-pose.pt", 17),
    ("yolo12s.pt", 17),
    ("yolo26s.pt", 18),
    ("yolo26s-cls.pt", 18),
    ("yolo26s-obb.pt", 18),
    ("yolo26s-seg.pt", 18),
    ("yolo26s-pose.pt", 18),
    ("rtdetr-l.pt", 17),
]

# Weights name suffix -> task shown in the progress line. Models without a suffix are detection models.
TASKS = {"-cls": "Classification", "-seg": "Segmentation", "-pose": "Pose", "-obb": "OBB"}


def onnx_name(weights: str) -> str:
    return Path(weights).stem + ".onnx"


def task_name(weights: str) -> str:
    return next((task for suffix, task in TASKS.items() if suffix in weights), "Detection")


def show(label: str, progress: float, status: str) -> None:
    """Redraws the current model's line in place: label, progress bar and status."""
    filled = int(progress * BAR_WIDTH)
    bar = ("=" * filled + ">")[:BAR_WIDTH].ljust(BAR_WIDTH)
    # \r returns to the start of the line and \033[K clears it, so a shorter status never leaves old text behind
    print(f"\r\033[K{label}  [{bar}] {progress:4.0%}  {status}", end="", flush=True)


@contextlib.contextmanager
def capture_output(log_file):
    """Sends all console output, including output from native code (PyTorch, ONNX), to log_file."""
    sys.stdout.flush()
    sys.stderr.flush()

    # Native code writes straight to file descriptors 1 and 2, so point them at the log file...
    saved = [os.dup(1), os.dup(2)]
    os.dup2(log_file.fileno(), 1)
    os.dup2(log_file.fileno(), 2)
    try:
        # ...and Python code writes through sys.stdout/sys.stderr, which on a Windows console bypass the descriptors.
        with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
            yield
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(saved[0], 1)
        os.dup2(saved[1], 2)
        os.close(saved[0])
        os.close(saved[1])


def user_accepts_license() -> bool:
    print(LICENSE_NOTICE)
    try:
        answer = input("Type 'yes' to confirm you have read and accept the model licenses: ")
    except EOFError:  # No interactive input available, e.g. stdin redirected
        return False
    return answer.strip().lower() == "yes"


def download(weights: str, target: Path, label: str) -> None:
    with urllib.request.urlopen(WEIGHTS_URL + weights) as response, open(target, "wb") as file:
        size = int(response.headers.get("Content-Length", 0))
        done = 0
        while chunk := response.read(256 * 1024):
            file.write(chunk)
            done += len(chunk)
            show(label, done / size if size else 0, "downloading")


def export_model(weights_path: Path, opset: int) -> None:
    # Imported here so that YOLO_VERBOSE, set in main(), is in place before Ultralytics loads
    from ultralytics import RTDETR, YOLO

    # The weights are already downloaded, so Ultralytics loads them from disk instead of downloading them again
    model = RTDETR(weights_path) if weights_path.name.startswith("rtdetr") else YOLO(weights_path)

    # nms=False is required for YOLO26 (keeps its built-in post-processing) and is the default for all other models.
    # dynamic=False gives the fixed input size YoloDotNet requires.
    onnx_path = model.export(format="onnx", opset=opset, nms=False, dynamic=False)

    # Copy, don't move: on Windows the temp folder only grants access to its owner, and a moved file keeps
    # those permissions. A copy is a new file that inherits the output folder's permissions.
    shutil.copyfile(onnx_path, OUTPUT_FOLDER / onnx_name(weights_path.name))


def process_model(weights: str, opset: int, label: str, work_folder: Path) -> bool:
    """Downloads and exports one model on a single progress line. Returns True on success."""
    with tempfile.TemporaryFile(mode="w+", encoding="utf-8", errors="replace") as log:
        try:
            weights_path = work_folder / weights
            download(weights, weights_path, label)

            show(label, 1, "exporting...")
            with capture_output(log):  # Hide Ultralytics' export output; it is shown only if the export fails
                export_model(weights_path, opset)

            show(label, 1, f"{GREEN}done{RESET}")
            print()
            return True
        except Exception as error:
            show(label, 1, f"{RED}FAILED: {error}{RESET}")
            print()
            log.seek(0)
            details = log.read().rstrip()
            if details:
                print(details)
            return False


def main() -> int:
    os.system("")  # Enables colours in the classic Windows console
    os.environ["YOLO_VERBOSE"] = "False"  # Silences Ultralytics' logging; must be set before Ultralytics is imported

    try:
        import ultralytics
    except ImportError:
        print('Ultralytics is not installed. Activate your Python environment or run: pip install "ultralytics>=8.4.0"')
        return 1

    missing = [m for m in MODELS if not (OUTPUT_FOLDER / onnx_name(m[0])).exists()]
    if not missing:
        print(f"All {len(MODELS)} models are already in {OUTPUT_FOLDER}. Delete a file to export it again.")
        return 0

    if not user_accepts_license():
        print("License not accepted. No models were downloaded.")
        return 1

    print(f"\nDownloading models from Ultralytics ({WEIGHTS_URL}) and exporting them with "
          f"Ultralytics {ultralytics.__version__}")
    print(f"{len(MODELS) - len(missing)} models already present, {len(missing)} to export\n")

    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    failed = []
    with tempfile.TemporaryDirectory() as work_folder:  # Downloaded weights are deleted when done
        for i, (weights, opset) in enumerate(missing, 1):
            label = f"[{i:2}/{len(missing)}] {task_name(weights):<14} {onnx_name(weights):<18} opset {opset}"
            if not process_model(weights, opset, label, Path(work_folder)):
                failed.append(onnx_name(weights))

    print(f"\nDone. {len(missing) - len(failed)} exported, {len(failed)} failed. Models are in: {OUTPUT_FOLDER}")
    if failed:
        print("Failed: " + ", ".join(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
