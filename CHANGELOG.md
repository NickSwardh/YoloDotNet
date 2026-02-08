# Changelog

All notable changes to **YoloDotNet** will be documented in this file.

## [4.2] — Usability, ROI & Segmentation Enhancements

### ✨ Added
- Option to define a **Region of Interest (ROI)** to run inference on specific areas of an image or video stream.
  - Useful for surveillance, monitoring zones, and performance optimization.
- Option to **draw edges** on segmented objects.
- Helper methods: `ToJson()` and `SaveJson()` for exporting inference results as JSON.
- Helper methods: `ToYoloFormat()` and `SaveYoloFormat()` for saving inference results as YOLO-annotated data.
- Helper method: `GetContourPoints()` for extracting ordered contour points from segmented objects.

### 🔄 Changed
- Updated **YOLOv26 inference execution** to align with other tasks, improving consistency and overall execution efficiency.

## [4.1] — Expanded Model Support & MIT Relicense

### ✨ Added
- Support for the **YOLOv26** model suite.
- Support for **RT-DETR** models.
- New **DirectML execution provider** for GPU acceleration on Windows.
- Improved video inference pipeline with better frame-to-frame stability.

### 🚀 Improved
- Reduced allocation pressure across all vision tasks.
- Lower per-frame latency for long-running inference workloads.
- General performance improvements across CPU and GPU execution providers.

### 🔄 Changed
- Project relicensed to **MIT** to maximize freedom for commercial and proprietary use.
  - No copyleft requirements
  - No network-use clauses
  - Model licensing remains the responsibility of the user

## [4.0] — Modular Execution Providers

### 🚨 Breaking Changes
- Execution providers are no longer bundled with the core package.
- Consumers **must reference exactly one execution provider NuGet package**.
- Provider setup code must be updated when upgrading from v3.x.

### ✨ Added
- Fully modular execution provider architecture.
- New execution providers:
  - **CPU** (baseline, cross-platform)
  - **CUDA / TensorRT** (NVIDIA GPUs, optional TensorRT acceleration)
  - **OpenVINO** (Intel CPUs and iGPUs on Windows/Linux)
  - **CoreML** (Apple Silicon on macOS)
- Support for **grayscale-only ONNX models**.
- Improved GPU execution behavior and predictability.
- Clear separation of native ONNX Runtime dependencies per provider.

### 🔧 Changed
- Core package is now execution-provider agnostic.
- Dependency graph simplified for predictable deployment.
- CUDA provider exposes clearer and more explicit GPU behavior.

### ⬆️ Updated
- **SkiaSharp** → 3.119.1
- **ONNX Runtime (CUDA)** → 1.23.2

---

## [Pre-4.0]

Changes prior to version **4.0** were released before a formal changelog was introduced.

For historical context, refer to:
- [NuGet package history](https://www.nuget.org/packages/YoloDotNet/#versions-body-tab)
