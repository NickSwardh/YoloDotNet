# Changelog

All notable changes to **YoloDotNet** will be documented in this file.

## [4.0] — 2025-12-14

### Breaking Changes
- Execution providers are no longer bundled with the core package.
- Consumers **must reference exactly one execution provider NuGet package**.
- Provider setup code must be updated when upgrading from v3.x.

### Added
- Fully modular execution provider architecture.
- New execution providers:
  - **CPU** (baseline, cross-platform)
  - **CUDA / TensorRT** (NVIDIA GPUs, optional TensorRT acceleration)
  - **OpenVINO** (Intel CPUs and iGPUs on Windows/Linux)
  - **CoreML** (Apple Silicon on macOS)
- Support for **grayscale-only ONNX models**.
- Improved GPU execution behavior and predictability.
- Clear separation of native ONNX Runtime dependencies per provider.

### Changed
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
