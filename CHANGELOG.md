# Changelog

All notable changes to **YoloDotNet** are documented in this file.

## [4.0] — 2025-12-14

### Added
- Fully modular execution provider architecture
- New execution providers:
  - Intel OpenVINO (Windows, Linux)
  - Apple CoreML (macOS)
- Support for grayscale (single-channel) ONNX models

### Changed
- Execution providers are now distributed as separate NuGet packages
- Core package is execution-provider agnostic
- CUDA execution provider behavior has been made more explicit and predictable
- Improved handling of TensorRT precision modes

### Updated
- SkiaSharp updated to 3.119.1
- ONNX Runtime (CUDA) updated to 1.23.2

### Migration Notes
- Projects upgrading from v3.x must explicitly reference one execution provider package
- Mixing execution providers is not supported due to native dependency conflicts

## [Pre-4.0]

Changes prior to version **4.0** were released before a formal changelog was introduced.

For historical context, refer to:
- [NuGet package history](https://www.nuget.org/packages/YoloDotNet/#versions-body-tab)