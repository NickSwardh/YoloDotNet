# Information
YoloDotNet uses modular execution providers to run inference on different hardware backends.
Each provider targets a specific platform or accelerator and may require additional system-level dependencies such as runtimes, drivers, or SDKs.

Installing the NuGet package alone is not always sufficient — proper setup depends on the selected provider and the target system.\
This document describes the installation, requirements, and usage of the **CPU execution provider**.


# Core Library Requirement

All execution providers require the core **YoloDotNet** package, which contains the shared inference pipeline, models, and APIs.

### NuGet Package
```
dotnet add package YoloDotNet
```

# Execution Provider - CPU

The CPU execution provider runs inference on the system CPU using ONNX Runtime’s built-in CPU backend.\
It is the most portable execution provider and requires no additional dependencies.

# Requirements:

- Any x64-compatible CPU
- Windows, Linux, or macOS

**Notes**:
- The CPU execution provider is always available and works on all supported platforms.
- Performance is lower compared to hardware-accelerated providers, but it is ideal for development, testing, and environments without GPU or NPU support.

# Installation

No additional installation or configuration is required.\
Install the NuGet package and run your application — that’s it.

### NuGet Package
```
dotnet add package YoloDotNet.ExecutionProvider.Cpu
```

# Usage Example:
```csharp
using YoloDotNet;
using YoloDotNet.ExecutionProvider.Cpu;

using var yolo = new Yolo(new YoloOptions
{
    ExecutionProvider = new CpuExecutionProvider(model: "path/to/model.onnx"),

    // ...other options
});
```