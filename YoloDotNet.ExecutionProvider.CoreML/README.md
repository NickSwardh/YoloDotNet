# Information
YoloDotNet uses modular execution providers to run inference on different hardware backends.
Each provider targets a specific platform or accelerator and may require additional system-level dependencies such as runtimes, drivers, or SDKs.

Installing the NuGet package alone is not always sufficient — proper setup depends on the selected provider and the target system.\
This document describes the installation and usage of the **CoreML execution provider**.

# Core Library Requirement

All execution providers require the core **YoloDotNet** package.

### NuGet Package
```
dotnet add package YoloDotNet
```

# Execution Provider - CoreML

The CoreML execution provider enables hardware-accelerated inference on Apple devices using Apple’s Core ML framework.\
It automatically leverages Apple Silicon accelerators such as the Neural Engine, GPU, or CPU depending on device capabilities.

> ⚠️ **Note**\
> CoreML is only available on Apple platforms and is supported on macOS 10.15 or later.

### Requirements

- macOS 10.15 **or later**

# Installation

No additional installation or configuration is required.\
CoreML is included with macOS and is automatically available at runtime on supported devices.

### NuGet Package
```
dotnet add package YoloDotNet.ExecutionProvider.CoreML
```

# Usage Example:
```csharp
using YoloDotNet;
using YoloDotNet.ExecutionProvider.CoreML;

using var yolo = new Yolo(new YoloOptions
{
    ExecutionProvider = new CoreMLExecutionProvider(
        model: "path/to/model.onnx",

        // Automatically selects the best available CoreML compute units
        // (Neural Engine, GPU, or CPU)
        adaptive: true
    ),

    // ...other options
});
```