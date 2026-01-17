# Information

YoloDotNet uses modular execution providers to run inference on
different hardware backends. Each provider targets a specific platform
or accelerator and may require additional system-level dependencies such
as runtimes, drivers, or operating system support.

Installing the NuGet package alone is not always sufficient --- proper
setup depends on the selected execution provider and the target system.\
This document describes the installation and usage of the **DirectML
execution provider**.

# Core Library Requirement

All execution providers require the core **YoloDotNet** package.

### NuGet Package

``` bash
dotnet add package YoloDotNet
```

# Execution Provider -- DirectML

The DirectML execution provider enables GPU-accelerated inference on
**Windows** using **Microsoft DirectML** via **ONNX Runtime**.

DirectML provides a hardware-agnostic GPU backend built on **DirectX
12**, allowing inference on a wide range of GPUs without vendor-specific SDKs.

This provider is well-suited for: - Windows desktop applications -
Systems without CUDA or OpenVINO - Environments where a unified GPU API
is preferred

> ⚠️ **Note**\
> This execution provider depends on **Windows DirectML support via
> DirectX 12**. No additional SDKs are required beyond a supported
> Windows version and GPU driver.

# Requirements

-   Windows 10 or newer (x64)
-   GPU with DirectX 12 support (NVIDIA, AMD, or Intel)
-   Up-to-date GPU drivers
-   ONNX model compatible with DirectML execution

# Installation

### NuGet Package

``` bash
dotnet add package YoloDotNet.ExecutionProvider.DirectML
```

No separate DirectML or DirectX SDK installation is required. DirectML
is provided as part of Windows and ONNX Runtime.

# Usage Example

``` csharp
using YoloDotNet;
using YoloDotNet.ExecutionProvider.DirectML;

using var yolo = new Yolo(new YoloOptions
{
    ExecutionProvider = new DirectMLExecutionProvider(
        model: "path/to/model.onnx",
        gpuId: 0
    ),

    // ...other options
});
```

# Notes on Performance & Compatibility

-   DirectML performance depends heavily on GPU driver quality and model
    structure.
-   For maximum throughput on NVIDIA hardware, the CUDA execution
    provider may offer higher performance.

DirectML prioritizes **broad compatibility and ease of deployment** over
vendor-specific optimizations.

# Troubleshooting

If inference fails to initialize:
- Verify that the GPU supports DirectX 12.
- Ensure GPU drivers are up to date.
- Confirm that only **one execution provider** package is referenced.
- Validate that the ONNX model is supported by ONNX Runtime DirectML.
