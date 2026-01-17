# Information
YoloDotNet uses modular execution providers to run inference on different hardware backends.
Each provider targets a specific platform or accelerator and may require additional system-level dependencies such as runtimes, drivers, or SDKs.

Installing the NuGet package alone is not always sufficient — proper setup depends on the selected provider and the target system.\
This document describes the installation and usage of the **OpenVINO execution provider**.

# Core Library Requirement

All execution providers require the core **YoloDotNet** package.

### NuGet Package
```
dotnet add package YoloDotNet
```

# Execution Provider - OpenVINO

The OpenVINO execution provider enables optimized and hardware-accelerated inference on supported Intel® CPUs, GPUs, and NPUs using Intel® OpenVINO™.
It is particularly well-suited for Intel hardware and edge deployments where efficient CPU and NPU utilization is critical.

> ⚠️ **Note**\
> This execution provider depends on the Intel® OpenVINO™ Runtime.

### Requirements

- Intel CPU, GPU, or NPU supported by OpenVINO
- Windows or Linux (x64)
- Intel® OpenVINO™ Runtime

# Download and Install
[Download and install Intel® OpenVINO™ for your platform](https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/download.html).

### NuGet Package
```
dotnet add package YoloDotNet.ExecutionProvider.OpenVino
```

> YoloDotNet.ExecutionProvider.OpenVino v1.1 requires YoloDotNet version 4.1.

# Usage Example:
```csharp
using var yolo = new Yolo(new YoloOptions
{
    ExecutionProvider = new OpenVinoExecutionProvider(
        model: "path/to/model.onnx",

        // If no OpenVINO configuration is provided, the OpenVINO CPU device is used by default.
        openVino: new OpenVino()
        {
            DeviceType = "GPU", // Device type: "CPU", "GPU", "MYRIAD", etc.
            Precision = Precision.FP16,
            CachePath = Path.Combine("path/to/cache/folder"),
            ModelPriority = ModelPriority.HIGH
        }
    ),

    // ...other options

});
```