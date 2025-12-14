# Information
YoloDotNet uses modular execution providers to run inference on different hardware backends.
Each provider targets a specific platform or accelerator and may require additional system-level dependencies such as runtimes, drivers, or SDKs.

Installing the NuGet package alone is not always sufficient — proper setup depends on the selected provider and the target system.\
This document describes the installation, requirements, and usage of the **CUDA & TensorRT execution provider**.

# Core Library Requirement

All execution providers require the core **YoloDotNet** package, which contains the shared inference pipeline, models, and APIs.

### NuGet Package
```
dotnet add package YoloDotNet
```

# Execution Provider - CUDA and TensorRT

The CUDA & TensorRT execution provider enables GPU-accelerated inference on NVIDIA GPUs using ONNX Runtime’s CUDA backend.\
Optionally, NVIDIA TensorRT can be enabled to further optimize models for maximum throughput and ultra-low latency.

> ⚠️ **Note**\
> This execution provider is supported on Windows and Linux only.\
> CUDA and TensorRT are not available on macOS.

### Requirements

- [CUDA Toolkit 12.x](https://developer.nvidia.com/cuda-toolkit-archive)
- [cuDNN 9.x](https://developer.nvidia.com/cudnn-archive)
- Windows or Linux (x64)

> **Important**\
> This execution provider depends on native CUDA and cuDNN libraries.\
> Installing the NuGet package alone is not sufficient — system-level dependencies must be installed correctly.

# Installation (Windows)

- ### CUDA
    Download and install the following from NVIDIA’s official websites:

    - [CUDA Toolkit 12.x](https://developer.nvidia.com/cuda-toolkit-archive)
    - [cuDNN 9.x](https://developer.nvidia.com/cudnn-archive)

    After installing cuDNN, locate the folder containing the cuDNN DLL files.
    This is typically:

    ```
    C:\Program Files\NVIDIA\CUDNN\v9.x\bin\v12.x
    ```

    *(Replace v9.x and v12.x with the versions installed on your system)*

    Add cuDNN to the System PATH

    1. Copy the full folder path to your cuDNN `bin\v12.x` folder

    2. Search `Edit the system environment variables` in Windows search and select it.

    3. Click `Environment Variables`.

    4. Under `System variables`, select `Path` and click Edit.

    5. Click `New` and paste the copied cuDNN path.

    6. Click `OK` to save and close all dialogs.

    7. Reboot your system.

- ### TensorRT (optional)

    TensorRT is NVIDIA’s high-performance inference engine and can significantly improve performance by optimizing models for your specific GPU.

    1. [Download](https://developer.nvidia.com/tensorrt) the `TensorRT 10.13.3` release for `CUDA 12.x`.

    2. Extract the archive to a folder on your system.

    3. Locate the `lib` folder inside the extracted TensorRT folder.

    4. Copy the full path to this lib folder.

    5. Add the path to your system's `PATH` environment variable (same process as described in the CUDA installation steps).

    6. Reboot your system.

# Installation (Linux)

- ### CUDA
    1. Install [CUDA Toolkit 12.x](https://developer.nvidia.com/cuda-toolkit-archive) for your Linux distribution

    2. Install [cuDNN 9.x](https://developer.nvidia.com/cudnn-archive) for your Linux distribution

    3. Reboot your system

- ### TensorRT (optional)
    1. [Download](https://developer.nvidia.com/tensorrt) the `TensorRT 10.13.3` release for `CUDA 12.x`.
    
    2. Follow NVIDIA’s [TensorRT installation instructions for Linux](https://docs.nvidia.com/deeplearning/tensorrt/latest/installing-tensorrt/installing.html).

# NuGet Package
```
dotnet add package YoloDotNet.ExecutionProvider.Cuda
```

# Usage Example:
```csharp
using YoloDotNet;
using YoloDotNet.ExecutionProvider.Cuda;

using var yolo = new Yolo(new YoloOptions
{
    ExecutionProvider = new CudaExecutionProvider(
        model: "path/to/model.onnx",

        // GPU device index (default: 0)
        gpuId: 0,

        // Optional TensorRT configuration for maximum performance
        trtConfig: new TensorRt
        {
            Precision = TrtPrecision.FP16,
            EngineCachePath = "path/to/cache/folder",
            EngineCachePrefix = "MyCachePrefix"
        }
    ),

    // ...other options
});

// See the TensorRT demo project for advanced configuration options.
```

# Notes & Recommendations

- Use CUDA only if you want simple GPU acceleration with minimal setup.
- Enable TensorRT if you need maximum performance and are comfortable managing engine caches.
- TensorRT engine generation happens once per model and configuration and is cached for subsequent runs.
- CUDA and TensorRT are not supported on macOS.