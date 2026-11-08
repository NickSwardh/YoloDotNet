# YoloDotNet Execution Provider: CUDA and TensorRT
**GPU-accelerated inference on NVIDIA GPUs, with optional TensorRT**

Execution providers are separate packages that decide where YoloDotNet runs inference.
The CUDA execution provider runs inference on NVIDIA GPUs with ONNX Runtime's CUDA backend.
Optionally, NVIDIA TensorRT optimizes the model for your GPU for the highest throughput and lowest latency.
With TensorRT, operations that TensorRT can't run fall back to CUDA, so they stay on the GPU.

## 📋 Requirements

- An NVIDIA GPU
- Windows or Linux (x64). CUDA and TensorRT are not available on macOS.
- [CUDA Toolkit 13.x](https://developer.nvidia.com/cuda-toolkit-archive)
- [cuDNN 9.x](https://developer.nvidia.com/cudnn-archive)
- Optional: [TensorRT 10.13.3](https://developer.nvidia.com/tensorrt) for CUDA 13.x

> ⚠️ **Important**\
> This execution provider depends on native CUDA and cuDNN libraries.\
> Installing the NuGet package alone is not enough: CUDA and cuDNN must be installed on the system, as described below.

## 📦 Installation

### 1. NuGet packages

```bash
dotnet add package YoloDotNet
dotnet add package YoloDotNet.ExecutionProvider.Cuda
```

> ⚠️ **Note**\
> Reference exactly one execution provider package in your project.\
> `YoloDotNet.ExecutionProvider.Cuda` 1.2 requires YoloDotNet 4.3.

### 2. CUDA and cuDNN on Windows

1. Install the [CUDA Toolkit 13.x](https://developer.nvidia.com/cuda-toolkit-archive) and [cuDNN 9.x](https://developer.nvidia.com/cudnn-archive).
2. Find the folder that contains the cuDNN DLL files, typically
   `C:\Program Files\NVIDIA\CUDNN\v9.x\bin\v13.x` (replace `v9.x` and `v13.x` with your installed versions).
3. Add that folder to the system `PATH`:
   1. Search for **Edit the system environment variables** in Windows and open it.
   2. Click **Environment Variables**.
   3. Under **System variables**, select `Path` and click **Edit**.
   4. Click **New**, paste the cuDNN folder path and click **OK** to close all dialogs.
4. Restart your computer.

### 3. CUDA and cuDNN on Linux

1. Install the [CUDA Toolkit 13.x](https://developer.nvidia.com/cuda-toolkit-archive) for your distribution.
2. Install [cuDNN 9.x](https://developer.nvidia.com/cudnn-archive) for your distribution.
3. Restart your computer.

### 4. TensorRT (optional)

TensorRT can significantly speed up inference by optimizing the model for your specific GPU.

1. [Download](https://developer.nvidia.com/tensorrt) the TensorRT 10.13.3 release for CUDA 13.x.
2. **Windows:** extract the archive, then add its `lib` folder to the system `PATH` (same steps as for cuDNN) and restart your computer.\
   **Linux:** follow NVIDIA's [TensorRT installation instructions](https://docs.nvidia.com/deeplearning/tensorrt/latest/installing-tensorrt/installing.html).

## 🚀 Usage

**CUDA:**

```csharp
using YoloDotNet;
using YoloDotNet.ExecutionProvider.Cuda;

using var yolo = new Yolo(new YoloOptions
{
    ExecutionProvider = new CudaExecutionProvider("path/to/model.onnx")
});
```

**TensorRT** (FP16, with the engine cached on disk):

```csharp
using var yolo = new Yolo(new YoloOptions
{
    ExecutionProvider = new CudaExecutionProvider(
        model: "path/to/model.onnx",
        trtConfig: new TensorRt
        {
            Precision = TrtPrecision.FP16,
            EngineCachePath = "path/to/cache/folder"
        })
});
```

The first run with TensorRT builds an engine for your GPU, which can take from several seconds to a few minutes.
Later runs load it from `EngineCachePath` and start quickly.

## ⚙️ Options

### `CudaExecutionProvider`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model` | – | Path to the ONNX model, or the model as a `byte[]`. |
| `gpuId` | `0` | The GPU to run on. `-1` runs on the CPU instead (not allowed with `enableCudaGraph`). |
| `trtConfig` | `null` | TensorRT settings (see below). Without them, inference runs on CUDA. |
| `enableCudaGraph` | `false` | Replays inference as a CUDA Graph for faster calls; see [CUDA Graphs](#-cuda-graphs). |

### `TensorRt`

| Property | Default | Description |
|----------|---------|-------------|
| `Precision` | `FP32` | `FP32`: full precision. `FP16`: faster, with minimal accuracy loss on supported GPUs. `INT8`: fastest, but needs a calibration cache. |
| `BuilderOptimizationLevel` | `3` | How much effort TensorRT spends optimizing when it builds the engine, from 0 to 5. Higher levels build slower but can run faster. |
| `EngineCachePath` | Not set | Folder where TensorRT stores and loads its engine. The folder must exist. If not set, the engine is rebuilt on every start. |
| `EngineCachePrefix` | `YoloDotNet` | File name prefix for the engine files, to keep engines for several models or settings in one folder. |
| `Int8CalibrationCacheFile` | Not set | Path to the INT8 calibration cache. Required with `INT8`; ignored otherwise. |

For details on the engine cache and on creating an INT8 calibration cache, see the [TensorRT demo](https://github.com/NickSwardh/YoloDotNet/tree/master/Demo/TensorRTDemo).

## ⚡ CUDA Graphs

CUDA Graph replay makes every inference after the first one faster, with CUDA and with TensorRT. It is off by default:

```csharp
new CudaExecutionProvider("path/to/model.onnx", enableCudaGraph: true);                     // CUDA
new CudaExecutionProvider("path/to/model.onnx", trtConfig: tensorRt, enableCudaGraph: true); // TensorRT
```

Instead of launching the GPU work step by step on every call, the first inference records it and later calls replay it in one launch.
The results are identical. Measured with yolo26s detection on an RTX 5070 Ti (ms per image):

| | Without CUDA Graph | With CUDA Graph |
|---|---|---|
| CUDA | 4.88 | 3.51 |
| TensorRT FP32 | 2.99 | 2.27 |
| TensorRT FP16 | 2.11 | 1.40 |

The first inference captures the graph and takes longer, so benchmark later calls after a warm-up.
Input and output stay on the GPU at fixed addresses; every call uploads fresh input and returns results that you own.
Calls on one provider run one at a time, including the transfers, so you can process the returned results while the next inference runs.

**Requirements and limits:**

- The model has one input and one or two outputs, in Float32 or Float16.
- The whole model must run on the GPU: on CUDA, or on TensorRT when `trtConfig` is set.
  If it doesn't, creating the provider fails; it never returns wrong results.
  Export models as described in [ONNX Model Export Requirements](https://github.com/NickSwardh/YoloDotNet#-onnx-model-export-requirements).
- The CUDA 13 runtime libraries must be found on the system (`cudart64_13.dll` on Windows, `libcudart.so.13` on Linux).

## 💡 Choosing a setup

- **CUDA** gives GPU acceleration with the least setup.
- **TensorRT** gives the highest performance, in exchange for an engine build on the first run and an engine cache to manage.
- **CUDA Graphs** lower the latency of both. The fastest setup is TensorRT FP16 with CUDA Graphs.

## 🛠️ Troubleshooting

- **The provider fails to load CUDA or cuDNN:** check that CUDA 13.x and cuDNN 9.x are installed and that the cuDNN folder is on the system `PATH`, then restart.
- **TensorRT fails to start:** check that the TensorRT `lib` folder is on the `PATH` (Windows) and that `EngineCachePath` points to an existing, writable folder.
- **Creating the provider with CUDA Graphs fails because some operations run on the CPU:** export the model as described in the main README.
  Otherwise use the model with TensorRT, which runs it fully on the GPU, or turn CUDA Graphs off.

## 📚 More information

- [YoloDotNet README](https://github.com/NickSwardh/YoloDotNet): getting started, supported models and the other execution providers
- [TensorRT demo](https://github.com/NickSwardh/YoloDotNet/tree/master/Demo/TensorRTDemo): TensorRT, the engine cache and INT8 calibration
- [Demos](https://github.com/NickSwardh/YoloDotNet/tree/master/Demo): ready-made examples for every task
