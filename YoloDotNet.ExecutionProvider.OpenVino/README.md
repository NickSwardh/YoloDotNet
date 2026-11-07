# YoloDotNet Execution Provider: OpenVINO
**Optimized inference on Intel GPUs and NPUs with Intel® OpenVINO™**

Execution providers are separate packages that decide where YoloDotNet runs inference.
The OpenVINO execution provider runs inference on Intel GPUs and NPUs with Intel® OpenVINO™,
which makes it well suited to Intel hardware and edge devices.

## 📋 Requirements

- An Intel GPU or NPU supported by OpenVINO
- Windows or Linux (x64)
- [Intel® OpenVINO™ Runtime](https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/download.html)

## 📦 Installation

1. [Download and install Intel® OpenVINO™](https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/download.html) for your platform.
2. Add the NuGet packages:

   ```bash
   dotnet add package YoloDotNet
   dotnet add package YoloDotNet.ExecutionProvider.OpenVino
   ```

> ⚠️ **Note**\
> Reference exactly one execution provider package in your project.\
> `YoloDotNet.ExecutionProvider.OpenVino` 1.2 requires YoloDotNet 4.3.

## 🚀 Usage

```csharp
using YoloDotNet;
using YoloDotNet.ExecutionProvider.OpenVino;

using var yolo = new Yolo(new YoloOptions
{
    ExecutionProvider = new OpenVinoExecutionProvider(
        model: "path/to/model.onnx",
        openVino: new OpenVino
        {
            DeviceType = "GPU",
            Precision = Precision.FP16,
            CachePath = "path/to/cache/folder"
        })
});
```

## ⚙️ Options

### `OpenVinoExecutionProvider`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model` | – | Path to the ONNX model, or the model as a `byte[]`. |
| `openVino` | `null` | OpenVINO settings (see below). Without them, inference runs on the CPU. |

### `OpenVino`

| Property | Default | Description |
|----------|---------|-------------|
| `DeviceType` | Not set | `"GPU"` or `"NPU"` runs inference with OpenVINO on that device. Not set, or `"CPU"`, runs on the CPU (see the note below). |
| `Precision` | `FP32` | `FP32`: full precision. `FP16`: faster, with minimal accuracy loss. |
| `Threads` | `8` | Number of inference threads. |
| `Streams` | `1` | Number of parallel inference streams. |
| `CachePath` | Not set | Folder where OpenVINO caches compiled models, so later starts are faster. Optional but recommended: without it, OpenVINO compiles the model on every start. |
| `ModelPriority` | `DEFAULT` | Priority of this model when several models share the device: `DEFAULT`, `LOW`, `MEDIUM` or `HIGH`. |

> ⚠️ **Note**\
> The OpenVINO settings apply to `"GPU"` and `"NPU"` only. With `DeviceType` not set or set to `"CPU"`,
> inference runs on ONNX Runtime's built-in CPU backend, the same as the CPU execution provider, and the other settings are ignored.

## 📚 More information

- [YoloDotNet README](https://github.com/NickSwardh/YoloDotNet): getting started, supported models and the other execution providers
- [OpenVINO documentation](https://docs.openvino.ai/): supported devices and performance tuning
- [Demos](https://github.com/NickSwardh/YoloDotNet/tree/master/Demo): ready-made examples for every task
