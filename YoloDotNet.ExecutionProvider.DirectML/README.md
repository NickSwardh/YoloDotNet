# YoloDotNet Execution Provider: DirectML
**GPU-accelerated inference on Windows, on any DirectX 12 GPU**

Execution providers are separate packages that decide where YoloDotNet runs inference.
The DirectML execution provider runs inference on the GPU with Microsoft DirectML, built on DirectX 12.
It works with GPUs from NVIDIA, AMD and Intel without vendor-specific SDKs, which makes it well suited to:

- Windows desktop applications
- Systems without CUDA or OpenVINO
- Applications that must run on whatever GPU the user has

## 📋 Requirements

- Windows 10 or later (x64)
- A GPU with DirectX 12 support (NVIDIA, AMD or Intel)
- Up-to-date GPU drivers

DirectML is part of Windows and ONNX Runtime, so there is no SDK to install.

## 📦 Installation

```bash
dotnet add package YoloDotNet
dotnet add package YoloDotNet.ExecutionProvider.DirectML
```

> ⚠️ **Note**\
> Reference exactly one execution provider package in your project.\
> `YoloDotNet.ExecutionProvider.DirectML` 1.2 requires YoloDotNet 4.3.

## 🚀 Usage

```csharp
using YoloDotNet;
using YoloDotNet.ExecutionProvider.DirectML;

using var yolo = new Yolo(new YoloOptions
{
    ExecutionProvider = new DirectMLExecutionProvider("path/to/model.onnx")
});
```

## ⚙️ Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model` | – | Path to the ONNX model, or the model as a `byte[]`. |
| `gpuId` | `0` | The GPU to run on. `-1` runs on the CPU instead. |

## 💡 Notes

- DirectML favors broad compatibility and easy deployment over vendor-specific optimizations.
- Performance depends on the GPU driver and the model. On NVIDIA GPUs, the CUDA execution provider is usually faster.

## 🛠️ Troubleshooting

If inference fails to start:

- Check that the GPU supports DirectX 12 and that its drivers are up to date.
- Check that your project references only one execution provider package.
- Check that the ONNX model is supported by ONNX Runtime's DirectML backend.

## 📚 More information

- [YoloDotNet README](https://github.com/NickSwardh/YoloDotNet): getting started, supported models and the other execution providers
- [CUDA execution provider](https://github.com/NickSwardh/YoloDotNet/tree/master/YoloDotNet.ExecutionProvider.Cuda): faster inference on NVIDIA GPUs
- [Demos](https://github.com/NickSwardh/YoloDotNet/tree/master/Demo): ready-made examples for every task
