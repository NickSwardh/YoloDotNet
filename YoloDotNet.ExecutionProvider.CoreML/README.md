# YoloDotNet Execution Provider: CoreML
**Hardware-accelerated inference on Apple devices with Core ML**

Execution providers are separate packages that decide where YoloDotNet runs inference.
The CoreML execution provider runs inference with Apple's Core ML framework,
which uses the Neural Engine, the GPU or the CPU, depending on the device and the model.

## 📋 Requirements

- macOS 10.15 or later

Core ML is part of macOS, so there is nothing else to install.

## 📦 Installation

```bash
dotnet add package YoloDotNet
dotnet add package YoloDotNet.ExecutionProvider.CoreML
```

> ⚠️ **Note**\
> Reference exactly one execution provider package in your project.\
> `YoloDotNet.ExecutionProvider.CoreML` 1.2 requires YoloDotNet 4.3.

## 🚀 Usage

```csharp
using YoloDotNet;
using YoloDotNet.ExecutionProvider.CoreML;

using var yolo = new Yolo(new YoloOptions
{
    ExecutionProvider = new CoreMLExecutionProvider("path/to/model.onnx")
});
```

## ⚙️ Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model` | – | Path to the ONNX model, or the model as a `byte[]`. |
| `adaptive` | `true` | `true` runs inference with Core ML, which picks the Neural Engine, the GPU or the CPU. `false` doesn't use Core ML: inference runs on ONNX Runtime's built-in CPU backend. |

## 📚 More information

- [YoloDotNet README](https://github.com/NickSwardh/YoloDotNet): getting started, supported models and the other execution providers
- [Accuracy & Configuration Guide](https://github.com/NickSwardh/YoloDotNet/blob/master/AccuracyAndConfiguration.md): resize mode, sampling and thresholds
- [Demos](https://github.com/NickSwardh/YoloDotNet/tree/master/Demo): ready-made examples for every task
