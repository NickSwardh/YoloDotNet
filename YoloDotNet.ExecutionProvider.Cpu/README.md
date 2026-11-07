# YoloDotNet Execution Provider: CPU
**Run YoloDotNet on any CPU, with no extra setup**

Execution providers are separate packages that decide where YoloDotNet runs inference.
The CPU execution provider uses ONNX Runtime's built-in CPU backend. It is the most portable execution provider:
it works on every supported platform and needs nothing beyond the NuGet packages.

## 📋 Requirements

- An x64 or ARM64 CPU
- Windows, Linux or macOS (including Apple Silicon)

## 📦 Installation

```bash
dotnet add package YoloDotNet
dotnet add package YoloDotNet.ExecutionProvider.Cpu
```

No other installation or configuration is needed.

> ⚠️ **Note**\
> Reference exactly one execution provider package in your project.\
> `YoloDotNet.ExecutionProvider.Cpu` 1.2 requires YoloDotNet 4.3.

## 🚀 Usage

```csharp
using YoloDotNet;
using YoloDotNet.ExecutionProvider.Cpu;

using var yolo = new Yolo(new YoloOptions
{
    ExecutionProvider = new CpuExecutionProvider("path/to/model.onnx")
});
```

## ⚙️ Options

| Parameter | Description |
|-----------|-------------|
| `model` | Path to the ONNX model, or the model as a `byte[]`. |

## 💡 Notes

- The CPU is slower than a GPU or NPU, but it is always available, which makes it a good choice for development, testing and machines without an accelerator.
- To process many images, run them in parallel on one `Yolo` instance; inference is thread-safe.
  See [Speed Tips](https://github.com/NickSwardh/YoloDotNet#-speed-tips) in the main README.

## 📚 More information

- [YoloDotNet README](https://github.com/NickSwardh/YoloDotNet): getting started, supported models and the other execution providers
- [Accuracy & Configuration Guide](https://github.com/NickSwardh/YoloDotNet/blob/master/AccuracyAndConfiguration.md): resize mode, sampling and thresholds
- [Demos](https://github.com/NickSwardh/YoloDotNet/tree/master/Demo): ready-made examples for every task
