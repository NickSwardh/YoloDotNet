# <img src="https://github.com/NickSwardh/YoloDotNet/assets/35733515/994287a9-556c-495f-8acf-1acae8d64ac0" height=24> YoloDotNet 🚀
**Blazing-fast, production-ready YOLO inference for .NET**

**YoloDotNet** is a fully featured C# library for real-time computer vision using **YOLOv5u–v12**, **YOLO-World**, and **YOLO-E** models.  
Built on **.NET 8** and **ONNX Runtime**, it delivers **high-performance, predictable inference** on **Windows, Linux, and macOS** — with explicit control over execution, memory, and preprocessing.

No Python. No magic. Just fast, deterministic YOLO — done properly for .NET.

## ⭐ Why YoloDotNet?

YoloDotNet is designed for developers who need:

- ✅ **Pure .NET** — no Python runtime, no scripts  
- ✅ **Real performance** — CPU, CUDA / TensorRT, OpenVINO, CoreML  
- ✅ **Explicit configuration** — predictable accuracy and memory usage  
- ✅ **Production readiness** — engine caching, long-running stability  
- ✅ **Large image support** — not limited to toy resolutions  
- ✅ **Multiple vision tasks** — detection, OBB, segmentation, pose, classification  

Ideal for **desktop apps, backend services, and real-time vision pipelines** that require deterministic behavior and full control.

## 🆕 What’s New v4

- Modular execution providers (CPU, CUDA/TensorRT, OpenVINO, CoreML)
- New OpenVINO and CoreML providers
- Cleaner dependency graph
- Improved GPU behavior and predictability
- Grayscale ONNX model support

📖 Full release history: [CHANGELOG.md](./CHANGELOG.md)


## 🚀 Quick Start

### 1️⃣ Install the core package

```bash
dotnet add package YoloDotNet
```

### 2️⃣ Install **exactly one** execution provider

```bash
# CPU (recommended starting point)
dotnet add package YoloDotNet.ExecutionProvider.Cpu

# Hardware-accelerated execution (choose one)
dotnet add package YoloDotNet.ExecutionProvider.Cuda
dotnet add package YoloDotNet.ExecutionProvider.OpenVino
dotnet add package YoloDotNet.ExecutionProvider.CoreML
```

> 💡 **Note:** The CUDA execution provider includes optional **TensorRT** acceleration.  
> No separate TensorRT package is required.

### 3️⃣ Run object detection

```csharp
using SkiaSharp;
using YoloDotNet;
using YoloDotNet.ExecutionProvider.Cpu;

using var yolo = new Yolo(new YoloOptions
{
    ExecutionProvider = new CpuExecutionProvider("model.onnx")
});

using var image = SKBitmap.Decode("image.jpg");

var results = yolo.RunObjectDetection(image, confidence: 0.25, iou: 0.7);

image.Draw(results);
image.Save("result.jpg");
```

You’re now running YOLO inference in **pure C#**.

## 💡 Important: Accuracy Depends on Configuration

YOLO inference accuracy is **not automatic**.

Preprocessing settings such as image resize mode, sampling method, and confidence/IoU thresholds **must match how the model was trained**.  
These settings directly control the accuracy–performance tradeoff and should be treated as part of the model itself.

📖 **Before tuning models or comparing results, read:**  
👉 [Accuracy & Configuration Guide](./AccuracyAndConfiguration.md)

## Supported Tasks

| Classification | Object Detection | OBB Detection | Segmentation | Pose Estimation |
|---------------|------------------|---------------|--------------|-----------------|
| <img src="https://user-images.githubusercontent.com/35733515/297393507-c8539bff-0a71-48be-b316-f2611c3836a3.jpg" width=300> | <img src="https://user-images.githubusercontent.com/35733515/273405301-626b3c97-fdc6-47b8-bfaf-c3a7701721da.jpg" width=300> | <img src="https://github.com/NickSwardh/YoloDotNet/assets/35733515/d15c5b3e-18c7-4c2c-9a8d-1d03fb98dd3c" width=300> | <img src="https://github.com/NickSwardh/YoloDotNet/assets/35733515/3ae97613-46f7-46de-8c5d-e9240f1078e6" width=300> | <img src="https://github.com/NickSwardh/YoloDotNet/assets/35733515/b7abeaed-5c00-4462-bd19-c2b77fe86260" width=300> |
| <sub>[pexels.com](https://www.pexels.com/photo/hummingbird-drinking-nectar-from-blooming-flower-in-garden-5344570/)</sub> | <sub>[pexels.com](https://www.pexels.com/photo/men-s-brown-coat-842912/)</sub> | <sub>[pexels.com](https://www.pexels.com/photo/bird-s-eye-view-of-watercrafts-docked-on-harbor-8117665/)</sub> | <sub>[pexels.com](https://www.pexels.com/photo/man-riding-a-black-touring-motorcycle-903972/)</sub> | <sub>[pexels.com](https://www.pexels.com/photo/woman-doing-ballet-pose-2345293/)</sub> |

## 📁 Demos

Hands-on examples are available in the demo folder:

👉 [Browse the demo projects](./Demo)

Includes image inference, video streams, GPU acceleration, segmentation, and large-image workflows.

## Execution Providers

| Provider | Windows | Linux | macOS | Documentation |
|--------|---------|-------|-------|---------------|
| CPU | ✅ | ✅ | ✅ | [CPU README](./YoloDotNet.ExecutionProvider.Cpu/README.md) |
| CUDA / TensorRT | ✅ | ✅ | ❌ | [CUDA README](./YoloDotNet.ExecutionProvider.Cuda/README.md) |
| OpenVINO | ✅ | ✅ | ❌ | [OpenVINO README](./YoloDotNet.ExecutionProvider.OpenVino/README.md) |
| CoreML | ❌ | ❌ | ✅ | [CoreML README](./YoloDotNet.ExecutionProvider.CoreML/README.md) |

> ℹ️ Only **one** execution provider package may be referenced.  
> Mixing providers will cause native runtime conflicts.

## ⚡ Performance Characteristics

YoloDotNet focuses on stable, low-overhead inference where runtime cost is dominated by the execution provider and model.

📊 Benchmarks: [/test/YoloDotNet.Benchmarks](./test/YoloDotNet.Benchmarks/README.md)

- Stable latency after warm-up  
- Clean scaling from CPU → GPU → TensorRT  
- Predictable allocation behavior  
- Suitable for real-time and long-running services  

## 🚀 Modular Execution Providers

- Core package is provider-agnostic  
- Execution providers are separate NuGet packages  
- Native ONNX Runtime dependencies are isolated  

**Why this matters:** fewer conflicts, predictable deployment, and production-safe behavior.

## Support YoloDotNet

⭐ Star the repo  
💬 Share feedback  
🤝 Sponsor development  

[GitHub Sponsors](https://github.com/sponsors/NickSwardh)  
[PayPal](https://paypal.me/nickswardh)

## License

YoloDotNet is licensed under the [MIT License](./LICENSE.txt) and provides an ONNX inference
engine for YOLO models exported using Ultralytics YOLO tooling.

This project does **not** include, distribute, download, or bundle any
pretrained models.

Users must supply their own ONNX models.

YOLO ONNX models produced using Ultralytics tooling are typically licensed
under **AGPL-3.0** or a separate commercial license from Ultralytics.

YoloDotNet does **not** impose, modify, or transfer any license terms related
to user-supplied models.

**Users are solely responsible** for ensuring that their use of any model
complies with the applicable license terms, including requirements related
to commercial use, distribution, or network deployment.