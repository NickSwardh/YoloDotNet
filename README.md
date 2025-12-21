# <img src="https://github.com/NickSwardh/YoloDotNet/assets/35733515/994287a9-556c-495f-8acf-1acae8d64ac0" height=24> YoloDotNet

🚀 **Blazing-fast, production-ready YOLO inference for .NET**

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

YoloDotNet is ideal for developers building **desktop apps, backend services, or real-time vision pipelines** in .NET who need predictable performance and explicit control.

## 🆕 What’s New in v4.0

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
### 2️⃣ Install exactly one execution provider
```bash
# CPU (recommended starting point)
dotnet add package YoloDotNet.ExecutionProvider.Cpu

# Optional GPU acceleration
dotnet add package YoloDotNet.ExecutionProvider.Cuda
dotnet add package YoloDotNet.ExecutionProvider.TensorRt
dotnet add package YoloDotNet.ExecutionProvider.OpenVino
dotnet add package YoloDotNet.ExecutionProvider.CoreML
```

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

You’re now running YOLO inference in pure C#.

## 💡 Important: Accuracy Depends on Configuration

YOLO inference accuracy is **not automatic**.

Preprocessing settings such as image resize mode, sampling method, and confidence/IoU thresholds **must match how the model was trained**. These settings directly control the accuracy–performance tradeoff and should be treated as part of the model itself.

📖 **Before tuning models or comparing results, read:**  
👉 [Accuracy & Configuration Guide](./AccuracyAndConfiguration.md)

## Supported Tasks

| Classification | Object Detection | OBB Detection | Segmentation | Pose Estimation |
|----------------|------------------|---------------|--------------|-----------------|
| <img src="https://user-images.githubusercontent.com/35733515/297393507-c8539bff-0a71-48be-b316-f2611c3836a3.jpg" width=300> | <img src="https://user-images.githubusercontent.com/35733515/273405301-626b3c97-fdc6-47b8-bfaf-c3a7701721da.jpg" width=300> | <img src="https://github.com/NickSwardh/YoloDotNet/assets/35733515/d15c5b3e-18c7-4c2c-9a8d-1d03fb98dd3c" width=300> | <img src="https://github.com/NickSwardh/YoloDotNet/assets/35733515/3ae97613-46f7-46de-8c5d-e9240f1078e6" width=300> | <img src="https://github.com/NickSwardh/YoloDotNet/assets/35733515/b7abeaed-5c00-4462-bd19-c2b77fe86260" width=300> |
| <sub>[image from pexels.com](https://www.pexels.com/photo/hummingbird-drinking-nectar-from-blooming-flower-in-garden-5344570/)</sub> | <sub>[image from pexels.com](https://www.pexels.com/photo/men-s-brown-coat-842912/)</sub> | <sub>[image from pexels.com](https://www.pexels.com/photo/bird-s-eye-view-of-watercrafts-docked-on-harbor-8117665/)</sub> | <sub>[image from pexels.com](https://www.pexels.com/photo/man-riding-a-black-touring-motorcycle-903972/)</sub> | <sub>[image from pexels.com](https://www.pexels.com/photo/woman-doing-ballet-pose-2345293/)</sub> |

## 📁 Demos

Hands-on examples are available in the demo folder, covering common real-world scenarios:

👉 [Browse the demo projects](./Demo)

Including image inference, video streams, GPU acceleration, segmentation, and large-image workflows.

## Execution Providers

| Execution Provider | Windows | Linux | macOS | Documentation |
| ------------------ | ------- | ----- | ----- | ------------- |
| CPU | ✅ | ✅ | ✅ | [CPU README](./YoloDotNet.ExecutionProvider.Cpu/README.md) |
| CUDA / TensorRT | ✅ | ✅ | ❌ | [CUDA README](./YoloDotNet.ExecutionProvider.Cuda/README.md) |
| OpenVINO | ✅ | ✅ | ❌ | [OpenVINO README](./YoloDotNet.ExecutionProvider.OpenVino/README.md) |
| CoreML | ❌ | ❌ | ✅ | [CoreML README](./YoloDotNet.ExecutionProvider.CoreML/README.md) |

Each execution provider has its own README covering installation, runtime requirements, and provider-specific configuration.  
Real-world usage examples and recommended settings can be found in the demo projects.

> ℹ️ Only **one** execution provider package may be referenced.  
> Each provider ships its own native ONNX Runtime binaries; mixing providers will cause runtime conflicts.

## ⚡ Performance Characteristics

YoloDotNet focuses on stable, low-overhead inference behavior, where runtime cost is dominated by the selected execution provider and model, not framework overhead.

📊 See: [Benchmark methodology and results](/test/YoloDotNet.Benchmarks/README.md).

Internal benchmarks using **BenchmarkDotNet** across classification, object detection, OBB, pose estimation, and segmentation show that:

- Inference latency is stable after warm-up
- Performance scales cleanly with the selected execution provider (CPU → GPU → TensorRT)
- TensorRT precision modes (FP32, FP16, INT8) behave as expected
- Allocation behavior is predictable and bounded by output complexity
- Overall throughput is determined primarily by the execution provider and model configuration

For GPU-based providers, the first inference may be slower due to initialization or engine creation; subsequent runs operate at steady-state performance.

YoloDotNet is suitable for:
- Real-time pipelines
- Long-running services
- High-resolution image processing
- Deterministic production workloads

## 🚀 Modular Execution Providers

YoloDotNet uses a **fully modular execution architecture** that gives developers explicit control over native dependencies and runtime behavior.

- The core package is execution-provider agnostic
- Execution providers are delivered as separate NuGet packages
- Native ONNX Runtime dependencies are isolated per provider

### Why this matters
- Fewer native dependency conflicts
- Cleaner and more predictable deployment
- Consistent behavior across platforms and runtimes
- Easier integration into production and long-running services

💡 **Note for existing users**  
Projects upgrading from earlier versions must reference exactly one execution provider package and update provider setup accordingly. Existing models remain fully compatible.

## Support YoloDotNet
YoloDotNet is built and maintained independently. If you’ve found my project helpful, consider supporting its development:

⭐ Star the repository\
💬 Share feedback\
🤝 Consider sponsoring development

[![GitHub Sponsors](https://img.shields.io/badge/Sponsor-GitHub-ea4aaa?logo=githubsponsors&logoColor=white)](https://github.com/sponsors/NickSwardh) [![PayPal](https://img.shields.io/badge/Support-PayPal-00457C?logo=paypal&logoColor=white)](https://paypal.me/nickswardh)


Thank you. ❤️

## References & Acknowledgements

https://github.com/ultralytics/ultralytics \
https://github.com/sstainba/Yolov8.Net \
https://github.com/mentalstack/yolov5-net

## License

YoloDotNet is © 2023–2025 Niklas Swärd ([GitHub](https://github.com/NickSwardh/YoloDotNet))  
Licensed under the **GNU General Public License v3.0 or later**.

Commercial use is permitted under the terms of the GPL v3; however, derivative works must comply with the same license.

![License: GPL v3 or later](https://img.shields.io/badge/License-GPL_v3_or_later-blue)  
See the [LICENSE](./LICENSE.txt) file for the full license text.

This software is provided “as is”, without warranty of any kind.  
The author is not liable for any damages arising from its use.