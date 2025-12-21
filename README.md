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

<!-- # ℹ️ Accuracy Depends on Configuration

**The accuracy of your results depends heavily on how you configure preprocessing and thresholds**. Even with a correctly trained model, mismatched settings can cause accuracy loss. There is no one-size-fits-all configuration — optimal values depend on your dataset, how your model was trained, and your specific application needs.

### 🔑 Key Factors
1. **Image Preprocessing & Resize Mode**
    * Controlled via `ImageResize`.
    * Must match the preprocessing used during training to ensure accurate detections.
      | **Proportional dataset** | **Stretched dataset** |
      |:------------:|:---------:|
      |<img width="160" height="107" alt="proportional" src="https://github.com/user-attachments/assets/e95a2c5c-8032-4dee-a05a-a73b062a4965" />|<img width="160" height="160" alt="stretched" src="https://github.com/user-attachments/assets/90fa31cf-89dd-41e4-871c-76ae3215171f" />|
      |Use `ImageResize.Proportional` (default) if the dataset images were not distorted during training and their aspect ratio was preserved.|Use `ImageResize.Stretched` if the dataset images were resized directly to the model’s input size (e.g., 640x640), ignoring the original aspect ratio.|
    * **Important:** Selecting the wrong resize mode can reduce detection accuracy.
2. **Sampling Options**
    * Controlled via `SamplingOptions`.
    * Define how pixel data is resampled when resizing (e.g., **`Cubic`**, **`NearestNeighbor`**, **`Bilinear`**). This choice has a direct impact on the accuracy of your detections, as different resampling methods can slightly alter object shapes and edges.
    * YoloDotNet default:
        ```csharp
        SamplingOptions = new SKSamplingOptions(SKFilterMode.Nearest, SKMipmapMode.None);
        ```
        💡 **Tip:** Check the [ResizeImage Benchmarks](./test/YoloDotNet.Benchmarks/ImageExtensionTests/ResizeImageTests.cs) for examples of different `SamplingOptions` and to help you choose the best settings for your needs.
3. **Confidence & IoU Thresholds**
    * Results are filtered based on thresholds you set during inference:
        * Confidence → Minimum probability for a detection to count.
        * IoU (**Intersection-over-Union**) → Overlap required to merge/suppress detections.
    * Too low → more false positives.
    * Too high → more missed detections.
    * Fine-tune these values for your dataset and application.

💡 **Tip**: Start with the defaults, then adjust `ImageResize`, `SamplingOptions`, and `Confidence/IoU` thresholds based on your dataset for optimal detection results. -->





















# 📖 Table of Contents
- [What's new in v4.0](#-yolodotnet-v40--modular-execution-maximum-performance)
- [Requirements](#requirements)
  - [ONNX Model (opset 17)](#onnx-model-opset-17)
  - [Installation](#installation)
- [Customization](#make-it-yours--customize-the-look)
- [Support YoloDotNet](#support-yolodotnet)
- [License](#license)
<!-- - [Export Yolo Models to ONNX](#export-yolo-models-to-onnx-with-opset17)
- [Quick Start & Demos](#-quick-start-dive-into-the-demos)
- [Bare Minimum Example](#bare-minimum--get-up-and-running-in-a-snap)
- 💡 [Accuracy Depends on Configuration](#%EF%B8%8F-accuracy-depends-on-configuration)
- [Customize Detection Overlay](#make-it-yours--customize-the-look)
- [Support YoloDotNet](#support-yolodotnet)
- [License](#license) -->

# 🚀 YoloDotNet v4.0 — Modular Execution, Maximum Performance

> ℹ️ **Breaking Change (v4.0)**  
> YoloDotNet v4.0 introduces a **new modular architecture** and is therefore a **breaking update**.

> The core **YoloDotNet** package is now completely execution-provider agnostic and no longer ships with any ONNX Runtime dependencies.  
> Execution providers (CPU, CUDA, OpenVINO, CoreML, etc.) are now **separate NuGet packages** and must be referenced explicitly.
>
> **What this means for you:**
> - You must install **YoloDotNet** and **exactly one execution provider**
> - Execution provider **selection and configuration code has changed**
> - Existing projects upgrading from v3.x will need to update their NuGet references and setup code
> - The inference *API* remains familiar, but provider wiring is now explicit and platform-aware
>
> The upside? Cleaner architecture, fewer native dependency conflicts, and a setup that behaves consistently across platforms.

## What's new

- ✅ **Fully modular execution providers**\
Execution providers have officially moved out and got their own place.
YoloDotNet Core now focuses on what to run, not how to run it — leaving all the platform-specific heavy lifting to dedicated NuGet packages.
The result? Cleaner projects, fewer native dependency tantrums, and behavior you can actually predict.

- ✅ **New execution providers: OpenVINO & CoreML**\
Two shiny new execution providers join the lineup:

    - **Intel OpenVINO** — Squeezes the most out of Intel CPUs and iGPUs on Windows and Linux
    - **Apple CoreML** — Taps directly into Apple’s native ML stack for fast, hardware-accelerated inference on macOS
    Plug them in, run inference, enjoy the speed — no code changes, no drama.

- ✅ **Improved CUDA execution provider**\
Smarter and more explicit GPU handling for CUDA and TensorRT, so your NVIDIA hardware does what you expect — and not what it feels like doing today.

- ✅ **Grayscale ONNX model support**\
Run inference on grayscale-only models when color is just wasted effort.
Less data, less work, same results — because sometimes black and white is all you need.

- 🔄 **Dependency updates**
    - **SkiaSharp** updated to **3.119.1**
    - **ONNX Runtime** (CUDA) updated to **1.23.2**\
    Fresh bits, fewer bugs, better vibes.

# Requirements

To use **YoloDotNet**, you need:

- A YOLO model exported to **ONNX format (opset 17)**
- The **YoloDotNet core** NuGet package
- **Exactly one** execution provider package suitable for your target platform

## ONNX Model (opset 17)

All models — including custom-trained models — must be exported to **ONNX format with `opset 17`** for best performance and compatibility.

For instructions on exporting YOLO models to ONNX, see:  
https://docs.ultralytics.com/modes/export/#usage-examples

> **Note:** Dynamic models are not supported.

## Installation

1. ### Install YoloDotNet Core
    ```bash
    dotnet add package YoloDotNet
    ```
2. ### Select an Execution Provider

    | Execution Provider | Windows | Linux | macOS | Documentation |
    | ------------------ | ------- | ----- | ----- | ------------- |
    | CPU | ✅ | ✅ | ✅ | [CPU README](./YoloDotNet.ExecutionProvider.Cpu/README.md) |
    | CUDA / TensorRT | ✅ | ✅ | ❌ | [CUDA README](./YoloDotNet.ExecutionProvider.Cuda/README.md) |
    | OpenVINO | ✅ | ✅ | ❌ | [OpenVINO README](./YoloDotNet.ExecutionProvider.OpenVino/README.md) |
    | CoreML | ❌ | ❌ | ✅ | [CoreML README](./YoloDotNet.ExecutionProvider.CoreML/README.md) |

    > ℹ️ **Important limitation**  
    > Only **one** execution provider package may be referenced.  
    > Execution providers ship their own ONNX Runtime native binaries, and mixing providers will overwrite shared DLLs and cause runtime errors.


# 🚀 Quick Start: Dive into the Demos
Can’t wait to see YoloDotNet in action? The [demo projects](./Demo) are the fastest way to get started and explore everything this library can do.

Each demo showcases:

- **Classification**
- **Object Detection**
- **OBB Detection**
- **Segmentation**
- **Pose Estimation**

Oh, and it doesn’t stop there — there’s a demo for [real-time video inference](./Demo/VideoStreamDemo) too! Whether you’re analyzing local video files or streaming live, the demos have you covered.

Each [demo](./Demo) is packed with inline comments to help you understand how everything works under the hood. From model setup, execution provider, preprocessing to video streaming and result rendering — it's all there.

> Pro tip:
> For detailed configuration options and usage guidance, check out the comments in the demo source files.

[Open the YoloDotNet.Demo projects](./Demo), build, run, and start detecting like a pro. ✨

# Bare Minimum — Get Up and Running in a Snap
Sometimes you just want to see the magic happen without the bells and whistles. Here’s the absolute simplest way to load a model, run inference on an image, and get your detected objects:
```csharp
using SkiaSharp;
using YoloDotNet;
using YoloDotNet.Enums;
using YoloDotNet.Models;
using YoloDotNet.Extensions;
using YoloDotNet.ExecutionProvider.Cpu;

public class Program
{
    // ⚠️ Note: The accuracy of inference results depends heavily on how you configure preprocessing and thresholds.
    // Make sure to read the README section "Accuracy Depends on Configuration":
    // https://github.com/NickSwardh/YoloDotNet/tree/master#%EF%B8%8F-accuracy-depends-on-configuration

    static void Main(string[] args)
    {
        // Instantiate yolo
        using var yolo = new Yolo(new YoloOptions
        {
            // Select an execution provider for your target system
            // 
            ExecutionProvider = new CpuExecutionProvider(model: "path/to/model.onnx"),

            // ...other options here
        });

        // Load image
        using var image = SKBitmap.Decode("image.jpg");

        // Run inference
        var results = yolo.RunObjectDetection(image, confidence: 0.25, iou: 0.7);

        image.Draw(results);         // Draw boxes and labels
        image.Save("result.jpg");    // Save – boom, done!
    }
}
```
That’s it! No fuss.

Of course, the real power lies in customizing the pipeline, streaming images/videos, or tweaking models… but this snippet gets you started in seconds.

**Want more?** Dive into the [demos and source code](./Demo) for full examples, from video streams to segmentation and pose estimation.

# ⚠️ Accuracy Depends on Configuration

**The accuracy of your results depends heavily on how you configure preprocessing and thresholds**. Even with a correctly trained model, mismatched settings can cause accuracy loss. There is no one-size-fits-all configuration — optimal values depend on your dataset, how your model was trained, and your specific application needs.

### 🔑 Key Factors
1. **Image Preprocessing & Resize Mode**
    * Controlled via `ImageResize`.
    * Must match the preprocessing used during training to ensure accurate detections.
      | **Proportional dataset** | **Stretched dataset** |
      |:------------:|:---------:|
      |<img width="160" height="107" alt="proportional" src="https://github.com/user-attachments/assets/e95a2c5c-8032-4dee-a05a-a73b062a4965" />|<img width="160" height="160" alt="stretched" src="https://github.com/user-attachments/assets/90fa31cf-89dd-41e4-871c-76ae3215171f" />|
      |Use `ImageResize.Proportional` (default) if the dataset images were not distorted during training and their aspect ratio was preserved.|Use `ImageResize.Stretched` if the dataset images were resized directly to the model’s input size (e.g., 640x640), ignoring the original aspect ratio.|
    * **Important:** Selecting the wrong resize mode can reduce detection accuracy.
2. **Sampling Options**
    * Controlled via `SamplingOptions`.
    * Define how pixel data is resampled when resizing (e.g., **`Cubic`**, **`NearestNeighbor`**, **`Bilinear`**). This choice has a direct impact on the accuracy of your detections, as different resampling methods can slightly alter object shapes and edges.
    * YoloDotNet default:
        ```csharp
        SamplingOptions = new SKSamplingOptions(SKFilterMode.Nearest, SKMipmapMode.None);
        ```
        💡 **Tip:** Check the [ResizeImage Benchmarks](./test/YoloDotNet.Benchmarks/ImageExtensionTests/ResizeImageTests.cs) for examples of different `SamplingOptions` and to help you choose the best settings for your needs.
3. **Confidence & IoU Thresholds**
    * Results are filtered based on thresholds you set during inference:
        * Confidence → Minimum probability for a detection to count.
        * IoU (**Intersection-over-Union**) → Overlap required to merge/suppress detections.
    * Too low → more false positives.
    * Too high → more missed detections.
    * Fine-tune these values for your dataset and application.

💡 **Tip**: Start with the defaults, then adjust `ImageResize`, `SamplingOptions`, and `Confidence/IoU` thresholds based on your dataset for optimal detection results.

# Make It Yours – Customize the Look
Want to give your detections a personal touch? Go ahead! If you're drawing bounding boxes on-screen, there’s full flexibility to style them just the way you like:

- **Custom Colors** – Use the built-in class-specific colors or define your own for every bounding box.
- **Font Style & Size** – Choose your favorite font, set the size, and even change the color for the labels.
- **Custom Fonts** – Yep, you can load your own font files to give your overlay a totally unique feel.

If that's not enough, check out the extension methods in the main YoloDotNet repository — a solid boilerplate for building even deeper customizations tailored exactly to your needs.

For practical examples on drawing and customization, don’t forget to peek at the demo project source code too!

# Support YoloDotNet
YoloDotNet is the result of countless hours of development, testing, and continuous improvement — all offered freely to the community.

If you’ve found my project helpful, consider supporting its development. Your contribution helps cover the time and resources needed to keep the project maintained, updated, and freely available to everyone.

Support the project:

[![GitHub Sponsors](https://img.shields.io/badge/Sponsor-GitHub-ea4aaa?logo=githubsponsors&logoColor=white)](https://github.com/sponsors/NickSwardh) [![PayPal](https://img.shields.io/badge/Support-PayPal-00457C?logo=paypal&logoColor=white)](https://paypal.me/nickswardh)


Whether it's a donation, sponsorship, or just spreading the word — every bit of support fuels the journey. Thank you for helping YoloDotNet grow! ❤️

# References & Acknowledgements

https://github.com/ultralytics/ultralytics \
https://github.com/sstainba/Yolov8.Net \
https://github.com/mentalstack/yolov5-net

# License

YoloDotNet is © 2023–2025 Niklas Swärd ([GitHub](https://github.com/NickSwardh/YoloDotNet))  
Licensed under the **GNU General Public License v3.0 or later**.

![License: GPL v3 or later](https://img.shields.io/badge/License-GPL_v3_or_later-blue)  
See the [LICENSE](./LICENSE.txt) file for the full license text.

This software is provided “as is”, without warranty of any kind.  
The author is not liable for any damages arising from its use.
