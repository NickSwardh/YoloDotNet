# <img src="https://github.com/NickSwardh/YoloDotNet/assets/35733515/994287a9-556c-495f-8acf-1acae8d64ac0" height=24> YoloDotNet v4.3 🚀
**Blazing-fast, production-ready YOLO inference for .NET**

[![NuGet](https://img.shields.io/nuget/v/YoloDotNet?logo=nuget)](https://www.nuget.org/packages/YoloDotNet)
[![NuGet downloads](https://img.shields.io/nuget/dt/YoloDotNet?logo=nuget)](https://www.nuget.org/packages/YoloDotNet)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue)](./LICENSE.txt)

> 🔥 **v4.3: up to 2.5× faster than v4.2** — see [What's New](#-whats-new-v43--yolodotnet-just-entered-beast-mode)

**YoloDotNet** is a modular, lightweight C# library for real-time YOLO inference in .NET.
It runs modern YOLO model families (`YOLOv5u` through `YOLOv26`, `YOLO-World`, `YOLO-E`, and `RT-DETR`) on **Windows, Linux, and macOS**,
with explicit control over execution, memory, and preprocessing.

Built on **.NET 8**, **ONNX Runtime**, and **SkiaSharp**, without heavy frameworks such as OpenCV and without hidden preprocessing.

No Python. No magic. Just fast, deterministic YOLO — done properly for .NET.

## ⭐ Why YoloDotNet?

YoloDotNet is designed for developers who need:

- ✅ **Pure .NET** — no scripts, no external runtimes  
- ✅ **Real performance** — CPU, CUDA / TensorRT, OpenVINO, CoreML, DirectML, with FP32 and FP16 models  
- ✅ **Explicit configuration** — predictable accuracy and memory usage  
- ✅ **Production readiness** — engine caching, long-running stability
- ✅ **Multiple vision tasks** — detection, OBB, segmentation, pose, classification  

Ideal for **desktop apps, backend services, and real-time vision pipelines** that require deterministic behavior and full control.

## 🚀 Quick Start

**Requirements:** .NET 8 or later, plus the setup for your execution provider (for example CUDA and cuDNN for NVIDIA GPUs); see its README under [Execution Providers](#️-execution-providers).
*Optional:* FFmpeg and FFprobe on `PATH`, only if you process video.

### 💡 ONNX Model Export Requirements

- For **YOLOv26 models**, export with **opset=18** and **nms=False**, so the model keeps its built-in post-processing (recent Ultralytics versions otherwise export a different output that YoloDotNet can't read)
- For **YOLOv5u–YOLOv12**, export with **opset=17**
- Models exported with `dynamic=True` are not supported.

> [!IMPORTANT]
> Using the correct opset ensures optimal compatibility and performance with ONNX Runtime.  
> For more information on how to export models to ONNX, refer to https://docs.ultralytics.com/modes/export/

**Example export commands (Ultralytics CLI):**
```bash
# For YOLOv5u–YOLOv12 (opset 17)
yolo export model=yolov8n.pt format=onnx opset=17

# For YOLOv26 (opset 18)
yolo export model=yolo26n.pt format=onnx opset=18 nms=False
```

> [!WARNING]
> **Model License Notice:**  
> YoloDotNet is MIT licensed, but **most Ultralytics YOLO models are AGPL-3.0 or require a commercial license for commercial use**.  
> You are responsible for ensuring your use of any model complies with its license.  
> See [Ultralytics Model Licensing](https://www.ultralytics.com/license/) for details.

### 1️⃣ Install the core package

```bash
dotnet add package YoloDotNet
```

### 2️⃣ Install **exactly one**(!) execution provider

```bash
# CPU (recommended starting point)
dotnet add package YoloDotNet.ExecutionProvider.Cpu

# Hardware-accelerated execution (choose one)
dotnet add package YoloDotNet.ExecutionProvider.Cuda      # NVIDIA GPUs, optional TensorRT for even faster inference
dotnet add package YoloDotNet.ExecutionProvider.OpenVino
dotnet add package YoloDotNet.ExecutionProvider.CoreML
dotnet add package YoloDotNet.ExecutionProvider.DirectML
```

> [!WARNING]
> Mixing execution providers causes native ONNX Runtime conflicts.

### 3️⃣ Run object detection

```csharp
using SkiaSharp;
using YoloDotNet;
using YoloDotNet.Models;
using YoloDotNet.Extensions;
using YoloDotNet.ExecutionProvider.Cpu; // Other provider? Use its namespace, e.g. YoloDotNet.ExecutionProvider.Cuda

using var yolo = new Yolo(new YoloOptions
{
    ExecutionProvider = new CpuExecutionProvider("model.onnx")
});

using var image = SKBitmap.Decode("image.jpg");

// Note: The IoU parameter is used for NMS-based models.
// For YOLOv10, YOLOv26 and end-to-end RT-DETR exports, IoU is ignored since post-processing is handled internally by the model.
var results = yolo.RunObjectDetection(image, confidence: 0.25, iou: 0.7);

image.Draw(results);
image.Save("result.jpg");
```

You’re now running YOLO inference in **pure C#**.

## 💡 Accuracy Depends on Configuration

The defaults match how YOLO models are normally trained (proportional/letterbox resize, bilinear sampling).
If your model was trained with different settings, set the same resize mode and sampling here, and pick confidence/IoU thresholds that fit your use case.
These settings control the accuracy–performance tradeoff and are part of the model itself.

📖 **Before tuning models or comparing results, read:**  
👉 [Accuracy & Configuration Guide](./AccuracyAndConfiguration.md)

## 🧩 Supported Tasks and Models

| Classification | Object Detection | OBB Detection | Segmentation | Pose Estimation |
|---------------|------------------|---------------|--------------|-----------------|
| <img src="https://user-images.githubusercontent.com/35733515/297393507-c8539bff-0a71-48be-b316-f2611c3836a3.jpg" width=300> | <img src="https://user-images.githubusercontent.com/35733515/273405301-626b3c97-fdc6-47b8-bfaf-c3a7701721da.jpg" width=300> | <img src="https://github.com/NickSwardh/YoloDotNet/assets/35733515/d15c5b3e-18c7-4c2c-9a8d-1d03fb98dd3c" width=300> | <img src="https://github.com/NickSwardh/YoloDotNet/assets/35733515/3ae97613-46f7-46de-8c5d-e9240f1078e6" width=300> | <img src="https://github.com/NickSwardh/YoloDotNet/assets/35733515/b7abeaed-5c00-4462-bd19-c2b77fe86260" width=300> |
| <sub>[pexels.com](https://www.pexels.com/photo/hummingbird-drinking-nectar-from-blooming-flower-in-garden-5344570/)</sub> | <sub>[pexels.com](https://www.pexels.com/photo/men-s-brown-coat-842912/)</sub> | <sub>[pexels.com](https://www.pexels.com/photo/bird-s-eye-view-of-watercrafts-docked-on-harbor-8117665/)</sub> | <sub>[pexels.com](https://www.pexels.com/photo/man-riding-a-black-touring-motorcycle-903972/)</sub> | <sub>[pexels.com](https://www.pexels.com/photo/woman-doing-ballet-pose-2345293/)</sub> |

![YOLOv5u](https://img.shields.io/badge/YOLOv5u-supported-2ea44f)
![YOLOv8-v26](https://img.shields.io/badge/YOLOv8--v26-supported-2ea44f)
![YOLO-RT-DETR](https://img.shields.io/badge/YOLO--RT--DETR-supported-2ea44f)
![YOLO-World](https://img.shields.io/badge/YOLO--World-supported-2ea44f)
![YOLO-E](https://img.shields.io/badge/YOLO--E-supported-2ea44f)
![FP32](https://img.shields.io/badge/FP32-supported-2ea44f)
![FP16 half](https://img.shields.io/badge/FP16_half-supported-2ea44f)
![FP16 full](https://img.shields.io/badge/FP16_full-supported-2ea44f)

Models can be **FP32**, **FP16 half** (FP16 input, FP32 outputs) or **FP16 full** (FP16 input and outputs).
YoloDotNet detects the precision from the model, so there is nothing to configure.

The following YOLO models have been **tested and verified** with YoloDotNet using
official Ultralytics exports and default heads.

| Classification | Object Detection | Segmentation | Pose Estimation | OBB Detection |
|----------------|------------------|--------------|------------------|---------------|
| YOLOv8-cls<br>YOLOv11-cls<br>YOLOv12-cls<br>YOLOv26-cls | YOLOv5u<br>YOLOv8<br>YOLOv9<br>YOLOv10<br>YOLOv11<br>YOLOv12<br>YOLOv26<br>RT-DETR | YOLOv8-seg<br>YOLOv11-seg<br>YOLOv12-seg<br>YOLOv26-seg<br>YOLO-World (v2) | YOLOv8-pose<br>YOLOv11-pose<br>YOLOv12-pose<br>YOLOv26-pose | YOLOv8-obb<br>YOLOv11-obb<br>YOLOv12-obb<br>YOLOv26-obb<br> |



## 🔥 What’s New v4.3 — YoloDotNet Just Entered Beast Mode

The fastest YoloDotNet yet, optimized for speed from start to finish: preprocessing, inference, post-processing and drawing are all faster and use less memory. The speed optimizations don't change the detections.

**⚡ Performance compared with v4.2**

| yolo26s, default settings | v4.2 | v4.3 | v4.3 + CUDA Graphs |
|---|---|---|---|
| Video, TensorRT FP16 | 290 fps | 455 fps | **714 fps (2.5× faster)** |
| Video, CUDA | 157 fps | 197 fps | **268 fps (1.7× faster)** |
| Detection, TensorRT FP16 | 2.47 ms | 2.11 ms | **1.40 ms (1.8× faster)** |
| Detection, CUDA | 5.27 ms | 4.88 ms | **3.51 ms (1.5× faster)** |
| Segmentation, CUDA | 7.55 ms | 6.48 ms | **4.91 ms (1.5× faster)** |
| Drawing segmentation results (masks, boxes, labels) | 1.68 ms | **0.71 ms (2.4× faster)** | – |

<sub>Measured on an RTX 5070 Ti, same machine and settings for both versions: ms per image, and frames per second for a video file with detection and drawing.
v4.3 video uses `StartVideoProcessing<T>`. Gains vary by hardware and model.</sub>

- **CUDA Graphs (opt-in), now for CUDA and TensorRT:** `enableCudaGraph: true` replays the recorded GPU work instead of launching it step by step
- **Faster video file processing** with `StartVideoProcessing<T>`: detects the next frames while you handle the current one, in frame order
- **Faster preprocessing:** single-pass AVX2 bilinear resize + normalize, multithreaded for large images
- **Faster, leaner segmentation:** masks are traced as outlines, using less memory per mask
- **Thread-safe parallel inference** on one `Yolo` instance, so you can process folders with many images using `Parallel.ForEach`
- **Lower memory:** fewer allocations, and trimming support (`PublishTrimmed`) makes apps less than half the size
- **Automatic AVX2 acceleration:** the new fast paths turn on by themselves on CPUs with AVX2; on other CPUs (including ARM) YoloDotNet automatically uses the regular code path, so no setup is needed

**🎯 Accuracy & Features**
- **Sharper segmentation:** masks are traced as smooth sub-pixel outlines and no longer shift or shrink. New `Contours` and `GetPixelMask()`
- **Pixel-accurate results:** bounding boxes and keypoints no longer sit half a pixel off, and images are resized the way YOLO models are trained (bilinear by default)
- **Pose & tracking:** define keypoints, colors and connections in a JSON file (`PoseKeyPoints.Load`), and steadier tracking of partly hidden objects (`highConfidenceThreshold`, `minHits`)
- **FP16 models:** both FP16 half (FP16 input, FP32 outputs) and FP16 full (FP16 input and outputs) are supported, alongside FP32

> [!IMPORTANT]
> **Upgrading from 4.2:** update your execution provider package to v1.2 too, because old providers stop at model load.
> The default sampling is now bilinear, `SortTracker` assigns Ids after 3 hits (`minHits: 1` for the old behaviour), and pose colors and
> connections now come from `PoseDrawingOptions.PoseKeyPoints`. See [CHANGELOG.md](./CHANGELOG.md) for all breaking changes and the full release history.

## 📁 Demos

Ready-to-run examples for every task. See the [Demo README](./Demo/README.md) for how to export the models and run the demos.

| Demo | Shows |
|------|-------|
| [ObjectDetectionDemo](./Demo/ObjectDetectionDemo) | Object detection on an image |
| [OBBDetectionDemo](./Demo/OBBDetectionDemo) | Oriented bounding boxes |
| [SegmentationDemo](./Demo/SegmentationDemo) | Segmentation masks and outlines |
| [PoseEstimationDemo](./Demo/PoseEstimationDemo) | Pose estimation with keypoints from JSON |
| [ClassificationDemo](./Demo/ClassificationDemo) | Image classification |
| [YoloE_SegmentationDemo](./Demo/YoloE_SegmentationDemo) | YOLO-E zero-shot detection with text or visual prompts |
| [VideoStreamDemo](./Demo/VideoStreamDemo) | Video files, livestreams and webcams, with tracking |
| [BatchDemo](./Demo/BatchDemo) | Processing folders with many images in parallel |
| [WebcamDemo](./Demo/WebcamDemo) | Real-time webcam detection in a WPF app |
| [TensorRTDemo](./Demo/TensorRTDemo) | GPU inference with TensorRT |
| [DockerDemo](./Demo/DockerDemo) | ASP.NET Core Web API in a Docker container |

## 🖥️ Execution Providers

| Provider           | Windows | Linux | macOS | Documentation |
|--------------------|---------|-------|-------|---------------|
| CPU                | ✅      | ✅    | ✅    | [CPU README](./YoloDotNet.ExecutionProvider.Cpu/README.md) |
| CUDA / TensorRT    | ✅      | ✅    | ❌    | [CUDA README](./YoloDotNet.ExecutionProvider.Cuda/README.md) |
| OpenVINO           | ✅      | ✅    | ❌    | [OpenVINO README](./YoloDotNet.ExecutionProvider.OpenVino/README.md) |
| CoreML             | ❌      | ❌    | ✅    | [CoreML README](./YoloDotNet.ExecutionProvider.CoreML/README.md) |
| DirectML           | ✅      | ❌    | ❌    | [DirectML README](./YoloDotNet.ExecutionProvider.DirectML/README.md) |

> ℹ️ Reference only one provider package (see Quick Start, step 2).

**Why separate packages:** the core package is provider-agnostic and each provider brings its own native ONNX Runtime, so there are fewer conflicts and deployment is predictable.

## ⚡ Speed Tips

📊 Benchmarks: [/test/YoloDotNet.Benchmarks](./test/YoloDotNet.Benchmarks/README.md)

- **Video files:** `StartVideoProcessing<T>` detects the next frames while you handle the current one, in frame order. Up to 2.5× faster than v4.2.
  See the [VideoStreamDemo](./Demo/VideoStreamDemo).
- **Image folders:** inference on one `Yolo` instance is thread-safe, so you can process many images in parallel with `Parallel.ForEach`.
  About 4.5× faster than one image at a time. See the [BatchDemo](./Demo/BatchDemo).
- **NVIDIA GPUs:** `enableCudaGraph: true` makes every inference faster, including single images and live streams: 14–33% on CUDA and 23–38% on TensorRT.
  See [CUDA Graphs](./YoloDotNet.ExecutionProvider.Cuda/README.md#-cuda-graphs) in the CUDA README.

The first two help with bulk work only; single images and live streams arrive one at a time.

## 💖 Support YoloDotNet

⭐ Star the repo  
💬 Share feedback  
🤝 Sponsor development  

[GitHub Sponsors](https://github.com/sponsors/NickSwardh)  
[PayPal](https://paypal.me/nickswardh)

## 📜 License

YoloDotNet is licensed under the [MIT License](./LICENSE.txt). Copyright (c) Niklas Swärd.

---

### Model Licensing & Responsibility

* YoloDotNet provides an ONNX inference
engine for YOLO models exported using Ultralytics YOLO tooling.

* YoloDotNet does **not** include, bundle, or redistribute any pretrained
models. Users must supply their own ONNX models.

* The demo projects and unit tests in this repository are meant to be run
locally, to help developers learn how to integrate YoloDotNet into their own
applications. To run them, the included script `tools/download_models.py`
downloads models directly from Ultralytics and exports them to ONNX on your
machine.

* These models are **not** part of YoloDotNet and are **not** covered by its
MIT license. The models remain the property of Ultralytics and are licensed
under **AGPL-3.0**. They must **not** be used in commercial or closed-source
software unless you either comply with AGPL-3.0, which requires releasing your
application's source code under the same license, or have a commercial license
from Ultralytics; see [Ultralytics Licensing](https://www.ultralytics.com/license/).

* YoloDotNet does **not** impose, modify, or transfer any license terms related
to user-supplied models.

* **Users are solely responsible** for ensuring that their use of any model
complies with the applicable license terms, including requirements related
to commercial use, distribution, or network deployment.