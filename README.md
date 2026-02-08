# <img src="https://github.com/NickSwardh/YoloDotNet/assets/35733515/994287a9-556c-495f-8acf-1acae8d64ac0" height=24> YoloDotNet v4.2 🚀
**Blazing-fast, production-ready YOLO inference for .NET**

**YoloDotNet** is a modular, lightweight C# library for real-time computer vision
and YOLO-based inference in .NET.

It provides high-performance inference for modern YOLO model families (`YOLOv5u` through `YOLOv26`, `YOLO-World`, `YOLO-E`, and `RT-DETR`), with explicit control over execution, memory, and preprocessing.

Built on **.NET 8**, **ONNX Runtime**, and **SkiaSharp**, YoloDotNet intentionally
avoids heavy computer vision frameworks such as OpenCV.  
There is no Python runtime, no hidden preprocessing, and no implicit behavior —
only the components required for fast, predictable inference on **Windows,
Linux, and macOS**.

No Python. No magic. Just fast, deterministic YOLO — done properly for .NET.

## ⭐ Why YoloDotNet?

![YOLOv5u](https://img.shields.io/badge/YOLOv5u-supported-2ea44f)
![YOLOv8-v26](https://img.shields.io/badge/YOLOv8--v26-supported-2ea44f)
![YOLO-RT-DETR](https://img.shields.io/badge/YOLO--RT--DETR-supported-2ea44f)
![YOLO-World](https://img.shields.io/badge/YOLO--World-supported-2ea44f)
![YOLO-E](https://img.shields.io/badge/YOLO--E-supported-2ea44f)

YoloDotNet is designed for developers who need:

- ✅ **Pure .NET** — no Python runtime, no scripts  
- ✅ **Real performance** — CPU, CUDA / TensorRT, OpenVINO, CoreML, DirectML  
- ✅ **Explicit configuration** — predictable accuracy and memory usage  
- ✅ **Production readiness** — engine caching, long-running stability
- ✅ **Multiple vision tasks** — detection, OBB, segmentation, pose, classification  

Ideal for **desktop apps, backend services, and real-time vision pipelines** that require deterministic behavior and full control.

## 🆕 What’s New v4.2

- Added **Region of Interest (ROI)** support, allowing inference to run on selected areas of an image or video stream  
  *(useful for surveillance, monitoring zones, and performance-focused pipelines)*
- Added the option to **draw edges on segmented objects** for improved visual clarity
- Added helper methods for **JSON export**:
  - `ToJson()` — convert inference results to JSON
  - `SaveJson()` — save inference results directly to a JSON file
- Added helper methods for **YOLO-formatted annotations**:
  - `ToYoloFormat()` — convert results to YOLO annotation format
  - `SaveYoloFormat()` — save results as YOLO-compatible training data
- Added `GetContourPoints()` helper for extracting **ordered contour points** from segmented objects
- Updated **YOLOv26 inference execution** to align with other tasks, improving **consistency and overall execution efficiency**

📖 Full release history: [CHANGELOG.md](./CHANGELOG.md)

> [!TIP]
> **See the demos**  
> Practical, runnable examples showcasing YoloDotNet features are available in the demo projects:  
> 👉 [Browse the demo folder](./Demo)

## 🚀 Quick Start

### 💡 ONNX Model Export Requirements

- For **YOLOv26 models**, export with **opset=18**
- For **YOLOv5u–YOLOv12**, export with **opset=17**

> [!IMPORTANT]
> Using the correct opset ensures optimal compatibility and performance with ONNX Runtime.  
> For more information on how to export models to ONNX, refer to https://docs.ultralytics.com/modes/export/

**Example export commands (Ultralytics CLI):**
```bash
# For YOLOv5u–YOLOv12 (opset 17)
yolo export model=yolov8n.pt format=onnx opset=17

# For YOLOv26 (opset 18)
yolo export model=yolo26n.pt format=onnx opset=18
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
dotnet add package YoloDotNet.ExecutionProvider.Cuda
dotnet add package YoloDotNet.ExecutionProvider.OpenVino
dotnet add package YoloDotNet.ExecutionProvider.CoreML
dotnet add package YoloDotNet.ExecutionProvider.DirectML
```

> 💡 **Note:** The CUDA execution provider includes optional **TensorRT** acceleration.  
> No separate TensorRT package is required.

### 3️⃣ Run object detection

```csharp
using SkiaSharp;
using YoloDotNet;
using YoloDotNet.Models;
using YoloDotNet.Extensions;
using YoloDotNet.ExecutionProvider.Cpu;

using var yolo = new Yolo(new YoloOptions
{
    ExecutionProvider = new CpuExecutionProvider("model.onnx")
});

using var image = SKBitmap.Decode("image.jpg");

// Note: The IoU parameter is used for NMS-based models.
// For YOLOv10 and YOLOv26, IoU is ignored since post-processing is handled internally by the model.
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


## ✅ Verified YOLO Models

The following YOLO models have been **tested and verified** with YoloDotNet using
official Ultralytics exports and default heads.

| Classification | Object Detection | Segmentation | Pose Estimation | OBB Detection |
|----------------|------------------|--------------|------------------|---------------|
| YOLOv8-cls<br>YOLOv11-cls<br>YOLOv12-cls<br>YOLOv26-cls | YOLOv5u<br>YOLOv8<br>YOLOv9<br>YOLOv10<br>YOLOv11<br>YOLOv12<br>YOLOv26<br>RT-DETR | YOLOv8-seg<br>YOLOv11-seg<br>YOLOv12-seg<br>YOLOv26-seg<br>YOLO-World (v2) | YOLOv8-pose<br>YOLOv11-pose<br>YOLOv12-pose<br>YOLOv26-pose | YOLOv8-obb<br>YOLOv11-obb<br>YOLOv12-obb<br>YOLOv26-obb<br> |



## 📁 Demos

Hands-on examples are available in the demo folder:

👉 [Browse the demo projects](./Demo)

Includes image inference, video streams, GPU acceleration, segmentation, and large-image workflows.

## Execution Providers

| Provider           | Windows | Linux | macOS | Documentation |
|--------------------|---------|-------|-------|---------------|
| CPU                | ✅      | ✅    | ✅    | [CPU README](./YoloDotNet.ExecutionProvider.Cpu/README.md) |
| CUDA / TensorRT    | ✅      | ✅    | ❌    | [CUDA README](./YoloDotNet.ExecutionProvider.Cuda/README.md) |
| OpenVINO           | ✅      | ✅    | ❌    | [OpenVINO README](./YoloDotNet.ExecutionProvider.OpenVino/README.md) |
| CoreML             | ❌      | ❌    | ✅    | [CoreML README](./YoloDotNet.ExecutionProvider.CoreML/README.md) |
| DirectML           | ✅      | ❌    | ❌    | [DirectML README](./YoloDotNet.ExecutionProvider.DirectML/README.md) |

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

MIT License

Copyright (c) Niklas Swärd

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

---

### Model Licensing & Responsibility

* YoloDotNet is licensed under the [MIT License](./LICENSE.txt) and provides an ONNX inference
engine for YOLO models exported using Ultralytics YOLO tooling.

* This project does **not** include, distribute, download, or bundle any
pretrained models.

* Users must supply their own ONNX models.

* YOLO ONNX models produced using Ultralytics tooling are typically licensed
under **AGPL-3.0** or a separate commercial license from Ultralytics.

* YoloDotNet does **not** impose, modify, or transfer any license terms related
to user-supplied models.

* **Users are solely responsible** for ensuring that their use of any model
complies with the applicable license terms, including requirements related
to commercial use, distribution, or network deployment.