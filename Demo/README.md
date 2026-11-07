# <img src="https://github.com/NickSwardh/YoloDotNet/assets/35733515/994287a9-556c-495f-8acf-1acae8d64ac0" height=24> YoloDotNet Demos
**Hands-on examples of YoloDotNet in action**

These demos cover every vision task YoloDotNet supports, plus video streams, parallel batch processing, TensorRT and Docker.
Each demo is a small, heavily commented program, so the source code doubles as documentation for the options it uses.

## 🚀 Quick Start

**Requirements:** .NET 8 SDK, and Python with Ultralytics 8.4.0 or later for the one-time model export.
Some demos run on an NVIDIA GPU by default; see [Execution providers](#-execution-providers).

### 1️⃣ Export the models

The demos use Ultralytics YOLO models, which are not stored in the repository.
A script in the repository downloads them and exports them to ONNX.

From the repository root, in your Python environment (for example, an activated conda environment), run:

```bash
pip install "ultralytics>=8.4.0"
python tools/download_models.py
```

The script:

- ✅ downloads the models directly from [Ultralytics](https://github.com/ultralytics/assets/releases)
- ✅ exports them to ONNX using Ultralytics' default export settings and the correct opset version for best performance
- ✅ saves them to `test/assets/Models`, where the demos and tests find them automatically
- ✅ skips models you already have (delete a file to export it again)

> [!NOTE]
> Python is only needed for this one-time export. YoloDotNet itself is pure .NET and doesn't use Python at runtime.

> [!WARNING]
> **Model License Notice:**
> The exported models are **not part of YoloDotNet**. They are owned by Ultralytics and licensed under **AGPL-3.0**,
> and YoloDotNet's MIT license does **not** cover them. Do not use them in commercial or closed-source software
> unless you have a commercial license from Ultralytics. The script asks you to accept this before it downloads anything.
> See [Ultralytics Model Licensing](https://www.ultralytics.com/license/) for details.

### 2️⃣ Run a demo

From the repository root:

```bash
dotnet run --project Demo/ObjectDetectionDemo
```

Alternatively, open `YoloDotNet.sln`, set a demo as the startup project and press **F5**.

Most demos save their output to a `YoloDotNet_Results` folder on your desktop. The test images and video they use are in `test/assets/Media`.

Each demo has a README in its folder that explains how it works, documents its options and lists any extra setup, such as FFmpeg, TensorRT or a model of your own.

## 💻 Execution providers

Each demo sets its execution provider in `Program.cs`: some run on the CPU, others on an NVIDIA GPU with CUDA.
The CUDA demos require CUDA 12 and cuDNN 9; see the [CUDA execution provider README](../YoloDotNet.ExecutionProvider.Cuda/README.md).

You can run any demo with another execution provider:

| Execution provider | Project | Namespace and class |
|--------------------|---------|---------------------|
| CPU | [YoloDotNet.ExecutionProvider.Cpu](../YoloDotNet.ExecutionProvider.Cpu/README.md) | `YoloDotNet.ExecutionProvider.Cpu.CpuExecutionProvider` |
| CUDA / TensorRT | [YoloDotNet.ExecutionProvider.Cuda](../YoloDotNet.ExecutionProvider.Cuda/README.md) | `YoloDotNet.ExecutionProvider.Cuda.CudaExecutionProvider` |
| OpenVINO | [YoloDotNet.ExecutionProvider.OpenVino](../YoloDotNet.ExecutionProvider.OpenVino/README.md) | `YoloDotNet.ExecutionProvider.OpenVino.OpenVinoExecutionProvider` |
| CoreML | [YoloDotNet.ExecutionProvider.CoreML](../YoloDotNet.ExecutionProvider.CoreML/README.md) | `YoloDotNet.ExecutionProvider.CoreML.CoreMLExecutionProvider` |
| DirectML | [YoloDotNet.ExecutionProvider.DirectML](../YoloDotNet.ExecutionProvider.DirectML/README.md) | `YoloDotNet.ExecutionProvider.DirectML.DirectMLExecutionProvider` |

> [!IMPORTANT]
> Each execution provider has its own requirements, such as drivers, runtimes or SDKs, and its own options.
> Read the provider's README, linked in the table, before you switch.

To switch a demo to another execution provider, make three changes:

1. In the demo's `.csproj`, replace the execution provider project reference with the one you want. Reference exactly one execution provider.
2. In `Program.cs`, replace the execution provider's `using` directive, for example `using YoloDotNet.ExecutionProvider.Cuda;` with `using YoloDotNet.ExecutionProvider.Cpu;`.
3. Replace the execution provider in `YoloOptions` with the new one for the same model. For example, to run on the CPU:

   ```csharp
   ExecutionProvider = new CpuExecutionProvider(SharedConfig.GetTestModelV26(ModelType.ObjectDetection)),
   ```

## 🧭 How the demos work

The image demos (classification, object detection, OBB detection, segmentation and pose estimation) follow the same four steps. Only the task changes:

```csharp
// 1. Load the model
using var yolo = new Yolo(new YoloOptions
{
    ExecutionProvider = new CpuExecutionProvider("yolo26s.onnx")
});

// 2. Run inference
using var image = SKBitmap.Decode("image.jpg");
var results = yolo.RunObjectDetection(image, confidence: 0.25, iou: 0.7);

// 3. Draw the results on the image
image.Draw(results);

// 4. Save the image
image.Save("result.jpg", SKEncodedImageFormat.Jpeg, 80);
```

The demos load their model and test image through `SharedConfig` from `YoloDotNet.Test.Common`, so they work without any paths to set.
In your own code, pass the path to your model and image instead, as above.

Each demo's README walks through its code and documents every option it uses, with defaults: the execution provider,
model settings, thresholds and drawing options. For guidance on choosing resize mode, sampling and thresholds,
see the [Accuracy & Configuration Guide](../AccuracyAndConfiguration.md).

## 🛠️ Troubleshooting

- **Model file not found:** run `python tools/download_models.py` from the repository root.
- **"Could not find the test/assets folder":** run the demo from inside the repository, so it can locate `test/assets`.
- **CUDA demo fails to start:** check your CUDA and cuDNN installation, or [switch the demo to another execution provider](#-execution-providers).
- **VideoStreamDemo can't open the video:** make sure FFmpeg and FFprobe are installed and on your `PATH`; `ffmpeg -version` should work in a terminal.

## 📜 License

The demo source code is part of YoloDotNet and licensed under the [MIT License](../LICENSE.txt).
The demos use models licensed by Ultralytics under **AGPL-3.0**, which are **not** part of the YoloDotNet MIT license.
These models are intended for running the demos only and must **not** be used commercially without a commercial license from Ultralytics.
See [Ultralytics Licensing](https://www.ultralytics.com/license/).
