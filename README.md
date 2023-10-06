# YoloDotNet

### YoloDotNet is a C# .NET 7.0 implementation of Yolov8 and ONNX runtime with CUDA

Yolov8 is a real-time object detection tool by Ultralytics. YoloDotNet is a .NET 7 implementation of Yolov8 for detecting objects in images using ML.NET and the ONNX runtime with GPU acceleration using CUDA.

# Requirements
When using YoloDotNet with GPU-acceleration, you need CUDA and cuDNN.

:information_source: Before you install CUDA and cuDNN, make sure to verify the ONNX runtime's [current compatibility with specific versions](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements).

- Download and install [CUDA](https://developer.nvidia.com/cuda-downloads)
- Download [cuDNN](https://developer.nvidia.com/cudnn) and follow the [installation instructions](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html#install-windows)
- Yolov8 model [exported to ONNX format](https://docs.ultralytics.com/modes/export/#usage-examples)

# Example

```csharp
using SixLabors.ImageSharp;
using YoloDotNet;
using YoloDotNet.Extensions;

// Instantiate a new Yolov8 object with your ONNX-model and CUDA
using var yolo = new Yolov8(@"path\to\model.onnx");

// Load image
using var image = Image.Load(@"path\to\image.jpg");

// Run inference
var results = yolo.RunInference(image);

// Draw boxes
image.DrawBoxes(results);

// Save image
image.Save(@"save\image.jpg");
```

# GPU

Object detection with GPU and GPU-Id = 0 is enabled by default

```csharp
// Default setup. GPU with GPU-Id 0
using var yolo = new Yolo(@"path\to\model.onnx");
```

With a specific GPU-Id

```csharp
// GPU with a user defined GPU-Id
using var yolo = new Yolo(@"path\to\model.onnx", true, 1);
```
# CPU

YoloDotNet detection with CPU

```csharp
// With CPU
using var yolo = new Yolo(@"path\to\model.onnx", false);
```

# Access ONNX metadata and labels

The internal ONNX metadata such as ***input & output parameters, version, author, description, date*** along with the labels can be accessed via the `yolo.OnnxModel` property.

Example:

```csharp
using var yolo = new Yolo(@"path\to\model.onnx");

// ONNX metadata and labels resides inside yolo.OnnxModel
Console.WriteLine(yolo.OnnxModel);
```

Example:

```csharp
// Instantiate a new object
using var yolo = new Yolo(@"path\to\model.onnx");

// Display metadata
foreach (var property in yolo.OnnxModel.GetType().GetProperties())
{
    var value = property.GetValue(yolo.OnnxModel);
    Console.WriteLine($"{property.Name,-20}{value!}");
}

// Get ONNX labels
var labels = yolo.OnnxModel.Labels;

Console.WriteLine();
Console.WriteLine($"Labels ({labels.Length}):");
Console.WriteLine(new string('-', 58));

// Display
for (var i = 0; i < labels.Length; i++)
    Console.WriteLine($"index: {i,-8} label: {labels[i].Name,20} color: {labels[i].Color}");

// Output:

// InputName            = images
// OutputName           = output0
// Date                 = 2023-10-03 11:32:15
// Description          = Ultralytics YOLOv8m model trained on coco.yaml
// Author               = Ultralytics
// Task                 = detect
// License              = AGPL-3.0 https://ultralytics.com/license
// Version              = 8.0.181
// Stride               = 32
// BatchSize            = 1
// ImageSize            = Size[Width = 640, Height = 640]
// Input                = Input { BatchSize = 1, Channels = 3, Width = 640, Height = 640 }
// Output               = Output { BatchSize = 1, Dimensions = 84, Channels = 8400 }
//
// Labels (80):
// ---------------------------------------------------------
// index: 0        label: person              color: #5d8aa8
// index: 1        label: bicycle             color: #f0f8ff
// index: 2        label: car                 color: #e32636
// index: 3        label: motorcycle          color: #efdecd
// ...
```

# References & Acknowledgements

https://github.com/ultralytics/ultralytics

https://github.com/sstainba/Yolov8.Net

https://github.com/mentalstack/yolov5-net
