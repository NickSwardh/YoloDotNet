# YoloDotNet

### YoloDotNet is a C# .NET 8.0 implementation of Yolov8 and ONNX runtime with CUDA

Yolov8 is a real-time object detection tool by Ultralytics. YoloDotNet is a .NET 8 implementation of Yolov8 for detecting objects in images and videos using ML.NET and the ONNX runtime with GPU acceleration using CUDA.

![result](https://github.com/NickSwardh/YoloDotNet/assets/35733515/626b3c97-fdc6-47b8-bfaf-c3a7701721da)

<sup>[image from pexels.com](https://www.pexels.com/photo/men-s-brown-coat-842912/)</sup>

# Requirements
When using YoloDotNet with GPU-acceleration, you need CUDA and cuDNN.

:information_source: Before you install CUDA and cuDNN, make sure to verify the ONNX runtime's [current compatibility with specific versions](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements).

- Download and install [CUDA](https://developer.nvidia.com/cuda-downloads)
- Download [cuDNN](https://developer.nvidia.com/cudnn) and follow the [installation instructions](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html#install-windows)
- Yolov8 model [exported to ONNX format](https://docs.ultralytics.com/modes/export/#usage-examples)

:information_source: For Video, you need FFmpeg and FFProbe
- Download [FFMPEG](https://ffmpeg.org/download.html)
- Add FFmpeg and ffprobe to the Path-variable in your Environment Variables
  
# Example - Image

```csharp
using SixLabors.ImageSharp;
using YoloDotNet;
using YoloDotNet.Extensions;

// Instantiate a new Yolo object with your ONNX-model and CUDA
using var yolo = new Yolo(@"path\to\model.onnx");

// Load image
using var image = Image.Load<Rgb32>(@"path\to\image.jpg");

// Run inference
var results = yolo.RunInference(image);

// Draw boxes
image.DrawBoundingBoxes(results);

// Save image
image.Save(@"save\image.jpg");
```

# Example - Video

```csharp
using SixLabors.ImageSharp;
using YoloDotNet;
using YoloDotNet.Extensions;

// Instantiate a new Yolo object with your ONNX-model and CUDA
using var yolo = new Yolo(@"path\to\model.onnx");

// Run inference
yolo.RunInference(new VideoOptions
{
    VideoFile = @"path\to\video.mp4",
    OutputDir = @"path\to\outputfolder"
});
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
