# YoloDotNet

### YoloDotNet is a C# .NET 8.0 implementation of Yolov8 and ONNX runtime with CUDA

YoloDotNet is a .NET 8 implementation of Yolov8 for detecting objects in images and videos using ML.NET and ONNX runtime with GPU acceleration using CUDA.
YoloDotNet currently supports `Classification` and `Object Detection` in both images and videos.

Classification<br> | Object Detection
:---:|:---:
Categorize an image or video frame | Detect multiple objects in a single image or video frame
![hummingbird](https://user-images.githubusercontent.com/35733515/297393507-c8539bff-0a71-48be-b316-f2611c3836a3.jpg)<br><sup>[image from pexels.com](https://www.pexels.com/photo/hummingbird-drinking-nectar-from-blooming-flower-in-garden-5344570/)</sup> | ![result](https://user-images.githubusercontent.com/35733515/273405301-626b3c97-fdc6-47b8-bfaf-c3a7701721da.jpg)<br><sup>[image from pexels.com](https://www.pexels.com/photo/men-s-brown-coat-842912/)</sup>

# Requirements

YoloDotNet with GPU-acceleration requires CUDA and cuDNN.

:information_source: Before installing CUDA and cuDNN, make sure to verify the ONNX runtime's [current compatibility with specific versions](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements).

- Download and install [CUDA](https://developer.nvidia.com/cuda-downloads)
- Download [cuDNN](https://developer.nvidia.com/cudnn) and follow the [installation instructions](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html#install-windows)
- Yolov8 model [exported to ONNX format](https://docs.ultralytics.com/modes/export/#usage-examples)<br>
  Currently, YoloDotNet supports `Classification` and `ObjectDetection` on both images and videos
  
  ## Verify your model
  
  ```csharp
  using YoloDotNet;
  
  // Instantiate a new Yolo object with your ONNX-model
  using var yolo = new Yolo(@"path\to\model.onnx");
  
  Console.WriteLine(yolo.OnnxModel.ModelType); // Output if valid: Classification or ObjectDetection
  ```
  

> [!NOTE]
> For Video, you need FFmpeg and FFProbe
> - Download [FFMPEG](https://ffmpeg.org/download.html)
> - Add FFmpeg and ffprobe to the Path-variable in your Environment Variables

# Example - Image Classification

```csharp
using SixLabors.ImageSharp;
using YoloDotNet;
using YoloDotNet.Extensions;

// Instantiate a new Yolo object with your ONNX-model and CUDA (default)
using var yolo = new Yolo(@"path\to\model_for_classification.onnx");

// Load image
using var image = Image.Load<Rgba32>(@"path\to\image.jpg");

// Run classification, draw labels and save image
var results = yolo.RunClassification(image, 5); // Get top 5 classifications, default = 1
image.DrawClassificationLabels(results);
image.Save(@"path\to\save\image.jpg");
```

# Example - Image Object Detection

```csharp
using SixLabors.ImageSharp;
using YoloDotNet;
using YoloDotNet.Extensions;

// Instantiate a new Yolo object with your ONNX-model and CUDA (default)
using var yolo = new Yolo(@"path\to\model_for_object_detection.onnx");

// Load image
using var image = Image.Load<Rgba32>(@"path\to\image.jpg");

// Run object detection, draw boxes and save image
var results = yolo.RunObjectDetection(image, 0.5); // Default threshold = 0.25
image.DrawBoundingBoxes(results);
image.Save(@"path\to\save\image.jpg");
```

# Example - Video Classification on frames

```csharp
using SixLabors.ImageSharp;
using YoloDotNet;
using YoloDotNet.Extensions;

// Instantiate a new Yolo object with your ONNX-model and CUDA
using var yolo = new Yolo(@"path\to\model_for_classification.onnx");

// Run classification and save video to output folder
yolo.RunClassification(new VideoOptions
{
    VideoFile = @"path\to\video.mp4",
    OutputDir = @"path\to\outputfolder"
}, 5); // Get top 5 classifications for each frame, default = 1
```

# Example - Video Object Detection on frames

```csharp
using SixLabors.ImageSharp;
using YoloDotNet;
using YoloDotNet.Extensions;

// Instantiate a new Yolo object with your ONNX-model and CUDA
using var yolo = new Yolo(@"path\to\model_for_classification.onnx");

// Run object detection and save video to output folder
yolo.RunObjectDetection(new VideoOptions
{
    VideoFile = @"path\to\video.mp4",
    OutputDir = @"path\to\outputfolder"
}, 0.3); // Default threshold = 0.25
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

    if (property.Name == nameof(yolo.OnnxModel.CustomMetaData))
        foreach (var data in (Dictionary<string, string>)value!)
            Console.WriteLine($"{"",-20}{data.Key,-20}{data.Value}");
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

// ModelType           ObjectDetection
// InputName           images
// OutputName          output0
// CustomMetaData      System.Collections.Generic.Dictionary`2[System.String,System.String]
//                     date                2023-11-07T13:33:33.565196
//                     description         Ultralytics YOLOv8n model trained on coco.yaml
//                     author              Ultralytics
//                     task                detect
//                     license             AGPL-3.0 https://ultralytics.com/license
//                     version             8.0.202
//                     stride              32
//                     batch               1
//                     imgsz               [640, 640]
//                     names               {0: 'person', 1: 'bicycle', 2: 'car' ... }
// ImageSize           Size [ Width=640, Height=640 ]
// Input               Input { BatchSize = 1, Channels = 3, Width = 640, Height = 640 }
// Output              ObjectDetectionShape { BatchSize = 1, Elements = 84, Channels = 8400 }
// Labels              YoloDotNet.Models.LabelModel[]
//
// Labels (80):
// ---------------------------------------------------------
// index: 0        label: person              color: #5d8aa8
// index: 1        label: bicycle             color: #f0f8ff
// index: 2        label: car                 color: #e32636
// index: 3        label: motorcycle          color: #efdecd
// ...
```

# Donate
[https://paypal.me/nickswardh](https://paypal.me/nickswardh?country.x=SE&locale.x=en_US)

# References & Acknowledgements

https://github.com/ultralytics/ultralytics

https://github.com/sstainba/Yolov8.Net

https://github.com/mentalstack/yolov5-net
