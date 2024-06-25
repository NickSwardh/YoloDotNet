# <img src="https://github.com/NickSwardh/YoloDotNet/assets/35733515/994287a9-556c-495f-8acf-1acae8d64ac0" height=24> YoloDotNet v1.7

YoloDotNet is a C# .NET 8 implementation of Yolov8 for real-time detection of objects in images and videos using ML.NET and ONNX runtime, with GPU acceleration using CUDA.

### YoloDotNet supports the following:

&nbsp;&nbsp;✓&nbsp;&nbsp;`   Classification   `&nbsp;&nbsp;Categorize an image<br>
&nbsp;&nbsp;✓&nbsp;&nbsp;`  Object Detection  `&nbsp;&nbsp;Detect multiple objects in a single image<br>
&nbsp;&nbsp;✓&nbsp;&nbsp;`   OBB Detection    `&nbsp;&nbsp;OBB (Oriented Bounding Box), like `Object Detection` but with rotated bounding boxes<br>
&nbsp;&nbsp;✓&nbsp;&nbsp;`   Segmentation     `&nbsp;&nbsp;Separate detected objects using pixel masks<br>
&nbsp;&nbsp;✓&nbsp;&nbsp;`  Pose Estimation   `&nbsp;&nbsp;Identifying location of specific keypoints in an image<br>

Batteries not included ;)

| Classification | Object Detection | OBB Detection | Segmentation | Pose Estimation |
|:---:|:---:|:---:|:---:|:---:|
| <img src="https://user-images.githubusercontent.com/35733515/297393507-c8539bff-0a71-48be-b316-f2611c3836a3.jpg" width=300> | <img src="https://user-images.githubusercontent.com/35733515/273405301-626b3c97-fdc6-47b8-bfaf-c3a7701721da.jpg" width=300> | <img src="https://github.com/NickSwardh/YoloDotNet/assets/35733515/d15c5b3e-18c7-4c2c-9a8d-1d03fb98dd3c" width=300> | <img src="https://github.com/NickSwardh/YoloDotNet/assets/35733515/3ae97613-46f7-46de-8c5d-e9240f1078e6" width=300> | <img src="https://github.com/NickSwardh/YoloDotNet/assets/35733515/b7abeaed-5c00-4462-bd19-c2b77fe86260" width=300> |
| <sub>[image from pexels.com](https://www.pexels.com/photo/hummingbird-drinking-nectar-from-blooming-flower-in-garden-5344570/)</sub> | <sub>[image from pexels.com](https://www.pexels.com/photo/men-s-brown-coat-842912/)</sub> | <sub>[image from pexels.com](https://www.pexels.com/photo/bird-s-eye-view-of-watercrafts-docked-on-harbor-8117665/)</sub> | <sub>[image from pexels.com](https://www.pexels.com/photo/man-riding-a-black-touring-motorcycle-903972/)</sub> | <sub>[image from pexels.com](https://www.pexels.com/photo/woman-doing-ballet-pose-2345293/)</sub> |

# What's new?
- Minor improvements and optimizations
- Updated dependencies to the latest version

# Nuget
```
> dotnet add package YoloDotNet
```

# Install CUDA (optional)
YoloDotNet with GPU-acceleration requires CUDA and cuDNN.

:information_source: Before installing CUDA and cuDNN, make sure to verify the ONNX runtime's [current compatibility with specific versions](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements).

- Download and install [CUDA v11.8](https://developer.nvidia.com/cuda-toolkit-archive)
- Download [cuDNN v8.9.7 ZIP for CUDA v11.x](https://developer.nvidia.com/rdp/cudnn-archive), unzip and copy the dll's in bin folder to your CUDA bin folder
- Add CUDA bin folder-path to your `Path` environment variables
- [Youtube Installation guide](https://www.youtube.com/watch?v=KC9-9L7FgPc)
- Optional: [Allocate memory to the GPU for faster initial inference](#gpu)

# Export Yolov8 model to ONNX
All models must be Yolov8-models. [How to export to ONNX format](https://docs.ultralytics.com/modes/export/#usage-examples).
  
  ## Verify your model
  
  ```csharp
  using YoloDotNet;
  
  // Instantiate a new Yolo object with your ONNX-model
  using var yolo = new Yolo(@"path\to\model.onnx");
  
  Console.WriteLine(yolo.OnnxModel.ModelType); // Output modeltype...
  ```

# Example - Image inference

```csharp
using YoloDotNet;
using YoloDotNet.Extensions;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

// Instantiate a new Yolo object with your ONNX-model and CUDA (default)
using var yolo = new Yolo(@"path\to\your_model.onnx");
//using var yolo = new Yolo(@"path\to\your_model.onnx", primeGpu: true); // Allocate GPU-memory for blazing fast inference

// Load image
using var image = Image.Load<Rgba32>(@"path\to\image.jpg");

// Run
var results = yolo.RunClassification(image, 5); // Top 5 classes
//var results = yolo.RunObjectDetection(image) // Example with default confidence (0.25) and IoU (0.45) threshold;
//var results = yolo.RunObbDetection(options, 0.35, 0.5);
//var results = yolo.RunSegmentation(image, 0.25, 0.5);
//var results = yolo.RunPoseEstimation(image, 0.25, 0.5);

image.Draw(results);
image.Save(@"path\to\save\image.jpg");
```

# Example - Video inference

> [!IMPORTANT]
> Processing video requires FFmpeg and FFProbe
> - Download [FFMPEG](https://ffmpeg.org/download.html)
> - Add FFmpeg and ffprobe to the Path-variable in your Environment Variables

```csharp
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using YoloDotNet;
using YoloDotNet.Extensions;

// Instantiate a new Yolo object with your ONNX-model and CUDA
using var yolo = new Yolo(@"path\to\your_model.onnx");

// Video options
var options = new VideoOptions
{
    VideoFile = @"path\to\video.mp4",
    OutputDir = @"path\to\output\folder",
    //GenerateVideo = true,
    //DrawLabels = true,
    //FPS = 30,
    //Width = 640, // Resize video...
    //Height = -2, // -2 automatically calculate dimensions to keep proportions
    //Quality = 28,
    //DrawConfidence = true,
    //KeepAudio = true,
    //KeepFrames = false,
    //DrawSegment = DrawSegment.Default,
    //PoseOptions = MyPoseMarkerConfiguration // Your own pose marker configuration...
};

// Run
var results = yolo.RunClassification(options, 5); // Top 5 classes
//var results = yolo.RunObjectDetection(options, 0.25);
//var results = yolo.RunObbDetection(options, 0.25);
//var results = yolo.RunSegmentation(options, 0.25);
//var results = yolo.RunPoseEstimation(options, 0.25);

// Do further processing with 'results'...
```

# GPU

Object detection with GPU and GPU-Id = 0 is enabled by default

```csharp
// Default setup. GPU with GPU-Id 0
using var yolo = new Yolo(@"path\to\model.onnx");
```

Allocate GPU memory for faster initial inference (disabled by default)

```csharp
// With CUDA and Allocated GPU memory
using var yolo = new Yolo(@"path\to\model.onnx", primeGpu: true);
```

With a specific GPU-Id

```csharp
// GPU with a user defined GPU-Id
using var yolo = new Yolo(@"path\to\model.onnx", gpuId: 1);
```

# CPU

YoloDotNet detection with CPU

```csharp
// With CPU
using var yolo = new Yolo(@"path\to\model.onnx", false);
```

# Custom Pose-marker configuration
[Example on how to configure PoseOptions for a Pose Estimation model](ConsoleDemo/Config/PoseSetup.cs)
 ```csharp
// Pass in a PoseOptions parameter to the Draw() extension method. Ex:
image.Draw(poseEstimationResults, poseOptions);
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

# Benchmarks

There are some benchmarks included in the project. To run them, you simply need to build the project and run the `YoloDotNet.Benchmarks` project.
The solution must be set to Release mode to run the benchmarks.

There is a if DEBUG section in the benchmark project that will run the benchmarks in Debug mode, but it is not recommended as it will not give accurate results.
This is however useful to debug and step through the code. Two examples have been left in place to show how to run the benchmarks in Debug mode, but have been commented out.

Because there is no persistant storage for benchmark results, the results below are in the form of starting point and ending point.
If one makes changes to the benchmarks, you would move the ending point to the starting point and run the benchmarks again to see the improvements and those values would 
be the new ending point.

Benchmark results would be very much based on the hardware used.
It is important to try run benchmarks on the same hardware for future comparisons. If different hardware is used, it is important to note the hardware used,
as the results would be different, thus the starting point and ending point would need to be updated. Hopefully in future a single hardware configuration can be used for benchmarks,
before updating documentation.

## Simple Benchmarks

Simple benchmarks were modeled around the test project. The test project uses the same images and models as the benchmarks. The benchmarks are run on the same images and models as the test project.
These benchmarks provide a good starting point to identify bottlenecks and areas for improvement.

The hardware these benchmarks used are detailed below, the graphics card used was a Nvidia RTX 4070 TI Super.

// * Summary *

BenchmarkDotNet v0.13.12, Windows 11 (10.0.22631.3593/23H2/2023Update/SunValley3)
Intel Core i9-10900K CPU 3.70GHz, 1 CPU, 20 logical and 10 physical cores
.NET SDK 8.0.300
  [Host]     : .NET 8.0.5 (8.0.524.21615), X64 RyuJIT AVX2
  DefaultJob : .NET 8.0.5 (8.0.524.21615), X64 RyuJIT AVX2


### Starting Point

The starting point means that the benchmarks are run without any optimizations or changes to the code.

| Method                     | Mean     | Error     | StdDev    | Gen0   | Allocated |
|--------------------------- |---------:|----------:|----------:|-------:|----------:|
| RunSimpleClassificationGpu | 6.397 ms | 0.1018 ms | 0.0952 ms | 7.8125 | 674.09 KB |
| RunSimpleClassificationCpu | 6.139 ms | 0.0889 ms | 0.0788 ms | 7.8125 | 674.14 KB |

| Method                      | Mean     | Error    | StdDev   | Gen0     | Gen1     | Gen2     | Allocated |
|---------------------------- |---------:|---------:|---------:|---------:|---------:|---------:|----------:|
| RunSimpleObjectDetectionGpu | 12.93 ms | 0.165 ms | 0.154 ms | 500.0000 | 484.3750 | 484.3750 |   4.84 MB |
| RunSimpleObjectDetectionCpu | 54.55 ms | 0.832 ms | 0.738 ms | 444.4444 | 444.4444 | 444.4444 |   4.84 MB |

| Method                   | Mean      | Error    | StdDev   | Gen0     | Gen1     | Gen2     | Allocated |
|------------------------- |----------:|---------:|---------:|---------:|---------:|---------:|----------:|
| RunSimpleObbDetectionGpu |  19.22 ms | 0.248 ms | 0.232 ms | 875.0000 | 875.0000 | 875.0000 |  12.22 MB |
| RunSimpleObbDetectionCpu | 176.15 ms | 3.402 ms | 4.178 ms | 333.3333 | 333.3333 | 333.3333 |  12.22 MB |

| Method                     | Mean     | Error    | StdDev   | Gen0     | Gen1     | Gen2     | Allocated |
|--------------------------- |---------:|---------:|---------:|---------:|---------:|---------:|----------:|
| RunSimplePoseEstimationGpu | 11.66 ms | 0.121 ms | 0.113 ms | 484.3750 | 484.3750 | 484.3750 |   4.82 MB |
| RunSimplePoseEstimationCpu | 55.34 ms | 0.917 ms | 0.813 ms | 444.4444 | 444.4444 | 444.4444 |   4.82 MB |

| Method                   | Mean     | Error   | StdDev  | Gen0      | Gen1     | Gen2     | Allocated |
|------------------------- |---------:|--------:|--------:|----------:|---------:|---------:|----------:|
| RunSimpleSegmentationGpu | 286.7 ms | 2.80 ms | 2.34 ms | 1000.0000 | 500.0000 | 500.0000 |  14.75 MB |
| RunSimpleSegmentationCpu | 306.4 ms | 5.18 ms | 4.84 ms | 1000.0000 | 500.0000 | 500.0000 |  14.76 MB |

| Method                         | Mean      | Error     | StdDev    | Gen0     | Gen1     | Gen2     | Allocated |
|------------------------------- |----------:|----------:|----------:|---------:|---------:|---------:|----------:|
| ObjectDetectionOriginalSizeGpu | 12.807 ms | 0.1655 ms | 0.1548 ms | 500.0000 | 484.3750 | 484.3750 |   4.84 MB |
| ObjectDetectionOriginalSizeCpu | 56.389 ms | 1.1204 ms | 1.3760 ms | 375.0000 | 375.0000 | 375.0000 |   4.84 MB |
| ObjectDetectionModelSizeGpu    |  8.104 ms | 0.0284 ms | 0.0237 ms | 484.3750 | 484.3750 | 484.3750 |   4.82 MB |
| ObjectDetectionModelSizeCpu    | 52.827 ms | 1.0047 ms | 1.2706 ms | 400.0000 | 400.0000 | 400.0000 |   4.82 MB |

| Method                  | Mean     | Error   | StdDev  | Allocated |
|------------------------ |---------:|--------:|--------:|----------:|
| NormalizePixelsToTensor | 801.9 us | 3.11 us | 2.91 us |     154 B |

| Method             | DrawConfidence | Mean     | Error   | StdDev  | Gen0    | Gen1    | Allocated |
|------------------- |--------------- |---------:|--------:|--------:|--------:|--------:|----------:|
| DrawClassification | False          | 450.8 us | 1.52 us | 1.35 us | 50.7813 | 10.2539 | 519.59 KB |
| DrawClassification | True           | 842.0 us | 4.16 us | 3.89 us | 93.7500 | 29.2969 | 957.68 KB |

| Method              | DrawConfidence | Mean     | Error    | StdDev   | Gen0      | Gen1     | Allocated |
|-------------------- |--------------- |---------:|---------:|---------:|----------:|---------:|----------:|
| DrawObjectDetection | False          | 13.88 ms | 0.103 ms | 0.096 ms | 1375.0000 | 171.8750 |   13.8 MB |
| DrawObjectDetection | True           | 34.17 ms | 0.405 ms | 0.378 ms | 3666.6667 | 666.6667 |  37.49 MB |

| Method                  | DrawConfidence | Mean     | Error     | StdDev    | Gen0     | Gen1     | Allocated |
|------------------------ |--------------- |---------:|----------:|----------:|---------:|---------:|----------:|
| DrawOrientedBoundingBox | False          | 2.892 ms | 0.0346 ms | 0.0323 ms | 277.3438 |  46.8750 |   2.78 MB |
| DrawOrientedBoundingBox | True           | 5.932 ms | 0.0206 ms | 0.0193 ms | 671.8750 | 171.8750 |   6.74 MB |

| Method             | DrawConfidence | Mean     | Error    | StdDev   | Gen0      | Gen1     | Allocated |
|------------------- |--------------- |---------:|---------:|---------:|----------:|---------:|----------:|
| DrawPoseEstimation | False          | 14.08 ms | 0.044 ms | 0.041 ms |  906.2500 | 109.3750 |   9.07 MB |
| DrawPoseEstimation | True           | 20.51 ms | 0.067 ms | 0.062 ms | 1656.2500 | 343.7500 |  16.63 MB |

| Method           | DrawConfidence | Mean     | Error    | StdDev   | Gen0      | Gen1     | Allocated |
|----------------- |--------------- |---------:|---------:|---------:|----------:|---------:|----------:|
| DrawSegmentation | False          | 15.33 ms | 0.267 ms | 0.237 ms | 1296.8750 | 187.5000 |  12.88 MB |
| DrawSegmentation | True           | 28.30 ms | 0.373 ms | 0.415 ms | 2666.6667 | 666.6667 |   28.7 MB |

| Method      | Mean     | Error   | StdDev  | Gen0   | Gen1   | Allocated |
|------------ |---------:|--------:|--------:|-------:|-------:|----------:|
| ResizeImage | 263.9 us | 1.28 us | 1.20 us | 4.8828 | 0.4883 |  52.35 KB |

## Ending Point 

1. Implement a custom array pool which is able to provide buffers for the NormalizePixelsToTensor method. This dramatically reduces the GC overhead.
Allocations and GC presure is greatly reduced.

| Method                     | Mean     | Error     | StdDev    | Gen0   | Allocated |
|--------------------------- |---------:|----------:|----------:|-------:|----------:|
| RunSimpleClassificationGpu | 6.356 ms | 0.1185 ms | 0.1163 ms | 7.8125 |  86.08 KB |
| RunSimpleClassificationCpu | 5.986 ms | 0.0341 ms | 0.0302 ms | 7.8125 |  86.15 KB |

| Method                      | Mean     | Error    | StdDev   | Allocated |
|---------------------------- |---------:|---------:|---------:|----------:|
| RunSimpleObjectDetectionGpu | 11.74 ms | 0.096 ms | 0.080 ms |  70.64 KB |
| RunSimpleObjectDetectionCpu | 53.92 ms | 0.947 ms | 0.839 ms |  71.16 KB |

| Method                   | Mean      | Error    | StdDev   | Allocated |
|------------------------- |----------:|---------:|---------:|----------:|
| RunSimpleObbDetectionGpu |  17.39 ms | 0.204 ms | 0.191 ms | 224.74 KB |
| RunSimpleObbDetectionCpu | 170.74 ms | 3.389 ms | 3.171 ms | 225.64 KB |

| Method                     | Mean     | Error    | StdDev   | Allocated |
|--------------------------- |---------:|---------:|---------:|----------:|
| RunSimplePoseEstimationGpu | 10.73 ms | 0.095 ms | 0.089 ms | 133.83 KB |
| RunSimplePoseEstimationCpu | 53.72 ms | 0.961 ms | 0.803 ms |  134.3 KB |

| Method                   | Mean     | Error   | StdDev   | Median   | Gen0     | Allocated |
|------------------------- |---------:|--------:|---------:|---------:|---------:|----------:|
| RunSimpleSegmentationGpu | 271.8 ms | 5.00 ms | 10.09 ms | 267.6 ms | 500.0000 |  10.06 MB |
| RunSimpleSegmentationCpu | 305.5 ms | 6.04 ms |  7.64 ms | 303.3 ms | 500.0000 |  10.06 MB |

| Method                         | Mean      | Error     | StdDev    | Gen0    | Allocated |
|------------------------------- |----------:|----------:|----------:|--------:|----------:|
| ObjectDetectionOriginalSizeGpu | 12.045 ms | 0.0715 ms | 0.0634 ms | 15.6250 | 159.63 KB |
| ObjectDetectionOriginalSizeCpu | 54.716 ms | 1.0659 ms | 1.5286 ms |       - | 159.84 KB |
| ObjectDetectionModelSizeGpu    |  7.795 ms | 0.0947 ms | 0.0886 ms |       - | 136.87 KB |
| ObjectDetectionModelSizeCpu    | 49.994 ms | 0.9314 ms | 0.8712 ms |       - | 137.76 KB |



| Method                       | DrawConfidence | Mean      | Error     | StdDev    | Median    | Ratio        | RatioSD | Gen0      | Gen1      | Allocated   | Alloc Ratio     |
|----------------------------- |--------------- |----------:|----------:|----------:|----------:|-------------:|--------:|----------:|----------:|------------:|----------------:|
| DrawObjectDetection          | False          | 15.065 ms | 0.2662 ms | 0.7012 ms | 14.793 ms |     baseline |         | 1000.0000 | 1000.0000 | 14135.67 KB |                 |
| DrawExperimentalSkiaSharp    | False          |  7.760 ms | 0.0723 ms | 0.0604 ms |  7.774 ms | 2.12x faster |   0.11x |         - |         - |    25.41 KB |   556.215x less |
|                              |                |           |           |           |           |              |         |           |           |             |                 |
| DrawObjectDetection          | True           | 36.689 ms | 0.6858 ms | 1.2011 ms | 36.091 ms |     baseline |         | 3000.0000 | 1000.0000 | 38392.75 KB |                 |
| DrawExperimentalSkiaSharp    | True           |  7.617 ms | 0.0614 ms | 0.0544 ms |  7.598 ms | 4.90x faster |   0.20x |         - |         - |     22.9 KB | 1,676.654x less |