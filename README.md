# <img src="https://github.com/NickSwardh/YoloDotNet/assets/35733515/994287a9-556c-495f-8acf-1acae8d64ac0" height=24> YoloDotNet v2.0

YoloDotNet is a C# .NET 8 implementation of Yolov8 & Yolov10 for real-time detection of objects in images and videos using ML.NET and ONNX runtime, with GPU acceleration using CUDA.

### YoloDotNet supports the following:

&nbsp;&nbsp;✓&nbsp;&nbsp;`   Classification   `&nbsp;&nbsp;Categorize an image\
&nbsp;&nbsp;✓&nbsp;&nbsp;`  Object Detection  `&nbsp;&nbsp;Detect multiple objects in a single image\
&nbsp;&nbsp;✓&nbsp;&nbsp;`   OBB Detection    `&nbsp;&nbsp;OBB (Oriented Bounding Box)\
&nbsp;&nbsp;✓&nbsp;&nbsp;`   Segmentation     `&nbsp;&nbsp;Separate detected objects using pixel masks\
&nbsp;&nbsp;✓&nbsp;&nbsp;`  Pose Estimation   `&nbsp;&nbsp;Identifying location of specific keypoints in an image

Batteries not included ;)

| Classification | Object Detection | OBB Detection | Segmentation | Pose Estimation |
|:---:|:---:|:---:|:---:|:---:|
| <img src="https://user-images.githubusercontent.com/35733515/297393507-c8539bff-0a71-48be-b316-f2611c3836a3.jpg" width=300> | <img src="https://user-images.githubusercontent.com/35733515/273405301-626b3c97-fdc6-47b8-bfaf-c3a7701721da.jpg" width=300> | <img src="https://github.com/NickSwardh/YoloDotNet/assets/35733515/d15c5b3e-18c7-4c2c-9a8d-1d03fb98dd3c" width=300> | <img src="https://github.com/NickSwardh/YoloDotNet/assets/35733515/3ae97613-46f7-46de-8c5d-e9240f1078e6" width=300> | <img src="https://github.com/NickSwardh/YoloDotNet/assets/35733515/b7abeaed-5c00-4462-bd19-c2b77fe86260" width=300> |
| <sub>[image from pexels.com](https://www.pexels.com/photo/hummingbird-drinking-nectar-from-blooming-flower-in-garden-5344570/)</sub> | <sub>[image from pexels.com](https://www.pexels.com/photo/men-s-brown-coat-842912/)</sub> | <sub>[image from pexels.com](https://www.pexels.com/photo/bird-s-eye-view-of-watercrafts-docked-on-harbor-8117665/)</sub> | <sub>[image from pexels.com](https://www.pexels.com/photo/man-riding-a-black-touring-motorcycle-903972/)</sub> | <sub>[image from pexels.com](https://www.pexels.com/photo/woman-doing-ballet-pose-2345293/)</sub> |

# What's new in YoloDotNet v2.0?

YoloDotNet 2.0 is a Speed Demon release where the main focus has been on supercharging performance to bring you the fastest and most efficient version yet. With major code optimizations, a switch to SkiaSharp for lightning-fast image processing, and added support for Yolov10 as a little extra ;) this release is set to redefine your YoloDotNet experience, here's what's new:

- **Speed Demon Mode:** YoloDotNet is now faster than ever!
- **Code Overhaul:** Tinkered and tweaked under the hood for blazing-fast execution.
- **Swapped Image Libraries:** Out with ImageSharp, in with SkiaSharp. The result? Crazy fast image processing!
- **Memory Efficiency:** Brutally more memory efficient, making the most of your system's resources.
- **Optimized GC Performance** Greatly reduced GC pressure resulting in a sweet performance boost (thanks to louislewis2).
- **Benchmarking** Benchmarking project added for testing and evaluating performance (thanks to louislewis2).
- **Yolov10 Support:** Now featuring support for `Yolov10` object detection. Because why not have the latest and greatest? ;)

### YoloDotNet v2.0 Performance Analysis
**Processor:** Intel(R) Core(TM) i7-7700K CPU @ 4.20GHz\
**Ram:** 16GB\
**Graphics:** NVIDIA GeForce RTX 3060 12GB\
**OS:** Windows 10

Performance was tested using the Yolov8s models in onnx format and test-images provided in the YoloDotNet project.

| Task               | v1.7 Mean (ms) | v2.0 Mean (ms) | Improvement (ms) | Improvement (%) |
|--------------------|---------------:|---------------:|-----------------:|----------------:|
| ClassificationCpu  |         12.730 |          5.734 |            6.996 |          54.95% |
| ClassificationGpu  |          7.708 |          2.255 |            5.453 |          70.73% |
| ObjectDetectionCpu |        147.487 |        113.954 |           33.533 |          22.74% |
| ObjectDetectionGpu |         39.935 |         13.751 |           26.184 |          65.56% |
| SegmentationCpu    |        623.313 |        178.411 |          444.902 |          71.37% |
| SegmentationGpu    |        477.539 |         37.857 |          439.682 |          92.07% |
| PoseEstimationCpu  |        140.823 |        116.557 |           24.266 |          17.23% |
| PoseEstimationGpu  |         31.588 |         12.582 |           19.006 |          60.16% |
| ObbDetectionCpu    |        401.694 |        346.193 |           55.501 |          13.82% |
| ObbDetectionGpu    |         71.935 |         27.591 |           44.344 |          61.62% |

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

# Export Yolo models to ONNX
All models must be Yolo models exported to ONNX format. [How to export to ONNX format](https://docs.ultralytics.com/modes/export/#usage-examples).
  
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
using YoloDotNet.Enums;
using YoloDotNet.Models;
using YoloDotNet.Extensions;
using SkiaSharp;

// Instantiate a new Yolo object
using var yolo = new Yolo(new YoloOptions
{
    OnnxModel = @"path\to\model.onnx",      // Your Yolov8 or Yolov10 model in onnx format
    ModelType = ModelType.ObjectDetection,  // Model type
    Cuda = false,                           // Use CPU or CUDA for GPU accelerated inference. Default = true
    GpuId = 0                               // Select Gpu by id. Default = 0
    PrimeGpu = false,                       // Pre-allocate GPU before first inference. Default = false
});

// Load image
using var image = SKImage.FromEncodedData(@"path\to\image.jpg");

// Run inference and get the results
var results = yolo.RunObjectDetection(image, confidence: 0.25, iou: 0.7);

// Draw results
using var resultsImage = image.Draw(results);

// Save to file
resultsImage.Save(@"save\as\new_image.jpg", SKEncodedImageFormat.Jpeg, 80);
```

# Example - Video inference

> [!IMPORTANT]
> Processing video requires FFmpeg and FFProbe
> - Download [FFMPEG](https://ffmpeg.org/download.html)
> - Add FFmpeg and ffprobe to the Path-variable in your Environment Variables

```csharp
using YoloDotNet;
using YoloDotNet.Enums;
using YoloDotNet.Models;

// Instantiate a new Yolo object
using var yolo = new Yolo(new YoloOptions
{
    OnnxModel = @"path\to\model.onnx",      // Your Yolov8 or Yolov10 model in onnx format
    ModelType = ModelType.ObjectDetection,  // Model type
    Cuda = false,                           // Use CPU or CUDA for GPU accelerated inference. Default = true
    GpuId = 0                               // Select Gpu by id. Default = 0
    PrimeGpu = false,                       // Pre-allocate GPU before first. Default = false
});

// Set video options
var options = new VideoOptions
{
    VideoFile = @"path\to\video.mp4",
    OutputDir = @"path\to\output\dir",
    //GenerateVideo = true,
    //DrawLabels = true,
    //FPS = 30,
    //Width = 640,  // Resize video...
    //Height = -2,  // -2 automatically calculate dimensions to keep proportions
    //Quality = 28,
    //DrawConfidence = true,
    //KeepAudio = true,
    //KeepFrames = false,
    //DrawSegment = DrawSegment.Default,
    //PoseOptions = MyPoseMarkerConfiguration // Your own pose marker configuration...
};

// Run inference on video
var results = yolo.RunObjectDetection(options, 0.25, 0.7);

// Do further processing with 'results'...
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
The solution must be set to `Release` mode to run the benchmarks.

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

The hardware these benchmarks used are detailed below, the graphics card used was a `NVIDIA GeForce RTX 3060 12GB`.

`* Summary *`

BenchmarkDotNet v0.13.12, Windows 10 (10.0.19045.4529/22H2/2022Update)\
Intel Core i7-7700K CPU 4.20GHz (Kaby Lake), 1 CPU, 8 logical and 4 physical cores\
.NET SDK 8.0.302\
[Host]     : .NET 8.0.6 (8.0.624.26715), X64 RyuJIT AVX2\
DefaultJob : .NET 8.0.6 (8.0.624.26715), X64 RyuJIT AVX2

### Starting Point, YoloDotNet v7.1

| Method             | Mean       | Error      | StdDev     | Gen0        | Gen1      | Gen2      | Allocated |
|--------------------|-----------:|-----------:|-----------:|------------:|----------:|----------:|----------:|
| ClassificationCpu  |  12.730 ms |  0.2525 ms |  0.2593 ms |   1546.8750 |  125.0000 |   93.7500 |    6.4 MB |
| ClassificationGpu  |   7.708 ms |  0.1509 ms |  0.2796 ms |   1546.8750 |  125.0000 |   93.7500 |    6.4 MB |
| ObjectDetectionCpu | 147.487 ms |  2.6940 ms |  2.6459 ms |  18666.6667 |  333.3333 |  333.3333 |  77.97 MB |
| ObjectDetectionGpu |  39.935 ms |  0.2201 ms |  0.2059 ms |  18846.1538 |  538.4615 |  461.5385 |  77.97 MB |
| SegmentationCpu    | 623.313 ms | 12.0823 ms | 13.4294 ms | 187000.0000 | 7000.0000 | 1000.0000 | 763.48 MB |
| SegmentationGpu    | 477.539 ms |  8.7532 ms |  9.3658 ms | 188000.0000 | 3000.0000 |         - | 763.41 MB |
| PoseEstimationCpu  | 140.823 ms |  2.6669 ms |  2.4946 ms |  12333.3333 |  333.3333 |  333.3333 |  53.26 MB |
| PoseEstimationGpu  |  31.588 ms |  0.2031 ms |  0.1900 ms |  12812.5000 |  718.7500 |  625.0000 |  53.26 MB |
| ObbDetectionCpu    | 401.694 ms |  6.6027 ms |  9.4694 ms |  35000.0000 | 1000.0000 | 1000.0000 | 147.65 MB |
| ObbDetectionGpu    |  71.935 ms |  0.5656 ms |  0.5291 ms |  34428.5714 |  571.4286 |  428.5714 | 147.64 MB |

### Ending Point, YoloDotNet v2.0

| Method             | Mean       | Error     | StdDev    | Gen0      | Gen1     | Gen2     | Allocated  |
|------------------- |-----------:|----------:|----------:|----------:|---------:|---------:|-----------:|
| ClassificationCpu  |   5.734 ms | 0.1100 ms | 0.0859 ms |    7.8125 |        - |        - |   59.98 KB |
| ClassificationGpu  |   2.255 ms | 0.0054 ms | 0.0059 ms |   11.7188 |        - |        - |   59.98 KB |
| ObjectDetectionCpu | 113.954 ms | 1.3054 ms | 1.0901 ms |         - |        - |        - |    67.5 KB |
| ObjectDetectionGpu |  13.751 ms | 0.2164 ms | 0.1918 ms |   15.6250 |        - |        - |   67.37 KB |
| SegmentationCpu    | 178.411 ms | 2.7077 ms | 2.5328 ms | 1000.0000 | 333.3333 |        - | 7453.61 KB |
| SegmentationGpu    |  37.857 ms | 0.7501 ms | 0.9212 ms | 1214.2857 | 714.2857 | 214.2857 | 7418.45 KB |
| PoseEstimationCpu  | 116.557 ms | 0.9387 ms | 1.1528 ms |         - |        - |        - |   39.71 KB |
| PoseEstimationGpu  |  12.582 ms | 0.1421 ms | 0.1187 ms |         - |        - |        - |   39.57 KB |
| ObbDetectionCpu    | 346.193 ms | 4.7002 ms | 4.3965 ms |         - |        - |        - |   16.48 KB |
| ObbDetectionGpu    |  27.591 ms | 0.2080 ms | 0.1844 ms |         - |        - |        - |   15.78 KB |
