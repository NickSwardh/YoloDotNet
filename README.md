# <img src="https://github.com/NickSwardh/YoloDotNet/assets/35733515/994287a9-556c-495f-8acf-1acae8d64ac0" height=24> YoloDotNet v2.3

YoloDotNet is a blazing-fast C# .NET 8 implementation of Yolo and Yolo-World models for real-time object detection in images and videos. Powered by ONNX Runtime, and supercharged with GPU acceleration using CUDA, this app is all about detecting objects at lightning speed!

### Supported Versions:
```Yolov8``` ```Yolov9``` ```Yolov10``` ```Yolov11``` ```Yolo-World```

### Supported Tasks:

&nbsp;&nbsp;✓&nbsp;&nbsp;`   Classification   `&nbsp;&nbsp;Categorize an image\
&nbsp;&nbsp;✓&nbsp;&nbsp;`  Object Detection  `&nbsp;&nbsp;Detect multiple objects in a single image\
&nbsp;&nbsp;✓&nbsp;&nbsp;`   OBB Detection    `&nbsp;&nbsp;OBB (Oriented Bounding Box)\
&nbsp;&nbsp;✓&nbsp;&nbsp;`   Segmentation     `&nbsp;&nbsp;Separate detected objects using pixel masks\
&nbsp;&nbsp;✓&nbsp;&nbsp;`  Pose Estimation   `&nbsp;&nbsp;Identifying location of specific keypoints in an image

Batteries not included.

| Classification | Object Detection | OBB Detection | Segmentation | Pose Estimation |
|:---:|:---:|:---:|:---:|:---:|
| <img src="https://user-images.githubusercontent.com/35733515/297393507-c8539bff-0a71-48be-b316-f2611c3836a3.jpg" width=300> | <img src="https://user-images.githubusercontent.com/35733515/273405301-626b3c97-fdc6-47b8-bfaf-c3a7701721da.jpg" width=300> | <img src="https://github.com/NickSwardh/YoloDotNet/assets/35733515/d15c5b3e-18c7-4c2c-9a8d-1d03fb98dd3c" width=300> | <img src="https://github.com/NickSwardh/YoloDotNet/assets/35733515/3ae97613-46f7-46de-8c5d-e9240f1078e6" width=300> | <img src="https://github.com/NickSwardh/YoloDotNet/assets/35733515/b7abeaed-5c00-4462-bd19-c2b77fe86260" width=300> |
| <sub>[image from pexels.com](https://www.pexels.com/photo/hummingbird-drinking-nectar-from-blooming-flower-in-garden-5344570/)</sub> | <sub>[image from pexels.com](https://www.pexels.com/photo/men-s-brown-coat-842912/)</sub> | <sub>[image from pexels.com](https://www.pexels.com/photo/bird-s-eye-view-of-watercrafts-docked-on-harbor-8117665/)</sub> | <sub>[image from pexels.com](https://www.pexels.com/photo/man-riding-a-black-touring-motorcycle-903972/)</sub> | <sub>[image from pexels.com](https://www.pexels.com/photo/woman-doing-ballet-pose-2345293/)</sub> |

# What's new in YoloDotNet v2.3?

Hold onto your GPUs, folks! YoloDotNet 2.3 is here, and it's bringing some serious upgrades! Whether you're fine-tuning your models or pushing for peak performance, this update has got you covered. Let's dive in!

***Yolo v12 Support*** – The latest YOLO v12 model is now at your fingertips!

***Custom Image Resizing*** – When creating a dataset for training, preserving small details is crucial. Depending on your dataset, you might get better results by squaring images instead of resizing them proportionally. Now, you can choose between "proportional" resizing (default) to maintain aspect ratio or "stretched" resizing to match the dataset and model input exactly.

***SkiaSharp Sampling Options*** –  Fine-tune image resizing like never before! This new option allows you to adjust SkiaSharp´s SamplingOptions, optimizing for either better rendering quality or faster performance. These adjustments can directly impact inference results, influencing detection accuracy and speed. You can tweak it to suit your needs or simply stick with the defaults—your call!\
([See ImageResize benchmark tests for more details](https://github.com/NickSwardh/YoloDotNet/blob/development/test/YoloDotNet.Benchmarks/ImageExtensionTests/ResizeImageTests.cs))

***Class Label Filtering*** – Tired of detecting everything? Now you can filter out unwanted classes and focus only on what matters.\
([See Example 1 - Image inference](#example-1---image-inference))

***Performance Boosts*** – Pixel normalization just got even faster. Because every millisecond counts.

***Dependency Updates*** –

Updated SkiaSharp to 3.116.1\
Updated OnnxRuntime to 1.21.0

So what are you waiting for? Get out there and start detecting like a pro!

# Nuget
```
> dotnet add package YoloDotNet
```

# Install CUDA (optional)
YoloDotNet with GPU-acceleration requires CUDA Toolkit 12.x and cuDNN 9.x.

ONNX runtime's [current compatibility with specific versions](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements).

- Install [CUDA v12.x](https://developer.nvidia.com/cuda-downloads)
- Install [cuDNN v9.x](https://developer.nvidia.com/cudnn-downloads)
- Update your system PATH-variable
1. Open File Explorer and navigate to the folder where the cuDNN-dll's are installed. The typical path looks like:\
```C:\Program Files\NVIDIA\CUDNN\v9.x\bin\v12.x``` (where x is your  version)

2. Once you are in this specific folder (which contains .dll files), copy the folder path from the address bar at the top of the window.

3. Add the cuDNN-Path to your System Variables:
    - Type ```env``` in windows search
    - Click on ```Edit the system environment variables```
    - Click on ```Environment Variables```
    - Under ```System Variables``` select the ```Path```-variable and click ```Edit```
    - Click on ```New``` and paste in your cuDNN dll-folder path
    - Click Ok a million times to save the changes

4. Super-duper-important! In order for Windows to pick up the changes in your Environment Variables, make sure to close all open programs before you continue with whatever you were doing ;)

# Export your Yolo models to ONNX
All models—including your own custom models or any other YOLO model—must be exported to the ONNX format.\
Need help? [Check out this guide](https://docs.ultralytics.com/modes/export/#usage-examples)

The ONNX-models included in this repo are from Ultralytics s-series (small). https://docs.ultralytics.com/models.
  
## Verify your model

```csharp
using YoloDotNet;

// Instantiate a new Yolo object with your ONNX-model
using var yolo = new Yolo(@"path\to\model.onnx");

Console.WriteLine(yolo.OnnxModel.ModelType); // Output modeltype...
```

# Example 1 - Image inference

```csharp
using YoloDotNet;
using YoloDotNet.Enums;
using YoloDotNet.Models;
using YoloDotNet.Extensions;
using SkiaSharp;

// Instantiate a new Yolo object
using var yolo = new Yolo(new YoloOptions
{
    OnnxModel = @"path\to\model.onnx",          // Your Yolo model in onnx format
    ModelType = ModelType.ObjectDetection,      // Set your model type
    Cuda = false,                               // Use CPU or CUDA for GPU accelerated inference. Default = true
    GpuId = 0,                                  // Select Gpu by id. Default = 0
    PrimeGpu = false,                           // Pre-allocate GPU before first inference. Default = false

    // ImageResize = ImageResize.Proportional   // Proportional = Default, Stretched = Squares the image
    // SamplingOptions =  new SKSamplingOptions(SKFilterMode.Linear, SKMipmapMode.None) // View benchmark-test examples: https://github.com/NickSwardh/YoloDotNet/blob/development/test/YoloDotNet.Benchmarks/ImageExtensionTests/ResizeImageTests.cs
});

// Load image
using var image = SKImage.FromEncodedData(@"path\to\image.jpg");

// Run inference and get the results
var results = yolo.RunObjectDetection(image, confidence: 0.25, iou: 0.7);

// Tip:
// Use the extension method FilterLabels([]) on any result if you only want specific labels.
// Example: Select only the labels you're interested in and exclude the rest.
// var results = yolo.RunObjectDetection(image).FilterLabels(["person", "car", "cat"]);

// Draw results
using var resultImage = image.Draw(results);

// Save to file
resultImage.Save(@"save\as\new_image.jpg", SKEncodedImageFormat.Jpeg, 80);
```

# Example 2 - Inference on a batch of images
```csharp
using System;
using System.IO;
using SkiaSharp;
using YoloDotNet;
using YoloDotNet.Enums;
using YoloDotNet.Models;
using YoloDotNet.Extensions;
using System.Threading.Tasks;

// Instantiate a new yolo-object
using var yolo = new Yolo(new YoloOptions()
{
    OnnxModel = @"path\to\model.onnx",          // Your Yolo model in onnx format
    ModelType = ModelType.ObjectDetection,      // Set your model type
    Cuda = true,                                // Use CPU or CUDA for GPU accelerated inference. Default = true
    GpuId = 0                                   // Select Gpu by id. Default = 0
    PrimeGpu = true,                            // Pre-allocate GPU before first inference. Default = false

    // ImageResize = ImageResize.Proportional   // Proportional = Default, Stretched = Squares the image
    // SamplingOptions =  new SKSamplingOptions(SKFilterMode.Linear, SKMipmapMode.None) // View benchmark-test examples: https://github.com/NickSwardh/YoloDotNet/blob/development/test/YoloDotNet.Benchmarks/ImageExtensionTests/ResizeImageTests.cs
});

// Collect images
var images = Directory.GetFiles(@"path\to\image\folder");

// Process images using parallelism for faster processing
Parallel.ForEach(images, image =>
{
    // Load image
    using var img = SKImage.FromEncodedData(image);

    // Run inference
    var results = yolo.RunObjectDetection(img, 0.25, 0.5);

    // Draw results
    using var resultImg = img.Draw(results);

    // Save results
    resultImg.Save(Path.Combine(@"path\to\save\folder", Path.GetFileName(image)));

    // Do further processing if needed...

});
```

# Example 3 - Video inference

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
    ModelType = ModelType.ObjectDetection,  // Set your model type
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

# Custom KeyPoint configuration for Pose Estimation
[Example on how to configure Keypoints for a Pose Estimation model](ConsoleDemo/Config/KeyPointSetup.cs)
 ```csharp
// Pass in a KeyPoint options parameter to the Draw() extension method. Ex:
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

The hardware these benchmarks used are detailed below, the graphics card used was a `NVIDIA GeForce RTX 4070 TI 12GB`.

`* Summary *`

### Starting Point, YoloDotNet v2.1
```
BenchmarkDotNet v0.13.12, Windows 10 (10.0.19045.4529/22H2/2022Update)
Intel Core i7-7700K CPU 4.20GHz (Kaby Lake), 1 CPU, 8 logical and 4 physical cores
.NET SDK 8.0.302
[Host]     : .NET 8.0.6 (8.0.624.26715), X64 RyuJIT AVX2
DefaultJob : .NET 8.0.6 (8.0.624.26715), X64 RyuJIT AVX2
```

##### CLASSIFICATION (Input image size: 1280x844)
| Method                    | Mean     | Error     | StdDev    | Median   | Gen0   | Allocated | Model Used
|-------------------------- |---------:|----------:|----------:|---------:|-------:|----------:|--------------:
| ClassificationYolov8Cpu   | 3.027 ms | 0.0603 ms | 0.1176 ms | 3.037 ms |      - |  40.17 KB | yolov8s-cls  |
| ClassificationYolov8Gpu   | 1.451 ms | 0.0290 ms | 0.0310 ms | 1.456 ms | 1.9531 |  40.17 KB | yolov8s-cls  |
| ClassificationYolov11Cpu  | 6.721 ms | 0.1341 ms | 0.2829 ms | 6.760 ms |      - |  41.17 KB | yolov11s-cls |
| ClassificationYolov11Gpu  | 3.850 ms | 0.1590 ms | 0.4689 ms | 3.610 ms |      - |  41.17 KB | yolov11s-cls |

##### OBJECT DETECTION (input image size: 1280x851)
| Method                    | Mean      | Error     | StdDev    | Allocated | Model Used
|-------------------------- |----------:|----------:|----------:|----------:|----------:
| ObjectDetectionYolov8Cpu  | 34.462 ms | 0.6583 ms | 0.8559 ms |  34.67 KB | yolov8s  |
| ObjectDetectionYolov8Gpu  |  8.089 ms | 0.0795 ms | 0.0705 ms |  34.63 KB | yolov8s  |
| ObjectDetectionYolov9Cpu  | 38.676 ms | 0.7529 ms | 0.7394 ms |  29.65 KB | yolov9s  |
| ObjectDetectionYolov9Gpu  |  9.730 ms | 0.1243 ms | 0.0971 ms |  29.61 KB | yolov9s  |
| ObjectDetectionYolov10Cpu | 31.709 ms | 0.6309 ms | 0.5901 ms |  24.67 KB | yolov10s |
| ObjectDetectionYolov10Gpu |  7.062 ms | 0.1392 ms | 0.1368 ms |  24.63 KB | yolov10s |
| ObjectDetectionYolov11Cpu | 31.856 ms | 0.6252 ms | 0.7678 ms |  32.79 KB | yolov11s |
| ObjectDetectionYolov11Gpu |  7.321 ms | 0.0445 ms | 0.0825 ms |  32.75 KB | yolov11s |

##### ORIENTED OBJECT DETECTION (OBB) (input image size: 1280x720)
| Method                 | Mean     | Error    | StdDev   | Allocated | Model Used
|----------------------- |---------:|---------:|---------:|----------:|--------------:
| ObbDetectionYolov8Cpu  | 91.81 ms | 1.734 ms | 1.622 ms |   8.43 KB | yolov8s-obb  |
| ObbDetectionYolov8Gpu  | 13.39 ms | 0.041 ms | 0.036 ms |   8.37 KB | yolov8s-obb  |
| ObbDetectionYolov11Cpu | 81.91 ms | 1.423 ms | 1.331 ms |   8.43 KB | yolov11s-obb |
| ObbDetectionYolov11Gpu | 14.00 ms | 0.027 ms | 0.025 ms |   8.37 KB | yolov11s-obb |

##### POSE ESTIMATION (input image size: 1280x720)
| Method                   | Mean      | Error     | StdDev    | Median    | Allocated | Model Used
|------------------------- |----------:|----------:|----------:|----------:|----------:|---------------:
| PoseEstimationYolov8Cpu  | 35.275 ms | 0.4895 ms | 0.4579 ms | 35.180 ms |  24.14 KB | yolov8s-pose  |
| PoseEstimationYolov8Gpu  |  7.445 ms | 0.1474 ms | 0.3415 ms |  7.586 ms |  24.11 KB | yolov8s-pose  |
| PoseEstimationYolov11Cpu | 32.237 ms | 0.6384 ms | 0.9938 ms | 32.056 ms |  22.15 KB | yolov11s-pose |
| PoseEstimationYolov11Gpu |  7.190 ms | 0.1401 ms | 0.1721 ms |  7.206 ms |  22.13 KB | yolov11s-pose |

##### SEGMENTATION (input image size: 1280x853)
| Method                 | Mean     | Error    | StdDev   | Gen0     | Gen1     | Gen2     | Allocated | Model Used
|----------------------- |---------:|---------:|---------:|---------:|---------:|---------:|----------:|--------------:
| SegmentationYolov8Cpu  | 56.79 ms | 1.121 ms | 1.246 ms | 444.4444 | 333.3333 | 111.1111 |   7.31 MB | yolov8s-seg  |
| SegmentationYolov8Gpu  | 31.50 ms | 0.630 ms | 1.198 ms | 468.7500 | 437.5000 | 156.2500 |   7.28 MB | yolov8s-seg  |
| SegmentationYolov11Cpu | 96.84 ms | 1.848 ms | 2.270 ms | 333.3333 | 166.6667 |        - |    6.8 MB | yolov11s-seg |
| SegmentationYolov11Gpu | 28.97 ms | 0.293 ms | 0.274 ms | 406.2500 | 375.0000 | 125.0000 |   6.72 MB | yolov11s-seg |

### Ending Point, YoloDotNet v2.1

```
BenchmarkDotNet v0.14.0, Windows 11 (10.0.22631.4169/23H2/2023Update/SunValley3)
Intel Core i7-14700KF, 1 CPU, 28 logical and 20 physical cores
.NET SDK 8.0.400
[Host]     : .NET 8.0.8 (8.0.824.36612), X64 RyuJIT AVX2
DefaultJob : .NET 8.0.8 (8.0.824.36612), X64 RyuJIT AVX2
```

##### CLASSIFICATION (Input image size: 1280x844)
| Method                    | Mean     | Error     | StdDev    | Median   | Gen0   | Allocated | Model Used
|-------------------------- |---------:|----------:|----------:|---------:|-------:|----------:|--------------:
| ClassificationYolov8Cpu   | 2.990 ms | 0.0475 ms | 0.0444 ms | 2.989 ms |      - |  59.92 KB | yolov8s-cls  |
| ClassificationYolov8Gpu   | 1.157 ms | 0.0392 ms | 0.1156 ms | 1.091 ms | 1.9531 |  59.92 KB | yolov8s-cls  |
| ClassificationYolov11Cpu  | 3.313 ms | 0.0637 ms | 0.0914 ms | 3.321 ms |      - |  59.92 KB | yolov11s-cls |
| ClassificationYolov11Gpu  | 1.252 ms | 0.0038 ms | 0.0035 ms | 1.253 ms | 1.9531 |  59.92 KB | yolov11s-cls |

##### OBJECT DETECTION (input image size: 1280x851)
| Method                    | Mean      | Error     | StdDev    | Allocated | Model Used
|-------------------------- |----------:|----------:|----------:|----------:|----------:
| ObjectDetectionYolov8Cpu  | 34.893 ms | 0.4399 ms | 0.4115 ms |  34.52 KB | yolov8s  |
| ObjectDetectionYolov8Gpu  |  6.670 ms | 0.0152 ms | 0.0142 ms |  34.47 KB | yolov8s  |
| ObjectDetectionYolov9Cpu  | 39.623 ms | 0.7737 ms | 1.0590 ms |  29.34 KB | yolov9s  |
| ObjectDetectionYolov9Gpu  | 10.037 ms | 0.1195 ms | 0.1117 ms |  29.32 KB | yolov9s  |
| ObjectDetectionYolov10Cpu | 32.120 ms | 0.6222 ms | 0.7406 ms |   24.4 KB | yolov10s |
| ObjectDetectionYolov10Gpu |  6.571 ms | 0.0377 ms | 0.0334 ms |  24.35 KB | yolov10s |
| ObjectDetectionYolov11Cpu | 32.133 ms | 0.6097 ms | 0.6524 ms |  32.62 KB | yolov11s |
| ObjectDetectionYolov11Gpu |  6.736 ms | 0.0241 ms | 0.0213 ms |  32.57 KB | yolov11s |
| ObjectDetectionYolov12Cpu | 39.184 ms | 0.7626 ms | 0.9644 ms |     31 KB | yolov12s |
| ObjectDetectionYolov12Gpu |  9.046 ms | 0.1107 ms | 0.1035 ms |  30.95 KB | yolov12s |

##### ORIENTED OBJECT DETECTION (OBB) (input image size: 1280x720)
| Method                 | Mean     | Error    | StdDev   | Allocated | Model Used
|----------------------- |---------:|---------:|---------:|----------:|--------------:
| ObbDetectionYolov8Cpu  | 93.61 ms | 1.086 ms | 0.963 ms |   8.39 KB | yolov8s-obb  |
| ObbDetectionYolov8Gpu  | 13.31 ms | 0.052 ms | 0.049 ms |   8.33 KB | yolov8s-obb  |
| ObbDetectionYolov11Cpu | 85.04 ms | 1.683 ms | 1.653 ms |   8.39 KB | yolov11s-obb |
| ObbDetectionYolov11Gpu | 13.27 ms | 0.060 ms | 0.056 ms |   8.33 KB | yolov11s-obb |

##### POSE ESTIMATION (input image size: 1280x720)
| Method                   | Mean      | Error     | StdDev    | Allocated | Model Used
|------------------------- |----------:|----------:|----------:|----------:|---------------:
| PoseEstimationYolov8Cpu  | 36.508 ms | 0.3856 ms | 0.3418 ms |  23.97 KB | yolov8s-pose  |
| PoseEstimationYolov8Gpu  |  6.617 ms | 0.0271 ms | 0.0254 ms |  23.97 KB | yolov8s-pose  |
| PoseEstimationYolov11Cpu | 33.325 ms | 0.5708 ms | 0.5339 ms |  21.98 KB | yolov11s-pose |
| PoseEstimationYolov11Gpu |  6.458 ms | 0.0307 ms | 0.0272 ms |  21.98 KB | yolov11s-pose |

##### SEGMENTATION (input image size: 1280x853)
| Method                 | Mean     | Error    | StdDev   | Gen0     | Gen1     | Gen2     | Allocated | Model Used
|----------------------- |---------:|---------:|---------:|---------:|---------:|---------:|----------:|--------------:
| SegmentationYolov8Cpu  | 60.07 ms | 1.050 ms | 0.983 ms | 444.4444 | 333.3333 | 111.1111 |   7.52 MB | yolov8s-seg  |
| SegmentationYolov8Gpu  | 34.93 ms | 0.692 ms | 1.963 ms | 468.7500 | 437.5000 | 156.2500 |   7.51 MB | yolov8s-seg  |
| SegmentationYolov11Cpu | 55.90 ms | 1.101 ms | 1.353 ms | 444.4444 | 333.3333 | 111.1111 |   7.01 MB | yolov11s-seg |
| SegmentationYolov11Gpu | 24.23 ms | 0.306 ms | 0.286 ms | 468.7500 | 437.5000 | 156.2500 |   7.01 MB | yolov11s-seg |