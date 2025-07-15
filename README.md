# <img src="https://github.com/NickSwardh/YoloDotNet/assets/35733515/994287a9-556c-495f-8acf-1acae8d64ac0" height=24> YoloDotNet v3.0

YoloDotNet is a blazing-fast, fully featured C# library for real-time object detection, obb, segmentation, classification, pose estimation — and tracking — using YOLOv5u–v12, YOLO-World, and YOLO-E models.

Built on .NET 8, powered by ONNX Runtime, and supercharged with GPU acceleration via CUDA, YoloDotNet delivers exceptional speed and flexibility for both image and video processing — with full support for live streams, frame skipping, and custom visualizations.

### Supported Versions:
```Yolov5u``` ```Yolov8``` ```Yolov9``` ```Yolov10``` ```Yolov11``` ```Yolov12``` ```Yolo-World``` ```YoloE```

### Supported Tasks:

&nbsp;&nbsp;✓&nbsp;&nbsp;`   Classification   `&nbsp;&nbsp;Categorize an image\
&nbsp;&nbsp;✓&nbsp;&nbsp;`  Object Detection  `&nbsp;&nbsp;Detect multiple objects in a single image\
&nbsp;&nbsp;✓&nbsp;&nbsp;`   OBB Detection    `&nbsp;&nbsp;OBB (Oriented Bounding Box)\
&nbsp;&nbsp;✓&nbsp;&nbsp;`    Segmentation    `&nbsp;&nbsp;Separate detected objects using pixel masks\
&nbsp;&nbsp;✓&nbsp;&nbsp;`  Pose Estimation   `&nbsp;&nbsp;Identifying location of specific keypoints in an image

Batteries not included.

| Classification | Object Detection | OBB Detection | Segmentation | Pose Estimation |
|:---:|:---:|:---:|:---:|:---:|
| <img src="https://user-images.githubusercontent.com/35733515/297393507-c8539bff-0a71-48be-b316-f2611c3836a3.jpg" width=300> | <img src="https://user-images.githubusercontent.com/35733515/273405301-626b3c97-fdc6-47b8-bfaf-c3a7701721da.jpg" width=300> | <img src="https://github.com/NickSwardh/YoloDotNet/assets/35733515/d15c5b3e-18c7-4c2c-9a8d-1d03fb98dd3c" width=300> | <img src="https://github.com/NickSwardh/YoloDotNet/assets/35733515/3ae97613-46f7-46de-8c5d-e9240f1078e6" width=300> | <img src="https://github.com/NickSwardh/YoloDotNet/assets/35733515/b7abeaed-5c00-4462-bd19-c2b77fe86260" width=300> |
| <sub>[image from pexels.com](https://www.pexels.com/photo/hummingbird-drinking-nectar-from-blooming-flower-in-garden-5344570/)</sub> | <sub>[image from pexels.com](https://www.pexels.com/photo/men-s-brown-coat-842912/)</sub> | <sub>[image from pexels.com](https://www.pexels.com/photo/bird-s-eye-view-of-watercrafts-docked-on-harbor-8117665/)</sub> | <sub>[image from pexels.com](https://www.pexels.com/photo/man-riding-a-black-touring-motorcycle-903972/)</sub> | <sub>[image from pexels.com](https://www.pexels.com/photo/woman-doing-ballet-pose-2345293/)</sub> |

# Nuget
```
> dotnet add package YoloDotNet
```

# What's new in YoloDotNet v3.0?
Buckle up — YoloDotNet v3.0 is here, and it’s a beast!

After countless hours of profiling, tightening bolts, and chasing down bottlenecks, this release delivers massive performance gains, streamlined APIs, and cutting-edge model support.

Whether you're streamlining inference pipelines, tracking objects like a pro, or customizing your visual output, this version delivers massive upgrades in speed, flexibility, and usability.

This is the fastest release yet — with performance boosts of up to 70% faster inference and up to 92% lower memory usage, depending on task and hardware.
From lightning-fast segmentation to smarter tracking and smoother rendering, v3.0 is built to crush bottlenecks.

| YoloDotNet v3.0     | Device | Speed Gain         | Memory Reduction        |
| ------------------- | ------ | ------------------ | ----------------------- |
| **Segmentation**    | GPU    | up to 70.8% faster | up to 92.7% less memory |
|                     | CPU    | up to 12.6% faster | up to 92.7% less memory |
| **Classification**  | GPU    | up to 28.5% faster | up to 46.1% less memory |
|                     | CPU    | up to 9.6% faster  | up to 46.1% less memory |
| **OBB Detection**   | GPU    | up to 28.7% faster | up to 2.2% less memory  |
|                     | CPU    | up to 4.7% faster  | up to 2.9% less memory  |
| **Pose Estimation** | GPU    | up to 27.6% faster | up to 0.7% less memory  |
|                     | CPU    | up to 4.6% faster  | up to 0.7% less memory  |
| **Detection**       | GPU    | up to 25.0% faster | up to 0.8% less memory  |
|                     | CPU    | up to 5.3% faster  | up to 1.0% less memory  |

> **Note:** This summary highlights the *maximum* observed speed and memory improvements between versions.
> For full benchmark results, hardware used, detailed per-task analysis, and raw data, please see the complete benchmarks.

- **Model Support That Keeps You Ahead**
    - YOLO-E (Zero-Shot Detection) — tap into cutting-edge open-world object
    - YOLOv5u (Ultralytics fork of YOLOv5 using the YOLOv8 architecture) is now fully supported.
    - Load ONNX models directly from byte[] — ideal for embedded, encrypted, or dynamic loading scenarios.
- **Streamlined Inference & Performance**
    - Simplified inference pipeline — fewer steps, more power, same accuracy.
    - Inference can now run directly on SKBitmap and SKImage types (SkiaSharp), unlocking powerful image workflows.
    - Improved drawing performance when rendering results onto images or frames.
- **Video & Tracking Enhancements**
    - Video processing has been redesigned and simplified — less boilerplate, more control.
    - Split video streams into chunks — perfect for long video processing or batching inference tasks.
    - Added SORT object tracking for maintaining object identity across frames. (See demo project)
    - Customizable frame intervals — process every frame, or every 2nd, 10th, 30th… you decide. This is perfect for high-frame-rate or surveillance streams where analyzing every single frame isn’t necessary.
- **Custom Visualizations**
    - Choose your own font type for result annotations.
    - Define custom colors per class and bounding box — full control over output styling.
- **Dependency Updates**
    - ONNX Runtime 1.22.1
    - SkiaSharp 3.119.0

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

4. Captain-Obvious-important! For Windows to recognize your new environment variables, be sure to close all open programs before continuing — or just give your system a quick restart. Otherwise, your changes might play hide-and-seek! ;)

# Export Yolo models to ONNX with opset=17
All models — including your own custom models — must be exported to the ONNX format with **`opset 17`** for best performance.\
For more information on how to export yolo-models to onnx [read this guide](https://docs.ultralytics.com/modes/export/#usage-examples).

The ONNX-models included in this repo for test and demo purposes are from Ultralytics s-series (small). https://docs.ultralytics.com/models.

# 🚀 Quick Start: Dive into the Demos
Can’t wait to see YoloDotNet in action? The demo projects are the fastest way to get started and explore everything this library can do.

Each demo showcases one of the supported tasks:

- Classification – What is this thing?
- Object Detection – What are all these things?
- OBB Detection – Rotated objects? No problem.
- Segmentation – Color between the lines… automatically.
- Pose Estimation – Find the limbs, strike a pose!

Oh, and it doesn’t stop there — there’s a demo for real-time video inference too! Whether you’re analyzing local video files or streaming live, the demos have you covered.

Each demo is packed with inline comments to help you understand how everything works under the hood. From model setup and preprocessing to video streaming and result rendering — it's all there.

> Pro tip:
> For detailed configuration options and usage guidance, check out the comments in the demo source files. That’s where the real magic happens.

Open the YoloDotNet.Demo projects, build, run, and start detecting at full speed. ✨

# Bare Minimum — Get Up and Running in a Snap
Sometimes you just want to see the magic happen without the bells and whistles. Here’s the absolute simplest way to load a model, run inference on an image, and get your detected objects:
```csharp
using SkiaSharp;
using YoloDotNet;
using YoloDotNet.Enums;
using YoloDotNet.Models;
using YoloDotNet.Extensions;

public class Program
{
    static void Main(string[] args)
    {
        // Fire it up! Create an instance of YoloDotNet and reuse it across your app's lifetime.
        // Prefer the 'using' pattern for automatic cleanup if you're done after a single run.
        var yolo = new Yolo(new YoloOptions
        {
            OnnxModel = "model.onnx",
            Cuda = true,
            PrimeGpu = true,
            GpuId = 0,
            ImageResize = ImageResize.Proportional
            // Choose between Proportional or Stretched resizing.
            // Use 'Proportional' if your model was trained with images that preserve aspect ratio (e.g., padded borders).
            // Use 'Stretched' if your training data was resized to fit the model's input dimensions directly.
            // This setting influence detection accuracy, so be sure it aligns with how the model was trained.
        });

        // Which YOLO magic is this? Let’s find out!
        Console.WriteLine($"Model Type: {yolo.OnnxModel.ModelType}");

        // Load image with SkiaSharp
        using var image = SKBitmap.Decode("image.jpg");

        // Run object detection with default values
        var results = yolo.RunObjectDetection(image, confidence: 0.20, iou = 0.7);

        image.Draw(results);        // Overlay results on image
        image.Save("result.jpg");   // Save to disk – boom, done!

        // Clean up – unless you're using 'using' above.
        yolo?.Dispose();
    }
}
```
That’s it! No fuss, just fast and easy detection.

Of course, the real power lies in customizing the pipeline, streaming videos, or tweaking models… but this snippet gets you started in seconds.

Want more? Dive into the demos and source code for full examples, from video streams to segmentation and pose estimation.

# Make It Yours – Customize the Look
Want to give your detections a personal touch? Go ahead! If you're drawing bounding boxes on-screen, there’s full flexibility to style them just the way you like:

- Custom Colors – Use the built-in class-specific colors or define your own for every bounding box.
- Font Style & Size – Choose your favorite font, set the size, and even change the color for the labels.
- Custom Fonts – Yep, you can load your own font files to give your overlay a totally unique feel.

If that's not enough, check out the extension methods in the main YoloDotNet repository — a solid boilerplate for building even deeper customizations tailored exactly to your needs.

For practical examples on drawing and customization, don’t forget to peek at the demo project source code too!

# Support YoloDotNet
Find this project useful? Buy me a beer or two! 🍻 [https://paypal.me/nickswardh](https://paypal.me/nickswardh?country.x=SE&locale.x=en_US)

# References & Acknowledgements

https://github.com/ultralytics/ultralytics \
https://github.com/sstainba/Yolov8.Net \
https://github.com/mentalstack/yolov5-net

# License

YoloDotNet is © 2023 Niklas Swärd ([GitHub](https://github.com/NickSwardh/YoloDotNet))\
Licensed under the GNU General Public License v3.0 or later.

See the [LICENSE](https://github.com/NickSwardh/YoloDotNet?tab=GPL-3.0-1-ov-file#readme) file for full license text.

This software is provided "as-is", without warranty of any kind.  
The author is not liable for any damages arising from its use.