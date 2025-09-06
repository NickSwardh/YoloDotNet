# <img src="https://github.com/NickSwardh/YoloDotNet/assets/35733515/994287a9-556c-495f-8acf-1acae8d64ac0" height=24> YoloDotNet v3.1.1

**YoloDotNet** is a blazing-fast, fully featured C# library for real-time object detection, OBB, segmentation, classification, pose estimation — and tracking — using YOLOv5u–v12, YOLO-World, and YOLO-E models.

Built on .NET 8, powered by ONNX Runtime, and supercharged with GPU acceleration via **CUDA** — or break the speed barrier entirely with **NVIDIA TensorRT support**, unleashing maximum inference performance through hardware-level optimization. YoloDotNet delivers exceptional speed and flexibility for both image and video processing, with full support for live streams, frame skipping, and custom visualizations.

### Supported Versions:
```Yolov5u``` ```Yolov8``` ```Yolov9``` ```Yolov10``` ```Yolov11``` ```Yolov12``` ```Yolo-World``` ```YoloE```

### Supported Tasks:

| Classification | Object Detection | OBB Detection | Segmentation | Pose Estimation |
|:---:|:---:|:---:|:---:|:---:|
| <img src="https://user-images.githubusercontent.com/35733515/297393507-c8539bff-0a71-48be-b316-f2611c3836a3.jpg" width=300> | <img src="https://user-images.githubusercontent.com/35733515/273405301-626b3c97-fdc6-47b8-bfaf-c3a7701721da.jpg" width=300> | <img src="https://github.com/NickSwardh/YoloDotNet/assets/35733515/d15c5b3e-18c7-4c2c-9a8d-1d03fb98dd3c" width=300> | <img src="https://github.com/NickSwardh/YoloDotNet/assets/35733515/3ae97613-46f7-46de-8c5d-e9240f1078e6" width=300> | <img src="https://github.com/NickSwardh/YoloDotNet/assets/35733515/b7abeaed-5c00-4462-bd19-c2b77fe86260" width=300> |
| <sub>[image from pexels.com](https://www.pexels.com/photo/hummingbird-drinking-nectar-from-blooming-flower-in-garden-5344570/)</sub> | <sub>[image from pexels.com](https://www.pexels.com/photo/men-s-brown-coat-842912/)</sub> | <sub>[image from pexels.com](https://www.pexels.com/photo/bird-s-eye-view-of-watercrafts-docked-on-harbor-8117665/)</sub> | <sub>[image from pexels.com](https://www.pexels.com/photo/man-riding-a-black-touring-motorcycle-903972/)</sub> | <sub>[image from pexels.com](https://www.pexels.com/photo/woman-doing-ballet-pose-2345293/)</sub> |

### Supported Execution Providers
![ONNX Runtime](https://img.shields.io/badge/Backend-ONNX_Runtime-1f65dc?style=flat&logo=onnx)
![CPU](https://img.shields.io/badge/CPU-Supported-lightgrey?style=flat&logo=intel)
![CUDA](https://img.shields.io/badge/GPU-CUDA-76B900?style=flat&logo=nvidia)
![TensorRT](https://img.shields.io/badge/Inference-TensorRT-00BFFF?style=flat&logo=nvidia)
# Nuget
```
> dotnet add package YoloDotNet
```

# 📖 Table of Contents
- [What's new in 3.1.1](#-yolodotnet-v311---full-throttle-tensorrt-inference)
- [Install CUDA & TensorRT (Optional)](#install-cuda--tensorrt-optional)
  - [CUDA Installation](#cuda---installation-steps)
  - [TensorRT Installation](#tensorrt---installation-steps)
- [Export Yolo Models to ONNX](#export-yolo-models-to-onnx-with-opset17)
- [Quick Start & Demos](#-quick-start-dive-into-the-demos)
- [Bare Minimum Example](#bare-minimum--get-up-and-running-in-a-snap)
- [Accuracy Depends on Configuration](#%EF%B8%8F-accuracy-depends-on-configuration)
- [Customize Detection Overlay](#make-it-yours--customize-the-look)
- [Support YoloDotNet](#support-yolodotnet)
- [References & Acknowledgements](#references--acknowledgements)
- [License](#license)

# 🚀 YoloDotNet v3.1.1 - Full-Throttle TensorRT Inference!
**Say hello to TensorRT support in YoloDotNet!**

Version 3.1.1 bolts on NVIDIA's high-performance inference engine to break the speed barrier and unleash maximum throughput on supported GPUs.

### ✨ Highlights:
New `TensorRtExecutionProvider` — Configure GPU ID, precision (`FP32`, `FP16`, or turbocharged `INT8`), builder optimizations, and engine cache handling — all from your C# code.
Give the [TensorRT demo](./Demo/TensorRTDemo/) project a spin to get you started in **no time**. No pun intended ;)

**Hardware-Level Optimization** — TensorRT builds a custom engine just for your machine, tailored to your GPU and inference settings. That means raw, uncompromising speed.

**Engine Cache Support** — Save and reuse compiled TensorRT engines between runs to skip the long optimization stage.

**INT8 Calibration Cache** — Drop your precision down to `INT8` for maximum speed and minimal accuracy loss. Generate the `.cache` file once, reuse it forever.

**Fully Configurable** — Engine cache path, file prefix, calibration cache file location — it's all in your control.
> 💡 **Heads up:**\
On the first run, TensorRT may take a couple of minutes to compile and optimize your model. But once the engine is built and cached, it’s warp-speed from there on out.

**Segmentation upgraded** — Now leaner with 85% less memory, and quicker than ever—8% CPU boost and 28% GPU turbocharged!

**OrtEnv fix in 3.1.1** — Reusing or recreating YoloDotNet objects? No more "OrtEnv singleton instance already exists" drama.

Bottom Line? Real-time YOLO inference at breakneck speed! Check the [Benchmarks](./test/YoloDotNet.Benchmarks/).

## Previously in v3.0
YoloDotNet 3.0 introduced massive performance upgrades and smarter APIs:

- Up to 70% faster inference and 92% less memory usage, depending on task and hardware
- Support for YOLOv5u, YOLO-E, and ONNX byte array loading
- Direct inference on SkiaSharp types (SKBitmap, SKImage)
- Improved video handling and built-in SORT tracking
- Custom fonts, class colors, and smarter drawing tools
- Dependency updates: ONNX Runtime 1.22.1, SkiaSharp 3.119.0

## Install CUDA & TensorRT (optional)

### CUDA - Installation Steps

1. Download and install the following from NVIDIA’s official sites:
    - [CUDA v12.x](https://developer.nvidia.com/cuda-downloads)
    - [cuDNN v9.x](https://developer.nvidia.com/cudnn-downloads)

2. Locate the folder containing the cuDNN DLL files, typically:
    ```
    C:\Program Files\NVIDIA\CUDNN\v9.x\bin\v12.x
    ```
    *(Replace v9.x and v12.x with your installed versions.)*

3. Copy the full path of this folder from the File Explorer address bar.

4. Add this path to your Windows system environment variables:
    - Search for `Edit the system environment variables` in the Start menu and open it.
    - Click on ```Environment Variables```
    - Under ```System Variables``` find and select ```Path```, then click ```Edit```
    - Click ```New``` and paste the copied `cuDNN` folder path.
    - Click OK to save and close all dialogs.
5. **Important:** Restart any open programs or reboot your computer to ensure Windows recognizes the 
new environment variable.

### TensorRT - Installation Steps
1. Download TensorRT from NVIDIA’s official site:\
    https://developer.nvidia.com/tensorrt

2. Click the `Download Now` button (you may need to log in with your NVIDIA account).

3. Select `TensorRT 10` and accept the terms and conditions.

4. Choose the latest release of `TensorRT 10`.

5. Download the Windows **ZIP package** for `CUDA 12.x`.

6. Once downloaded, unzip the TensorRT archive to a folder on your system.

7. Locate the `lib` folder inside the extracted TensorRT directory.

8. Copy the full path of this lib folder.

9. Add this folder path to your system’s `PATH environment variable`, following the same process as with CUDA (see CUDA installation step #4).

10. **Installation complete!** Restart any open programs or reboot your system to ensure the environment variables take effect.

>💡 **Tip:** Curious about how TensorRT works in practice? Check out the [TensorRT Demo project](./Demo/TensorRTDemo/) and its [README](./Demo/TensorRTDemo/) for a hands-on example and setup instructions.

# Export Yolo models to ONNX with opset=17
**All models** — including your own custom models — must be exported to the ONNX format with **`opset 17`** for best performance.\
For more information on how to export yolo-models to onnx [read this guide](https://docs.ultralytics.com/modes/export/#usage-examples).

>**Note:** Dynamic models are not supported.

The ONNX-models included in this repo for test and demo purposes are from Ultralytics s-series (small). https://docs.ultralytics.com/models.

# 🚀 Quick Start: Dive into the Demos
Can’t wait to see YoloDotNet in action? The [demo projects](./Demo) are the fastest way to get started and explore everything this library can do.

Each demo showcases one of the supported tasks:

- **Classification** – What is this thing?
- **Object Detection** – What are all these things?
- **OBB Detection** – Rotated objects? No problem.
- **Segmentation** – Color between the lines… automatically.
- **Pose Estimation** – Find the limbs, strike a pose!

Oh, and it doesn’t stop there — there’s a demo for [real-time video inference](./Demo/VideoStreamDemo) too! Whether you’re analyzing local video files or streaming live, the demos have you covered.

Each [demo](./Demo) is packed with inline comments to help you understand how everything works under the hood. From model setup and preprocessing to video streaming and result rendering — it's all there.

> Pro tip:
> For detailed configuration options and usage guidance, check out the comments in the demo source files. That’s where the real magic happens.

[Open the YoloDotNet.Demo projects](./Demo), build, run, and start detecting at full speed. ✨

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
    // ⚠️ Note: The accuracy of inference results depends heavily on how you configure preprocessing and thresholds.
    // Make sure to read the README section "Accuracy Depends on Configuration":
    // https://github.com/NickSwardh/YoloDotNet/tree/master#%EF%B8%8F-accuracy-depends-on-configuration

    static void Main(string[] args)
    {
        // Fire it up! Create an instance of YoloDotNet and reuse it across your app's lifetime.
        // Prefer the 'using' pattern for automatic cleanup if you're done after a single run.
        var yolo = new Yolo(new YoloOptions
        {
            OnnxModel = "model.onnx",
            // Path to your trained model.
            // Ensure this model matches the preprocessing and training settings you use below.

            // OnnxModelBytes = modelBytes
            // Load model in byte[] format (e.g. for embedded scenarios)

            ExecutionProvider = new CudaExecutionProvider(GpuId: 0, PrimeGpu: true),
            // Sets the execution backend.
            // Available options:
            //   - CpuExecutionProvider         → CPU-only (no GPU required)
            //   - CudaExecutionProvider        → GPU via CUDA (NVIDIA required)
            //   - TensorRtExecutionProvider    → GPU via NVIDIA TensorRT for maximum performance

            ImageResize = ImageResize.Proportional,
            // IMPORTANT: Match this to your model's training preprocessing.
            // Proportional = the dataset images were not distorted; their aspect ratio was preserved.
            // Stretched = the dataset images were resized directly to the model's input size, ignoring aspect ratio.

            SamplingOptions = new SKSamplingOptions(SKFilterMode.Nearest, SKMipmapMode.None) // YoloDotNet default
            // IMPORTANT: This defines how pixel data is resampled when resizing the image.
            // The choice of sampling method can directly affect detection accuracy, 
            // as different resampling methods (Nearest, Bilinear, Cubic, etc.) slightly alter object shapes and edges.
            // Check the benchmarks for examples and guidance: 
            // https://github.com/NickSwardh/YoloDotNet/tree/master/test/YoloDotNet.Benchmarks
        });

        // Display model metadata
        Console.WriteLine($"Model Type: {yolo.ModelInfo}");

        // Load image using SkiaSharp
        using var image = SKBitmap.Decode("image.jpg");

        // Run object detection
        var results = yolo.RunObjectDetection(image, confidence: 0.20, iou: 0.7);

        image.Draw(results);         // Draw boxes and labels
        image.Save("result.jpg");    // Save to file – boom, done!

        yolo?.Dispose();             // Cleanup if not using 'using'
    }
}
```
That’s it! No fuss, just fast and easy detection.

Of course, the real power lies in customizing the pipeline, streaming videos, or tweaking models… but this snippet gets you started in seconds.

**Want more?** Dive into the [demos and source code](./Demo) for full examples, from video streams to segmentation and pose estimation.

# ⚠️ Accuracy Depends on Configuration

**The accuracy of your results in YoloDotNet depends heavily on how you configure preprocessing and thresholds**. Even with a correctly trained model, mismatched settings can cause accuracy loss. There is no one-size-fits-all configuration — optimal values depend on your dataset, how your model was trained, and your specific application needs.

### 🔑 Key Factors
1. **Image Preprocessing & Resize Mode**
    * Controlled via `ImageResize`.
    * Must match the preprocessing used during training to ensure accurate detections.
      | **Proportional dataset** | **Stretched dataset** |
      |:------------:|:---------:|
      |<img width="160" height="107" alt="proportional" src="https://github.com/user-attachments/assets/e95a2c5c-8032-4dee-a05a-a73b062a4965" />|<img width="160" height="160" alt="stretched" src="https://github.com/user-attachments/assets/90fa31cf-89dd-41e4-871c-76ae3215171f" />|
      |Use `ImageResize.Proportional` (default) if the dataset images were not distorted during training and their aspect ratio was preserved.|Use `ImageResize.Stretched` if the dataset images were resized directly to the model’s input size (e.g., 640x640), ignoring the original aspect ratio.|
    * **Important:** Selecting the wrong resize mode can reduce detection accuracy.
2. **Sampling Options**
    * Controlled via `SamplingOptions`.
    * Define how pixel data is resampled when resizing (e.g., **`Cubic`**, **`NearestNeighbor`**, **`Bilinear`**). This choice has a direct impact on the accuracy of your detections, as different resampling methods can slightly alter object shapes and edges.
    * Default in `SamplingOptions` in YoloDotNet:
        ```csharp
        SamplingOptions = new SKSamplingOptions (SKFilterMode.Nearest, SKMipmapMode.None);
        ```
        💡 **Tip:** Check the [ResizeImage Benchmarks](./test/YoloDotNet.Benchmarks/ImageExtensionTests/ResizeImageTests.cs) for examples of different `SamplingOptions` and to help you choose the best settings for your needs.
3. **Confidence & IoU Thresholds**
    * Results are filtered based on thresholds you set during inference:
        * Confidence → Minimum probability for a detection to count.
        * IoU (**Intersection-over-Union**) → Overlap required to merge/suppress detections.
    * Too low → more false positives.
    * Too high → more missed detections.
    * Fine-tune these values for your dataset and application.

💡 **Tip**: Start with the defaults, then adjust `ImageResize`, `SamplingOptions`, and `Confidence/IoU` thresholds based on your dataset for optimal detection results.

# Make It Yours – Customize the Look
Want to give your detections a personal touch? Go ahead! If you're drawing bounding boxes on-screen, there’s full flexibility to style them just the way you like:

- **Custom Colors** – Use the built-in class-specific colors or define your own for every bounding box.
- **Font Style & Size** – Choose your favorite font, set the size, and even change the color for the labels.
- **Custom Fonts** – Yep, you can load your own font files to give your overlay a totally unique feel.

If that's not enough, check out the extension methods in the main YoloDotNet repository — a solid boilerplate for building even deeper customizations tailored exactly to your needs.

For practical examples on drawing and customization, don’t forget to peek at the demo project source code too!

# Support YoloDotNet
YoloDotNet is the result of countless hours of development, testing, and continuous improvement — all offered freely to the community.

If you’ve found my project helpful, consider supporting its development. Your contribution helps cover the time and resources needed to keep the project maintained, updated, and freely available to everyone.

Support the project:

[![GitHub Sponsors](https://img.shields.io/badge/Sponsor-GitHub-ea4aaa?logo=githubsponsors&logoColor=white)](https://github.com/sponsors/NickSwardh) [![PayPal](https://img.shields.io/badge/Support-PayPal-00457C?logo=paypal&logoColor=white)](https://paypal.me/nickswardh)


Whether it's a donation, sponsorship, or just spreading the word — every bit of support fuels the journey. Thank you for helping YoloDotNet grow! ❤️

# References & Acknowledgements

https://github.com/ultralytics/ultralytics \
https://github.com/sstainba/Yolov8.Net \
https://github.com/mentalstack/yolov5-net

# License

YoloDotNet is © 2023–2025 Niklas Swärd ([GitHub](https://github.com/NickSwardh/YoloDotNet))  
Licensed under the **GNU General Public License v3.0 or later**.

![License: GPL v3 or later](https://img.shields.io/badge/License-GPL_v3_or_later-blue)  
See the [LICENSE](./LICENSE.txt) file for the full license text.

This software is provided “as is”, without warranty of any kind.  
The author is not liable for any damages arising from its use.
