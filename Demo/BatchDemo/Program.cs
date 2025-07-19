// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2025 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

using SkiaSharp;
using System.Diagnostics;
using YoloDotNet;
using YoloDotNet.Core;
using YoloDotNet.Enums;
using YoloDotNet.Extensions;
using YoloDotNet.Models;
using YoloDotNet.Test.Common;

namespace BatchDemo
{
    /// <summary>
    /// Demonstrates batch object detection on multiple static images using the YoloDotNet library,
    /// leveraging parallel processing for efficient inference.
    ///
    /// This demo loads a collection of images from a specified folder, runs object detection with YOLO segmentation,
    /// draws detection results (bounding boxes, labels, confidence scores),
    /// and saves the annotated images to disk with unique filenames.
    ///
    /// It showcases:
    /// - Parallel processing to speed up detection across large image batches
    /// - Model configuration with hardware acceleration (CUDA) and preprocessing settings
    /// - Customizable rendering of detection output including text styling and bounding box appearance
    /// - Automated output management with results saved to a desktop folder
    /// - Console output indicating the loaded model type
    ///
    /// Execution providers:
    /// - CpuExecutionProvider: runs inference on CPU, universally supported but slower.
    /// - CudaExecutionProvider: uses NVIDIA GPU via CUDA for faster inference, with optional GPU warm-up.
    /// - TensorRtExecutionProvider: leverages NVIDIA TensorRT for highly optimized GPU inference with FP32, FP16, INT8
    ///   precision modes, delivering significant speed improvements.
    ///
    /// Important notes:
    /// - Choose the execution provider based on your hardware and performance requirements.
    /// - Annotated output images are saved in a "YoloDotNet_Results\batch" folder on the desktop.
    /// </summary>
    internal class Program
    {
        private static string _outputFolder = default!;
        private static DetectionDrawingOptions _drawingOptions = default!;

        static void Main(string[] args)
        {
            CreateOutputFolder();
            SetDrawingOptions();

            // Initialize YoloDotNet.
            // YoloOptions configures the model, hardware settings, and image processing behavior.
            using var yolo = new Yolo(new YoloOptions
            {
                // Path or byte[] to the ONNX model file. 
                // SharedConfig.GetTestModelV11 loads a YOLOv11 classification model.
                OnnxModel = SharedConfig.GetTestModelV11(ModelType.ObjectDetection),

                // Select execution provider (determines how and where inference is executed).
                // Available execution providers:
                //
                //   - CpuExecutionProvider()  
                //     Runs inference entirely on the CPU.
                //     Universally compatible but generally the slowest option.
                //
                //   - CudaExecutionProvider(GpuId: 0, PrimeGpu: true)  
                //     Executes inference on an NVIDIA GPU using CUDA.
                //     Optionally primes the GPU with a warm-up run to reduce first-inference latency.
                //
                //   - TensorRtExecutionProvider() { ... }
                //     Executes inference using NVIDIA TensorRT for highly optimized GPU acceleration.
                //     Supports FP32 and FP16 precision modes, and optionally INT8 if calibration data is provided.
                //     Offers significant speed-ups by leveraging TensorRT engine optimizations.
                //
                //     See the TensorRTDemo and documentation for detailed configuration and best practices.
                ExecutionProvider = new CudaExecutionProvider(GpuId: 0, PrimeGpu: true),

                // Resize mode applied before inference. Proportional maintains the aspect ratio (adds padding if needed),
                // while Stretch resizes the image to fit the target size without preserving the aspect ratio.
                // Set this accordingly, as it directly impacts the inference results.
                ImageResize = ImageResize.Proportional,

                // Sampling options for resizing; affects inference speed and quality.
                // For examples of other sampling options, see benchmarks: https://github.com/NickSwardh/YoloDotNet/tree/master/test/YoloDotNet.Benchmarks
                SamplingOptions = new(SKFilterMode.Nearest, SKMipmapMode.None) // YoloDotNet default
            });

            // Print model type
            Console.WriteLine($"Loaded ONNX Model: {yolo.ModelInfo}");

            // Collect images
            var images = Directory.GetFiles(@"path\to\image\folder");

            Parallel.ForEach(images, image =>
            {
                // Load input image as SKBitmap (or SKImage)
                using var img = SKBitmap.Decode(image);

                // Run object detection inference
                var results = yolo.RunObjectDetection(img, 0.25, 0.7);

                // Draw results using custom _drawingOptions (optional)
                img.Draw(results, _drawingOptions);

                // Save image
                var fileName = Path.Combine(_outputFolder, $"ObjectDetection_{Guid.NewGuid()}.jpg");
                img.Save(fileName, SKEncodedImageFormat.Jpeg, 80);
            });

            DisplayOutputFolder();
        }

        private static void SetDrawingOptions()
        {
            // Set options for drawing
            _drawingOptions = new DetectionDrawingOptions
            {
                DrawBoundingBoxes = true,
                DrawConfidenceScore = true,
                DrawLabels = true,
                EnableFontShadow = true,

                // SKTypeface defines the font used for text rendering.
                // SKTypeface.Default uses the system default font.
                // To load a custom font:
                //   - Use SKTypeface.FromFamilyName("fontFamilyName", SKFontStyle) to load by font family name (if installed).
                //   - Use SKTypeface.FromFile("path/to/font.ttf") to load a font directly from a file.
                // Example:
                //   Font = SKTypeface.FromFamilyName("Arial", SKFontStyle.Normal)
                //   Font = SKTypeface.FromFile("C:\\Fonts\\CustomFont.ttf")
                Font = SKTypeface.Default,

                FontSize = 18,
                FontColor = SKColors.White,
                DrawLabelBackground = true,
                EnableDynamicScaling = true,
                BorderThickness = 2,

                // By default, YoloDotNet automatically assigns colors to bounding boxes.
                // To override these default colors, you can define your own array of hexadecimal color codes.
                // Each element in the array corresponds to the class index in your model.
                // Example:
                //   BoundingBoxHexColors = ["#00ff00", "#547457", ...] // Color per class id

                BoundingBoxOpacity = 128,

                // The following options configure tracked object tails, which visualize 
                // the movement path of detected objects across a sequence of frames or images.
                //
                // ⚠ Tracking is not recommended when running batch inference with parallelism.
                // For tracking (e.g., using SortTracker) to work correctly, frames must be processed
                // sequentially so that the tracker can maintain object state across frames.
                // Running tracking in parallel on independent frames will produce incorrect or unpredictable results.
                //
                // This functionality is demonstrated in the VideoStream demo, where sequential processing is enforced.
                //
                // DrawTrackedTail = false,
                // TailPaintColorEnd = new(),
                // TailPaintColorStart = new(),
                // TailThickness = 0,
            };
        }

        private static void CreateOutputFolder()
        {
            _outputFolder = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.Desktop), "YoloDotNet_Results");
            _outputFolder = Path.Combine(_outputFolder, "batch");

            if (Directory.Exists(_outputFolder) is false)
                Directory.CreateDirectory(_outputFolder);
        }

        private static void DisplayOutputFolder()
            => Process.Start("explorer.exe", _outputFolder);
    }
}
