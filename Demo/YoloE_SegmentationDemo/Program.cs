// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2025 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

using SkiaSharp;
using System.Diagnostics;
using System.Globalization;
using YoloDotNet;
using YoloDotNet.Core;
using YoloDotNet.Enums;
using YoloDotNet.Extensions;
using YoloDotNet.Models;
using YoloDotNet.Test.Common;
using YoloDotNet.Test.Common.Enums;

namespace YoloE_SegmentationDemo
{
    /// <summary>
    /// Demonstrates semantic segmentation on static images using the YoloDotNet library with the YoloE model.
    ///
    /// YoloE is a next-generation, real-time, multi-task vision model designed to "see anything".
    /// This demo showcases YoloE's segmentation capability by processing static images, drawing segmentation masks
    /// along with bounding boxes, labels, and confidence scores, and then saving the annotated result to disk.
    ///
    /// 🔔 Important:
    /// - YoloDotNet requires YoloE models to be exported to ONNX **with embedded text prompts** for zero-shot inference.
    ///   See the README for instructions on how to export YoloE models with custom prompts.
    ///
    /// Key features demonstrated:
    /// - Model initialization using a custom YoloE ONNX model (zero-shot with text prompts), configurable for CPU or GPU.
    /// - Pixel-level segmentation with optional bounding boxes, labels, confidence scores, and colored overlays.
    /// - Real-time capable architecture demonstrated in a static context.
    /// - Adjustable thresholds for object and pixel confidence.
    /// - Console logging of detected objects with confidence scores.
    /// - Output image saving with adjustable quality and output path customization.
    /// - Support for custom font rendering, bounding box colors, and dynamic scaling.
    /// - Extensible drawing pipeline via SegmentationDrawingOptions.
    ///
    /// Execution providers:
    /// - CpuExecutionProvider: runs inference on CPU, universally supported but slower.
    /// - CudaExecutionProvider: uses NVIDIA GPU via CUDA for faster inference, with optional GPU warm-up.
    /// - TensorRtExecutionProvider: leverages NVIDIA TensorRT for highly optimized GPU inference with FP32, FP16, INT8
    ///   precision modes, delivering significant speed improvements.
    ///
    /// Important notes:
    /// - Choose the execution provider based on your hardware and performance requirements.
    /// - This example uses a static image for simplicity, but YoloE is optimized for real-time video as well.
    /// - Object tracking is supported in YoloDotNet but not demonstrated here. See the VideoStream demo for details.
    /// </summary>
    internal class Program
    {
        private static string _outputFolder = default!;
        private static SegmentationDrawingOptions _drawingOptions = default!;

        static void Main(string[] args)
        {
            CreateOutputFolder();
            SetDrawingOptions();

            // Initialize YoloDotNet.
            // YoloOptions configures the model, hardware settings, and image processing behavior.
            using var yolo = new Yolo(new YoloOptions
            {
                // Path or byte[] to the ONNX model file.
                OnnxModel = @"path\to\your\yoloE_model.onnx",

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
                ExecutionProvider = new CpuExecutionProvider(),

                // Resize mode applied before inference. Proportional maintains the aspect ratio (adds padding if needed),
                // while Stretch resizes the image to fit the target size without preserving the aspect ratio.
                // Set this accordingly, as it directly impacts the inference results.
                ImageResize = ImageResize.Stretched,

                // Sampling options for resizing; affects inference speed and quality.
                // For examples of other sampling options, see benchmarks: https://github.com/NickSwardh/YoloDotNet/tree/master/test/YoloDotNet.Benchmarks
                SamplingOptions = new(SKFilterMode.Nearest, SKMipmapMode.None) // YoloDotNet default
            });

            // Print model type
            Console.WriteLine($"Loaded ONNX Model: {yolo.ModelInfo}");

            // Load input image as SKBitmap (or SKImage)
            // The image is sourced from SharedConfig for test/demo purposes.
            using var image = SKBitmap.Decode(SharedConfig.GetTestImage(ImageType.People));

            // Run inference
            var results = yolo.RunSegmentation(image, confidence: 0.20, pixelConfedence: 0.5, iou: 0.7);

            // Draw results
            image.Draw(results, _drawingOptions);

            // If using SKImage, the Draw method returns a new SKBitmap with the drawn results.
            // Example:
            // using var resultImage = image.Draw(results, _drawingOptions);

            // Save image
            var fileName = Path.Combine(_outputFolder, $"Segmentation.jpg");
            image.Save(fileName, SKEncodedImageFormat.Jpeg, 80);

            PrintResults(results);
            DisplayOutputFolder();
        }

        private static void SetDrawingOptions()
        {
            // Set options for drawing
            _drawingOptions = new SegmentationDrawingOptions
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
                DrawSegmentationPixelMask = true

                // The following options configure tracked object tails, which visualize 
                // the movement path of detected objects across a sequence of frames or images.
                // Drawing the tail only works when tracking is enabled (e.g., using SortTracker).
                // This is demonstrated in the VideoStream demo.

                // DrawTrackedTail = false,
                // TailPaintColorEnd = new(),
                // ailPaintColorStart = new(),
                // TailThickness = 0,
            };
        }

        private static void PrintResults(List<Segmentation> results)
        {
            Console.WriteLine();
            Console.WriteLine($"Inference Results: {results.Count} objects");
            Console.WriteLine(new string('=', 80));

            Console.ForegroundColor = ConsoleColor.Blue;

            foreach (var result in results)
            {
                var label = result.Label.Name;
                var confidence = (result.Confidence * 100).ToString("0.##", CultureInfo.InvariantCulture);
                Console.WriteLine($"{label} ({confidence}%)");
            }

            Console.ForegroundColor = ConsoleColor.Gray;
        }

        private static void CreateOutputFolder()
        {
            _outputFolder = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.Desktop), "YoloDotNet_Results");

            if (Directory.Exists(_outputFolder) is false)
                Directory.CreateDirectory(_outputFolder);
        }

        private static void DisplayOutputFolder()
            => Process.Start("explorer.exe", _outputFolder);
    }
}
