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

namespace SegmentationDemo
{
    /// <summary>
    /// Demonstrates semantic segmentation on static images using the YoloDotNet library.
    /// 
    /// This demo loads a sample image, performs segmentation inference to detect pixel-level object masks,
    /// draws the segmentation masks along with bounding boxes, labels, and confidence scores,
    /// and saves the annotated image to disk.
    /// 
    /// Key features showcased include:
    /// - Model initialization with flexible hardware options (CPU/GPU) and image preprocessing settings
    /// - Static image segmentation inference with adjustable confidence and mask thresholds
    /// - Comprehensive rendering options for segmentation masks, bounding boxes, labels, and confidence scores
    /// - Saving output images in a standard format with customizable compression
    /// - Console output of detected objects and their confidence levels
    /// - Automatic creation of an output folder on the desktop to store results
    /// 
    /// Execution providers:
    /// - CpuExecutionProvider: runs inference on CPU, universally supported but slower.
    /// - CudaExecutionProvider: uses NVIDIA GPU via CUDA for faster inference, with optional GPU warm-up.
    /// - TensorRtExecutionProvider: leverages NVIDIA TensorRT for highly optimized GPU inference with FP32, FP16, INT8
    ///   precision modes, delivering significant speed improvements.
    ///
    /// Important notes:
    /// - Choose the execution provider based on your hardware and performance requirements.
    /// - SegmentationDrawingOptions provides extensive customization for visual output,
    ///   including font styling, colors, opacity, and mask rendering.
    /// - Segmentation masks are drawn as pixel-level overlays, providing precise object outlines.
    /// - Tail visualization for tracking is mentioned but not enabled in this static image demo.
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
                // SharedConfig.GetTestModelV11 loads a YOLOv11 model.
                OnnxModel = SharedConfig.GetTestModelV11(ModelType.Segmentation),

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
            var results = yolo.RunSegmentation(image, confidence: 0.24, pixelConfedence: 0.5, iou: 0.7);

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
