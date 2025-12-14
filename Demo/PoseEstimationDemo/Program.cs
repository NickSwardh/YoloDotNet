// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2025 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

using SkiaSharp;
using System.Diagnostics;
using System.Globalization;
using YoloDotNet;
using YoloDotNet.Enums;
using YoloDotNet.ExecutionProvider.Cpu;
using YoloDotNet.Extensions;
using YoloDotNet.Models;
using YoloDotNet.Test.Common;
using YoloDotNet.Test.Common.Enums;

namespace PoseEstimationDemo
{
    /// <summary>
    /// Demonstrates pose estimation on static images using the YoloDotNet library.
    /// 
    /// This demo loads a sample image, runs pose estimation inference to detect human keypoints and skeletal structure,
    /// overlays the detected poses with keypoints, skeleton connections, bounding boxes, labels, and confidence scores,
    /// and saves the annotated image to disk.
    /// 
    /// Features included:
    /// - Model initialization with configurable hardware acceleration and image preprocessing options
    /// - Static image inference returning detailed keypoints for detected persons
    /// - Flexible drawing options for rendering keypoints, skeleton connections, bounding boxes, labels, and confidence
    /// - Support for custom keypoint markers and adjustable confidence thresholds
    /// - Saving output images with quality control and automated output folder management
    /// - Console reporting of detected poses and their confidence scores
    /// 
    /// Execution providers:
    /// - CpuExecutionProvider: runs inference entirely on the CPU. Universally supported but slower.
    /// - CudaExecutionProvider: executes inference on an NVIDIA GPU using CUDA for accelerated performance.  
    ///   Optionally integrates with TensorRT for further optimization, supporting FP32, FP16, and INT8 precision modes.
    /// 
    /// Important notes:
    /// - Choose the execution provider that matches your available hardware and performance requirements.
    /// - PoseDrawingOptions allows customization of font, colors, opacity, keypoint visualization, and label styling.
    /// - Tail visualization for tracking motion paths is supported but not enabled in this static image demo (see VideoStream demo).
    /// - For setup instructions and best practices, see the README:  
    ///   https://github.com/NickSwardh/YoloDotNet
    /// </summary>
    internal class Program
    {
        private static string _outputFolder = default!;
        private static PoseDrawingOptions _drawingOptions = default!;

        static void Main(string[] args)
        {
            CreateOutputFolder();
            SetDrawingOptions();

            // Initialize YoloDotNet.
            // YoloOptions configures the model, hardware settings, and image processing behavior.
            using var yolo = new Yolo(new YoloOptions
            {
                // Select execution provider (determines how and where inference is executed).
                // Available execution providers:
                // 
                //   - CpuExecutionProvider
                //     Runs inference entirely on the CPU. Universally supported on all hardware.
                //
                //   - CudaExecutionProvider
                //     Executes inference on an NVIDIA GPU using CUDA for accelerated performance.  
                //     Optionally integrates with TensorRT for further optimization, supporting FP32, FP16,  
                //     and INT8 precision modes. This delivers significant speed improvements on compatible GPUs.  
                //     See the TensorRT demo and documentation for detailed configuration and best practices.
                //
                //   - OpenVinoExecutionProvider
                //     Runs inference using Intel's OpenVINO toolkit for optimized performance on Intel hardware.
                //
                //   - CoreMLExecutionProvider
                //     Executes inference using Apple's CoreML framework for efficient performance on macOS and iOS devices.
                //
                //   Important:  
                //     - Choose the provider that matches your available hardware and performance requirements.  
                //     - If using CUDA with TensorRT enabled, ensure your environment has a compatible CUDA, cuDNN, and TensorRT setup.
                //     - For detailed setup instructions and examples, see the README:
                //
                //   More information about execution providers and setup instructions can be found in the README:
                //   https://github.com/NickSwardh/YoloDotNet

                ExecutionProvider = new CpuExecutionProvider(SharedConfig.GetTestModelV11(ModelType.PoseEstimation)),

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

            // Load input image as SKBitmap (or SKImage)
            // The image is sourced from SharedConfig for test/demo purposes.
            using var image = SKBitmap.Decode(SharedConfig.GetTestImage(ImageType.Crosswalk));

            // Run inference
            var results = yolo.RunPoseEstimation(image, confidence: 0.25, iou: 0.7);

            // Draw results (optional)
            image.Draw(results, _drawingOptions);

            // If using SKImage, the Draw method returns a new SKBitmap with the drawn results.
            // Example:
            // using var resultImage = image.Draw(results, _drawingOptions);

            // Save image (optional)
            var fileName = Path.Combine(_outputFolder, "PoseEstimation.jpg");
            image.Save(fileName, SKEncodedImageFormat.Jpeg, 80);

            PrintResults(results);
            DisplayOutputFolder();
        }

        private static void SetDrawingOptions()
        {
            // Set options for drawing
            _drawingOptions = new PoseDrawingOptions
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

                // Specifies the keypoints and their connection map used for drawing.
                KeyPointMarkers = CustomKeyPointColorMap.KeyPoints,

                // Draw keypoints and their connections by defining an array of keypoint markers.
                PoseConfidence = 0.65f

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

        private static void PrintResults(List<PoseEstimation> results)
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
        {
            var shell = OperatingSystem.IsWindows() ? "explorer"
                     : OperatingSystem.IsLinux() ? "xdg-open"
                     : OperatingSystem.IsMacOS() ? "open"
                     : null;

            if (shell is not null)
                Process.Start(shell, _outputFolder);
            else
                Console.WriteLine($"Results saved to: {_outputFolder}");
        }
    }
}
