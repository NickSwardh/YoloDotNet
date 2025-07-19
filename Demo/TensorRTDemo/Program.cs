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

namespace TensorRTDemo
{
    /// <summary>
    /// Demonstrates how to use YoloDotNet with GPU acceleration via the TensorRT execution provider.
    ///
    /// This demo performs object detection on a static image using a YOLOv11 ONNX model,
    /// accelerated by TensorRT with configurable precision (FP32, FP16, or INT8).
    /// The image is annotated with bounding boxes, class labels, and confidence scores,
    /// and the result is saved to disk.
    ///
    /// It showcases:
    /// - TensorRT-backed GPU inference with configurable numeric precision (FP32, FP16, INT8)
    /// - Model initialization with configurable hardware and preprocessing options
    /// - Static image inference for detecting objects with standard bounding boxes
    /// - Customizable rendering of detection results, including labels, confidence scores, and boxes
    /// - Saving annotated output to disk
    /// - Console reporting of inference results
    ///
    /// Note:
    /// - The TensorRT engine cache is saved to disk to avoid rebuilding the engine on each run.
    /// - INT8 mode requires a calibration cache file. See <see cref="YoloOptions.Int8CalibrationCacheFile"/> for details.
    /// - Requires a compatible NVIDIA GPU and TensorRT runtime support.
    /// </summary>
    internal class Program
    {
        private static string _outputFolder = default!;
        private static string _trtEngineCacheFolder = default!;
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

                // Available execution providers:
                //   new CpuExecutionProvider()
                //   new CudaExecutionProvider(GpuId: 0, PrimeGpu: true)
                //   new TensorRTExecutionProvider() { ... }
                ExecutionProvider = new TensorRtExecutionProvider
                {
                    GpuId = 0,
                    // Specifies which GPU device index to use for TensorRT execution. 0 = default.

                    Precision = TrtPrecision.FP32,
                    // - FP32: Full precision (32-bit float). Default mode. Highest accuracy, default execution.
                    // - FP16: Half precision (16-bit float). Offers improved performance on supported GPUs with minimal accuracy loss.
                    // - INT8: Integer precision (8-bit). Fastest inference performance, but requires calibration.
                    //
                    //   Note: INT8 mode enables **mixed precision execution**.
                    //   TensorRT will use INT8 precision where supported, and automatically fall back to FP16 or FP32
                    //   for layers or operations that are not quantizable — due to model structure, unsupported ops,
                    //   dynamic ranges, or numerical stability concerns.
                    //
                    //   See Int8CalibrationCacheFile for calibration file requirements.

                    BuilderOptimizationLevel = 3,
                    // Set the builder optimization level to use when building a new engine cache. A higher level
                    // allows TensorRT to spend more building time on more optimization options.
                    //
                    // WARNING: levels below 3 do not guarantee good engine performance, but greatly improve
                    // build time. Default 3, valid range[0 - 5].

                    EngineCachePath = _trtEngineCacheFolder,
                    // Specifies the directory where TensorRT will store and load engine cache files.
                    //
                    // The engine cache avoids rebuilding the TensorRT engine on every startup,
                    // significantly improving initialization time.
                    //
                    // If cache files already exist for the current model, hardware, precision, and configuration,
                    // they will be reused automatically. Otherwise, a new engine cache will be built and stored here.
                    //
                    // Note: Existing cache files are never deleted automatically.
                    // You must manually remove outdated or unused cache files from this directory as needed.

                    EngineCachePrefix = "YoloDotNet",
                    // Sets a filename prefix for the generated TensorRT engine and profile cache files.
                    //
                    // This helps distinguish between cache files from different models, versions, or configurations,
                    // especially when multiple engines are stored in the same EngineCachePath.
                    //
                    // If left empty, a default internal prefix will be used.

                    Int8CalibrationCacheFile = Path.Join(SharedConfig.AbsoluteAssetsPath, "cache", "yolov11s.cache"),
                    // Path to calibration cache required for INT8 precision mode. Leave empty if INT8 is not used.
                    //
                    // Specifies the path to the INT8 calibration cache file used during engine building.
                    // This file is required when using non-quantized models in INT8 mode.
                    // TensorRT uses it to assign dynamic ranges to tensors.
                    //
                    // The calibration cache must be pre-generated using the original model.pt data.
                    //
                    // 🔧 To generate the calibration cache, export the model using the Ultralytics CLI:
                    //
                    //   yolo export model=your_model.pt format=engine int8=true simplify=true data=your_model_dataset.yaml opset=17
                    //
                    // This command generates:
                    //   - A standard ONNX model (unquantized, FP32-based)
                    //   - A TensorRT engine optimized for INT8 precision
                    //   - A calibration cache file: <model_name>.cache
                    //
                    // The path to <model_name>.cache must be specified to run YOLO ONNX models in INT8 mixed precision mode.
                    // Example:
                    //   Int8CalibrationCacheFile = @"path\to\<model_name>.cache"
                },

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
            using var image = SKBitmap.Decode(SharedConfig.GetTestImage(ImageType.Street));

            // Run object detection inference
            var results = yolo.RunObjectDetection(image, confidence: 0.15, iou: 0.7);

            // Draw results
            image.Draw(results, _drawingOptions);

            // If using SKImage, the Draw method returns a new SKBitmap with the drawn results.
            // Example:
            // using var resultImage = image.Draw(results, _drawingOptions);

            // Save image
            var fileName = Path.Combine(_outputFolder, "ObjectDetection.jpg");
            image.Save(fileName, SKEncodedImageFormat.Jpeg, 80);

            PrintResults(results);
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
                // Drawing the tail only works when tracking is enabled (e.g., using SortTracker).
                // This is demonstrated in the VideoStream demo.

                // DrawTrackedTail = false,
                // TailPaintColorEnd = new(),
                // ailPaintColorStart = new(),
                // TailThickness = 0,
            };
        }

        private static void PrintResults(List<ObjectDetection> results)
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
            _outputFolder = Path.Join(Environment.GetFolderPath(Environment.SpecialFolder.Desktop), "YoloDotNet_Results");
            _trtEngineCacheFolder = Path.Join(_outputFolder, "TensorRT_Engine_Cache");

            var folder = _trtEngineCacheFolder;

            if (Directory.Exists(folder) is false)
                Directory.CreateDirectory(folder);
        }

        private static void DisplayOutputFolder()
            => Process.Start(new ProcessStartInfo
            {
                FileName = _outputFolder,
                UseShellExecute = true
            });
        //=> Process.Start("explorer.exe", _outputFolder);
    }
}
