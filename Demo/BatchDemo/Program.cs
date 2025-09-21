// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2025 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

using SkiaSharp;
using System.Diagnostics;
using YoloDotNet;
using YoloDotNet.Enums;
using YoloDotNet.ExecutionProvider.Cuda;
using YoloDotNet.Extensions;
using YoloDotNet.Models;
using YoloDotNet.Test.Common;

namespace BatchDemo
{
    /// <summary>
    /// Demonstrates batch object detection on static images using YoloDotNet, with parallel
    /// processing for faster inference across large datasets.
    ///
    /// This demo:
    /// - Loads images from a folder
    /// - Runs YOLO object detection in parallel
    /// - Draws detection results (boxes, labels, confidence scores)
    /// - Saves annotated images to a desktop results folder
    ///
    /// Key highlights:
    /// - Parallel batch processing to accelerate inference
    /// - Flexible execution provider configuration (CPU, CUDA, TensorRT, OpenVINO)
    /// - Customizable drawing options for text, colors, and bounding box styling
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
                // Select execution provider (determines how and where inference is executed).
                // Available execution providers:
                // 
                // - CpuExecutionProvider  
                //   Runs inference entirely on the CPU. Universally supported but typically slower.
                // 
                // - CudaExecutionProvider  
                //   Executes inference on an NVIDIA GPU using CUDA for accelerated performance.  
                //   Optionally integrates with TensorRT for further optimization, supporting FP32, FP16,  
                //   and INT8 precision modes. This delivers significant speed improvements on compatible GPUs.  
                //   See the TensorRT demo and documentation for detailed configuration and best practices.
                // 
                // Important:  
                // - Choose the provider that matches your available hardware and performance requirements.  
                // - If using CUDA with TensorRT enabled, ensure your environment has a compatible CUDA, cuDNN, and TensorRT setup.
                // - For detailed setup instructions and examples, see the README:  
                //   https://github.com/NickSwardh/YoloDotNet

                ExecutionProvider = new CudaExecutionProvider(

                    // Path or byte[] to the ONNX model file.
                    model: SharedConfig.GetTestModelV11(ModelType.ObjectDetection),

                    // GPU device Id to use for inference. -1 = CPU, 0+ = GPU device Id.
                    gpuId: 0),

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
