// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2025 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

using SkiaSharp;
using System.Diagnostics;
using System.Globalization;
using YoloDotNet;
using YoloDotNet.Enums;
using YoloDotNet.Extensions;
using YoloDotNet.Models;
using YoloDotNet.Test.Common;
using YoloDotNet.Test.Common.Enums;

namespace ClassificationDemo
{
    /// <summary>
    /// Demonstrates image classification on static images using the YoloDotNet library.
    /// 
    /// This demo loads a sample image, runs classification inference to identify the most probable class(es),
    /// draws the classification labels and confidence scores on the image,
    /// and saves the annotated image to disk.
    /// 
    /// Features included:
    /// - Model initialization with configurable hardware acceleration and image preprocessing settings
    /// - Static image classification inference with configurable top-N class results
    /// - Rendering of classification labels and confidence scores with customizable drawing options
    /// - Saving output images with quality control and automated output folder creation
    /// - Console reporting of classification results with label and confidence display
    /// 
    /// Important notes:
    /// - CUDA (GPU acceleration) is disabled by default but can be enabled for faster inference.
    /// - ClassificationDrawingOptions allows customization of font, color, scaling, and label background.
    /// - The number of classes to return can be limited to focus on the most confident predictions.
    /// </summary>
    internal class Program
    {
        private static string _outputFolder = default!;
        private static ClassificationDrawingOptions _drawingOptions = default!;

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
                OnnxModel = SharedConfig.GetTestModelV11(ModelType.Classification),

                // Use CUDA (Nvidia GPU acceleration) if available. Set to true for GPU inference.
                Cuda = false,

                // If true, will prime (warm up) the GPU to reduce the latency of the first inference.
                PrimeGpu = false,

                // Index of GPU device to use (0 = first GPU).
                GpuId = 0,

                // Resize mode applied before inference. Proportional maintains the aspect ratio (adds padding if needed),
                // while Stretch resizes the image to fit the target size without preserving the aspect ratio.
                // Set this accordingly, as it directly impacts the inference results.
                ImageResize = ImageResize.Proportional,

                // Sampling options for resizing; affects inference speed and quality.
                // For examples of other sampling options, see benchmarks: https://github.com/NickSwardh/YoloDotNet/blob/development/test/YoloDotNet.Benchmarks/ImageExtensionTests/ResizeImageTests.cs
                SamplingOptions = new(SKFilterMode.Nearest, SKMipmapMode.None) // YoloDotNet default
            });

            Console.WriteLine($"Onnx Model: {yolo.OnnxModel.ModelType}");

            // Load input image as SKBitmap (or SKImage)
            // The image is sourced from SharedConfig for test/demo purposes.
            using var image = SKBitmap.Decode(SharedConfig.GetTestImage(ImageType.Hummingbird));

            // Perform classification inference.
            // The 'classes' parameter limits the results to the top-N classes.
            List<Classification>? results = yolo.RunClassification(image, classes: 1);

            // Draw results (optional)
            image.Draw(results, _drawingOptions);

            // If using SKImage, the Draw method returns a new SKBitmap with the drawn results.
            // Example:
            // using var resultImage = image.Draw(results, _drawingOptions);

            // Save image (optional)
            var fileName = Path.Combine(_outputFolder, "Classification.jpg");
            image.Save(fileName, SKEncodedImageFormat.Jpeg, 80);

            PrintResults(results);
            DisplayOutputFolder();
        }

        private static void SetDrawingOptions()
        {
            // Set options for drawing
            _drawingOptions = new ClassificationDrawingOptions
            {
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
                EnableFontShadow = true,
                DrawLabelBackground = true,
                EnableDynamicScaling = true,
            };
        }

        private static void PrintResults(List<Classification> results)
        {
            Console.WriteLine();
            Console.WriteLine("Inference Results");
            Console.WriteLine(new string('=', 80));

            Console.ForegroundColor = ConsoleColor.Blue;

            foreach (var result in results)
            {
                var label = result.Label;
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
