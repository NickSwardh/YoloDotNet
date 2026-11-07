// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2025-2026 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

using System.Diagnostics;
using System.Globalization;
using SkiaSharp;
using YoloDotNet;
using YoloDotNet.Enums;
using YoloDotNet.ExecutionProvider.Cpu;
using YoloDotNet.Extensions;
using YoloDotNet.Models;
using YoloDotNet.Test.Common;
using YoloDotNet.Test.Common.Enums;

namespace OBBDetectionDemo
{
    /// <summary>
    /// Detects objects with oriented (rotated) bounding boxes in an image and saves the image with the results drawn on it.
    /// See Demo/README.md for how the demos work and how to customize them.
    /// </summary>
    internal class Program
    {
        private static void Main()
        {
            // 1. Load the model
            using var yolo = new Yolo(new YoloOptions
            {
                ExecutionProvider = new CpuExecutionProvider(SharedConfig.GetTestModelV26(ModelType.ObbDetection)),

                // Resize the image to the model's input size without keeping the aspect ratio
                ImageResize = ImageResize.Stretched
            });

            Console.WriteLine($"Loaded model: {yolo.ModelInfo}");

            // 2. Run inference
            using var image = SKBitmap.Decode(SharedConfig.GetTestImage(ImageType.ObbDetection));
            var results = yolo.RunObbDetection(image, confidence: 0.25, iou: 0.7);

            PrintResults(results);

            // 3. Draw the results on the image
            image.Draw(results);

            // 4. Save the image
            SaveAndShow(image, "OBBDetection.jpg");
        }

        private static void PrintResults(List<OBBDetection> results)
        {
            Console.WriteLine($"\nFound {results.Count} objects:");

            foreach (var result in results)
                Console.WriteLine($"  {result.Label.Name} ({result.Confidence.ToString("0%", CultureInfo.InvariantCulture)})");
        }

        private static void SaveAndShow(SKBitmap image, string fileName)
        {
            var outputFolder = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.Desktop), "YoloDotNet_Results");
            Directory.CreateDirectory(outputFolder);

            var filePath = Path.Combine(outputFolder, fileName);
            image.Save(filePath, SKEncodedImageFormat.Jpeg, 80);
            Console.WriteLine($"\nSaved: {filePath}");

            // Open the output folder in the system's file browser
            var fileBrowser = OperatingSystem.IsWindows() ? "explorer"
                            : OperatingSystem.IsMacOS() ? "open"
                            : "xdg-open";

            Process.Start(fileBrowser, outputFolder);
        }
    }
}
