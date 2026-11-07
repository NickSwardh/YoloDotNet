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

namespace YoloE_SegmentationDemo
{
    /// <summary>
    /// Finds and segments objects with a YOLO-E model, using the text or visual prompts exported into the model,
    /// and saves the image with the results drawn on it.
    /// See Demo/README.md for how the demos work, and this demo's README.md for how to export a YOLO-E model.
    /// </summary>
    internal class Program
    {
        // Set this to your exported YOLO-E model. No model is included: export one with your own prompts first (see README.md).
        private const string MODEL_PATH = "path/to/model.onnx";

        private static void Main()
        {
            // 1. Load the model
            using var yolo = new Yolo(new YoloOptions
            {
                ExecutionProvider = new CpuExecutionProvider(MODEL_PATH),
                ImageResize = ImageResize.Stretched
            });

            Console.WriteLine($"Loaded model: {yolo.ModelInfo}");

            // 2. Run inference
            using var image = SKBitmap.Decode(SharedConfig.GetTestImage(ImageType.Segmentation));
            var results = yolo.RunSegmentation(image, confidence: 0.20, pixelConfedence: 0.5);

            PrintResults(results);

            // 3. Draw the results on the image
            image.Draw(results);

            // 4. Save the image
            SaveAndShow(image, "YoloE_Segmentation.jpg");
        }

        private static void PrintResults(List<Segmentation> results)
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
