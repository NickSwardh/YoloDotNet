// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2025-2026 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

using System.Diagnostics;
using System.Globalization;
using SkiaSharp;
using YoloDotNet;
using YoloDotNet.Enums;
using YoloDotNet.ExecutionProvider.Cuda;
using YoloDotNet.ExecutionProvider.Cuda.TensorRT;
using YoloDotNet.Extensions;
using YoloDotNet.Models;
using YoloDotNet.Test.Common;
using YoloDotNet.Test.Common.Enums;

namespace TensorRTDemo
{
    /// <summary>
    /// Detects objects in an image with TensorRT (FP16, CUDA Graph) and saves the image with the results drawn on it.
    /// See Demo/TensorRTDemo/README.md for the setup and the TensorRT options.
    /// </summary>
    internal class Program
    {
        private static readonly string _outputFolder =
            Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.Desktop), "YoloDotNet_Results");

        private static void Main()
        {
            // TensorRT stores its optimized engine here, so it is only built on the first run
            var engineCacheFolder = Path.Combine(_outputFolder, "TensorRT_Engine_Cache");
            Directory.CreateDirectory(engineCacheFolder);

            Console.WriteLine("Loading the TensorRT engine. The first run builds it, which can take a few minutes.");

            // 1. Load the model with TensorRT
            using var yolo = new Yolo(new YoloOptions
            {
                ExecutionProvider = new CudaExecutionProvider(
                    model: SharedConfig.GetTestModelV26(ModelType.ObjectDetection),
                    trtConfig: new TensorRt
                    {
                        Precision = TrtPrecision.FP16,
                        EngineCachePath = engineCacheFolder,

                        // Only used with TrtPrecision.INT8. Generate the cache for your model first; see the README.
                        Int8CalibrationCacheFile = Path.Join(SharedConfig.AbsoluteAssetsPath, "Cache", "yolo26s.cache")
                    },
                    enableCudaGraph: true)
            });

            Console.WriteLine($"Loaded model: {yolo.ModelInfo}");

            // 2. Run inference
            using var image = SKBitmap.Decode(SharedConfig.GetTestImage(ImageType.ObjectDetection));
            var results = yolo.RunObjectDetection(image, confidence: 0.15);

            PrintResults(results);

            // 3. Draw the results on the image
            image.Draw(results);

            // 4. Save the image
            SaveAndShow(image, "TensorRT.jpg");
        }

        private static void PrintResults(List<ObjectDetection> results)
        {
            Console.WriteLine($"\nFound {results.Count} objects:");

            foreach (var result in results)
                Console.WriteLine($"  {result.Label.Name} ({result.Confidence.ToString("0%", CultureInfo.InvariantCulture)})");
        }

        private static void SaveAndShow(SKBitmap image, string fileName)
        {
            var filePath = Path.Combine(_outputFolder, fileName);
            image.Save(filePath, SKEncodedImageFormat.Jpeg, 80);
            Console.WriteLine($"\nSaved: {filePath}");

            // Open the output folder in the system's file browser
            var fileBrowser = OperatingSystem.IsWindows() ? "explorer"
                            : OperatingSystem.IsMacOS() ? "open"
                            : "xdg-open";

            Process.Start(fileBrowser, _outputFolder);
        }
    }
}
