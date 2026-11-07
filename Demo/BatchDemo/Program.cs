// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2025-2026 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

using System.Diagnostics;
using SkiaSharp;
using YoloDotNet;
using YoloDotNet.Enums;
using YoloDotNet.ExecutionProvider.Cuda;
using YoloDotNet.Extensions;
using YoloDotNet.Models;
using YoloDotNet.Test.Common;

namespace BatchDemo
{
    /// <summary>
    /// Detects objects in every image in a folder, processing several images in parallel,
    /// and saves each image with the results drawn on it.
    /// See Demo/README.md for how the demos work, and this demo's README.md for parallel versus sequential processing.
    /// </summary>
    internal class Program
    {
        // Set this to the folder with the images to process.
        private const string IMAGE_FOLDER = @"path\to\image\folder";

        // How many images to process at once. Tune this for your machine (see README.md).
        private const int MAX_PARALLEL_IMAGES = 8;

        private static void Main()
        {
            // 1. Load the model
            using var yolo = new Yolo(new YoloOptions
            {
                ExecutionProvider = new CudaExecutionProvider(SharedConfig.GetTestModelV26(ModelType.ObjectDetection))
            });

            Console.WriteLine($"Loaded model: {yolo.ModelInfo}");

            // 2. Find the images
            var imagePaths = Directory.GetFiles(IMAGE_FOLDER);
            var outputFolder = CreateOutputFolder();

            // 3. Detect, draw and save each image, several at a time.
            // Inference on one Yolo instance is thread-safe, so all threads share it.
            Parallel.ForEach(imagePaths, new ParallelOptions { MaxDegreeOfParallelism = MAX_PARALLEL_IMAGES }, imagePath =>
            {
                using var image = SKBitmap.Decode(imagePath);
                var results = yolo.RunObjectDetection(image, confidence: 0.25);

                image.Draw(results);

                var filePath = Path.Combine(outputFolder, $"ObjectDetection_{Guid.NewGuid()}.jpg");
                image.Save(filePath, SKEncodedImageFormat.Jpeg, 80);
            });

            Console.WriteLine($"\nProcessed {imagePaths.Length} images. Saved to: {outputFolder}");
            OpenFolder(outputFolder);
        }

        private static string CreateOutputFolder()
        {
            var outputFolder = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.Desktop), "YoloDotNet_Results", "batch");
            Directory.CreateDirectory(outputFolder);

            return outputFolder;
        }

        private static void OpenFolder(string folder)
        {
            // Open the folder in the system's file browser
            var fileBrowser = OperatingSystem.IsWindows() ? "explorer"
                            : OperatingSystem.IsMacOS() ? "open"
                            : "xdg-open";

            Process.Start(fileBrowser, folder);
        }
    }
}
