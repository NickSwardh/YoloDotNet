// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2025-2026 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

using SkiaSharp;
using System.Diagnostics;
using System.Globalization;
using System.Text.Json;
using YoloDotNet;
using YoloDotNet.Enums;
using YoloDotNet.ExecutionProvider.Cuda;
using YoloDotNet.ExecutionProvider.Cuda.TensorRT;
using YoloDotNet.Extensions;
using YoloDotNet.Models;
using YoloDotNet.Test.Common;

namespace RFDETRNanoDemo
{
    /// <summary>
    /// Demonstrates RF-DETR inference with CUDA + TensorRT FP16,
    /// timing each image under test/assets.
    /// </summary>
    internal class Program
    {
        private static string _outputFolder = default!;
        private static string _trtEngineCacheFolder = default!;
        private static DetectionDrawingOptions _detectionDrawingOptions = default!;
        private static SegmentationDrawingOptions _segmentationDrawingOptions = default!;

        private const int WarmupRuns = 2;
        private static readonly string[] ImageExtensions = [".jpg", ".jpeg", ".png", ".bmp", ".webp"];

        static void Main(string[] args)
        {
            CreateOutputFolder();
            SetDrawingOptions();

            var runSegmentation = true;
            var modelPath = Path.Join(SharedConfig.ModelsFolder, "de251c-45%.onnx");
            var classNamesPath = Path.Join(SharedConfig.ModelsFolder, "de251c-45%_class_names.txt");

            if (EnsureDemoAssets(modelPath, classNamesPath, true) is false)
                return;

            var imagePaths = CollectTestImages();
            if (imagePaths.Count == 0)
            {
                Console.ForegroundColor = ConsoleColor.Yellow;
                Console.WriteLine($"No images found under: {GetPreferredImageFolder()}");
                Console.ForegroundColor = ConsoleColor.Gray;
                return;
            }

            Console.WriteLine("Loading or building TensorRT FP16 engine cache...\n" +
                "- Existing compatible cache will be reused automatically.\n" +
                "- First build may take several minutes for RF-DETR Seg Medium.\n" +
                $"- Cache directory: {_trtEngineCacheFolder}\n");

            using var yolo = new Yolo(new YoloOptions
            {
                ExecutionProvider = new CudaExecutionProvider(
                    model: modelPath,
                    gpuId: 0,
                    trtConfig: new TensorRt
                    {
                        Precision = TrtPrecision.FP16,
                        BuilderOptimizationLevel = 3,
                        EngineCachePath = _trtEngineCacheFolder,
                        EngineCachePrefix = "de251c-45%",
                        Int8CalibrationCacheFile = Path.Join(SharedConfig.AbsoluteAssetsPath, "cache", "de251c-45%.cache"),
                    },
                    modelType:     ModelType.Segmentation ,
                    modelVersion: ModelVersion.RFDETR,
                    labels: LoadLabels(classNamesPath)),

                ImageResize = ImageResize.Stretched,
                // Source photos here are large (e.g. ~3072x4096) and get downscaled ~7-9x to the
                // model's 432x432 input. Plain bilinear without mipmapping aliases badly on
                // high-frequency fabric weave texture at that reduction ratio, which was flipping
                // classification on ambiguous defects (e.g. "yellowish stain" being misread as
                // "dragging filth"). Enabling mipmapping fixes the aliasing and matches PIL/
                // torchvision's higher-quality downscale used by the official rfdetr Python path.
                //SamplingOptions = new(SKFilterMode.Linear, SKMipmapMode.Linear),
                SamplingOptions = new(SKFilterMode.Linear, SKMipmapMode.None) ,
                // RF-DETR uses ImageNet normalization after scaling RGB values to 0..1.
                ImageMean = [0.485f, 0.456f, 0.406f],
                ImageStd = [0.229f, 0.224f, 0.225f]
            });

            Console.WriteLine($"Loaded ONNX Model: {yolo.ModelInfo}");
            Console.WriteLine($"Test images: {imagePaths.Count} from {GetPreferredImageFolder()}");
            Console.WriteLine();

            Warmup(yolo, imagePaths[0], runSegmentation);

            var timingsMs = new List<double>(imagePaths.Count);

            for (var i = 0; i < imagePaths.Count; i++)
            {
                var imagePath = imagePaths[i];
                using var image = SKBitmap.Decode(imagePath);
                if (image is null)
                {
                    Console.WriteLine($"[{i + 1}/{imagePaths.Count}] skip (decode failed): {Path.GetFileName(imagePath)}");
                    continue;
                }

                var sw = Stopwatch.StartNew();
                if (runSegmentation)
                {
                    var results = yolo.RunSegmentation(image, confidence: 0.2, pixelConfedence: 0.5, iou: 0.7);
                    sw.Stop();

                    timingsMs.Add(sw.Elapsed.TotalMilliseconds);
                    SaveSegmentationResult(image, results, imagePath);
                    PrintImageResult(i + 1, imagePaths.Count, imagePath, results.Count, sw.Elapsed.TotalMilliseconds);
                }
                else
                {
                    var results = yolo.RunObjectDetection(image, confidence: 0.2, iou: 0.7);
                    sw.Stop();

                    timingsMs.Add(sw.Elapsed.TotalMilliseconds);
                    SaveObjectDetectionResult(image, results, imagePath);
                    PrintImageResult(i + 1, imagePaths.Count, imagePath, results.Count, sw.Elapsed.TotalMilliseconds);
                }
            }

            PrintTimingSummary(timingsMs);
            DisplayOutputFolder();
        }

        private static void Warmup(Yolo yolo, string imagePath, bool runSegmentation)
        {
            using var image = SKBitmap.Decode(imagePath);
            if (image is null)
                return;

            Console.WriteLine($"Warmup x{WarmupRuns} on {Path.GetFileName(imagePath)}...");
            for (var i = 0; i < WarmupRuns; i++)
            {
                if (runSegmentation)
                    _ = yolo.RunSegmentation(image, confidence: 0.4, pixelConfedence: 0.5, iou: 0.7);
                else
                    _ = yolo.RunObjectDetection(image, confidence: 0.4, iou: 0.7);
            }

            Console.WriteLine("Warmup done.\n");
        }

        private static string GetPreferredImageFolder()
        {
            var testFolder = Path.Join(SharedConfig.AbsoluteAssetsPath, "Test");
            if (Directory.Exists(testFolder))
                return testFolder;

            return SharedConfig.MediaFolder;
        }

        private static List<string> CollectTestImages()
        {
            var folder = GetPreferredImageFolder();
            if (Directory.Exists(folder) is false)
                return [];

            return Directory
                .EnumerateFiles(folder, "*.*", SearchOption.TopDirectoryOnly)
                .Where(path => ImageExtensions.Contains(Path.GetExtension(path), StringComparer.OrdinalIgnoreCase))
                .OrderBy(path => path, StringComparer.OrdinalIgnoreCase)
                .ToList();
        }

        private static void SaveSegmentationResult(SKBitmap image, List<Segmentation> results, string imagePath)
        {
            image.Draw(results, _segmentationDrawingOptions);
            var fileName = Path.Combine(_outputFolder, $"seg_{Path.GetFileNameWithoutExtension(imagePath)}.jpg");
            image.Save(fileName, SKEncodedImageFormat.Jpeg, 80);
        }

        private static void SaveObjectDetectionResult(SKBitmap image, List<ObjectDetection> results, string imagePath)
        {
            image.Draw(results, _detectionDrawingOptions);
            var fileName = Path.Combine(_outputFolder, $"det_{Path.GetFileNameWithoutExtension(imagePath)}.jpg");
            image.Save(fileName, SKEncodedImageFormat.Jpeg, 80);
        }

        private static void PrintImageResult(int index, int total, string imagePath, int detectionCount, double elapsedMs)
        {
            var ms = elapsedMs.ToString("0.00", CultureInfo.InvariantCulture);
            Console.WriteLine($"[{index}/{total}] {Path.GetFileName(imagePath)}  detections={detectionCount}  inference={ms} ms");
        }

        private static void PrintTimingSummary(List<double> timingsMs)
        {
            Console.WriteLine();
            Console.WriteLine(new string('=', 80));
            Console.WriteLine("Inference Timing Summary (excluding warmup / draw / save)");
            Console.WriteLine(new string('=', 80));

            if (timingsMs.Count == 0)
            {
                Console.WriteLine("No successful inferences.");
                return;
            }

            var avg = timingsMs.Average();
            var min = timingsMs.Min();
            var max = timingsMs.Max();
            var total = timingsMs.Sum();
            var fps = avg > 0 ? 1000.0 / avg : 0;

            Console.WriteLine($"Images      : {timingsMs.Count}");
            Console.WriteLine($"Total      : {total.ToString("0.00", CultureInfo.InvariantCulture)} ms");
            Console.WriteLine($"Avg        : {avg.ToString("0.00", CultureInfo.InvariantCulture)} ms  (~{fps.ToString("0.00", CultureInfo.InvariantCulture)} FPS)");
            Console.WriteLine($"Min        : {min.ToString("0.00", CultureInfo.InvariantCulture)} ms");
            Console.WriteLine($"Max        : {max.ToString("0.00", CultureInfo.InvariantCulture)} ms");
        }

          private static string[] LoadLabels(string path)
            => File.ReadAllLines(path)
                .Where(label => string.IsNullOrWhiteSpace(label) is false)
                .ToArray();
        private static bool EnsureDemoAssets(string modelPath, string classNamesPath, bool segmentation)
        {
            if (File.Exists(modelPath) && File.Exists(classNamesPath))
                return true;

            var mode = segmentation ? "segmentation" : "object detection";

            Console.ForegroundColor = ConsoleColor.Yellow;
            Console.WriteLine($"RF-DETR Nano {mode} demo assets are missing.");
            Console.WriteLine($"Expected model: {modelPath}");
            Console.WriteLine($"Expected labels: {classNamesPath}");
            Console.WriteLine("Run detection asset download with: python Demo/RFDETRNanoDemo/download-rfdetr-nano.py");
            Console.WriteLine("For segmentation, place an RF-DETR segmentation ONNX model at test/assets/Models/rfdetr-nano-seg.onnx.");
            Console.ForegroundColor = ConsoleColor.Gray;

            return false;
        }

        private static void SetDrawingOptions()
        {
            _detectionDrawingOptions = new DetectionDrawingOptions
            {
                DrawBoundingBoxes = true,
                DrawConfidenceScore = true,
                DrawLabels = true,
                EnableFontShadow = true,
                Font = SKTypeface.Default,
                FontSize = 18,
                FontColor = SKColors.White,
                DrawLabelBackground = true,
                EnableDynamicScaling = true,
                BorderThickness = 2,
                BoundingBoxOpacity = 128,
            };

            _segmentationDrawingOptions = new SegmentationDrawingOptions
            {
                DrawBoundingBoxes = true,
                DrawConfidenceScore = true,
                DrawLabels = true,
                EnableFontShadow = true,
                Font = SKTypeface.Default,
                FontSize = 18,
                FontColor = SKColors.White,
                DrawLabelBackground = true,
                EnableDynamicScaling = true,
                BorderThickness = 2,
                BoundingBoxOpacity = 128,
                DrawSegmentationPixelMask = true,
                PixelMaskOpacity = 128,
                DrawContour = true,
                ContourThickness = 2,
            };
        }

        private static void CreateOutputFolder()
        {
            _outputFolder = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.Desktop), "YoloDotNet_Results");
            _trtEngineCacheFolder = Path.Combine(_outputFolder, "TensorRT_Engine_Cache");

            if (Directory.Exists(_trtEngineCacheFolder) is false)
                Directory.CreateDirectory(_trtEngineCacheFolder);
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
