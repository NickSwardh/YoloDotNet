// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2025-2026 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

using SkiaSharp;
using YoloDotNet;
using YoloDotNet.Extensions;
using YoloDotNet.Models;
using YoloDotNet.ExecutionProvider.MIGraphX;

namespace MIGraphXDemo
{
    internal class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("YoloDotNet - MIGraphX (AMD ROCm) Demo");
            Console.WriteLine("=====================================\n");

            // Check if model and image paths are provided
            if (args.Length < 2)
            {
                Console.WriteLine("Usage: MIGraphXDemo <model.onnx> <image.jpg> [deviceId]");
                Console.WriteLine("\nExample:");
                Console.WriteLine("  MIGraphXDemo yolov8n.onnx image.jpg 0");
                Console.WriteLine("\nParameters:");
                Console.WriteLine("  model.onnx  - Path to ONNX model file");
                Console.WriteLine("  image.jpg   - Path to input image");
                Console.WriteLine("  deviceId    - GPU device ID (default: 0, use -1 for CPU fallback)");
                return;
            }

            var modelPath = args[0];
            var imagePath = args[1];
            var deviceId = args.Length > 2 ? int.Parse(args[2]) : 0;

            // Validate files exist
            if (!File.Exists(modelPath))
            {
                Console.WriteLine($"Error: Model file not found: {modelPath}");
                return;
            }

            if (!File.Exists(imagePath))
            {
                Console.WriteLine($"Error: Image file not found: {imagePath}");
                return;
            }

            Console.WriteLine($"Model: {modelPath}");
            Console.WriteLine($"Image: {imagePath}");
            Console.WriteLine($"Device ID: {deviceId} {(deviceId == -1 ? "(CPU fallback)" : "(AMD GPU)")}\n");

            try
            {
                // Create MIGraphX execution provider
                Console.WriteLine("Initializing MIGraphX execution provider...");
                var provider = new MIGraphXExecutionProvider(modelPath, deviceId);

                // Create YOLO instance
                using var yolo = new Yolo(new YoloOptions
                {
                    ExecutionProvider = provider
                });

                Console.WriteLine($"Model loaded: {provider.OnnxData.ModelType}");
                Console.WriteLine($"Input shape: {string.Join("x", provider.OnnxData.InputShapes.First().Value)}");
                Console.WriteLine($"Data type: {provider.OnnxData.ModelDataType}\n");

                // Load image
                Console.WriteLine("Loading image...");
                using var image = SKBitmap.Decode(imagePath);
                Console.WriteLine($"Image size: {image.Width}x{image.Height}\n");

                // Run inference
                Console.WriteLine("Running inference...");
                var sw = System.Diagnostics.Stopwatch.StartNew();

                var results = yolo.RunObjectDetection(image, confidence: 0.25, iou: 0.7);

                sw.Stop();
                Console.WriteLine($"Inference completed in {sw.ElapsedMilliseconds}ms");
                Console.WriteLine($"Detections: {results.Count}\n");

                // Display results
                if (results.Count > 0)
                {
                    Console.WriteLine("Detected objects:");
                    foreach (var result in results.Take(10))
                    {
                        Console.WriteLine($"  - {result.Label.Name}: {result.Confidence:P2} at ({result.BoundingBox.Left}, {result.BoundingBox.Top})");
                    }

                    if (results.Count > 10)
                        Console.WriteLine($"  ... and {results.Count - 10} more");
                }

                // Draw results and save
                Console.WriteLine("\nDrawing results...");
                image.Draw(results);

                var outputPath = Path.Combine(
                    Path.GetDirectoryName(imagePath) ?? ".",
                    $"{Path.GetFileNameWithoutExtension(imagePath)}_migraphx_result.jpg"
                );

                image.Save(outputPath);
                Console.WriteLine($"Result saved to: {outputPath}");

                // Warm-up and benchmark
                Console.WriteLine("\nRunning benchmark (10 iterations)...");
                var times = new List<long>();

                for (int i = 0; i < 10; i++)
                {
                    sw.Restart();
                    yolo.RunObjectDetection(image, confidence: 0.25, iou: 0.7);
                    sw.Stop();
                    times.Add(sw.ElapsedMilliseconds);
                }

                Console.WriteLine($"Average: {times.Average():F2}ms");
                Console.WriteLine($"Min: {times.Min()}ms");
                Console.WriteLine($"Max: {times.Max()}ms");

                Console.WriteLine("\n✓ Demo completed successfully!");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"\n✗ Error: {ex.Message}");
                Console.WriteLine("\nTroubleshooting:");
                Console.WriteLine("  1. Ensure ROCm is installed: rocm-smi");
                Console.WriteLine("  2. Check GPU availability: rocminfo");
                Console.WriteLine("  3. Verify ONNX model format and opset version");
                Console.WriteLine("  4. Try CPU fallback with deviceId=-1");
            }
        }
    }
}
