// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2026 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

using SkiaSharp;
using YoloDotNet;
using YoloDotNet.ExecutionProvider.Cpu;
using YoloDotNet.Extensions;
using YoloDotNet.Models;

namespace DockerDemo
{
    /// <summary>
    /// A minimal ASP.NET Core Web API that runs YoloDotNet object detection on the CPU, inside a Docker container.
    /// POST an image to /detect to get the detections as JSON, or add ?download=true to get the image with the results drawn on it.
    /// See Demo/DockerDemo/README.md for how to build and run it.
    /// </summary>
    public class Program
    {
        // The model is copied to the "models" folder of the build output; see the README
        private static readonly string _modelPath = Path.Combine(AppContext.BaseDirectory, "models", "model.onnx");

        public static void Main(string[] args)
        {
            if (File.Exists(_modelPath) is false)
                throw new FileNotFoundException($"Model file not found: {_modelPath}");

            var builder = WebApplication.CreateBuilder(args);

            // Load the model once and share it between requests; inference on one Yolo instance is thread-safe
            builder.Services.AddSingleton(_ => new Yolo(new YoloOptions
            {
                ExecutionProvider = new CpuExecutionProvider(_modelPath)
            }));

            var app = builder.Build();

            app.UseHttpsRedirection();
            app.MapPost("/detect", Detect);

            app.Run();
        }

        private static async Task<IResult> Detect(HttpRequest request, Yolo yolo)
        {
            try
            {
                // 1. Read the uploaded image from the "image" form field
                if (request.HasFormContentType is false)
                    return Results.BadRequest(new { error = "Send the image as multipart/form-data in the 'image' field." });

                var form = await request.ReadFormAsync();
                var file = form.Files.GetFile("image");

                if (file is null || file.Length == 0)
                    return Results.BadRequest(new { error = "Form field 'image' is missing or empty." });

                await using var stream = file.OpenReadStream();
                using var image = SKBitmap.Decode(stream);

                if (image is null)
                    return Results.BadRequest(new { error = "Invalid or unsupported image." });

                // 2. Run inference
                var detections = yolo.RunObjectDetection(image);

                // 3. Return the image with the results drawn on it, or the results as JSON
                var download = string.Equals(request.Query["download"], "true", StringComparison.OrdinalIgnoreCase);

                return download ? DrawnImage(image, detections) : Json(detections);
            }
            catch (Exception ex)
            {
                return Results.Problem(detail: ex.Message);
            }
        }

        private static IResult DrawnImage(SKBitmap image, List<ObjectDetection> detections)
        {
            image.Draw(detections);

            using var encoded = image.Encode(SKEncodedImageFormat.Jpeg, 90);
            return Results.File(encoded.ToArray(), "image/jpeg");
        }

        private static IResult Json(List<ObjectDetection> detections)
            => Results.Json(detections.Select(d => new
            {
                label = d.Label.Name,
                confidence = d.Confidence,
                bbox = new
                {
                    x = d.BoundingBox.Left,
                    y = d.BoundingBox.Top,
                    width = d.BoundingBox.Width,
                    height = d.BoundingBox.Height
                }
            }));
    }
}
