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
    /// Demonstrates object detection using the YoloDotNet library inside a Dockerized
    /// ASP.NET Core Web API, intentionally designed to be minimal, explicit, and easy to follow.
    ///
    /// This demo exposes a single HTTP endpoint that accepts image uploads, performs object detection
    /// using a YOLOv11 ONNX model, and returns either structured detection results or an annotated
    /// image with detections drawn on top.
    ///
    /// The implementation intentionally uses minimal code and runs inference exclusively on the CPU
    /// to keep the demo simple, predictable, and universally compatible across platforms and
    /// container environments.
    ///
    /// Features included:
    /// - Minimal ASP.NET Core Web API for object detection
    /// - CPU-only inference using the CpuExecutionProvider
    /// - Automatic inclusion of the YOLOv11 ONNX model as part of the Release build output
    /// - Image inference via multipart/form-data uploads
    /// - JSON responses containing labels, confidence scores, and bounding box coordinates
    /// - Optional image output with rendered detection overlays via a query parameter
    ///
    /// Execution providers:
    /// - CpuExecutionProvider: runs inference entirely on the CPU.
    ///   This is intentionally chosen to reduce complexity and avoid hardware-specific setup,
    ///   making the demo easy to run in Docker without additional configuration.
    ///
    /// Important notes:
    /// - This project is intended as a getting-started reference, not a production-ready service.
    /// - The project must be built in Release mode, as the Docker image relies on the Release output.
    /// - The API exposes a single endpoint to keep usage and integration straightforward.
    /// - For setup instructions, Docker usage, and API examples, see the README:
    ///   https://github.com/NickSwardh/YoloDotNet
    /// </summary>

    public class Program
    {

        private static DetectionDrawingOptions _drawingOptions = default!;

        // Load the YOLOv11s ONNX model from the "models" directory
        private static string _modelSource = Path.Combine(AppContext.BaseDirectory, "models", "model.onnx");

        public static void Main(string[] args)
        {
            var builder = WebApplication.CreateBuilder(args);

            ConfigureServices(builder);

            var app = builder.Build();

            ConfigureApp(app);

            ConfigureDrawingOptions();

            app.Run();
        }

        /// <summary>
        /// Configures application services
        /// object detection.
        /// </summary>
        private static void ConfigureServices(WebApplicationBuilder builder)
        {
            builder.Services.AddControllers();

            // Ensure the model file exists
            if (!File.Exists(_modelSource))
            {
                throw new FileNotFoundException($"Model file not found: {_modelSource}");
            }

            // Register Yolo service with CPU execution provider
            builder.Services.AddSingleton(sp =>
            {
                return new Yolo(new YoloOptions
                {
                    ExecutionProvider = new CpuExecutionProvider(_modelSource),
                });
            });
        }

        /// <summary>
        /// Configures the request pipeline and endpoint mappings.
        /// </summary>
        private static void ConfigureApp(WebApplication app)
        {
            app.UseHttpsRedirection();
            app.UseAuthorization();
            app.MapControllers();

            app.MapPost("/detect", async (HttpRequest req, Yolo yolo) => await HandleDetect(req, yolo));
        }

        /// <summary>
        /// Processes an HTTP request containing an image and performs object detection using the specified Yolo model.
        /// </summary>
        /// <remarks>If the query parameter 'download' is set to 'true', the response will contain the
        /// input image with detected objects drawn on it as a downloadable file. Otherwise, the response contains the
        /// detection results in JSON format.</remarks>
        private static async Task<IResult> HandleDetect(HttpRequest request, Yolo yolo)
        {
            SKBitmap? bitmap = null;

            try
            {
                if (request.HasFormContentType)
                {
                    var form = await request.ReadFormAsync();
                    var file = form.Files.GetFile("image");

                    if (file is null || file.Length == 0)
                        return Results.BadRequest(new { error = "Form field 'image' is missing or empty." });

                    await using var imageStream = file.OpenReadStream();
                    bitmap = SKBitmap.Decode(imageStream);
                }

                if (bitmap is null)
                    return Results.BadRequest(new { error = "Invalid or unsupported image." });

                var detections = yolo.RunObjectDetection(bitmap, confidence: 0.2, iou: 0.7);

                if (request.Query.TryGetValue("download", out var val) && string.Equals(val, "true", StringComparison.OrdinalIgnoreCase))
                {
                    bitmap.Draw(detections, _drawingOptions ?? default!);
                    return GetImageResultForDownload(bitmap);
                }

                return GetDetectionResults(detections);
            }
            catch (Exception ex)
            {
                return Results.Problem(detail: ex.Message);
            }
            finally
            {
                bitmap?.Dispose();
            }
        }

        /// <summary>
        /// Creates a file result containing the specified bitmap image encoded as a JPEG for download.
        /// </summary>
        private static IResult GetImageResultForDownload(SKBitmap bitmap)
        {
            using var img = SKImage.FromBitmap(bitmap);
            using var data = img.Encode(SKEncodedImageFormat.Jpeg, 90);

            return Results.File(data.ToArray(), "image/jpeg");
        }

        /// <summary>
        /// Creates a JSON result containing the detection results with label, confidence, and bounding box information
        /// for each detected object.
        /// </summary>
        private static IResult GetDetectionResults(List<ObjectDetection> detections)
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

        /// <summary>
        /// Optional. Configures the default drawing options for detection overlays.
        /// </summary>
        private static void ConfigureDrawingOptions()
        {
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

                // Option for video streams:
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
    }
}
