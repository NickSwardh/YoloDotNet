using SkiaSharp;
using System.Diagnostics;
using YoloDotNet;
using YoloDotNet.Enums;
using YoloDotNet.Extensions;
using YoloDotNet.Models;
using YoloDotNet.Test.Common;
using YoloDotNet.Test.Common.Enums;
using YoloDotNet.Trackers;
using YoloDotNet.Video;

namespace VideoStreamDemo
{
    /// <summary>
    /// Demonstrates object detection and tracking on videos or livestreams using the YoloDotNet library.
    ///
    /// This demo loads a video source (file, livestream, or webcam), runs object detection and optional tracking on each frame,
    /// draws the results (bounding boxes, labels, confidence scores, and tracked tails), and optionally saves the output video.
    ///
    /// It showcases:
    /// - Model initialization with configurable hardware and preprocessing options
    /// - Real-time object detection using YoloDotNet on video streams
    /// - Filtering detections by class labels
    /// - Multi-object tracking across frames using the SORT tracker
    /// - Rendering detection and tracking results directly on video frames
    /// - Saving processed video output and optionally splitting into chunks
    /// - Progress reporting and end-of-stream handling with customizable callbacks
    ///
    /// Example video sources:
    /// - Local file (e.g., C:\videos\test.mp4)
    /// - Livestream (e.g., rtmp://your.server/stream)
    /// - Webcam (e.g., device:Your_Webcam_Name)
    ///
    /// Note:
    /// - CUDA acceleration is required.
    /// - The demo creates an output folder on the desktop to store processed results.
    /// </summary>
    internal class Program
    {
        private static string _outputFolder = default!;
        private static DetectionDrawingOptions _drawingOptions = default!;
        private static SortTracker _sortTracker = default!;

        static void Main(string[] args)
        {
            // Instantiate a tracker object for tracking objects (optional)
            _sortTracker = new SortTracker(0.5f, 5, 60);

            CreateOutputFolder();
            SetDrawingOptions();
            Console.CursorVisible = false;

            // Initialize YoloDotNet.
            // YoloOptions configures the model, hardware settings, and image processing behavior.
            using var yolo = new Yolo(new YoloOptions
            {
                // Path or byte[] to the ONNX model file. 
                // SharedConfig.GetTestModelV11 loads a YOLOv11 classification model.
                OnnxModel = SharedConfig.GetTestModelV11(ModelType.ObjectDetection),

                // NOTE: CUDA is required for video inference!
                Cuda = true,

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
                SamplingOptions = new(SKFilterMode.Linear, SKMipmapMode.None) // YoloDotNet default
            });

            // Print model type.
            Console.WriteLine($"Onnx Model: {yolo.OnnxModel.ModelType}");

            // Set the video options.
            yolo.InitializeVideo(new VideoOptions
            {
                // 💡 The input video source. This can be:
                // - A file path to a local video file.
                // - A URL to a livestream (e.g., RTMP, HTTP).
                // - A video device (e.g., webcam) using the format "device:<device_name>".
                //
                // Examples:
                // VideoInput = @"C:\videos\test.mp4"
                // VideoInput = "rtmp://your.rtmp.server/stream"
                // VideoInput = "device:Realtek_Webcam"
                VideoInput = SharedConfig.GetTestVideo(VideoType.PeopleWalking),

                // 💡 Optional: Path to save the processed output video file.
                // Leave unset (null or empty) if you do not want to save output.
                VideoOutput = Path.Combine(_outputFolder, "video_output.mp4"),

                // 💡 Frame rate for the output video.
                // FrameRate.AUTO will attempt to match the input video’s frame rate.
                FrameRate = FrameRate.AUTO,

                // 💡 Output video width in pixels.
                // Leave unset (0) to use the original width.
                Width = 720,

                // 💡 Output video height in pixels.
                // Leave unset (0) to use the original height.
                // Set to:
                //   -2  : Automatically calculate to preserve aspect ratio based on Width
                Height = -2,

                // 💡 Compression quality for the output video (1-51).
                // Lower values = better quality, larger file size.
                // Higher values = stronger compression, smaller file size, lower quality.
                // Recommended range: 20-35 for reasonable balance.
                CompressionQuality = 30,

                // 💡 Optional: Automatically split output video into chunks.
                // Duration in seconds for each chunk.
                // Example: 600 = split into 10-minute segments.
                // 0 = do not split (generate a single file).
                VideoChunkDuration = 0,

                // 💡 Process every Nth frame.
                // 0 = process all frames (default).
                // Example: 30 = process every 30th frame (useful for surveillance where full-frame detection is unnecessary).
                FrameInterval = 0
            });

            // Display basic video metadata before processing begins.
            var metadata = yolo.GetVideoMetaData();
            PrintMetaData(metadata);

            var progressStats = "Progress: ";
            var progressStatsLength = progressStats.Length;
            var textRow = 15;
            var progress = 0;

            Console.WriteLine();
            Console.WriteLine("Running Object Detection on Video with YOLOv11");
            Console.WriteLine(new string('=', 80));
            Console.Write(progressStats);

            // Assign a handler for incoming video frames.
            // 💡 This Action is invoked *once per processed frame*.
            // It provides:
            //   - `frame`: The current video frame as an SKBitmap.
            //   - `frameIndex`: The zero-based index of the frame in the sequence.
            //
            // You can assign either a method or a lambda expression.
            yolo.OnVideoFrameReceived = (SKBitmap frame, long frameIndex) =>
            {
                // 💡 Run object detection on the current frame.
                // Parameters:
                //   - confidence: Minimum confidence threshold for detections (0.0 - 1.0).
                //   - iou: Intersection-over-union threshold for non-maximum suppression.
                //
                // This will return a list of detected objects with bounding boxes and scores.
                var result = yolo.RunObjectDetection(frame, confidence: 0.25, iou: 0.5)

                    // 💡 (Optional) Filter results to include only specified class labels.
                    // In this case: keep only detections of "person".
                    .FilterLabels([ "person" ])

                    // 💡 (Optional) Apply object tracking to maintain object identities across frames.
                    .Track(_sortTracker);

                // 💡 (Optional) Draw detection and tracking results directly onto the current frame.
                // `_drawingOptions` controls appearance (e.g., color, thickness, font).
                // If not provided, default drawing settings will be applied.
                frame.Draw(result, _drawingOptions);

                // Additional processing logic here if needed...

                // 💡 (Optional) Save the processed frame as an image file.
                // Useful for debugging, auditing, or generating image datasets.
                // Example:
                // var framePath = Path.Combine(_outputFolder, $"frame_{frameIndex}.jpg");
                // frame.Save(framePath, SKEncodedImageFormat.Jpeg, 80);

                // Display progress.
                progress = (int)((double)(frameIndex) / metadata.TargetTotalFrames * 100);
                var str = $"{progress}% [frame {frameIndex} of {metadata.TargetTotalFrames}]";

                Console.SetCursorPosition(progressStatsLength, textRow);
                Console.Write(new string(' ', str.Length));
                Console.SetCursorPosition(progressStatsLength, textRow);
                Console.Write(str);
            };

            // Assign a handler for when video processing finishes.
            // 💡 This Action is invoked *exactly once* at the end of video processing.
            // It is useful for cleanup, reporting, logging, or triggering downstream actions.
            //
            // You can assign either a method or a lambda expression.
            yolo.OnVideoEnd = () =>
            {
                Console.WriteLine();
                Console.WriteLine();

                if (progress == 100)
                {
                    Console.WriteLine("Video processing completed successfully.");
                }
                else
                {
                    Console.ForegroundColor = ConsoleColor.Red;
                    Console.WriteLine("Warning: Video processing did not complete successfully.");
                }

                Console.ForegroundColor = ConsoleColor.Gray;
            };

            // Start processing the video stream.
            yolo.StartVideoProcessing();

            DisplayOutputFolder();

            Console.CursorVisible = true;
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

                // Properties for drawing tracked tail. Default values:
                DrawTrackedTail = true,
                TailPaintColorStart = new SKColor(255, 105, 180),   // #FF69B4 - Blazing Bubblegum Bomber Pink
                TailPaintColorEnd = SKColor.Empty.WithAlpha(0),              // Fade end of tail.
                TailThickness = 4,
            };
        }

        private static void PrintMetaData(VideoMetadata metaData)
        {
            Console.WriteLine();
            Console.WriteLine($"Video MetaData:");
            Console.WriteLine(new string('=', 80));

            Console.ForegroundColor = ConsoleColor.Blue;
            Console.WriteLine($"width           : {metaData.Width}");
            Console.WriteLine($"height          : {metaData.Height}");
            Console.WriteLine($"fps             : {metaData.FPS}");
            Console.WriteLine($"duration        : {metaData.Duration}");
            Console.WriteLine();
            Console.WriteLine($"target fps      : {metaData.FPS}");
            Console.WriteLine($"target width    : {metaData.TargetWidth}");
            Console.WriteLine($"target height   : {metaData.TargetHeight}");

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
