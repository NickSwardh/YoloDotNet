// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2025-2026 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

using System.Diagnostics;
using YoloDotNet;
using YoloDotNet.Enums;
using YoloDotNet.ExecutionProvider.Cuda;
using YoloDotNet.Extensions;
using YoloDotNet.Models;
using YoloDotNet.Test.Common;
using YoloDotNet.Test.Common.Enums;
using YoloDotNet.Trackers;
using YoloDotNet.Video;

namespace VideoStreamDemo
{
    /// <summary>
    /// Detects and tracks objects in a video and saves the video with the results drawn on it.
    /// Requires FFmpeg and FFprobe on the system PATH.
    /// See Demo/README.md for how the demos work and how to customize them.
    /// </summary>
    internal class Program
    {
        private static void Main()
        {
            var outputFolder = CreateOutputFolder();

            // 1. Load the model
            using var yolo = new Yolo(new YoloOptions
            {
                ExecutionProvider = new CudaExecutionProvider(SharedConfig.GetTestModelV26(ModelType.ObjectDetection))
            });

            Console.WriteLine($"Loaded model: {yolo.ModelInfo}");
            PrintVideoDevices();

            // 2. Set up the video input and output
            yolo.InitializeVideo(new VideoOptions
            {
                VideoInput = SharedConfig.GetTestVideo(VideoType.Default),
                VideoOutput = Path.Combine(outputFolder, "video_output.mp4"),
                VideoEncoder = VideoEncoder.H264Nvenc,  // NVIDIA GPU encoder
                Width = 720,
                Height = -2,                            // -2 keeps the aspect ratio
                VideoChunkDuration = 0                  // 0 saves one file instead of 10-minute chunks
            });

            var metadata = yolo.GetVideoMetaData();
            PrintMetadata(metadata);

            // 3. Create a tracker that follows objects across frames
            var tracker = new SortTracker(costThreshold: 0.7f, maxAge: 30, tailLength: 60, highConfidenceThreshold: 0.25f);

            var progress = 0;

            yolo.OnVideoEnd = () =>
            {
                Console.WriteLine();
                Console.WriteLine(progress == 100
                    ? "\nVideo processing completed successfully."
                    : "\nWarning: video processing did not complete successfully.");
            };

            // 4. Process the video: detect, filter, track and draw every frame
            Console.WriteLine("\nProcessing video...");
            Console.CursorVisible = false;

            yolo.StartVideoProcessing<ObjectDetection>((frame, results, frameIndex) =>
            {
                results = results
                    .FilterLabels(["person", "cat", "dog"])
                    .Track(tracker);

                frame.Draw(results);

                progress = (int)((double)frameIndex / metadata.TargetTotalFrames * 100);
                Console.Write($"\rProgress: {progress}% [frame {frameIndex} of {metadata.TargetTotalFrames}]");
            },
            confidence: 0.1,  // Below the tracker's highConfidenceThreshold, so weak detections can keep tracks alive
            iou: 0.5);

            Console.CursorVisible = true;
            OpenFolder(outputFolder);
        }

        private static void PrintVideoDevices()
        {
            var devices = Yolo.GetVideoDevices();

            Console.WriteLine("\nVideo input devices:");

            if (devices.Count == 0)
                Console.WriteLine("  None found");

            foreach (var device in devices)
                Console.WriteLine($"  {device}");
        }

        private static void PrintMetadata(VideoMetadata metadata)
        {
            Console.WriteLine("\nVideo:");
            Console.WriteLine($"  Input:  {metadata.Width}x{metadata.Height}, {metadata.FPS:0.##} fps, {metadata.Duration:0.#} s");
            Console.WriteLine($"  Output: {metadata.TargetWidth}x{metadata.TargetHeight}, {metadata.TargetFPS:0.##} fps");
        }

        private static string CreateOutputFolder()
        {
            var outputFolder = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.Desktop), "YoloDotNet_Results");
            Directory.CreateDirectory(outputFolder);
            return outputFolder;
        }

        private static void OpenFolder(string folder)
        {
            Console.WriteLine($"\nSaved to: {folder}");

            // Open the output folder in the system's file browser
            var fileBrowser = OperatingSystem.IsWindows() ? "explorer"
                            : OperatingSystem.IsMacOS() ? "open"
                            : "xdg-open";

            Process.Start(fileBrowser, folder);
        }
    }
}
