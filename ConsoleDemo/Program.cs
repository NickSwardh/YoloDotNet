using ConsoleDemo.Config;
using SkiaSharp;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using YoloDotNet;
using YoloDotNet.Enums;
using YoloDotNet.Extensions;
using YoloDotNet.Models;
using YoloDotNet.Test.Common;
using YoloDotNet.Test.Common.Enums;
using YoloDotNet.Trackers;
using YoloDotNet.Video;

int lineCounter = 0;
Console.CursorVisible = false;

CreateOutputFolder();

Action<ModelType, ModelVersion, ImageType, bool, bool> runDemoAction = RunDemo;
runDemoAction(ModelType.Classification, ModelVersion.V8, ImageType.Hummingbird, false, false);
runDemoAction(ModelType.Classification, ModelVersion.V11, ImageType.Hummingbird, false, false);

runDemoAction(ModelType.ObjectDetection, ModelVersion.V5U, ImageType.Street, false, false);
runDemoAction(ModelType.ObjectDetection, ModelVersion.V8, ImageType.Street, false, false);
runDemoAction(ModelType.ObjectDetection, ModelVersion.V9, ImageType.Street, false, false);
runDemoAction(ModelType.ObjectDetection, ModelVersion.V10, ImageType.Street, false, false);
runDemoAction(ModelType.ObjectDetection, ModelVersion.V11, ImageType.Street, false, false);
runDemoAction(ModelType.ObjectDetection, ModelVersion.V12, ImageType.Street, false, false);

runDemoAction(ModelType.ObbDetection, ModelVersion.V8, ImageType.Island, false, false);
runDemoAction(ModelType.ObbDetection, ModelVersion.V11, ImageType.Island, false, false);

runDemoAction(ModelType.Segmentation, ModelVersion.V8, ImageType.People, false, false);
runDemoAction(ModelType.Segmentation, ModelVersion.V11, ImageType.People, false, false);

runDemoAction(ModelType.PoseEstimation, ModelVersion.V8, ImageType.Crosswalk, false, false);
runDemoAction(ModelType.PoseEstimation, ModelVersion.V11, ImageType.Crosswalk, false, false);

ObjectDetectionOnVideo();
DisplayOutputFolder();
DisplayOnnxMetaDataExample();

#region Helper methods

static void CreateOutputFolder()
{
    var outputFolder = DemoSettings.OUTPUT_FOLDER;

    if (Directory.Exists(outputFolder) is false)
        Directory.CreateDirectory(outputFolder);
}

void RunDemo(ModelType modelType, ModelVersion modelVersion, ImageType imageType, bool cuda = false, bool primeGpu = false)
{
    var modelPath = modelVersion switch
    {
        ModelVersion.V5U => SharedConfig.GetTestModelV5U(modelType),
        ModelVersion.V8 => SharedConfig.GetTestModelV8(modelType),
        ModelVersion.V9 => SharedConfig.GetTestModelV9(modelType),
        ModelVersion.V10 => SharedConfig.GetTestModelV10(modelType),
        ModelVersion.V11 => SharedConfig.GetTestModelV11(modelType),
        ModelVersion.V12 => SharedConfig.GetTestModelV12(modelType),
        _ => throw new ArgumentException("Unkown yolo version")
    };

    var imagePath = SharedConfig.GetTestImage(imageType);

    using var yolo = new Yolo(new YoloOptions()
    {
        OnnxModel = modelPath,
        Cuda = cuda,
        PrimeGpu = primeGpu,
        ImageResize = ImageResize.Proportional
        // ImageResize = ImageResize.Proportional, // default
        // SamplingOptions = new SKSamplingOptions(SKFilterMode.Linear, SKMipmapMode.None) // default

        // 💡 Tip: For usage examples of SamplingOptions and how it can be used to fine-tune SkiaSharp rendering
        // and performance, affecting inference results, see the YOLODotNet benchmark examples:  
        // https://github.com/NickSwardh/YoloDotNet/blob/development/test/YoloDotNet.Benchmarks/ImageExtensionTests/ResizeImageTests.cs
    });

    using var image = SKBitmap.Decode(imagePath);

    var device = cuda ? "GPU" : "CPU";
    device += device == "CPU" ? "" : primeGpu ? ", primed: yes" : ", primed: no";

    Console.Write($"{yolo.OnnxModel.ModelType,-16} {modelVersion,-5}device: {device}");
    Console.WriteLine();

    List<LabelModel> labels = new();

    switch (modelType)
    {
        case ModelType.Classification:
            {
                var result = yolo.RunClassification(image, 1);
                labels = [.. result.Select(x => new LabelModel { Name = x.Label })];
                image.Draw(result);
                break;
            }
        case ModelType.ObjectDetection:
            {
                var result = yolo.RunObjectDetection(image, 0.23, 0.7);
                labels = [.. result.Select(x => x.Label)];
                image.Draw(result);
                break;
            }
        case ModelType.ObbDetection:
            {
                var result = yolo.RunObbDetection(image, 0.23, 0.7);
                labels = [.. result.Select(x => x.Label)];
                image.Draw(result);
                break;
            }
        case ModelType.Segmentation:
            {
                var result = yolo.RunSegmentation(image, 0.25, 0.65, 0.5);
                labels = [.. result.Select(x => x.Label)];
                image.Draw(result);
                break;
            }
        case ModelType.PoseEstimation:
            {
                var result = yolo.RunPoseEstimation(image,0.23, 0.7);
                labels = [.. result.Select(x => x.Label)];
                image.Draw(result, CustomKeyPointColorMap.KeyPointOptions);
                break;
            }
    }

    DisplayDetectedLabels(labels);
    image.Save(Path.Combine(DemoSettings.OUTPUT_FOLDER,  $"{ modelType}_{modelVersion}.jpg"));
}

void ObjectDetectionOnVideo()
{
    var sortTrack = new SortTrack(0.5f, 3, 30);

    // Initialize new yolo object
    using var yolo = new Yolo(new YoloOptions
    {
        OnnxModel = SharedConfig.GetTestModelV11(ModelType.ObjectDetection),
        Cuda = true,
        ImageResize = ImageResize.Proportional
    });

    // Initialize video
    yolo.InitializeVideo(new VideoOptions
    {
        VideoInput = SharedConfig.GetTestImage("walking.mp4"),
        VideoOutput = DemoSettings.OUTPUT_FOLDER,
        SaveProcessedFramesToVideo = true,
        FrameRate = FrameRate.AUTO,
        Width = 640, // Resize video...
        Height = -2, // -2 = automatically calculate dimensions to keep proportions
        CompressionQuality = 30,
        SegmentDuration = 0,
    });

    var metadata = yolo.GetVideoMetaData();

    var message = "Running Object Detection on Video with YOLOv11: ";
    var messageLength = message.Length;

    Console.WriteLine();
    Console.Write(message);

    lineCounter += 1;

    var progress = 0;
    // Subscribe to event for running inference on frames
    yolo.OnVideoFrameReceived = (SKBitmap image, long frameIndex) =>
    {
        // Run inference on incoming frame
        var result = yolo.RunObjectDetection(image).FilterLabels(["person"]);

        // Draw results on frame
        image.Draw(result);

        // Do some more processing if needed...

        // Optional: Save frame to folder
        // image.Save($@"path_to_folder/frame_{frameIndex}.jpg");

        // Calculate progress in %
        progress = (int)((double)(frameIndex) / metadata.TargetTotalFrames * 100);

        var str = $"{progress}% [frame {frameIndex} of {metadata.TargetTotalFrames}]";

        Console.SetCursorPosition(messageLength, lineCounter);
        Console.Write(new string(' ', str.Length));
        Console.SetCursorPosition(messageLength, lineCounter);
        Console.Write(str);
    };

    yolo.OnVideoEnd = () =>
    {
        Console.WriteLine();

        if (progress == 100)
        {
            Console.ForegroundColor = ConsoleColor.Green;
            Console.WriteLine("Video processing completed successfully.");
        }
        else
        {
            Console.ForegroundColor = ConsoleColor.Red;
            Console.WriteLine("Warning: Video processing did not complete successfully.");
        }

        Console.ForegroundColor = ConsoleColor.Gray;
    };

    // Start processing video
    yolo.StartVideoProcessing();

    Console.WriteLine();
}


void DisplayDetectedLabels(IEnumerable<LabelModel> labels)
{
    if (!labels.Any())
        return;

    var ls = labels.GroupBy(x => x.Name)
        .ToDictionary(x => x.Key, x => x.Count());

    Console.WriteLine(new string('-', 33));
    Console.ForegroundColor = ConsoleColor.Blue;

    lineCounter += 2;

    foreach (var label in ls)
    {
        Console.WriteLine($"{label.Key,16} ({label.Value})");
        lineCounter++;
    }

    Console.ForegroundColor = ConsoleColor.Gray;
    Console.WriteLine();
    lineCounter++;
}

static void DisplayOnnxMetaDataExample()
{
    Console.WriteLine();
    Console.WriteLine("Internal ONNX properties");
    Console.WriteLine(new string('-', 58));

    using var yolo = new Yolo(new YoloOptions
    {
        OnnxModel = SharedConfig.GetTestModelV8(ModelType.ObjectDetection)
    });

    // Display internal ONNX properties...
    foreach (var property in yolo.OnnxModel.GetType().GetProperties())
    {
        var value = property.GetValue(yolo.OnnxModel);
        Console.WriteLine($"{property.Name,-20}{value!}");

        if (property.Name == nameof(yolo.OnnxModel.CustomMetaData))
        {
            var customMetaData = (Dictionary<string, string>)value!;

            foreach (var data in customMetaData)
                Console.WriteLine($"{"",-20}{data.Key,-20}{data.Value}");
        }
    }

    var labels = yolo.OnnxModel.Labels;

    Console.WriteLine();
    Console.WriteLine($"Labels ({labels.Length}):");
    Console.WriteLine(new string('-', 58));

    // Display labels and its corresponding color
    for (var i = 0; i < 3; i++)
    {
        // Capitalize first letter in label
        var label = char.ToUpper(labels[i].Name[0]) + labels[i].Name[1..];
        Console.WriteLine($"index: {i,-8} label: {label,-20} color: {labels[i].Color}");
    }

    Console.WriteLine("...");
    Console.WriteLine();
}

static void DisplayOutputFolder()
    => Process.Start("explorer.exe", DemoSettings.OUTPUT_FOLDER);

#endregion