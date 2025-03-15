using System;
using System.IO;
using System.Linq;
using System.Diagnostics;
using System.Collections.Generic;
using YoloDotNet;
using YoloDotNet.Enums;
using YoloDotNet.Models;
using ConsoleDemo.Config;
using YoloDotNet.Extensions;
using YoloDotNet.Test.Common;
using YoloDotNet.Test.Common.Enums;
using SkiaSharp;

Console.CursorVisible = false;

CreateOutputFolder();

Action<ModelType, ModelVersion, ImageType, bool, bool> runDemoAction = RunDemo;
runDemoAction(ModelType.Classification, ModelVersion.V8, ImageType.Hummingbird, false, false);
runDemoAction(ModelType.Classification, ModelVersion.V11, ImageType.Hummingbird, false, false);

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

static void RunDemo(ModelType modelType, ModelVersion modelVersion, ImageType imageType, bool cuda = false, bool primeGpu = false)
{
    var modelPath = modelVersion switch
    {
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
        ModelType = modelType,
        // ImageResize = ImageResize.Proportional, // default
    });

    using var image = SKImage.FromEncodedData(imagePath);

    var device = cuda ? "GPU" : "CPU";
    device += device == "CPU" ? "" : primeGpu ? ", primed: yes" : ", primed: no";

    Console.Write($"{yolo.OnnxModel.ModelType,-16} {modelVersion, -5}device: {device}");
    Console.WriteLine();

    SKImage resultImage = SKImage.Create(new SKImageInfo());
    List<LabelModel> labels = new();
    
    switch (modelType)
    {
        case ModelType.Classification:
            {
                var result = yolo.RunClassification(image, 1);
                labels = result.Select(x => new LabelModel { Name = x.Label }).ToList();
                resultImage = image.Draw(result);
                break;
            }
        case ModelType.ObjectDetection:
            {
                var result = yolo.RunObjectDetection(image, 0.23, 0.7);
                labels = result.Select(x => x.Label).ToList();
                resultImage = image.Draw(result);
                break;
            }
        case ModelType.ObbDetection:
            {
                var result = yolo.RunObbDetection(image, 0.23, 0.7);
                labels = result.Select(x => x.Label).ToList();
                resultImage = image.Draw(result);
                break;
            }
        case ModelType.Segmentation:
            {
                var result = yolo.RunSegmentation(image, 0.23, 0.65, 0.7);
                labels = result.Select(x => x.Label).ToList();

                resultImage = image.Draw(result);
                break;
            }
        case ModelType.PoseEstimation:
            {
                var result = yolo.RunPoseEstimation(image, 0.23, 0.7);
                labels = result.Select(x => x.Label).ToList();
                resultImage = image.Draw(result, CustomKeyPointColorMap.KeyPointOptions);
                break;
            }
    }

    DisplayDetectedLabels(labels);
    resultImage.Save(Path.Combine(DemoSettings.OUTPUT_FOLDER, $"{modelType}_{modelVersion}.jpg"), SKEncodedImageFormat.Jpeg);
    
}

static void ObjectDetectionOnVideo()
{
    var videoOptions = new VideoOptions
    {
        VideoFile = SharedConfig.GetTestImage("walking.mp4"),
        OutputDir = DemoSettings.OUTPUT_FOLDER,
        //GenerateVideo = false,
        //DrawLabels = false,
        //FPS = 30,
        //Width = 640, // Resize video...
        //Height = -2, // -2 = automatically calculate dimensions to keep proportions
        //Quality = 28,
        //DrawConfidence = true,
        //KeepAudio = true,
        //KeepFrames = false,
        DrawSegment = DrawSegment.Default,
        KeyPointOptions = CustomKeyPointColorMap.KeyPointOptions
    };

    Console.WriteLine();
    Console.WriteLine("Running Object Detection on video with Yolo v8...");

    using var yolo = new Yolo(new YoloOptions
    {
        OnnxModel = SharedConfig.GetTestModelV8(ModelType.ObjectDetection),
        ModelType = ModelType.ObjectDetection,
        Cuda = true
    });

    int currentLineCursor = 0;

    // Listen to events...
    yolo.VideoStatusEvent += (sender, e) =>
    {
        Console.WriteLine();
        Console.Write((string)sender!);
        currentLineCursor = Console.CursorTop;
    };

    yolo.VideoProgressEvent += (object sender, EventArgs e) =>
    {
        Console.SetCursorPosition(20, currentLineCursor);
        Console.Write(new string(' ', 4));
        Console.SetCursorPosition(20, currentLineCursor);
        Console.Write("{0}%", (int)sender!);
    };

    yolo.VideoCompleteEvent += (object sender, EventArgs e) =>
    {
        Console.WriteLine();
        Console.WriteLine();
        Console.WriteLine("Complete!");
    };

    Dictionary<int, List<ObjectDetection>> detections = yolo.RunObjectDetection(videoOptions, 0.25);
    Console.WriteLine();
}

static void DisplayDetectedLabels(IEnumerable<LabelModel> labels)
{
    if (!labels.Any())
        return;

    var ls = labels.GroupBy(x => x.Name)
        .ToDictionary(x => x.Key, x => x.Count());

    Console.WriteLine(new string('-', 33));
    Console.ForegroundColor = ConsoleColor.Blue;

    foreach (var label in ls)
        Console.WriteLine($"{label.Key,16} ({label.Value})");

    Console.ForegroundColor = ConsoleColor.Gray;
    Console.WriteLine();
}

static void DisplayOnnxMetaDataExample()
{
    Console.WriteLine();
    Console.WriteLine("Internal ONNX properties");
    Console.WriteLine(new string('-', 58));

    using var yolo = new Yolo(new YoloOptions
    {
        OnnxModel = SharedConfig.GetTestModelV8(ModelType.ObjectDetection),
        ModelType = ModelType.ObjectDetection
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