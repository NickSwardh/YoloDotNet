using ConsoleDemo.Config;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using System.Diagnostics;
using YoloDotNet;
using YoloDotNet.Enums;
using YoloDotNet.Extensions;
using YoloDotNet.Models;

Console.CursorVisible = false;

CreateOutputFolder();

Action<string, string, bool, bool> runDemoAction = RunDemo;
runDemoAction("yolov8s-cls.onnx", "hummingbird.jpg", false, false);
runDemoAction("yolov8s.onnx", "street.jpg", false, false);
runDemoAction("yolov8s-obb.onnx", "island.jpg", false, false);
runDemoAction("yolov8s-seg.onnx", "people.jpg", false, false);
runDemoAction("yolov8s-pose.onnx", "crosswalk.jpg", false, false);

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

static void RunDemo(string model, string inferenceImage, bool cuda = false, bool primeGpu = false)
{
    using var yolo = new Yolo(Path.Combine(DemoSettings.MODELS_FOLDER, model), cuda, primeGpu);
    using var image = Image.Load<Rgba32>(Path.Combine(DemoSettings.MEDIA_FOLDER, inferenceImage));

    var device = cuda ? "GPU" : "CPU";
    device += device == "CPU" ? "" : primeGpu ? ", primed: yes" : ", primed: no";

    Console.Write($"{yolo.OnnxModel.ModelType,-20}device: {device,-20}");

    object results = yolo.OnnxModel.ModelType switch
    {
        ModelType.Classification => yolo.RunClassification(image, 5),   // Top 5 classes, default: classes = 1
        ModelType.ObjectDetection => yolo.RunObjectDetection(image),    // Example with default confidence (0.25) and IoU (0.45) threshold;
        ModelType.ObbDetection => yolo.RunObbDetection(image, 0.35, 0.5),
        ModelType.Segmentation => yolo.RunSegmentation(image, 0.25, 0.5),
        ModelType.PoseEstimation => yolo.RunPoseEstimation(image, 0.25, 0.5),
        _ => throw new NotImplementedException()
    };

    Console.Write("done!");
    Console.WriteLine();

    // Draw results and save image
    if (results is List<Classification> classification)
    {
        image.Draw(classification);
    }

    if (results is List<ObjectDetection> objectDetection)
    {
        DisplayDetectedLabels(objectDetection.Select(x => x.Label));
        image.Draw(objectDetection);
    }

    if (results is List<OBBDetection> obbDetection)
    {
        DisplayDetectedLabels(obbDetection.Select(x => x.Label));
        image.Draw(obbDetection);
    }

    if (results is List<Segmentation> segmentation)
    {
        DisplayDetectedLabels(segmentation.Select(x => x.Label));
        image.Draw(segmentation);
    }

    if (results is List<PoseEstimation> poseEstimation)
    {
        DisplayDetectedLabels(poseEstimation.Select(x => x.Label));
        image.Draw(poseEstimation, CustomPoseMarkerColorMap.PoseMarkerOptions);
    }

    image.Save(Path.Combine(DemoSettings.OUTPUT_FOLDER, $"{yolo.OnnxModel.ModelType}.jpg"));
}

static void ObjectDetectionOnVideo()
{
    var videoOptions = new VideoOptions
    {
        VideoFile = Path.Combine(DemoSettings.MEDIA_FOLDER, "walking.mp4"),
        OutputDir = DemoSettings.OUTPUT_FOLDER,
        //GenerateVideo = true,
        //DrawLabels = true,
        //FPS = 30,
        //Width = 640, // Resize video...
        //Height = -2, // -2 = automatically calculate dimensions to keep proportions
        //Quality = 28,
        //DrawConfidence = true,
        //KeepAudio = true,
        //KeepFrames = false,
        DrawSegment = DrawSegment.Default,
        PoseOptions = CustomPoseMarkerColorMap.PoseMarkerOptions
    };

    Console.WriteLine();
    Console.WriteLine("Running Object Detection on video...");
    using var yolo = new Yolo(Path.Combine(DemoSettings.MODELS_FOLDER, "yolov8s.onnx"));

    int currentLineCursor = 0;

    // Listen to events...
    yolo.VideoStatusEvent += (sender, e) =>
    {
        Console.WriteLine();
        Console.Write((string)sender!);
        currentLineCursor = Console.CursorTop;
    };

    yolo.VideoProgressEvent += (object? sender, EventArgs e) =>
    {
        Console.SetCursorPosition(20, currentLineCursor);
        Console.Write(new string(' ', 4));
        Console.SetCursorPosition(20, currentLineCursor);
        Console.Write("{0}%", (int)sender!);
    };

    yolo.VideoCompleteEvent += (object? sender, EventArgs e) =>
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

    using var yolo = new Yolo(Path.Combine(DemoSettings.MODELS_FOLDER, "yolov8s.onnx"), false);

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
