using ConsoleDemo.Config;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using System.Diagnostics;
using YoloDotNet;
using YoloDotNet.Enums;
using YoloDotNet.Extensions;
using YoloDotNet.Models;

// Use models and testdata from assets in YoloDotNet.Tests
const string ASSETS_FOLDER = @"..\..\..\..\YoloDotNet.Tests\Assets";
const string MODELS_FOLDER = ASSETS_FOLDER + @"\models";
const string MEDIA_FOLDER = ASSETS_FOLDER + @"\media";
string outputFolder = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.Desktop), "YoloDotNet_Results");
Console.CursorVisible = false;

// Build pipeline
var pipeline = new List<Action<string>>
{
    CreateOutputFolder,
    Classification,
    ObjectDetection,
    Segmentation,
    PoseEstimation,
    ObjectDetectionOnVideo,
    DisplayOutputFolder
};

// Run pipeline demo
foreach (var item in pipeline)
{
    item.Invoke(outputFolder);
}

DisplayOnnxMetaDataExample();

#region Methods

static void CreateOutputFolder(string outputFolder)
{
    if (Directory.Exists(outputFolder) is false)
        Directory.CreateDirectory(outputFolder);
}


static void Classification(string outputFolder)
{
    Console.Write("Running Classification...\t");
    using var yolo = new Yolo(Path.Combine(MODELS_FOLDER, "yolov8s-cls.onnx"), false);

    using var image = Image.Load<Rgba32>(Path.Combine(MEDIA_FOLDER, "hummingbird.jpg"));

    List<Classification> results = yolo.RunClassification(image, 3); // Get top 5 classifications. Default = 1

    image.Draw(results);
    image.Save(Path.Combine(outputFolder, $"{nameof(Classification)}.jpg"));
    Console.Write("complete!");
    Console.WriteLine();
}


static void ObjectDetection(string outputFolder)
{
    Console.Write("Running Object Detection...\t");
    using var yolo = new Yolo(Path.Combine(MODELS_FOLDER, "yolov8s.onnx"), false);

    using var image = Image.Load<Rgba32>(Path.Combine(MEDIA_FOLDER, "street.jpg"));

    List<ObjectDetection> results = yolo.RunObjectDetection(image, 0.25);

    image.Draw(results);
    image.Save(Path.Combine(outputFolder, $"{nameof(ObjectDetection)}.jpg"));
    Console.Write("complete!");
    Console.WriteLine();
}


static void Segmentation(string outputFolder)
{
    Console.Write("Running Segmentation...\t\t");
    using var yolo = new Yolo(Path.Combine(MODELS_FOLDER, "yolov8s-seg.onnx"), false);

    using var image = Image.Load<Rgba32>(Path.Combine(MEDIA_FOLDER, "people.jpg"));

    List<Segmentation> results = yolo.RunSegmentation(image, 0.25);

    image.Draw(results, DrawSegment.PixelMaskOnly);
    image.Save(Path.Combine(outputFolder, $"{nameof(Segmentation)}.jpg"));
    Console.Write("complete!");
    Console.WriteLine();
}


static void PoseEstimation(string outputFolder)
{
    Console.Write("Running Pose Estimation...\t");
    using var yolo = new Yolo(Path.Combine(MODELS_FOLDER, "yolov8s-pose.onnx"), false);

    using var image = Image.Load<Rgba32>(Path.Combine(MEDIA_FOLDER, "crosswalk.jpg"));

    var results = yolo.RunPoseEstimation(image, 0.25);

    // Draw the connected pose-markers and colors according to a custom configuration for the model
    image.Draw(results, CustomPoseMarkerColorMap.MyCustomPoseMarkerMap);
    image.Save(Path.Combine(outputFolder, $"{nameof(YoloDotNet.Models.PoseEstimation)}.jpg"));
    Console.Write("complete!");
    Console.WriteLine();
}


static void ObjectDetectionOnVideo(string outputFolder)
{
    var videoOptions = new VideoOptions
    {
        VideoFile = Path.Combine(MEDIA_FOLDER, "walking.mp4"),
        OutputDir = outputFolder,
        //GenerateVideo = true,
        //DrawLabels = true,
        //FPS = 30,
        //Width = 640, // Resize video...
        //Height = -2, // -2 = automatically calculate dimensions to keep proportions
        //DrawConfidence = true,
        //KeepAudio = true,
        //KeepFrames = false,
        DrawSegment = DrawSegment.Default,
        PoseOptions = CustomPoseMarkerColorMap.MyCustomPoseMarkerMap
    };

    Console.WriteLine();
    Console.WriteLine("Running Object Detection on video...");
    using var yolo = new Yolo(Path.Combine(MODELS_FOLDER, "yolov8s.onnx"));

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

static void DisplayOnnxMetaDataExample()
{
    Console.WriteLine();
    Console.WriteLine("Internal ONNX properties");
    Console.WriteLine(new string('-', 58));

    using var yolo = new Yolo(Path.Combine(MODELS_FOLDER, "yolov8s.onnx"), false);

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


static void DisplayOutputFolder(string outputFolder)
    => Process.Start("explorer.exe", outputFolder);

#endregion
