using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using YoloDotNet;
using YoloDotNet.Enums;
using YoloDotNet.Extensions;
using YoloDotNet.Models;

// Instantiate a new Yolo object
using var yolo = new Yolo(@"path\to\model.onnx");
Console.WriteLine($"Loaded ONNX-model is of type: {yolo.OnnxModel.ModelType}");

// Display ONNX metadata
DisplayOnnxMetaData(yolo);

// Image
var inputImage = @"path\to\image.jpg";
var saveFolder = @"path\to\output\folder";

// Run Yolo on image
ProcessImage(yolo, inputImage, saveFolder);

// Run Yolo on video
ProcessVideo(yolo, new VideoOptions
{
    VideoFile = @"path\to\video.mp4",
    OutputDir = @"path\to\output\folder",
    //GenerateVideo = true,
    //DrawLabels = true,
    //FPS = 30,
    //Width = 1280,
    //Height = 720,
    //DrawConfidence = true,
    //KeepAudio = true,
    //KeepFrames = false
});

#region Methods
static void ProcessImage(Yolo yolo, string imgPath, string saveFolder)
{
    if (File.Exists(imgPath) is false)
    {
        Console.WriteLine($"{imgPath} not found.");
        return;
    }

    Console.WriteLine($"Running inference on {imgPath}\r\n");

    // Load image as RGBA
    using var image = Image.Load<Rgba32>(imgPath);

    switch (yolo.OnnxModel.ModelType)
    {
        case ModelType.Classification:
            List<Classification> classifications = yolo.RunClassification(image, 5); // Get top 5 classifications. Default = 1
            image.Draw(classifications);
            break;
        case ModelType.ObjectDetection:
            List<ObjectDetection> detections = yolo.RunObjectDetection(image, 0.25);
            image.Draw(detections);
            break;
        case ModelType.Segmentation:
            List<Segmentation> segments = yolo.RunSegmentation(image, 0.25);
            image.Draw(segments);
            break;
    }

    // Save image
    var filename = Path.Combine(saveFolder) + @"\result.jpg";
    image.Save(filename);

    Console.WriteLine("Done!");
    Console.WriteLine("Saved as {0}", filename);
}

static void ProcessVideo(Yolo yolo, VideoOptions videoOptions)
{
    if (File.Exists(videoOptions.VideoFile) is false)
    {
        Console.WriteLine($"{videoOptions.VideoFile} not found.");
        return;
    }

    int currentLineCursor = 0;
    Console.WriteLine();
    Console.WriteLine(@"Running inference on {0}", videoOptions.VideoFile);
    Console.CursorVisible = false;

    // Listen to events...
    yolo.VideoStatusEvent += OnStatusChangeEvent;
    yolo.VideoProgressEvent += OnVideoProgressEvent;
    yolo.VideoCompleteEvent += OnCompleteEvent;

    // Run inference on video
    try
    {
        switch (yolo.OnnxModel.ModelType)
        {
            case ModelType.Classification:
                Dictionary<int, List<Classification>> classifications = yolo.RunClassification(videoOptions, 5);
                // classifications contains all frames (int) and a list of classes for each frame
                break;
            case ModelType.ObjectDetection:
                Dictionary<int, List<ObjectDetection>> detections = yolo.RunObjectDetection(videoOptions, 0.25);
                // detections contains all frames (int) and a list of detected object for each frame
                break;
            case ModelType.Segmentation:
                Dictionary<int, List<Segmentation>> segmentations = yolo.RunSegmentation(videoOptions, 0.25);
                // detections contains all frames (int) and a list of segmentationns for each frame
                break;
        }
    }
    catch (Exception ex)
    {
        Console.SetCursorPosition(20, currentLineCursor);
        Console.Write("{0}", ex.Message);
    }

    static void OnCompleteEvent(object? sender, EventArgs e)
    {
        Console.WriteLine();
        Console.WriteLine("All done!");
    }

    void OnStatusChangeEvent(object? sender, EventArgs e)
    {
        Console.WriteLine();
        Console.Write((string)sender!);
        currentLineCursor = Console.CursorTop;
    }

    void OnVideoProgressEvent(object? sender, EventArgs e)
    {
        Console.SetCursorPosition(20, currentLineCursor);
        Console.Write(new string(' ', 4));
        Console.SetCursorPosition(20, currentLineCursor);
        Console.Write("{0}%", (int)sender!);
    }

    Console.WriteLine();
    Console.CursorVisible = true;
}

static void DisplayOnnxMetaData(Yolo yolo)
{
    Console.WriteLine();
    Console.WriteLine("Internal ONNX properties");
    Console.WriteLine(new string('-', 58));

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
#endregion