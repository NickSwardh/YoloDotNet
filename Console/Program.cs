using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using System.Globalization;
using YoloDotNet;
using YoloDotNet.Enums;
using YoloDotNet.Extensions;
using YoloDotNet.Models;

// Instantiate a new Yolo object
using var yolo = new Yolo(@"path\to\model.onnx");

// Display ONNX metadata
DisplayOnnxMetaData(yolo);

// Run inference on image
InferenceOnImage(yolo, @"path\to\image.jpg");

// Run inference on video
InferenceOnVideo(yolo, new VideoOptions
{
    VideoFile = @"path\to\video.mp4",
    OutputDir = @"path\to\outputfolder",
    //FPS = 30,
    //Width = 1280,
    //Height = 720,
    //DrawConfidence = true,
    //KeepAudio = true
});

#region Methods
static void InferenceOnImage(Yolo yolo, string imgPath)
{
    if (File.Exists(imgPath) is false)
    {
        Console.WriteLine($"{imgPath} not found.");
        return;
    }

    Console.WriteLine($"Running inference on {imgPath}\r\n");

    // Load image as RGBA
    using var image = Image.Load<Rgba32>(imgPath);

    // Run Classification or ObjectDetection depending on ONNX-modeltype
    if (yolo.OnnxModel.ModelType == ModelType.Classification)
    {
        var detections = yolo.RunClassification(image, 5); // default classes = 1

        Console.WriteLine("Classification result:");
        foreach (var label in detections)
        {
            Console.WriteLine($"{label.Confidence.ToString("0.00", CultureInfo.InvariantCulture)}% {label.Label}");
        }

        Console.WriteLine();
        image.DrawClassificationLabels(detections);
    }
    else if (yolo.OnnxModel.ModelType == ModelType.ObjectDetection)
    {
        var detections = yolo.RunObjectDetection(image, 0.25); // default threshold = 0.25
        image.DrawBoundingBoxes(detections);
    }

    // Save image
    var filename = Path.GetDirectoryName(imgPath) + @"\result.jpg";
    image.Save(filename);

    Console.WriteLine("Done!");
    Console.WriteLine("Saved as {0}", filename);
}

static void InferenceOnVideo(Yolo yolo, VideoOptions videoOptions)
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
        if (yolo.OnnxModel.ModelType == ModelType.Classification)
            yolo.RunClassification(videoOptions, 5); // default classes = 1
        else if (yolo.OnnxModel.ModelType == ModelType.ObjectDetection)
            yolo.RunObjectDetection(videoOptions, 0.3); // default threshold = 0.25
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
    }

    var labels = yolo.OnnxModel.Labels;

    Console.WriteLine();
    Console.WriteLine($"Labels ({labels.Length}):");
    Console.WriteLine(new string('-', 58));

    // Display labels and its corresponding color
    for (var i = 0; i < 3; i++)
    {
        // Capitalize first letter in label
        var label = string.Concat(labels[i].Name[0].ToString().ToUpper(), labels[i].Name.AsSpan(1));
        Console.WriteLine($"index: {i,-8} label: {label,-20} color: {labels[i].Color}");
    }

    Console.WriteLine("...");
    Console.WriteLine();
}
#endregion