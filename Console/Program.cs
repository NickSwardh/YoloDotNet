using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using YoloDotNet;
using YoloDotNet.Extensions;
using YoloDotNet.Models;

// Instantiate a new Yolo object
using var yolo = new Yolo(@"path\to\model.onnx");

// Run inference on image
RunInferenceOnImage(yolo, @"path\to\image.jpg");

// Run inference on video
RunInferenceOnVideo(yolo, new VideoOptions
{
    VideoFile = @"path\to\video.mp4",
    OutputDir = @"path\to\outputfolder"
});

// Display ONNX metadata
DisplayOnnxMetaData(yolo);

#region Methods
static void RunInferenceOnImage(Yolo yolo, string imgPath)
{
    Console.WriteLine("Running inference on {0}\r\n", imgPath);

    // Load image
    using var image = Image.Load<Rgba32>(imgPath);

    // Run inference
    var detections = yolo.RunInference(image);

    // Draw boxes
    image.DrawBoundingBoxes(detections);

    var filename = Path.GetDirectoryName(imgPath) + @"\result.jpg";

    // Save image
    image.Save(filename);

    Console.WriteLine("Done!");
    Console.WriteLine("Saved as {0}", filename);
}

static void RunInferenceOnVideo(Yolo yolo, VideoOptions videoOptions)
{
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
        yolo.RunInference(videoOptions);
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
    for (var i = 0; i < labels.Length; i++)
    {
        // Capitalize first letter in label
        var label = string.Concat(labels[i].Name[0].ToString().ToUpper(), labels[i].Name.AsSpan(1));
        Console.WriteLine($"index: {i,-8} label: {label,-20} color: {labels[i].Color}");
    }
}
#endregion