using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using YoloDotNet;
using YoloDotNet.Extensions;

// Instantiate a new Yolo object
using var yolo = new Yolo(@"path\to\model.onnx");

// Load image
using var image = Image.Load<Rgba32>(@"path\to\image.jpg");

// Run inference
var detections = yolo.RunInference(image);

// Draw boxes
image.DrawBoundingBoxes(detections);

// Save image
image.Save(@"save\image.jpg");

// Optional - displaying ONNX metadata...
#region Display internal ONNX parameters
{
    Console.WriteLine();
    Console.WriteLine("Internal ONNX properties");
    Console.WriteLine(new string('-', 58));

    // Display internal ONNX properties using reflection
    foreach (var property in yolo.OnnxModel.GetType().GetProperties())
    {
        var value = property.GetValue(yolo.OnnxModel);
        Console.WriteLine($"{property.Name,-20}{value!}");
    }

    // Print internal ONNX labels with added hexadecimal colors from YoloDotNet
    var labels = yolo.OnnxModel.Labels;

    Console.WriteLine();
    Console.WriteLine($"Labels ({labels.Length}):");
    Console.WriteLine(new string('-', 58));

    for (var i = 0; i < labels.Length; i++)
    {
        // Capitalize first letter in label
        var label = string.Concat(labels[i].Name[0].ToString().ToUpper(), labels[i].Name.AsSpan(1));
        Console.WriteLine($"index: {i,-8} label: {label,-20} color: {labels[i].Color}");
    }
}
#endregion