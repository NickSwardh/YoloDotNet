
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using System;
using System.IO;
using YoloDotNet;
using YoloDotNet.Extensions;
using YoloDotNet.Models;

class Program
{
    static void Main()
    {
        // Specify the folder containing the images
        string folderPath = @"folder_path";

        // Load the YOLO model
        using var yolo = new Yolo("your_model.onnx", false);

        // Get all image files in the folder
        string[] imageFiles = Directory.GetFiles(folderPath, "*.JPG");

        // Run inference on each image
        foreach (var imgPath in imageFiles)
        {
            RunInferenceAndSaveResult(yolo, imgPath);
        }

        // Display ONNX metadata
        DisplayOnnxMetaData(yolo);
    }

    static void RunInferenceAndSaveResult(Yolo yolo, string imgPath)
    {
        Console.WriteLine("Running inference on {0}\r\n", imgPath);

        // Load image
        using var image = Image.Load<Rgba32>(imgPath);

        // Run inference
        var detections = yolo.RunInference(image);

        // Draw boxes
        image.DrawBoundingBoxes(detections);

        // Save the result with the same file name
        var resultFileName = Path.Combine(Path.GetDirectoryName(imgPath), Path.GetFileNameWithoutExtension(imgPath) + "_result.jpg");

        // Save image
        image.Save(resultFileName);

        Console.WriteLine("Done!");
        Console.WriteLine("Saved as {0}", resultFileName);
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
}
