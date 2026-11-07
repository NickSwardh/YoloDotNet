// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024-2026 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

using System;
using System.IO;
using YoloDotNet.Enums;
using YoloDotNet.Test.Common.Enums;

namespace YoloDotNet.Test.Common
{
    /// <summary>
    /// Paths to the models, media and other assets used by the tests, benchmarks and demos.
    /// All assets live in the repository's test/assets folder. The models are created by tools/download_models.py.
    /// </summary>
    public static class SharedConfig
    {
        private static string _assetsFolder;

        /// <summary>
        /// The repository's test/assets folder.
        /// </summary>
        public static string AbsoluteAssetsPath => _assetsFolder ??= FindAssetsFolder();

        /// <summary>
        /// The pose estimation keypoint definitions.
        /// </summary>
        public static string PoseKeyPointsFile => Path.Join(AbsoluteAssetsPath, "PoseKeyPoints", "pose-keypoints.json");

        /// <summary>
        /// Test models for YOLOv5u.
        /// </summary>
        public static string GetTestModelV5U(ModelType modelType) => modelType switch
        {
            ModelType.ObjectDetection => Model("yolov5su.onnx"),
            _ => throw UnknownModelType(modelType)
        };

        /// <summary>
        /// Test models for YOLOv8.
        /// </summary>
        public static string GetTestModelV8(ModelType modelType) => modelType switch
        {
            ModelType.Classification => Model("yolov8s-cls.onnx"),
            ModelType.ObjectDetection => Model("yolov8s.onnx"),
            ModelType.ObbDetection => Model("yolov8s-obb.onnx"),
            ModelType.Segmentation => Model("yolov8s-seg.onnx"),
            ModelType.PoseEstimation => Model("yolov8s-pose.onnx"),
            _ => throw UnknownModelType(modelType)
        };

        /// <summary>
        /// Test models for YOLOv9.
        /// </summary>
        public static string GetTestModelV9(ModelType modelType) => modelType switch
        {
            ModelType.ObjectDetection => Model("yolov9s.onnx"),
            _ => throw UnknownModelType(modelType)
        };

        /// <summary>
        /// Test models for YOLOv10.
        /// </summary>
        public static string GetTestModelV10(ModelType modelType) => modelType switch
        {
            ModelType.ObjectDetection => Model("yolov10s.onnx"),
            _ => throw UnknownModelType(modelType)
        };

        /// <summary>
        /// Test models for YOLO11.
        /// </summary>
        public static string GetTestModelV11(ModelType modelType) => modelType switch
        {
            ModelType.Classification => Model("yolo11s-cls.onnx"),
            ModelType.ObjectDetection => Model("yolo11s.onnx"),
            ModelType.ObbDetection => Model("yolo11s-obb.onnx"),
            ModelType.Segmentation => Model("yolo11s-seg.onnx"),
            ModelType.PoseEstimation => Model("yolo11s-pose.onnx"),
            _ => throw UnknownModelType(modelType)
        };

        /// <summary>
        /// Test models for YOLO12.
        /// </summary>
        public static string GetTestModelV12(ModelType modelType) => modelType switch
        {
            ModelType.ObjectDetection => Model("yolo12s.onnx"),
            _ => throw UnknownModelType(modelType)
        };

        /// <summary>
        /// Test models for YOLO26.
        /// </summary>
        public static string GetTestModelV26(ModelType modelType) => modelType switch
        {
            ModelType.Classification => Model("yolo26s-cls.onnx"),
            ModelType.ObjectDetection => Model("yolo26s.onnx"),
            ModelType.ObbDetection => Model("yolo26s-obb.onnx"),
            ModelType.Segmentation => Model("yolo26s-seg.onnx"),
            ModelType.PoseEstimation => Model("yolo26s-pose.onnx"),
            _ => throw UnknownModelType(modelType)
        };

        /// <summary>
        /// Test models for RT-DETR.
        /// </summary>
        public static string GetTestModelRTDETR(ModelType modelType) => modelType switch
        {
            ModelType.ObjectDetection => Model("rtdetr-l.onnx"),
            _ => throw UnknownModelType(modelType)
        };

        /// <summary>
        /// Test image for each task.
        /// </summary>
        public static string GetTestImage(ImageType imageType) => imageType switch
        {
            ImageType.Classification => Media("hummingbird.jpg"),
            ImageType.ObjectDetection => Media("street.jpg"),
            ImageType.PoseEstimation => Media("crosswalk.jpg"),
            ImageType.Segmentation => Media("people.jpg"),
            ImageType.ObbDetection => Media("island.jpg"),
            _ => throw new ArgumentException($"Unknown image type: {imageType}.", nameof(imageType))
        };

        /// <summary>
        /// Test videos.
        /// </summary>
        public static string GetTestVideo(VideoType videoType) => videoType switch
        {
            VideoType.Default => Media("walking.mp4"),
            _ => throw new ArgumentException($"Unknown video type: {videoType}.", nameof(videoType))
        };

        private static string Model(string fileName) => Path.Join(AbsoluteAssetsPath, "Models", fileName);

        private static string Media(string fileName) => Path.Join(AbsoluteAssetsPath, "Media", fileName);

        private static ArgumentException UnknownModelType(ModelType modelType)
            => new($"No test model for model type: {modelType}.", nameof(modelType));

        /// <summary>
        /// Walks up from the build output folder (e.g. test/YoloDotNet.Tests/bin/Debug/net8.0)
        /// until it finds the repository's test/assets folder.
        /// </summary>
        private static string FindAssetsFolder()
        {
            for (var folder = new DirectoryInfo(AppContext.BaseDirectory); folder is not null; folder = folder.Parent)
            {
                var assets = Path.Join(folder.FullName, "test", "assets");
                if (Directory.Exists(assets))
                    return assets;
            }

            throw new DirectoryNotFoundException(
                $"Could not find the test/assets folder above '{AppContext.BaseDirectory}'. " +
                "Run the tests, benchmarks and demos from inside the YoloDotNet repository.");
        }
    }
}
