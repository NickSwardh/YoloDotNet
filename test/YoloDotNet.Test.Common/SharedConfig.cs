// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2024-2025 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

namespace YoloDotNet.Test.Common
{
    using System;
    using System.IO;

    using YoloDotNet.Enums;
    using YoloDotNet.Test.Common.Enums;

    public static class SharedConfig
    {
        private const string ASSETS_FOLDER = @".\assets";
        private const string BASE_MODELS = ASSETS_FOLDER + @"\models";
        private const string BASE_MEDIA = ASSETS_FOLDER + @"\media";
        private static readonly string ABSOLUTE_PATH = GetAbsolutePath();

        // Define an absolute path to the assets folder.
        public static readonly string AbsoluteAssetsPath = GetAbsoluteAssetsPath();

        /// <summary>
        /// Test models for Yolo V5U
        /// </summary>
        /// <param name="modelType"></param>
        /// <returns></returns>
        /// <exception cref="ArgumentException"></exception>
        public static string GetTestModelV5U(ModelType modelType) => modelType switch
        {
            ModelType.ObjectDetection => Path.Combine(BASE_MODELS, "yolov5su.onnx"),
            _ => throw new ArgumentException("Unknown modeltype.")
        };

        /// <summary>
        /// Test models for Yolo V8
        /// </summary>
        /// <param name="modelType"></param>
        /// <returns></returns>
        /// <exception cref="ArgumentException"></exception>
        public static string GetTestModelV8(ModelType modelType) => modelType switch
        {
            ModelType.Classification => Path.Combine(BASE_MODELS, "yolov8s-cls.onnx"),
            ModelType.ObjectDetection => Path.Combine(BASE_MODELS, "yolov8s.onnx"),
            ModelType.ObbDetection => Path.Combine(BASE_MODELS, "yolov8s-obb.onnx"),
            ModelType.Segmentation => Path.Combine(BASE_MODELS, "yolov8s-seg.onnx"),
            ModelType.PoseEstimation => Path.Combine(BASE_MODELS, "yolov8s-pose.onnx"),
            _ => throw new ArgumentException("Unknown modeltype.")
        };

        /// <summary>
        /// Test models for Yolo V9
        /// </summary>
        /// <param name="modelType"></param>
        /// <returns></returns>
        /// <exception cref="ArgumentException"></exception>
        public static string GetTestModelV9(ModelType modelType) => modelType switch
        {
            ModelType.ObjectDetection => Path.Combine(BASE_MODELS, "yolov9s.onnx"),
            _ => throw new ArgumentException("Unknown modeltype.")
        };

        /// <summary>
        /// Test models for Yolo V10
        /// </summary>
        /// <param name="modelType"></param>
        /// <returns></returns>
        /// <exception cref="ArgumentException"></exception>
        public static string GetTestModelV10(ModelType modelType) => modelType switch
        {
            ModelType.ObjectDetection => Path.Combine(BASE_MODELS, "yolov10s.onnx"),
            _ => throw new ArgumentException("Unknown modeltype.")
        };

        /// <summary>
        /// Test models for Yolo V11
        /// </summary>
        /// <param name="modelType"></param>
        /// <returns></returns>
        /// <exception cref="ArgumentException"></exception>
        public static string GetTestModelV11(ModelType modelType) => modelType switch
        {
            ModelType.Classification => Path.Combine(BASE_MODELS, "yolov11s-cls.onnx"),
            ModelType.ObjectDetection => Path.Combine(BASE_MODELS, "yolov11s.onnx"),
            ModelType.ObbDetection => Path.Combine(BASE_MODELS, "yolov11s-obb.onnx"),
            ModelType.Segmentation => Path.Combine(BASE_MODELS, "yolov11s-seg.onnx"),
            ModelType.PoseEstimation => Path.Combine(BASE_MODELS, "yolov11s-pose.onnx"),
            _ => throw new ArgumentException("Unknown modeltype.")
        };

        /// <summary>
        /// Test models for Yolo V12
        /// </summary>
        /// <param name="modelType"></param>
        /// <returns></returns>
        /// <exception cref="ArgumentException"></exception>
        public static string GetTestModelV12(ModelType modelType) => modelType switch
        {
            ModelType.ObjectDetection => Path.Combine(BASE_MODELS, "yolov12s.onnx"),
            _ => throw new ArgumentException("Unknown modeltype.")
        };

        /// <summary>
        /// Test images
        /// </summary>
        /// <param name="imageType"></param>
        /// <returns></returns>
        /// <exception cref="ArgumentException"></exception>
        public static string GetTestImage(ImageType imageType) => imageType switch
        {
            ImageType.Hummingbird => Path.Combine(BASE_MEDIA, "hummingbird.jpg"),
            ImageType.Street => Path.Combine(BASE_MEDIA, "street.jpg"),
            ImageType.People => Path.Combine(BASE_MEDIA, "people.jpg"),
            ImageType.Crosswalk => Path.Combine(BASE_MEDIA, "crosswalk.jpg"),
            ImageType.Island => Path.Combine(BASE_MEDIA, "island.jpg"),
            _ => throw new ArgumentException("Unknown ImageType.")
        };

        /// <summary>
        /// Test Videos
        /// </summary>
        /// <param name="videoType"></param>
        /// <returns></returns>
        /// <exception cref="ArgumentException"></exception>
        public static string GetTestVideo(VideoType videoType) => videoType switch
        {
            VideoType.PeopleWalking => Path.Join(AbsoluteAssetsPath, "media", "walking.mp4"),
            _ => throw new ArgumentException("Unknown VideoType.")
        };

        /// <summary>
        /// Get absolute path to the assets folder.
        /// </summary>
        /// <returns></returns>
        /// <exception cref="InvalidOperationException"></exception>
        private static string GetAbsoluteAssetsPath()
        {
            // Get executable base dir
            var dir = new DirectoryInfo(AppContext.BaseDirectory);

            // Traverse back untill we find .git file marker
            while (dir != null && !Directory.Exists(Path.Combine(dir.FullName, ".git")))
        {
                dir = dir.Parent;
            }

            // If no .git marker was found, throw an exception.
            if (dir == null)
                throw new InvalidOperationException("Could not find Git repository root.");

            // Add the assets folder
            return Path.Join(dir.FullName, "test", "assets");
        }

        public static string GetTestImage(string imageName) => Path.Join(BaseMedia, imageName);
    }
}
