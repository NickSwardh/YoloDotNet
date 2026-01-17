// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024-2026 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

namespace YoloDotNet.Test.Common
{
    using System;
    using System.IO;
    using System.Text.Json;

    using YoloDotNet.Enums;
    using YoloDotNet.Test.Common.Enums;

    public static class SharedConfig
    {
        private const string EnvAssetsFolder = "YOLODOTNET_ASSETS_FOLDER";

        private static string? s_assetsBaseOverride;
        private static string? s_modelsFolderOverride;
        private static string? s_imagesFolderOverride;

        private static string? s_modelsFolder;
        private static string? s_mediaFolder;

        /// <summary>
        /// Programmatically set a single base assets folder. The code will use
        /// {base}\Models and {base}\Images for models and images respectively.
        /// </summary>
        public static void SetBaseFolder(string? assetsBaseFolder)
        {
            s_assetsBaseOverride = string.IsNullOrWhiteSpace(assetsBaseFolder) ? null : assetsBaseFolder;
            s_modelsFolder = null;
            s_mediaFolder = null;
        }

        /// <summary>
        /// Backwards-compatible programmatic setter for explicit model/image folders.
        /// </summary>
        public static void SetFolders(string? modelsFolder, string? imagesFolder)
        {
            s_modelsFolderOverride = string.IsNullOrWhiteSpace(modelsFolder) ? null : modelsFolder;
            s_imagesFolderOverride = string.IsNullOrWhiteSpace(imagesFolder) ? null : imagesFolder;
            s_modelsFolder = null;
            s_mediaFolder = null;
        }

        public static string ModelsFolder => s_modelsFolder ??= ResolveModelsFolder();
        public static string MediaFolder => s_mediaFolder ??= ResolveImagesFolder();

        /// <summary>
        /// Path to the repository-level test assets folder (e.g. &lt;repo&gt;/test/assets).
        /// Falls back to ModelsFolder parent when repo root cannot be found.
        /// </summary>
        public static string AbsoluteAssetsPath => ResolveAbsoluteAssetsPath();

        private static string ResolveModelsFolder()
        {
            // explicit per-folder override
            if (!string.IsNullOrWhiteSpace(s_modelsFolderOverride))
                return s_modelsFolderOverride!;

            // if an assets base is provided, use Models subfolder
            var baseFolder = ResolveAssetsBaseFolder();
            if (!string.IsNullOrWhiteSpace(baseFolder))
                return Path.Join(baseFolder, "Models");

            // config per-folder
            var cfg = TryLoadConfigFromRepoRoot();
            if (!string.IsNullOrWhiteSpace(cfg?.ModelsFolder))
                return cfg.ModelsFolder!;

            // last resort: attempt to find repo assets folder
            var repoAssets = TryGetRepoAssetsPath();
            if (!string.IsNullOrWhiteSpace(repoAssets))
                return Path.Join(repoAssets, "Models");

            throw new InvalidOperationException($"Could not determine models folder. Provide it by calling SharedConfig.SetBaseFolder(baseFolder) or " +
                $"SharedConfig.SetFolders(modelsFolder, imagesFolder), setting the environment variable {EnvAssetsFolder}, or adding " +
                $"a 'yolodotnet.config.json' with a ModelsFolder property at the repository root.");
        }

        private static string ResolveImagesFolder()
        {
            // explicit per-folder override
            if (!string.IsNullOrWhiteSpace(s_imagesFolderOverride))
                return s_imagesFolderOverride!;

            // if an assets base is provided, use Images subfolder
            var baseFolder = ResolveAssetsBaseFolder();
            if (!string.IsNullOrWhiteSpace(baseFolder))
                return Path.Join(baseFolder, "Media");

            // config per-folder
            var cfg = TryLoadConfigFromRepoRoot();
            if (!string.IsNullOrWhiteSpace(cfg?.ImagesFolder))
                return cfg.ImagesFolder!;

            // last resort: attempt to find repo assets folder
            var repoAssets = TryGetRepoAssetsPath();
            if (!string.IsNullOrWhiteSpace(repoAssets))
                return Path.Join(repoAssets, "Media");

            throw new InvalidOperationException($"Could not determine images folder. Provide it by calling SharedConfig.SetBaseFolder(baseFolder) or" +
                $"SharedConfig.SetFolders(modelsFolder, imagesFolder), setting the environment variable {EnvAssetsFolder}, or adding " +
                $"a 'yolodotnet.config.json' with an ImagesFolder property at the repository root.");
        }

        private static string? ResolveAssetsBaseFolder()
        {
            // programmatic override
            if (!string.IsNullOrWhiteSpace(s_assetsBaseOverride))
                return s_assetsBaseOverride;

            // preferred single env var
            var env = Environment.GetEnvironmentVariable(EnvAssetsFolder);
            if (!string.IsNullOrWhiteSpace(env))
                return env!;

            // config: if config contains both ModelsFolder and ImagesFolder and they share a parent,
            // return that parent.
            var cfg = TryLoadConfigFromRepoRoot();
            if (cfg != null)
            {
                if (!string.IsNullOrWhiteSpace(cfg.ModelsFolder) && !string.IsNullOrWhiteSpace(cfg.ImagesFolder))
                {
                    var parentModels = Path.GetDirectoryName(cfg.ModelsFolder!);
                    var parentImages = Path.GetDirectoryName(cfg.ImagesFolder!);
                    if (!string.IsNullOrWhiteSpace(parentModels) && string.Equals(parentModels, parentImages, StringComparison.OrdinalIgnoreCase))
                        return parentModels;
                }
            }

            // repo layout fallback handled in TryGetRepoAssetsPath by returning test/assets
            return null;
        }

        private sealed class RepoConfig
        {
            public string? ModelsFolder { get; set; }
            public string? ImagesFolder { get; set; }
        }

        private static RepoConfig? TryLoadConfigFromRepoRoot()
        {
            try
            {
                var dir = new DirectoryInfo(AppContext.BaseDirectory);
                while (dir != null && !Directory.Exists(Path.Combine(dir.FullName, ".git")))
                {
                    dir = dir.Parent;
                }

                if (dir == null)
                    return null;

                var cfgPath = Path.Join(dir.FullName, "yolodotnet.config.json");
                if (!File.Exists(cfgPath))
                    return null;

                var json = File.ReadAllText(cfgPath);
                return JsonSerializer.Deserialize<RepoConfig>(json, new JsonSerializerOptions { PropertyNameCaseInsensitive = true });
            }
            catch
            {
                return null;
            }
        }

        private static string? TryGetRepoAssetsPath()
        {
            try
            {
                var dir = new DirectoryInfo(AppContext.BaseDirectory);
                while (dir != null && !Directory.Exists(Path.Combine(dir.FullName, ".git")))
                {
                    dir = dir.Parent;
                }

                if (dir == null)
                    return null;

                var assets = Path.Join(dir.FullName, "test", "assets");
                return Directory.Exists(assets) ? assets : null;
            }
            catch
            {
                return null;
            }
        }

        private static string ResolveAbsoluteAssetsPath()
        {
            var repo = TryGetRepoAssetsPath();
            if (!string.IsNullOrWhiteSpace(repo))
                return repo!;

            // Fallback: return parent of ModelsFolder if available
            try
            {
                var parent = Path.GetDirectoryName(ModelsFolder);
                return parent ?? ModelsFolder;
            }
            catch
            {
                return ModelsFolder;
            }
        }

        /// <summary>
        /// Test models for Yolo V5U
        /// </summary>
        /// <param name="modelType"></param>
        /// <returns></returns>
        /// <exception cref="ArgumentException"></exception>
        public static string GetTestModelV5U(ModelType modelType) => modelType switch
        {
            ModelType.ObjectDetection => Path.Join(ModelsFolder, "yolov5su.onnx"),
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
            ModelType.Classification => Path.Join(ModelsFolder, "yolov8s-cls.onnx"),
            ModelType.ObjectDetection => Path.Join(ModelsFolder, "yolov8s.onnx"),
            ModelType.ObbDetection => Path.Join(ModelsFolder, "yolov8s-obb.onnx"),
            ModelType.Segmentation => Path.Join(ModelsFolder, "yolov8s-seg.onnx"),
            ModelType.PoseEstimation => Path.Join(ModelsFolder, "yolov8s-pose.onnx"),
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
            ModelType.ObjectDetection => Path.Join(ModelsFolder, "yolov9s.onnx"),
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
            ModelType.ObjectDetection => Path.Join(ModelsFolder, "yolov10s.onnx"),
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
            ModelType.Classification => Path.Join(ModelsFolder, "yolo11s-cls.onnx"),
            ModelType.ObjectDetection => Path.Join(ModelsFolder, "yolo11s.onnx"),
            ModelType.ObbDetection => Path.Join(ModelsFolder, "yolo11s-obb.onnx"),
            ModelType.Segmentation => Path.Join(ModelsFolder, "yolo11s-seg.onnx"),
            ModelType.PoseEstimation => Path.Join(ModelsFolder, "yolo11s-pose.onnx"),
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
            ModelType.ObjectDetection => Path.Join(ModelsFolder, "yolov12s.onnx"),
            _ => throw new ArgumentException("Unknown modeltype.")
        };

        /// <summary>
        /// Test models for Yolo V26
        /// </summary>
        /// <param name="modelType"></param>
        /// <returns></returns>
        /// <exception cref="ArgumentException"></exception>
        public static string GetTestModelV26(ModelType modelType) => modelType switch
        {
            ModelType.Classification => Path.Join(ModelsFolder, "yolo26s-cls.onnx"),
            ModelType.ObjectDetection => Path.Join(ModelsFolder, "yolo26s.onnx"),
            ModelType.ObbDetection => Path.Join(ModelsFolder, "yolo26s-obb.onnx"),
            ModelType.Segmentation => Path.Join(ModelsFolder, "yolo26s-seg.onnx"),
            ModelType.PoseEstimation => Path.Join(ModelsFolder, "yolo26s-pose.onnx"),
            _ => throw new ArgumentException("Unknown modeltype.")
        };

        /// <summary>
        /// Test models for Yolo RT-DETR
        /// </summary>
        /// <param name="modelType"></param>
        /// <returns></returns>
        /// <exception cref="ArgumentException"></exception>
        public static string GetTestModelRTDETR(ModelType modelType) => modelType switch
        {
            ModelType.ObjectDetection => Path.Join(ModelsFolder, "rtdetr-l.onnx"),
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
            ImageType.Classification => Path.Join(MediaFolder, "classification.jpg"),
            ImageType.ObjectDetection => Path.Join(MediaFolder, "object_detection.jpg"),
            ImageType.PoseEstimation => Path.Join(MediaFolder, "pose_estimation.jpg"),
            ImageType.Segmentation => Path.Join(MediaFolder, "segmentation.jpg"),
            ImageType.ObbDetection => Path.Join(MediaFolder, "obb_detection.jpg"),
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
            VideoType.Default => Path.Join(MediaFolder, "video.mp4"),
            _ => throw new ArgumentException("Unknown VideoType.")
        };

        public static string GetTestImage(string imageName) => Path.Join(MediaFolder, imageName);
    }
}
