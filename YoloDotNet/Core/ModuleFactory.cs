// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023-2025 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

namespace YoloDotNet.Core
{
    /// <summary>
    /// Factory class to create YOLO detection modules based on model version and type.
    /// </summary>
    internal class ModuleFactory
    {
        // Dictionary mapping model versions and types to their respective module creation functions.
        private static readonly Dictionary<ModelVersion, Dictionary<ModelType, Func<YoloCore, IModule>>> _versionModuleMap =
        new()
        {
            {
                ModelVersion.V5U, new Dictionary<ModelType, Func<YoloCore, IModule>>
                {
                    { ModelType.Classification, core =>throw new NotImplementedException() },
                    { ModelType.ObjectDetection, core => new ObjectDetectionModuleV5U(core) },
                    { ModelType.ObbDetection, core => throw new NotImplementedException() },
                    { ModelType.Segmentation, core => throw new NotImplementedException() },
                    { ModelType.PoseEstimation, core => throw new NotImplementedException() }
                }
            },
            {
                ModelVersion.V8, new Dictionary<ModelType, Func<YoloCore, IModule>>
                {
                    { ModelType.Classification, core => new ClassificationModuleV8(core) },
                    { ModelType.ObjectDetection, core => new ObjectDetectionModuleV8(core) },
                    { ModelType.ObbDetection, core => new OBBDetectionModuleV8(core) },
                    { ModelType.Segmentation, core => new SegmentationModuleV8(core) },
                    { ModelType.PoseEstimation, core => new PoseEstimationModuleV8(core) }
                }
            },
            {
                ModelVersion.V8E, new Dictionary<ModelType, Func<YoloCore, IModule>>
                {
                    { ModelType.Classification, core => throw new NotImplementedException() },
                    { ModelType.ObjectDetection, core => throw new NotImplementedException() },
                    { ModelType.ObbDetection, core => throw new NotImplementedException() },
                    { ModelType.Segmentation, core => new SegmentationModuleV8E(core) },
                    { ModelType.PoseEstimation, core => throw new NotImplementedException() }
                }
            },
            {
                ModelVersion.V9, new Dictionary<ModelType, Func<YoloCore, IModule>>
                {
                    { ModelType.Classification, core => throw new NotImplementedException() },
                    { ModelType.ObjectDetection, core => new ObjectDetectionModuleV9(core) },
                    { ModelType.ObbDetection, core => throw new NotImplementedException() },
                    { ModelType.Segmentation, core => throw new NotImplementedException() },
                    { ModelType.PoseEstimation, core => throw new NotImplementedException() }
                }
            },
            {
                ModelVersion.V10, new Dictionary<ModelType, Func<YoloCore, IModule>>
                {
                    { ModelType.Classification, core => throw new NotImplementedException() },
                    { ModelType.ObjectDetection, core => new ObjectDetectionModuleV10(core) },
                    { ModelType.ObbDetection, core => throw new NotImplementedException() },
                    { ModelType.Segmentation, core => throw new NotImplementedException() },
                    { ModelType.PoseEstimation, core => throw new NotImplementedException() }
                }
            },
            {
                ModelVersion.V11, new Dictionary<ModelType, Func<YoloCore, IModule>>
                {
                    { ModelType.Classification, core => new ClassificationModuleV11(core) },
                    { ModelType.ObjectDetection, core => new ObjectDetectionModuleV11(core) },
                    { ModelType.ObbDetection, core => new OBBDetectionModuleV11(core) },
                    { ModelType.Segmentation, core => new SegmentationModuleV11(core) },
                    { ModelType.PoseEstimation, core => new PoseEstimationModuleV11(core) }
                }
            },
            {
                ModelVersion.V11E, new Dictionary<ModelType, Func<YoloCore, IModule>>
                {
                    { ModelType.Classification, core => throw new NotImplementedException() },
                    { ModelType.ObjectDetection, core => throw new NotImplementedException() },
                    { ModelType.ObbDetection, core => throw new NotImplementedException() },
                    { ModelType.Segmentation, core => new SegmentationModuleV11E(core) },
                    { ModelType.PoseEstimation, core => throw new NotImplementedException() }
                }
            },
            {
                ModelVersion.V12, new Dictionary<ModelType, Func<YoloCore, IModule>>
                {
                    { ModelType.Classification, core => new ClassificationModuleV12(core) },
                    { ModelType.ObjectDetection, core => new ObjectDetectionModuleV12(core) },
                    { ModelType.ObbDetection, core => new OBBDetectionModuleV12(core) },
                    { ModelType.Segmentation, core => new SegmentationModuleV12(core) },
                    { ModelType.PoseEstimation, core => new PoseEstimationModuleV12(core) }
                }
            },
            {
                ModelVersion.WORLDV2, new Dictionary<ModelType, Func<YoloCore, IModule>>
                {
                    { ModelType.Classification, core => throw new NotImplementedException() },
                    { ModelType.ObjectDetection, core => new ObjectDetectionModuleWorldV2(core) },
                    { ModelType.ObbDetection, core =>  throw new NotImplementedException() },
                    { ModelType.Segmentation, core =>  throw new NotImplementedException() },
                    { ModelType.PoseEstimation, core =>  throw new NotImplementedException() }
                }
            }
        };

        /// <summary>
        /// Creates a detection module based on the specified YOLO options.
        /// </summary>
        /// <param name="options">The options for creating the YOLO detection module.</param>
        /// <returns>An instance of the appropriate detection module.</returns>
        /// <exception cref="YoloDotNetModelException">Thrown if the model version or type is unsupported.</exception>
        public static IModule CreateModule(YoloOptions options)
        {
            var yoloCore = InitializeYoloCore(options);

            // Get model version and type
            var modelVersion = yoloCore.OnnxModel.ModelVersion;
            var modelType = yoloCore.ModelType;

            // Get dictionary from module map based on model version
            var versionSelected = _versionModuleMap.TryGetValue(modelVersion, out var moduleMap);
            var moduleSelected = moduleMap!.TryGetValue(modelType, out var createModule);

            if (versionSelected && moduleSelected)
                return createModule!(yoloCore);

            throw new YoloDotNetModelException($"Unsupported detection type {modelType} or model version {modelVersion}.");
        }

        /// <summary>
        /// Initializes the YoloCore based on the specified options.
        /// </summary>
        /// <param name="options">The options for initializing the Yolo model.</param>
        /// <returns>An initialized YoloCore instance.</returns>
        private static YoloCore InitializeYoloCore(YoloOptions options)
        {
            //var yoloCore = new YoloCore(options.OnnxModel, options.Cuda, options.PrimeGpu, options.GpuId);
            var yoloCore = new YoloCore(options);
            yoloCore.InitializeYolo();
            return yoloCore;
        }
    }
}
