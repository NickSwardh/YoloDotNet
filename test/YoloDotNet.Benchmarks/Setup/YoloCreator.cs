// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2026 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

namespace YoloDotNet.Benchmarks.Setup
{
    public class YoloCreator
    {
        public static Yolo Create(YoloType type)
        {
            var part = type.ToString().Split('_');

            var modelKey = part[0] + "_" + part[1];
            var providerKey = part[2];

            var model = GetModel(modelKey);
            var provider = GetExecutionProvider(providerKey, model);

            return new Yolo(new YoloOptions { ExecutionProvider = provider });
        }

        #region Helper Methods
        private static string GetModel(string model) => model switch
        {
            "V5u_Cls" => SharedConfig.GetTestModelV5U(ModelType.Classification),
            "V5u_Obj" => SharedConfig.GetTestModelV5U(ModelType.ObjectDetection),
            "V5u_Obb" => SharedConfig.GetTestModelV5U(ModelType.ObbDetection),
            "V5u_Pos" => SharedConfig.GetTestModelV5U(ModelType.PoseEstimation),
            "V5u_Seg" => SharedConfig.GetTestModelV5U(ModelType.Segmentation),

            "V8_Cls" => SharedConfig.GetTestModelV8(ModelType.Classification),
            "V8_Obj" => SharedConfig.GetTestModelV8(ModelType.ObjectDetection),
            "V8_Obb" => SharedConfig.GetTestModelV8(ModelType.ObbDetection),
            "V8_Pos" => SharedConfig.GetTestModelV8(ModelType.PoseEstimation),
            "V8_Seg" => SharedConfig.GetTestModelV8(ModelType.Segmentation),

            "V9_Cls" => SharedConfig.GetTestModelV9(ModelType.Classification),
            "V9_Obj" => SharedConfig.GetTestModelV9(ModelType.ObjectDetection),
            "V9_Obb" => SharedConfig.GetTestModelV9(ModelType.ObbDetection),
            "V9_Pos" => SharedConfig.GetTestModelV9(ModelType.PoseEstimation),
            "V9_Seg" => SharedConfig.GetTestModelV9(ModelType.Segmentation),

            "V10_Cls" => SharedConfig.GetTestModelV10(ModelType.Classification),
            "V10_Obj" => SharedConfig.GetTestModelV10(ModelType.ObjectDetection),
            "V10_Obb" => SharedConfig.GetTestModelV10(ModelType.ObbDetection),
            "V10_Pos" => SharedConfig.GetTestModelV10(ModelType.PoseEstimation),
            "V10_Seg" => SharedConfig.GetTestModelV10(ModelType.Segmentation),

            "V11_Cls" => SharedConfig.GetTestModelV11(ModelType.Classification),
            "V11_Obj" => SharedConfig.GetTestModelV11(ModelType.ObjectDetection),
            "V11_Obb" => SharedConfig.GetTestModelV11(ModelType.ObbDetection),
            "V11_Pos" => SharedConfig.GetTestModelV11(ModelType.PoseEstimation),
            "V11_Seg" => SharedConfig.GetTestModelV11(ModelType.Segmentation),

            "V12_Cls" => SharedConfig.GetTestModelV12(ModelType.Classification),
            "V12_Obj" => SharedConfig.GetTestModelV12(ModelType.ObjectDetection),
            "V12_Obb" => SharedConfig.GetTestModelV12(ModelType.ObbDetection),
            "V12_Pos" => SharedConfig.GetTestModelV12(ModelType.PoseEstimation),
            "V12_Seg" => SharedConfig.GetTestModelV12(ModelType.Segmentation),

            "V26_Cls" => SharedConfig.GetTestModelV26(ModelType.Classification),
            "V26_Obj" => SharedConfig.GetTestModelV26(ModelType.ObjectDetection),
            "V26_Obb" => SharedConfig.GetTestModelV26(ModelType.ObbDetection),
            "V26_Pos" => SharedConfig.GetTestModelV26(ModelType.PoseEstimation),
            "V26_Seg" => SharedConfig.GetTestModelV26(ModelType.Segmentation),
            _ => throw new ArgumentException("Unknown model")
        };

        private static IExecutionProvider GetExecutionProvider(string provider, string model) => provider switch
        {
            "CPU" => new CudaExecutionProvider(model, -1), // Sentinel value -1 = CPU execution in CudaExecutionProvider.
            "GPU" => new CudaExecutionProvider(model),
            "TRT32" => new CudaExecutionProvider(model, 0, new TensorRt
            {
                Precision = TrtPrecision.FP32,
                EngineCachePath = TensorRtConfig.TRT_ENGINE_CACHE_PATH,
            }),
            "TRT16" => new CudaExecutionProvider(model, 0, new TensorRt
            {
                Precision = TrtPrecision.FP16,
                EngineCachePath = TensorRtConfig.TRT_ENGINE_CACHE_PATH,
            }),
            "TRT8" => new CudaExecutionProvider(model, 0, new TensorRt
            {
                Precision = TrtPrecision.INT8,
                EngineCachePath = TensorRtConfig.TRT_ENGINE_CACHE_PATH,
                Int8CalibrationCacheFile = Path.Join(SharedConfig.AbsoluteAssetsPath, "cache", $"{Path.GetFileNameWithoutExtension(model)}.cache"),
            }),
            _ => throw new ArgumentException("Unknown execution provider")
        };

        #endregion
    }
}
