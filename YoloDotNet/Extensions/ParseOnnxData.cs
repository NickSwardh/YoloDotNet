// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023-2026 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

namespace YoloDotNet.Extensions
{
    public static class ParseOnnxData
    {
        /// <summary>
        /// Creates the model information from the values an execution provider reads from its ONNX Runtime session.
        /// </summary>
        /// <param name="inputs">Shape of each model input, by tensor name.</param>
        /// <param name="outputs">Shape of each model output, by tensor name.</param>
        /// <param name="metadata">The model's custom metadata (Ultralytics exports store the task, description and class names here).</param>
        /// <param name="modelDataType">Element type of the model's input tensor.</param>
        public static OnnxModel CreateOnnxModel(Dictionary<string, long[]> inputs, Dictionary<string, int[]> outputs,
            Dictionary<string, string> metadata, ModelDataType modelDataType)
            => new()
            {
                InputShapes = inputs,
                OutputShapes = outputs,
                CustomMetaData = metadata,
                ModelDataType = modelDataType,
                ModelType = GetModelType(metadata["task"]),
                ModelVersion = GetModelVersion(metadata["description"]),
                Labels = MapLabelsAndColors(metadata["names"]),
                InputShapeSize = CalculateTotalInputShapeSize(inputs.First().Value)
            };

        /// <summary>
        /// Removed: execution providers now read the model information themselves and call <see cref="CreateOnnxModel"/>.
        /// Kept only so that execution providers built for an earlier YoloDotNet version fail with a clear message.
        /// </summary>
        [EditorBrowsable(EditorBrowsableState.Never)]
        [Obsolete("Execution providers call CreateOnnxModel instead.", error: true)]
        public static OnnxModel ParseOnnx(this object onnxData)
            => throw new YoloDotNetModelException("This execution provider is not compatible with this version of YoloDotNet. " +
                "Update your YoloDotNet.ExecutionProvider package to the latest version.");

        #region Helper Methods
        /// <summary>
        /// Maps ONNX labels to corresponding colors for visualization.
        /// </summary>
        private static LabelModel[] MapLabelsAndColors(string onnxLabelData)
        {
            // Labels to Dictionary
            var onnxLabels = onnxLabelData
                .Trim('{', '}')
                .Replace("'", "")
                .Split(", ")
                .Select(x => x.Split(": "))
                .ToDictionary(x => int.Parse(x[0]), x => x[1]);

            return [.. onnxLabels!.Select((label, index) => new LabelModel
            {
                Index = index,
                Name = label.Value,
            })];
        }

        /// <summary>
        /// Calculates the total number of elements in a tensor based on its shape dimensions.
        /// </summary>
        private static int CalculateTotalInputShapeSize(long[] shape)
        {
            if (shape.Length == 0)
                return 0;

            int shapeSize = 1;

            foreach (var dimension in shape)
            {
                if (dimension <= 0)
                    throw new YoloDotNetException(
                        $"All shape dimensions must be positive. Found invalid value: {dimension}",
                        nameof(shape));

                shapeSize *= (int)dimension;
            }

            return shapeSize;
        }

        private static ModelType GetModelType(string modelType) => modelType switch
        {
            "classify" => ModelType.Classification,
            "detect" => ModelType.ObjectDetection,
            "obb" => ModelType.ObbDetection,
            "pose" => ModelType.PoseEstimation,
            "segment" => ModelType.Segmentation,
            _ => throw new YoloDotNetModelException("Unsupported task")
        };

        /// <summary>
        /// Get ONNX model version
        /// </summary>
        private static ModelVersion GetModelVersion(string modelDescription) => modelDescription.ToLower() switch
        {
            // YOLOv5
            var version when version.StartsWith("ultralytics yolov5") => ModelVersion.V5U,

            // YOLOv8
            var version when version.StartsWith("ultralytics yolov8") => ModelVersion.V8,
            var version when version.StartsWith("ultralytics yoloe-v8") => ModelVersion.V8E,

            // YOLOv9
            var version when version.StartsWith("ultralytics yolov9") => ModelVersion.V9,

            // YOLOv10
            var version when version.StartsWith("ultralytics yolov10") => ModelVersion.V10,

            // YOLOv11
            var version when version.StartsWith("ultralytics yolo11") => ModelVersion.V11,      // Note the missing v in Yolo11
            var version when version.StartsWith("ultralytics yoloe-11") => ModelVersion.V11E,   // Note the missing v in Yoloe-11

            // YOLOv12
            var version when version.StartsWith("ultralytics yolov12") => ModelVersion.V12,

            // YOLOv26
            var version when version.StartsWith("ultralytics yolo26") => ModelVersion.V26,

            // YOLO WorldV2
            var version when version.Contains("worldv2") => ModelVersion.V11,

            // RT-DETR
            var version when version.StartsWith("ultralytics rt-detr") => ModelVersion.RTDETR,

            // Fallback: if version metadata is missing, treat the model as YOLOv8.
            var version when version.StartsWith("ultralytics") && !version.Contains("yolo") => ModelVersion.V8,

            _ => throw new YoloDotNetModelException("Onnx model not supported!")
        };
        #endregion
    }
}
