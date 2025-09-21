// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2023-2025 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

namespace YoloDotNet.Extensions
{
    internal static class OnnxPropertiesExtension
    {
        /// <summary>
        /// Extracts and retrieves metadata properties from the ONNX model.
        /// </summary>
        /// <param name="session">The ONNX model inference session.</param>
        /// <returns>An instance of OnnxModel containing extracted metadata properties.</returns>
        public static OnnxModel GetOnnxProperties(this OnnxDataRecord onnxData)
        {
            var metaData = onnxData.MetaData;

            var modelType = GetModelType(metaData[NameOf(MetaData.Task)]);
            var modelVersion = GetModelVersion(metaData[NameOf(MetaData.Description)]);

            var inputName = onnxData.InputName;
            var outputNames = onnxData.OutputNames.ToList();

            return new OnnxModel()
            {
                ModelType = modelType,
                ModelVersion = modelVersion,
                InputName = inputName,
                OutputNames = outputNames,
                CustomMetaData = metaData,
                Input = GetModelInputShape(onnxData.InputShape),
                Outputs = GetOutputShapes(onnxData.OutputShapes, modelType),
                Labels = MapLabelsAndColors(metaData[NameOf(MetaData.Names)], modelType),

                // Input shape in NCHW order: [N (batch), Channels, Height, Width]
                InputShape =
                [
                    onnxData.InputShape[0], // Batch (nr of images the model can process)
                    onnxData.InputShape[1], // Color channels
                    onnxData.InputShape[2], // Required image height
                    onnxData.InputShape[3], // Required image width
                ]
            };
        }

        #region Helper methods

        /// <summary>
        /// Maps ONNX labels to corresponding colors for visualization.
        /// </summary>
        private static LabelModel[] MapLabelsAndColors(string onnxLabelData, ModelType modelType)
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

        private static string NameOf(dynamic metadata)
            => metadata.ToString().ToLower();

        /// <summary>
        /// Retrieves the input shape of a model based on metadata
        /// </summary>
        private static Input GetModelInputShape(int[] inputShapes)
        {
            // Check for any dynamic dimension (-1 means dynamic in ONNX)
            if (inputShapes.Any(d => d == -1))
                throw new YoloDotNetModelException("Dynamic ONNX models are not supported.");

            return Input.Shape(inputShapes);
        }

        /// <summary>
        /// Retrieves the output shapes of a model based on metadata and model type.
        /// </summary>
        private static List<Output> GetOutputShapes(int[][] outputDimensions, ModelType modelType)
        {
            var (output0, output1) = modelType switch
            {
                ModelType.Classification => (Output.Classification(outputDimensions[0]), Output.Empty()),
                ModelType.ObjectDetection => (Output.Detection(outputDimensions[0]), Output.Empty()),
                ModelType.ObbDetection => (Output.Detection(outputDimensions[0]), Output.Empty()),
                ModelType.Segmentation => (Output.Detection(outputDimensions[0]), Output.Segmentation(outputDimensions[1])),
                ModelType.PoseEstimation => (Output.Detection(outputDimensions[0]), Output.Empty()),
                _ => throw new YoloDotNetModelException($"Error getting output shapes. Unknown ONNX model type.", nameof(modelType))
            };

            return [output0, output1];
        }

        /// <summary>
        /// Get ONNX model type
        /// </summary>
        private static ModelType GetModelType(string modelType) => (ModelType)(typeof(ModelType)
            .GetFields()
            .FirstOrDefault((x => Attribute.GetCustomAttribute(x, typeof(EnumMemberAttribute)) is EnumMemberAttribute attribute && attribute.Value == modelType)))!
            .GetValue(null)!;

        /// <summary>
        /// Get ONNX model version
        /// </summary>
        private static ModelVersion GetModelVersion(string modelDescription) => modelDescription.ToLower() switch
        {
            // Fallback: if version metadata is missing, treat the model as YOLOv8.
            var version when version.Contains("yolo") is false => ModelVersion.V8,

            var version when version.StartsWith("ultralytics yolov5") => ModelVersion.V5U,
            var version when version.StartsWith("ultralytics yolov8") => ModelVersion.V8,
            var version when version.StartsWith("ultralytics yoloe-v8") => ModelVersion.V8E,
            var version when version.StartsWith("ultralytics yolov9") => ModelVersion.V9,
            var version when version.StartsWith("ultralytics yolov10") => ModelVersion.V10,
            var version when version.StartsWith("ultralytics yolo11") => ModelVersion.V11,      // Note the missing v in Yolo11
            var version when version.StartsWith("ultralytics yoloe-11") => ModelVersion.V11E,   // Note the missing v in Yoloe-11
            var version when version.StartsWith("ultralytics yolov12") => ModelVersion.V12,
            var version when version.Contains("worldv2") => ModelVersion.V11,
            _ => throw new YoloDotNetModelException("Onnx model not supported!")
        };

        #endregion
    }
}