// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023-2026 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

namespace YoloDotNet.Extensions
{
    public static class ParseOnnxData
    {

        public static OnnxModel ParseOnnx(this object onnxData)
        {
            var inputs = GetShape<long>(onnxData, "InputMetadata");
            var outputs = GetShape<int>(onnxData, "OutputMetadata");

            var metadata = GetMetadata(onnxData);
            var model = new OnnxModel
            {
                InputShapes = inputs,
                OutputShapes = outputs,
                CustomMetaData = metadata,
                ModelDataType = GetModelDataType(onnxData),
                ModelType = GetModelType(metadata["task"]),
                ModelVersion = GetModelVersion(metadata["description"]),
                Labels = MapLabelsAndColors(metadata["names"]),
                InputShapeSize = CalculateTotalInputShapeSize(inputs.First().Value)
            };

            return model;
        }

        #region Helper Methods
        /// <summary>
        /// Retrieves the shape dimensions for each input or output from an ONNX model property and returns them as a dictionary.
        /// </summary>
        private static Dictionary<string, T[]> GetShape<T>(object onnxData, string propertyName)
        {
            try
            {
                var metadataProperty = onnxData.GetType().GetProperty(propertyName)
                    ?? throw new YoloDotNetModelException($"{propertyName} property not found on ONNX model.");

                var metadata = metadataProperty.GetValue(onnxData)
                    ?? throw new YoloDotNetModelException($"{propertyName} value is null.");

                var shapes = new Dictionary<string, T[]>();

                foreach (var item in (dynamic)metadata)
                {
                    var tensorName = (string)item.Key;

                    var valueProperty = item.GetType().GetProperty("Value");
                    var valueData = valueProperty.GetValue(item);

                    var dimensionsProperty = valueData.GetType().GetProperty("Dimensions");
                    var dimensions = dimensionsProperty.GetValue(valueData);

                    var dimensionsList = new List<T>();

                    foreach (var dimension in (dynamic)dimensions)
                    {
                        var dim = (T)Convert.ChangeType(dimension, typeof(T));

                        dimensionsList.Add(dim);
                    }

                    shapes.Add(tensorName, [.. dimensionsList]);
                }

                return shapes;
            }
            catch (YoloDotNetModelException)
            {
                throw;
            }
            catch (Exception ex)
            {
                throw new YoloDotNetModelException($"Failed to retrieve {propertyName} from ONNX model.", ex);
            }
        }

        /// <summary>
        /// Retrieves the custom metadata key-value pairs from the specified ONNX model data object.
        /// </summary>
        private static Dictionary<string, string> GetMetadata(object onnxData)
        {
            try
            {
                var modelMetadataProperty = onnxData.GetType().GetProperty("ModelMetadata")
                    ?? throw new InvalidOperationException("ModelMetadata property not found.");

                var modelMetadata = modelMetadataProperty.GetValue(onnxData)
                    ?? throw new InvalidOperationException("ModelMetadata value is null.");

                var customMetadataMapProperty = modelMetadata.GetType().GetProperty("CustomMetadataMap")
                    ?? throw new InvalidOperationException("CustomMetadataMap property not found.");

                var customMetadataMap = customMetadataMapProperty.GetValue(modelMetadata)
                    ?? throw new InvalidOperationException("CustomMetadataMap value is null.");

                var metadataCollection = new Dictionary<string, string>();

                foreach (var item in (dynamic)customMetadataMap)
                {
                    metadataCollection.Add(item.Key, item.Value);
                }

                return metadataCollection;
            }
            catch (Exception)
            {
                // Return empty dictionary if metadata cannot be retrieved
                return [];
            }
        }

        /// <summary>
        /// Determines the data type used by the ONNX model's input tensor.
        /// </summary>
        private static ModelDataType GetModelDataType(object onnxData)
        {
            try
            {
                var inputMetadataProperty = onnxData.GetType().GetProperty("InputMetadata")
                    ?? throw new YoloDotNetModelException("InputMetadata could not be retrieved from ONNX model.");

                var inputMetadata = inputMetadataProperty.GetValue(onnxData)
                    ?? throw new YoloDotNetModelException("InputMetadata value is null.");

                // Get the first input's element data type
                foreach (var item in (dynamic)inputMetadata)
                {
                    var itemType = item.GetType();
                    var valueProperty = itemType.GetProperty("Value");
                    var valueData = valueProperty.GetValue(item);

                    var elementDataTypeProperty = valueData.GetType().GetProperty("ElementDataType");
                    var elementDataType = elementDataTypeProperty.GetValue(valueData)?.ToString();

                    // Check if the element data type is Float16
                    return string.Equals(elementDataType, "Float16", StringComparison.OrdinalIgnoreCase)
                        ? ModelDataType.Float16
                        : ModelDataType.Float;
                }

                // Default to Float if no inputs found
                return ModelDataType.Float;
            }
            catch (YoloDotNetModelException)
            {
                throw;
            }
            catch (Exception ex)
            {
                throw new YoloDotNetModelException("Failed to retrieve model data type from ONNX model.", ex);
            }
        }

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
