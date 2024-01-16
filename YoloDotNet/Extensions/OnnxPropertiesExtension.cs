using Microsoft.ML.OnnxRuntime;
using Newtonsoft.Json;
using SixLabors.ImageSharp;
using YoloDotNet.Data;
using YoloDotNet.Enums;
using YoloDotNet.Models;

namespace YoloDotNet.Extensions
{
    public static class OnnxPropertiesExtension
    {
        /// <summary>
        /// Extracts and retrieves metadata properties from the ONNX model.
        /// </summary>
        /// <param name="session">The ONNX model inference session.</param>
        /// <returns>An instance of OnnxModel containing extracted metadata properties.</returns>
        public static OnnxModel GetOnnxProperties(this InferenceSession session)
        {
            var metaData = session.ModelMetadata.CustomMetadataMap;

            var inputName = session.InputNames[0];
            var outputName = session.OutputNames[0];
            var inputMetaData = session.InputMetadata[inputName];
            var (modelType, outputShape) = GetModelOutputShape(session.OutputMetadata[outputName].Dimensions);

            var model = new OnnxModel()
            {
                ModelType = modelType,
                InputName = inputName,
                OutputName = outputName,
                CustomMetaData = metaData,
                ImageSize = new Size(
                    inputMetaData.Dimensions[2],
                    inputMetaData.Dimensions[3]
                    ),
                Input = new(
                    inputMetaData.Dimensions[0],
                    inputMetaData.Dimensions[1],
                    inputMetaData.Dimensions[2],
                    inputMetaData.Dimensions[3]
                    ),
                Output = outputShape, 
                    session.OutputMetadata[outputName].Dimensions[0],
                    session.OutputMetadata[outputName].Dimensions[1],
                    session.OutputMetadata[outputName].Dimensions[2]
                    ),
                Labels = MapLabelsAndColors(session.ModelMetadata.CustomMetadataMap[NameOf(MetaData.Names)])
            };

            return model;
        }

        /// <summary>
        /// Maps ONNX labels to corresponding colors for visualization.
        /// </summary>
        /// <param name="onnxLabelData">The JSON-encoded ONNX label data.</param>
        /// <returns>An array of LabelModel objects with names and associated colors.</returns>
        private static LabelModel[] MapLabelsAndColors(string onnxLabelData)
        {
            var onnxLabels = JsonConvert.DeserializeObject<Dictionary<int, string>>(onnxLabelData);

            var colors = YoloDotNetColors.Get();

            if (onnxLabels!.Count > colors.Length)
                throw new("There are more labels than available colors.");

            // Map ONNX labels to corresponding colors
            return onnxLabels.Zip(colors, (label, color)
                => new LabelModel
                {
                    Name = label.Value,
                    Color = color
                }).ToArray();
        }

        private static string NameOf(dynamic metadata)
            => metadata.ToString().ToLower();

        private static (ModelType, IOutputShape) GetModelOutputShape(int[] outputDimensions)
        {
            var modelType = (ModelType)outputDimensions.Length;

            IOutputShape outputShape = modelType switch
            {
                ModelType.Classification => new ClassificationShape(
                    outputDimensions[0],
                    outputDimensions[1]
                    ),
                ModelType.ObjectDetection => new ObjectDetectionShape(
                    outputDimensions[0],
                    outputDimensions[1],
                    outputDimensions[2]
                    ),
                _ => throw new ArgumentOutOfRangeException(
                    nameof(outputDimensions),
                    $"Unknown ONNX model. Please provide a model for classification or Object Detection.")
            };

            return (modelType, outputShape);
        }
    }
}
