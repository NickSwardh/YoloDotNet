using Microsoft.ML.OnnxRuntime;
using Newtonsoft.Json;
using SixLabors.ImageSharp;
using System.Runtime.Serialization;
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

            var modelType = GetModelType(metaData[NameOf(MetaData.Task)]);

            var inputName = session.InputNames[0];
            var outputNames = session.OutputNames.ToList();

            return new OnnxModel()
            {
                ModelType = modelType,
                InputName = inputName,
                OutputNames = outputNames,
                CustomMetaData = metaData,
                Input = GetModelInputShape(session.InputMetadata[inputName]),
                Outputs = GetOutputShapes(session.OutputMetadata, modelType),
                Labels = MapLabelsAndColors(metaData[NameOf(MetaData.Names)], modelType)
            };
        }

        #region Helper methods
        /// <summary>
        /// Maps ONNX labels to corresponding colors for visualization.
        /// </summary>
        private static LabelModel[] MapLabelsAndColors(string onnxLabelData, ModelType modelType)
        {
            var onnxLabels = JsonConvert.DeserializeObject<Dictionary<int, string>>(onnxLabelData);

            var colors = modelType == ModelType.Classification
                ? Enumerable.Repeat(YoloDotNetColors.Default(), onnxLabels!.Count).ToArray()
                : YoloDotNetColors.Get();

            if (onnxLabels!.Count > colors.Length)
                throw new("There are more labels than available colors.");

            return onnxLabels.Zip(colors, (label, color) => new LabelModel
            {
                Name = label.Value,
                Color = color
            }).ToArray();
        }

        private static string NameOf(dynamic metadata)
            => metadata.ToString().ToLower();

        /// <summary>
        /// Retrieves the input shape of a model based on metadata
        /// </summary>
        private static Input GetModelInputShape(NodeMetadata metaData)
            => Input.Shape(metaData.Dimensions);

        /// <summary>
        /// Retrieves the output shapes of a model based on metadata and model type.
        /// </summary>
        private static List<Output> GetOutputShapes(IReadOnlyDictionary<string, NodeMetadata> metaData, ModelType modelType)
        {
            var dimensions = metaData.Values.Select(x => x.Dimensions).ToArray();

            var (output0, output1) = modelType switch
            {
                ModelType.Classification => (Output.Classification(dimensions[0]), Output.Empty()),
                ModelType.ObjectDetection => (Output.Detection(dimensions[0]), Output.Empty()),
                ModelType.Segmentation => (Output.Detection(dimensions[0]), Output.Segmentation(dimensions[1])),
                _ => throw new ArgumentException($"Error getting output shapes. Unknown ONNX model type.", nameof(modelType))
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

        #endregion
    }
}
