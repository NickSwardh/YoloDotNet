using Microsoft.ML.OnnxRuntime;
using Newtonsoft.Json;
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

            var model = new OnnxModel()
            {
                InputName = inputName,
                OutputName = outputName,
                Date = DateTime.Parse(metaData[NameOf(MetaData.Date)]),
                Description = metaData[NameOf(MetaData.Description)],
                Author = metaData[NameOf(MetaData.Author)],
                Version = metaData[NameOf(MetaData.Version)],
                License = metaData[NameOf(MetaData.License)],
                Stride = int.Parse(metaData[NameOf(MetaData.Stride)]),
                Task = metaData[NameOf(MetaData.Task)],
                BatchSize = int.Parse(metaData[NameOf(MetaData.Batch)]),
                ImageSize = new Size(
                    session.InputMetadata[inputName].Dimensions[2],
                    session.InputMetadata[inputName].Dimensions[3]
                    ),
                Input = new(
                    session.InputMetadata[inputName].Dimensions[0],
                    session.InputMetadata[inputName].Dimensions[1],
                    session.InputMetadata[inputName].Dimensions[2],
                    session.InputMetadata[inputName].Dimensions[3]
                    ),
                Output = new(
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
    }
}
