using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using YoloDotNet.Data;
using YoloDotNet.Models;

namespace YoloDotNet
{
    /// <summary>
    /// Detects objects in an image based on ONNX model.
    /// </summary>
    /// <remarks>
    public class Yolo : YoloBase
    {
        /// <summary>
        /// Initializes a new instance of the Yolo object detection model.
        /// </summary>
        /// <param name="onnxModel">The path to the ONNX model.</param>
        /// <param name="cuda">Optional. Indicates whether to use CUDA for GPU acceleration (default is true).</param>
        /// <param name="gpuId">Optional. The GPU device ID to use when CUDA is enabled (default is 0).</param>
        /// <remarks>
        public Yolo(string onnxModel, bool cuda = true, int gpuId = 0)
            : base(onnxModel, cuda, gpuId) { }

        /// <summary>
        /// Detects objects in a tensor and returns a list of result models.
        /// </summary>
        /// <param name="tensor">The input tensor containing object detection data.</param>
        /// <param name="image">The image associated with the tensor data.</param>
        /// <param name="threshold">The confidence threshold for accepting object detections.</param>
        /// <returns>A list of result models representing detected objects.</returns>
        public override List<ObjectDetection> DetectObjectsInTensor(Tensor<float> tensor, Image image, double threshold)
        {
            var result = new List<ObjectDetection>();

            var (w, h) = (image.Width, image.Height);
            var (xGain, yGain) = (OnnxModel.Input.Width / (float)w, OnnxModel.Input.Height / (float)h);
            var (xPad, yPad) = ((OnnxModel.Input.Width - w * xGain) / 2, (OnnxModel.Input.Height - h * yGain) / 2);

            var dimensions = OnnxModel.Output.Dimensions - 4;
            var batchSize = tensor.Dimensions[0];
            var elementsPerPrediction = (int)(tensor.Length / tensor.Dimensions[1]); //divide total length by the elements per prediction

            Parallel.For(0, batchSize, i =>
            {
                for (var j = 0; j < elementsPerPrediction; j++)
                {
                    // Calculate coordinates of the bounding box in the original image
                    var xMin = ((tensor[i, 0, j] - tensor[i, 2, j] / 2) - xPad) / xGain;
                    var yMin = ((tensor[i, 1, j] - tensor[i, 3, j] / 2) - yPad) / yGain;
                    var xMax = ((tensor[i, 0, j] + tensor[i, 2, j] / 2) - xPad) / xGain;
                    var yMax = ((tensor[i, 1, j] + tensor[i, 3, j] / 2) - yPad) / yGain;

                    // Keep bounding box coordinates within the image boundaries
                    xMin = Math.Max(xMin, 0);
                    yMin = Math.Max(yMin, 0);
                    xMax = Math.Min(xMax, w - 1);
                    yMax = Math.Min(yMax, h - 1);

                    for (int l = 0; l < dimensions; l++)
                    {
                        var confidence = tensor[i, 4 + l, j];

                        if (confidence < threshold) continue;

                        result.Add(new ObjectDetection()
                        {
                            Label = OnnxModel.Labels[l],
                            Confidence = confidence,
                            Rectangle = new RectangleF(xMin, yMin, xMax - xMin, yMax - yMin)
                        });
                    }
                }
            });

            return result;
        }
    }
}