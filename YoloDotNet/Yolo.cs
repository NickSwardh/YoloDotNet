using Microsoft.ML.OnnxRuntime.Tensors;
using System.Collections.Concurrent;
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
        public override List<ResultModel> DetectObjectsInTensor(Tensor<float> tensor, Image image, double threshold)
        {
            var result = new ConcurrentBag<ResultModel>();

            var (w, h) = (image.Width, image.Height); // image w and h
            var (xGain, yGain) = (OnnxModel.Input.Width / (float)w, OnnxModel.Input.Height / (float)h); // x, y gains
            var (xPad, yPad) = ((OnnxModel.Input.Width - w * xGain) / 2, (OnnxModel.Input.Height - h * yGain) / 2); // left, right pads

            var dimensions = OnnxModel.Output.Dimensions - 4;
            var batchSize = tensor.Dimensions[0];
            var elementsPerPrediction = (int)(tensor.Length / tensor.Dimensions[1]); //divide total length by the elements per prediction

            //for each batch
            for (var i = 0; i < batchSize; i++)
            {
                Parallel.For(0, elementsPerPrediction, j =>
                {
                    float xMin = ((tensor[i, 0, j] - tensor[i, 2, j] / 2) - xPad) / xGain; // unpad bbox tlx to original
                    float yMin = ((tensor[i, 1, j] - tensor[i, 3, j] / 2) - yPad) / yGain; // unpad bbox tly to original
                    float xMax = ((tensor[i, 0, j] + tensor[i, 2, j] / 2) - xPad) / xGain; // unpad bbox brx to original
                    float yMax = ((tensor[i, 1, j] + tensor[i, 3, j] / 2) - yPad) / yGain; // unpad bbox bry to original

                    xMin = Clamp(xMin, 0, w - 0); // clip bbox tlx to boundaries
                    yMin = Clamp(yMin, 0, h - 0); // clip bbox tly to boundaries
                    xMax = Clamp(xMax, 0, w - 1); // clip bbox brx to boundaries
                    yMax = Clamp(yMax, 0, h - 1); // clip bbox bry to boundaries

                    for (int l = 0; l < dimensions; l++)
                    {
                        var confidence = tensor[i, 4 + l, j];

                        //skip low confidence values
                        if (confidence < threshold) continue;

                        result.Add(new ResultModel()
                        {
                            Label = OnnxModel.Labels[l],
                            Confidence = confidence,
                            Rectangle = new RectangleF(xMin, yMin, xMax - xMin, yMax - yMin)
                        });
                    }
                });
            }

            return RemoveOverlappingBoxes(result.ToList());
        }
    }
}