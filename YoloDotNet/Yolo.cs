namespace YoloDotNet
{
    /// <summary>
    /// Detects objects in an image based on ONNX model.
    /// </summary>
    /// <remarks>
    /// Initializes a new instance of the Yolo object detection model.
    /// </remarks>
    /// <param name="onnxModel">The path to the ONNX model.</param>
    /// <param name="cuda">Optional. Indicates whether to use CUDA for GPU acceleration (default is true).</param>
    /// <param name="primeGpu">Optional. Indicates whether to prime the GPU by allocating memory for faster initial inference (default is false).</param>
    /// <param name="gpuId">Optional. The GPU device ID to use when CUDA is enabled (default is 0).</param>
    public class Yolo(string onnxModel, bool cuda = true, bool primeGpu = false, int gpuId = 0)
        : YoloBase(onnxModel, cuda, primeGpu, gpuId)
    {
        /// <summary>
        /// Run image classification on an Image.
        /// </summary>
        /// <param name="img">The image to classify.</param>
        /// <param name="classes">The number of classes to return (default is 1).</param>
        /// <returns>A list of classification results.</returns>
        public override List<Classification> RunClassification(Image img, int classes = 1)
            => Run<Classification>(img, classes, 0, ModelType.Classification);

        /// <summary>
        /// Run object detection on the provided image.
        /// </summary>
        /// <param name="img">The image to perform object detection on.</param>
        /// <param name="confidence">Confidence threshold value for detected objects (default: 0.25).</param>
        /// <param name="iou">IoU (Intersection Over Union) overlap threshold value for removing overlapping bounding boxes (default: 0.45).</param>
        /// <returns>A list of detected objects.</returns>
        public override List<ObjectDetection> RunObjectDetection(Image img, double confidence = 0.25, double iou = 0.45)
            => Run<ObjectDetection>(img, confidence, iou, ModelType.ObjectDetection);

        /// <summary>
        /// Run oriented bounding bBox detection on an image.
        /// </summary>
        /// <param name="img">The image to obb detect.</param>
        /// <param name="confidence">The confidence threshold for detected objects (default is 0.25).</param>
        /// <param name="iou">IoU (Intersection Over Union) overlap threshold value for removing overlapping bounding boxes (default: 0.45).</param>
        /// <returns>A list of Segmentation results.</returns>
        public override List<OBBDetection> RunObbDetection(Image img, double confidence = 0.25, double iou = 0.45)
            => Run<OBBDetection>(img, confidence, iou, ModelType.ObbDetection);

        /// <summary>
        /// Run segmentation on an image.
        /// </summary>
        /// <param name="img">The image to segmentate.</param>
        /// <param name="confidence">The confidence threshold for detected objects (default is 0.25).</param>
        /// <param name="iou">IoU (Intersection Over Union) overlap threshold value for removing overlapping bounding boxes (default: 0.45).</param>
        /// <returns>A list of Segmentation results.</returns>
        public override List<Segmentation> RunSegmentation(Image img, double confidence = 0.25, double iou = 0.45)
            => Run<Segmentation>(img, confidence, iou, ModelType.Segmentation);

        /// <summary>
        /// Run pose estimation on an image.
        /// </summary>
        /// <param name="img">The image to pose estimate.</param>
        /// <param name="threshold">The confidence threshold for detected objects (default is 0.25).</param>
        /// <param name="iou">IoU (Intersection Over Union) overlap threshold value for removing overlapping bounding boxes (default: 0.45).</param>
        /// <returns>A list of Segmentation results.</returns>
        public override List<PoseEstimation> RunPoseEstimation(Image img, double confidence = 0.25, double iou = 0.45)
            => Run<PoseEstimation>(img, confidence, iou, ModelType.PoseEstimation);

        /// <summary>
        /// Run image classification on a video file.
        /// </summary>
        /// <param name="options">Options for video processing.</param>
        /// <param name="classes">The number of classes to return for each frame (default is 1).</param>
        public override Dictionary<int, List<Classification>> RunClassification(VideoOptions options, int classes = 1)
            => RunVideo<Classification>(options, classes, 0, ModelType.Classification);

        /// <summary>
        /// Run object detection on a video file.
        /// </summary>
        /// <param name="options">Options for video processing.</param>
        /// <param name="confidence">The confidence threshold for detected objects (default is 0.25).</param>
        /// <param name="iou">IoU (Intersection Over Union) overlap threshold value for removing overlapping bounding boxes (default: 0.45).</param>
        public override Dictionary<int, List<ObjectDetection>> RunObjectDetection(VideoOptions options, double confidence = 0.25, double iou = 0.45)
            => RunVideo<ObjectDetection>(options, confidence, iou, ModelType.ObjectDetection);

        /// <summary>
        /// Run oriented bounding box detection on a video file.
        /// </summary>
        /// <param name="options">Options for video processing.</param>
        /// <param name="confidence">The confidence threshold for detected objects (default is 0.25).</param>
        /// <param name="iou">IoU (Intersection Over Union) overlap threshold value for removing overlapping bounding boxes (default: 0.45).</param>
        public override Dictionary<int, List<OBBDetection>> RunObbDetection(VideoOptions options, double confidence = 0.25, double iou = 0.45)
            => RunVideo<OBBDetection>(options, confidence, iou, ModelType.ObbDetection);

        /// <summary>
        /// Run object detection on a video file.
        /// </summary>
        /// <param name="options">Options for video processing.</param>
        /// <param name="confidence">The confidence threshold for detected objects (default is 0.25).</param>
        /// <param name="iou">IoU (Intersection Over Union) overlap threshold value for removing overlapping bounding boxes (default: 0.45).</param>
        public override Dictionary<int, List<Segmentation>> RunSegmentation(VideoOptions options, double confidence = 0.25, double iou = 0.45)
            => RunVideo<Segmentation>(options, confidence, iou, ModelType.Segmentation);

        /// <summary>
        /// Run pose estimation on a video file.
        /// </summary>
        /// <param name="options">Options for video processing.</param>
        /// <param name="confidence">The confidence threshold for detected objects (default is 0.25).</param>
        /// <param name="iou">IoU (Intersection Over Union) overlap threshold value for removing overlapping bounding boxes (default: 0.45).</param>
        public override Dictionary<int, List<PoseEstimation>> RunPoseEstimation(VideoOptions options, double confidence = 0.25, double iou = 0.45)
            => RunVideo<PoseEstimation>(options, confidence, iou, ModelType.PoseEstimation);

        #region Tensor methods

        /// <summary>
        /// Classifies a tensor and returns a Classification list 
        /// </summary>
        /// <param name="tensor"></param>
        /// <param name="numberOfClasses"></param>
        /// <returns></returns>
        protected override List<Classification> ClassifyTensor(int numberOfClasses) => Tensors[OnnxModel.OutputNames[0]]
            .GetTensorDataAsSpan<float>()
            .ToArray()
            .Select((score, index) => new Classification
        {
            Confidence = score,
            Label = OnnxModel.Labels[index].Name
        })
            .OrderByDescending(x => x.Confidence)
            .Take(numberOfClasses)
            .ToList();

        /// <summary>
        /// Detects objects in a tensor and returns a ObjectDetection list.
        /// </summary>
        /// <param name="tensor">The input tensor containing object detection data.</param>
        /// <param name="image">The image associated with the tensor data.</param>
        /// <param name="confidenceThreshold">The confidence threshold for accepting object detections.</param>
        /// <returns>A list of result models representing detected objects.</returns>
        protected override ObjectResult[] ObjectDetectImage(Image image, double confidenceThreshold, double overlapThreshold)
        {
            var (xPad, yPad, gain) = CalculateGain(image, OnnxModel);

            var labels = OnnxModel.Labels.Length;
            var elements = OnnxModel.Outputs[0].Elements;
            var channels = OnnxModel.Outputs[0].Channels;

            var boxes = new ObjectResult[channels];

            // Get tensor as a flattened Span for faster processing.
            var ortSpan = Tensors[OnnxModel.OutputNames[0]].GetTensorDataAsSpan<float>();

            for (int i = 0; i < channels; i++)
                    {
                // Move forward to confidence value of first label
                var labelOffset = i + channels * 4;

                // Iterate through all labels...
                for (var l = 0; l < labels; l++, labelOffset += channels)
                {
                    var boxConfidence = ortSpan[labelOffset];

                        if (boxConfidence < confidenceThreshold) continue;

                    float x = ortSpan[i];                   // x
                    float y = ortSpan[i + channels];        // y
                    float w = ortSpan[i + channels * 2];    // w
                    float h = ortSpan[i + channels * 3];    // h

                    var xMin = (int)((x - w / 2 - xPad) * gain);
                    var yMin = (int)((y - h / 2 - yPad) * gain);
                    var xMax = (int)((x + w / 2 - xPad) * gain);
                    var yMax = (int)((y + h / 2 - yPad) * gain);

                    boxes[i] = new ObjectResult
                        {
                            Label = OnnxModel.Labels[l],
                            Confidence = boxConfidence,
                            BoundingBox = new Rectangle(xMin, yMin, xMax - xMin, yMax - yMin),
                        BoundingBoxIndex = i,
                        OrientationAngle = OnnxModel.ModelType == ModelType.ObbDetection ? CalculateRadianToDegree(ortSpan[i + channels * (4 + labels)]) : 0, // Angle (radian) for OBB is located at the end of the labels.
                    };
                    }
            }

            var results = boxes.Where(x => x is not null).ToArray();

            return RemoveOverlappingBoxes(results, overlapThreshold);
        }

        /// <summary>
        /// Performs segmentation on the input image
        /// </summary>
        /// <param name="image">The input image for segmentation.</param>
        /// <param name="boundingBoxes">List of bounding boxes for segmentation.</param>
        /// <returns>List of Segmentation objects corresponding to the input bounding boxes.</returns>
        protected override List<Segmentation> SegmentImage(Image image, ObjectResult[] boundingBoxes)
        {
            Output? output = OnnxModel.Outputs[1]; // Segmentation output

            // Get tensors as a flattened Span for faster processing.
            var ortSpan0 = Tensors[OnnxModel.OutputNames[0]].GetTensorDataAsSpan<float>();
            var ortSpan1 = Tensors[OnnxModel.OutputNames[1]].GetTensorDataAsSpan<float>();

            var elements = OnnxModel.Labels.Length + 4; // 4 = the boundingbox dimension (x, y, width, height)
            var channels = OnnxModel.Outputs[0].Channels;

            var boxSpan = boundingBoxes.AsSpan();
            for (var i = 0; i < boxSpan.Length; i++)
            {
                var box = boxSpan[i];

                // Collect mask weights from the first tensor based on the bounding box index
                var maskWeights = new float[output.Channels];
                var maskOffset = box.BoundingBoxIndex + (channels * elements);

                for (var m = 0; m < output.Channels; m++, maskOffset += channels)
                    maskWeights[m] = ortSpan0[maskOffset];

                // Create an empty image with the same size as the output shape
                using var segmentedImage = new Image<L8>(output.Width, output.Height);

                var heightOffset = 0;

                // Iterate over each empty pixel...
                for (var y = 0; y < output.Height; y++, heightOffset += output.Height)
                    for (int x = 0; x < output.Width; x++)
                    {
                        float pixelWeight = 0;

                        // Iterate over each channel and calculate pixel location (x, y) with its maskweight, collected from first tensor.
                        for (var (p, off) = (0, x + heightOffset); p < output.Channels; p++, off += output.Width * output.Height)
                            pixelWeight += ortSpan1[off] * maskWeights[p];

                        // Calculate and update the pixel luminance value
                        var pixelLuminance = CalculatePixelLuminance(Sigmoid(pixelWeight));
                        segmentedImage[x, y] = new L8(pixelLuminance);
                    }

                segmentedImage.CropResizedSegmentedArea(image, box.BoundingBox);
                box.SegmentedPixels = segmentedImage.GetSegmentedPixels(p => CalculatePixelConfidence(p.PackedValue));
            }

            return boundingBoxes.Select(x => (Segmentation)x).ToList();
        }

        protected override List<PoseEstimation> PoseEstimateImage(Image image, double threshold, double overlapThrehshold)
        {
            var boxes = ObjectDetectImage(image, threshold, overlapThrehshold);
            var (xPad, yPad, gain) = CalculateGain(image, OnnxModel);

            // Get tensor as a flattened Span for faster processing.
            var ortSpan = Tensors[OnnxModel.OutputNames[0]].GetTensorDataAsSpan<float>();

            var labels = OnnxModel.Labels.Length;
            var ouputChannels = OnnxModel.Outputs[0].Channels;
            var totalKeypoints = (int)Math.Floor(((double)OnnxModel.Outputs[0].Elements / OnnxModel.Input.Channels)) - labels;

            for (int i = 0; i < boxes.Length; i++)
            {
                var box = boxes[i];
                var poseEstimations = new Pose[totalKeypoints];
                var keypointOffset = box.BoundingBoxIndex + (ouputChannels * (4 + labels)); // Skip boundingbox + labels (4 + labels) and move forward to the first keypoint

                for (var j = 0; j < totalKeypoints; j++)
                {
                    var xIndex = keypointOffset;
                    var yIndex = xIndex + ouputChannels;
                    var cIndex = yIndex + ouputChannels;
                    keypointOffset += ouputChannels * 3;

                    var x = (int)((ortSpan[xIndex] - xPad) * gain);  // Keypoint x
                    var y = (int)((ortSpan[yIndex] - yPad) * gain);  // Keypoint y
                    var confidence = ortSpan[cIndex];               // Keypoint confidence

                    poseEstimations[j] = new Pose(x, y, confidence);
                }

                box.PoseMarkers = poseEstimations;
            }

            return boxes.Select(x => (PoseEstimation)x).ToList();
        }
        #endregion
    }
}