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
        protected override List<Classification> ClassifyTensor(int numberOfClasses) => Tensors[OnnxModel.OutputNames[0]].Select((score, index) => new Classification
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
        protected override List<ObjectResult> ObjectDetectImage(Image image, double confidenceThreshold, double overlapThreshold)
        {
            var result = new ConcurrentBag<ObjectResult>();

            var (w, h) = (image.Width, image.Height);

            var gain = Math.Max((float)w / OnnxModel.Input.Width, (float)h / OnnxModel.Input.Height);
            var ratio = Math.Min(OnnxModel.Input.Width / (float)image.Width, OnnxModel.Input.Height / (float)image.Height);
            var (xPad, yPad) = ((int)(OnnxModel.Input.Width - w * ratio) / 2, (int)(OnnxModel.Input.Height - h * ratio) / 2);

            var labels = OnnxModel.Labels.Length;
            var batchSize = OnnxModel.Outputs[0].BatchSize;
            var elements = OnnxModel.Outputs[0].Elements;
            var channels = OnnxModel.Outputs[0].Channels;

            var tensor = Tensors[OnnxModel.OutputNames[0]];
            for (var i = 0; i < batchSize; i++)
            {
                Parallel.For(0, channels, j =>
                {
                    // Calculate coordinates of the bounding box in the original image
                    var xMin = (int)((tensor[i, 0, j] - tensor[i, 2, j] / 2 - xPad) * gain);
                    var yMin = (int)((tensor[i, 1, j] - tensor[i, 3, j] / 2 - yPad) * gain);
                    var xMax = (int)((tensor[i, 0, j] + tensor[i, 2, j] / 2 - xPad) * gain);
                    var yMax = (int)((tensor[i, 1, j] + tensor[i, 3, j] / 2 - yPad) * gain);

                    // Keep bounding box coordinates within the image boundaries
                    xMin = Math.Clamp(xMin, 0, w);
                    yMin = Math.Clamp(yMin, 0, h);
                    xMax = Math.Clamp(xMax, 0, w);
                    yMax = Math.Clamp(yMax, 0, h);

                    for (int l = 0; l < labels; l++)
                    {
                        var boxConfidence = tensor[i, l + 4, j];

                        if (boxConfidence < confidenceThreshold) continue;

                        result.Add(new ObjectResult
                        {
                            Label = OnnxModel.Labels[l],
                            Confidence = boxConfidence,
                            BoundingBox = new Rectangle(xMin, yMin, xMax - xMin, yMax - yMin),
                            BoundingBoxIndex = j,
                            OrientationAngle = OnnxModel.ModelType == ModelType.ObbDetection ? CalculateRadianToDegree(tensor[i, elements - 1, j]) : 0 // Angle (radian) for OBB is the last item in elements.
                        });

                        break;
                    }
                });
            }

            return RemoveOverlappingBoxes([.. result], overlapThreshold);
        }

        /// <summary>
        /// Performs segmentation on the input image
        /// </summary>
        /// <param name="image">The input image for segmentation.</param>
        /// <param name="boundingBoxes">List of bounding boxes for segmentation.</param>
        /// <returns>List of Segmentation objects corresponding to the input bounding boxes.</returns>
        protected override List<Segmentation> SegmentImage(Image image, List<ObjectResult> boundingBoxes)
        {
            var output = OnnxModel.Outputs[1]; // Segmentation output
            var tensor0 = Tensors[OnnxModel.OutputNames[0]];
            var tensor1 = Tensors[OnnxModel.OutputNames[1]];

            var elements = OnnxModel.Labels.Length + 4; // 4 = the boundingbox dimension (x, y, width, height)

            Parallel.ForEach(boundingBoxes, _parallelOptions, box =>
            {
                // Collect mask weights from the first tensor based on the bounding box index
                var maskWeights = Enumerable.Range(0, output.Channels)
                    .Select(i => tensor0[0, elements + i, box.BoundingBoxIndex])
                    .ToArray();

                // Create an empty image with the same size as the output shape
                using var segmentedImage = new Image<L8>(output.Width, output.Height);

                // Iterate over each empty pixel...
                for (int y = 0; y < output.Height; y++)
                    for (int x = 0; x < output.Width; x++)
                    {
                        // Iterate over each channel and calculate pixel location (x, y) with its maskweight, collected from first tensor.
                        var value = Enumerable.Range(0, output.Channels).Sum(i => tensor1[0, i, y, x] * maskWeights[i]);

                        // Calculate and update the pixel luminance value
                        var pixelLuminance = CalculatePixelLuminance(Sigmoid(value));
                        segmentedImage[x, y] = new L8(pixelLuminance);
                    }

                segmentedImage.CropResizedSegmentedArea(image, box.BoundingBox);
                box.SegmentedPixels = segmentedImage.GetSegmentedPixels(p => CalculatePixelConfidence(p.PackedValue));
            });

            return boundingBoxes.Select(x => (Segmentation)x).ToList();
        }

        protected override List<PoseEstimation> PoseEstimateImage(Image image, double threshold, double overlapThrehshold)
        {
            var (w, h) = (image.Width, image.Height);

            var gain = Math.Max((float)w / OnnxModel.Input.Width, (float)h / OnnxModel.Input.Height);
            var ratio = Math.Min(OnnxModel.Input.Width / (float)image.Width, OnnxModel.Input.Height / (float)image.Height);
            var (xPad, yPad) = ((int)(OnnxModel.Input.Width - w * ratio) / 2, (int)(OnnxModel.Input.Height - h * ratio) / 2);

            var boxes = ObjectDetectImage(image, threshold, overlapThrehshold);

            var tensor = Tensors[OnnxModel.OutputNames[0]];

            var labels = OnnxModel.Labels.Length;
            var channels = OnnxModel.Input.Channels;
            var elements = OnnxModel.Outputs[0].Elements;
            var markers = (int)Math.Floor(((double)elements / channels)) - labels;

            for (int i = 0; i < boxes.Count; i++)
            {
                var poseEstimations = new Pose[markers];
                var box = boxes[i];

                for (int j = 0; j < markers; j++)
                {
                    var offset = j * channels + labels + 4; // 4 = dimensions of the boundingbox (w, h, x, y)

                    var x = (int)((tensor[0, offset + 0, box.BoundingBoxIndex] - xPad) * gain);
                    var y = (int)((tensor[0, offset + 1, box.BoundingBoxIndex] - yPad) * gain);
                    var confidence = tensor[0, offset + 2, box.BoundingBoxIndex];

                    poseEstimations[j] = new Pose(x, y, confidence);
                }

                box.PoseMarkers = poseEstimations;
            }

            return boxes.Select(x => (PoseEstimation)x).ToList();
        }

        #endregion
    }
}