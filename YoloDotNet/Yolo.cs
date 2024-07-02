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
        #region Exposed entry methods

        /// <summary>
        /// Run image classification on an Image.
        /// </summary>
        /// <param name="img">The image to classify.</param>
        /// <param name="classes">The number of classes to return (default is 1).</param>
        /// <returns>A list of classification results.</returns>
        public override List<Classification> RunClassification(SKImage img, int classes = 1)
            => Run<Classification>(img, classes, 0, ModelType.Classification);

        /// <summary>
        /// Run object detection on the provided image.
        /// </summary>
        /// <param name="img">The image to perform object detection on.</param>
        /// <param name="confidence">Confidence threshold value for detected objects (default: 0.25).</param>
        /// <param name="iou">IoU (Intersection Over Union) overlap threshold value for removing overlapping bounding boxes (default: 0.45).</param>
        /// <returns>A list of detected objects.</returns>
        public override List<ObjectDetection> RunObjectDetection(SKImage img, double confidence = 0.25, double iou = 0.45)
           => Run<ObjectDetection>(img, confidence, iou, ModelType.ObjectDetection);

        /// <summary>
        /// Run oriented bounding bBox detection on an image.
        /// </summary>
        /// <param name="img">The image to obb detect.</param>
        /// <param name="confidence">The confidence threshold for detected objects (default is 0.25).</param>
        /// <param name="iou">IoU (Intersection Over Union) overlap threshold value for removing overlapping bounding boxes (default: 0.45).</param>
        /// <returns>A list of Segmentation results.</returns>
        public override List<OBBDetection> RunObbDetection(SKImage img, double confidence = 0.25, double iou = 0.45)
            => Run<OBBDetection>(img, confidence, iou, ModelType.ObbDetection);

        /// <summary>
        /// Run segmentation on an image.
        /// </summary>
        /// <param name="img">The image to segmentate.</param>
        /// <param name="confidence">The confidence threshold for detected objects (default is 0.25).</param>
        /// <param name="iou">IoU (Intersection Over Union) overlap threshold value for removing overlapping bounding boxes (default: 0.45).</param>
        /// <returns>A list of Segmentation results.</returns>
        public override List<Segmentation> RunSegmentation(SKImage img, double confidence = 0.25, double iou = 0.45)
            => Run<Segmentation>(img, confidence, iou, ModelType.Segmentation);

        /// <summary>
        /// Run pose estimation on an image.
        /// </summary>
        /// <param name="img">The image to pose estimate.</param>
        /// <param name="confidence">The confidence threshold for detected objects (default is 0.25).</param>
        /// <param name="iou">IoU (Intersection Over Union) overlap threshold value for removing overlapping bounding boxes (default: 0.45).</param>
        /// <returns>A list of Segmentation results.</returns>
        public override List<PoseEstimation> RunPoseEstimation(SKImage img, double confidence = 0.25, double iou = 0.45)
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

        #endregion

        #region Tensor methods

        /// <summary>
        /// Classifies a tensor and returns a Classification list 
        /// </summary>
        /// <param name="numberOfClasses"></param>
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
        /// <param name="image">The image associated with the tensor data.</param>
        /// <param name="confidenceThreshold">The confidence threshold for accepting object detections.</param>
        /// <param name="overlapThreshold">The threshold for overlapping boxes to filter detections.</param>
        /// <returns>A list of result models representing detected objects.</returns>
        protected override ObjectResult[] ObjectDetectImage(SKImage image, double confidenceThreshold, double overlapThreshold)
        {
            var (xPad, yPad, gain) = CalculateGain(image, OnnxModel);

            var labels = OnnxModel.Labels.Length;
            var channels = OnnxModel.Outputs[0].Channels;

            var boxes = _customSizeObjectResultPool.Rent(channels);

            try
            {
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

                        // Scaled coordinates for original image
                        var xMin = (int)((x - w / 2 - xPad) * gain);
                        var yMin = (int)((y - h / 2 - yPad) * gain);
                        var xMax = (int)((x + w / 2 - xPad) * gain);
                        var yMax = (int)((y + h / 2 - yPad) * gain);

                        // Unscaled coordinates for resized input image
                        var sxMin = (int)(x - w / 2);
                        var syMin = (int)(y - h / 2);
                        var sxMax = (int)(x + w / 2);
                        var syMax = (int)(y + h / 2);
                        
                        if (boxes[i] is null || boxConfidence > boxes[i].Confidence)
                        {
                            boxes[i] = new ObjectResult
                            {
                                Label = OnnxModel.Labels[l],
                                Confidence = boxConfidence,
                                BoundingBox = new SKRectI (xMin, yMin, xMax, yMax),
                                BoundingBoxOrg = new SKRectI (sxMin, syMin, sxMax, syMax),
                                BoundingBoxIndex = i,
                                OrientationAngle = OnnxModel.ModelType == ModelType.ObbDetection ? ortSpan[i + channels * (4 + labels)] : 0 //CalculateRadianToDegree(ortSpan[i + channels * (4 + labels)]) : 0, // Angle (radian) for OBB is located at the end of the labels.
                            };
                        }
                    }
                }

                var results = boxes.Where(x => x is not null).ToArray();

                return RemoveOverlappingBoxes(results, overlapThreshold);
            }
            finally
            {
                _customSizeObjectResultPool.Return(boxes, clearArray: true);
            }
        }

        /// <summary>
        /// Performs segmentation on the input image
        /// </summary>
        /// <param name="image">The input image for segmentation.</param>
        /// <param name="boundingBoxes">List of bounding boxes for segmentation.</param>
        /// <returns>List of Segmentation objects corresponding to the input bounding boxes.</returns>
        protected override List<Segmentation> SegmentImage(SKImage image, ObjectResult[] boundingBoxes)
        {
            // Get tensors as a flattened Span for faster processing.
            var ortSpan0 = Tensors[OnnxModel.OutputNames[0]].GetTensorDataAsSpan<float>();
            var ortSpan1 = Tensors[OnnxModel.OutputNames[1]].GetTensorDataAsSpan<float>();

            // Get input size
            var input = OnnxModel.Input;
            var (inputWidth, inputHeight) = (input.Width, input.Height);

            // Segmentation output size
            var output = OnnxModel.Outputs[1];
            var (outputWidth, outputHeight, outputChannels) = (output.Width, output.Height, output.Channels);

            // Calculate scaling factors for resizing padded bounding boxes 
            float scalingFactorW = (float)outputWidth / inputWidth;
            float scalingFactorH = (float)outputHeight / inputHeight;

            var elements = OnnxModel.Labels.Length + 4; // 4 = the boundingbox dimension (x, y, width, height)
            var channels = OnnxModel.Outputs[0].Channels;

            // Container to hold segmented pixels
            var pixels = new ConcurrentBag<Pixel>();

            // Bitmaps for reuse to minimize memory consumption
            var croppedImage = new SKBitmap();
            var resizedBitmap = new SKBitmap();

            // Create a new image for so 
            using var segmentedBitmap = new SKBitmap(outputWidth, outputHeight, SKColorType.Gray8, SKAlphaType.Opaque);

            var totalBoundingboxes = boundingBoxes.Length;
            for (var i = 0; i < totalBoundingboxes; i++)
            {
                var box = boundingBoxes[i];

                // Scale padded boundingbox to fit within the dimensions of the output
                var scaledLeft = (int)Math.Round(box.BoundingBoxOrg.Left * scalingFactorW);
                var scaledTop = (int)Math.Round(box.BoundingBoxOrg.Top * scalingFactorH);
                var scaledRight = (int)Math.Round(box.BoundingBoxOrg.Right * scalingFactorW);
                var scaledBottom = (int)Math.Round(box.BoundingBoxOrg.Bottom * scalingFactorH);
                var miniRect = new SKRectI(scaledLeft, scaledTop, scaledRight, scaledBottom);

                // Create an empty image with the same size as the output shape
                using var canvas = new SKCanvas(segmentedBitmap);
                canvas.Clear(SKColors.White);
                IntPtr pixelsPtr = segmentedBitmap.GetPixels();

                // Collect mask weights from the first tensor based on the bounding box index
                var maskWeights = new float[outputChannels];
                var maskOffset = box.BoundingBoxIndex + (channels * elements);

                // Get maskweights for current bounding box from first tensor
                for (var m = 0; m < output.Channels; m++, maskOffset += channels)
                    maskWeights[m] = ortSpan0[maskOffset];

                var heightOffset = 0;

                // Iterate through segmentation and calculate 
                for (int y = 0; y < outputHeight; y++, heightOffset += outputHeight)
                {
                    for (int x = 0; x < outputWidth; x++)
                    {
                        // Skip pixels that are outside the boundary of the bounding box
                        if (x < miniRect.Left || x > miniRect.Right || y < miniRect.Top || y > miniRect.Bottom)
                            continue;

                        float pixelWeight = 0;

                        // Iterate over each channel and calculate pixel location with its maskweight collected from first tensor.
                        var off = x + heightOffset;

                        for (var p = 0; p < outputChannels; p++, off += outputWidth * outputHeight)
                            pixelWeight += ortSpan1[off] * maskWeights[p];

                        // Update current pixel
                        unsafe
                        {
                            byte* pixelData = (byte*)pixelsPtr.ToPointer();
                            pixelData[y * outputWidth + x] = CalculatePixelLuminance(Sigmoid(pixelWeight)); //pixelLuminance;
                        }
                    }
                }

                /////////////////////////////////////////////////////////////////////////////
                ///
                /// Crop bounding box
                /// 
                /////////////////////////////////////////////////////////////////////////////
                // Create an empty image
                croppedImage.Reset();
                croppedImage = new SKBitmap(new SKImageInfo(miniRect.Width, miniRect.Height, SKColorType.Gray8));

                // Access pixels directly
                IntPtr croppedPixelsPtr = croppedImage.GetPixels();

                // Crop
                Parallel.For(0, miniRect.Height, y =>
                {
                    int srcIndex = ((miniRect.Top + y) * outputWidth + miniRect.Left) * 1;
                    int dstIndex = y * miniRect.Width * 1;
                    unsafe
                    {
                        byte* srcPixelData = (byte*)pixelsPtr.ToPointer();
                        byte* dstPixelData = (byte*)croppedPixelsPtr.ToPointer();
                        Buffer.MemoryCopy(srcPixelData + srcIndex, dstPixelData + dstIndex, miniRect.Width * 1, miniRect.Width * 1);
                    }
                });

                /////////////////////////////////////////////////////////////////////////////
                ///
                /// Rescale the cropped bounding box to original size
                /// 
                /////////////////////////////////////////////////////////////////////////////
                resizedBitmap.Reset();
                resizedBitmap = new SKBitmap(box.BoundingBox.Width, box.BoundingBox.Height, SKColorType.Gray8, SKAlphaType.Opaque);
                using var recanvas = new SKCanvas(resizedBitmap);

                // Draw the cropped bitmap onto the canvas, scaling it to the original size
                var destRect = new SKRect(0, 0, box.BoundingBox.Width, box.BoundingBox.Height);
                recanvas.DrawBitmap(croppedImage, destRect, new SKPaint { FilterQuality = SKFilterQuality.Low, IsAntialias = false });


                /////////////////////////////////////////////////////////////////////////////
                ///
                /// Get segmented pixels
                /// 
                /////////////////////////////////////////////////////////////////////////////
                // Define the bounding box
                int bboxX = box.BoundingBox.Left;
                int bboxY = box.BoundingBox.Top;
                int bboxWidth = box.BoundingBox.Width;
                int bboxHeight = box.BoundingBox.Height;
                float confidenceThreshold = 0.65f;

                // Access the pixels directly
                IntPtr resizedPtr = resizedBitmap.GetPixels();

                pixels.Clear();
                Parallel.For(0, resizedBitmap.Height, y =>
                {
                    unsafe
                    {
                        byte* resizedPixelData = (byte*)resizedPtr.ToPointer();

                        for (int x = 0; x < resizedBitmap.Width; x++)
                        {
                            int resizedIndex = (y * resizedBitmap.Width + x);
                            byte pixelValue = resizedPixelData[resizedIndex];

                            float confidence = CalculatePixelConfidence(pixelValue);

                            // If the confidence is higher than the threshold, draw the overlay
                            if (confidence > confidenceThreshold)
                                pixels.Add(new Pixel(bboxX + x, bboxY + y, confidence));
                        }
                    }
                });

                box.SegmentedPixels = [.. pixels];
            }

            segmentedBitmap.Dispose();
            croppedImage.Dispose();
            resizedBitmap.Dispose();

            return boundingBoxes.Select(x => (Segmentation)x).ToList();
        }
       
        protected override List<PoseEstimation> PoseEstimateImage(SKImage image, double threshold, double overlapThrehshold)
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
                var poseEstimations = new KeyPoint[totalKeypoints];
                var keypointOffset = box.BoundingBoxIndex + (ouputChannels * (4 + labels)); // Skip boundingbox + labels (4 + labels) and move forward to the first keypoint

                for (var j = 0; j < totalKeypoints; j++)
                {
                    var xIndex = keypointOffset;
                    var yIndex = xIndex + ouputChannels;
                    var cIndex = yIndex + ouputChannels;
                    keypointOffset += ouputChannels * 3;

                    var x = (int)((ortSpan[xIndex] - xPad) * gain);
                    var y = (int)((ortSpan[yIndex] - yPad) * gain);
                    var confidence = ortSpan[cIndex];

                    poseEstimations[j] = new KeyPoint(x, y, confidence);
                }

                box.KeyPoints = poseEstimations;
            }

            return boxes.Select(x => (PoseEstimation)x).ToList();
        }
        #endregion
    }
}