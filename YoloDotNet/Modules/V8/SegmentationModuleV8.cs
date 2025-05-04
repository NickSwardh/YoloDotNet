namespace YoloDotNet.Modules.V8
{
    internal class SegmentationModuleV8 : ISegmentationModule
    {
        private readonly YoloCore _yoloCore;
        private readonly ObjectDetectionModuleV8 _objectDetectionModule;

        public OnnxModel OnnxModel => _yoloCore.OnnxModel;

        public SegmentationModuleV8(YoloCore yoloCore)
        {
            _yoloCore = yoloCore;
            _objectDetectionModule = new ObjectDetectionModuleV8(_yoloCore);
        }

        public List<Segmentation> ProcessImage(SKBitmap image, double confidence, double pixelConfidence, double iou)
        {
            using IDisposableReadOnlyCollection<OrtValue>? ortValues = _yoloCore.Run(image);
            return RunSegmentation(image, ortValues, confidence, pixelConfidence, iou);
        }

        #region Segmentation

        /// <summary>
        /// <para><strong>Current segmentation process overview:</strong></para>
        /// <list type="number">
        ///     <item>
        ///         <description>Perform regular object detection to obtain the bounding boxes.</description>
        ///     </item>
        ///     <item>
        ///         <description>Rescale each bounding box to fit the object within the ONNX model dimensions for segmentation (default: 160x160 for YOLO).</description>
        ///     </item>
        ///     <item>
        ///         <description>Calculate pixel mask weights for the rescaled bounding box.</description>
        ///     </item>
        ///     <item>
        ///         <description>Apply the mask weights to the pixels within the rescaled bounding box area.</description>
        ///     </item>
        ///     <item>
        ///         <description>Crop the bounding box with the applied mask weights.</description>
        ///     </item>
        ///     <item>
        ///         <description>Rescale the cropped bounding box back to its original size.</description>
        ///     </item>
        ///     <item>
        ///         <description>Iterate through all pixels and collect those with confidence values greater than the threshold.</description>
        ///     </item>
        /// </list>
        /// </summary>
        /// <param name="image">The input image for segmentation.</param>
        /// <param name="ortValues">A read-only collection of OrtValue objects used for segmentation.</param>
        /// <param name="confidence">The confidence threshold for object detection.</param>
        /// <param name="iou">The Intersection over Union (IoU) threshold for excluding bounding boxes.</param>
        /// <returns>A list of Segmentation objects corresponding to the input bounding boxes.</returns> 
        private List<Segmentation> RunSegmentation(SKBitmap image, IDisposableReadOnlyCollection<OrtValue> ortValues, double confidence, double pixelConfidence, double iou)
        {
            var ortSpan0 = ortValues[0].GetTensorDataAsSpan<float>();
            var ortSpan1 = ortValues[1].GetTensorDataAsSpan<float>();

            var boundingBoxes = _objectDetectionModule.ObjectDetection(image, ortSpan0, confidence, iou);
            var pixels = new ConcurrentBag<Pixel>();
            var croppedImage = new SKBitmap();
            var resizedBitmap = new SKBitmap();
            using var segmentedBitmap = new SKBitmap(_yoloCore.OnnxModel.Outputs[1].Width, _yoloCore.OnnxModel.Outputs[1].Height, SKColorType.Gray8, SKAlphaType.Opaque);

            var elements = _yoloCore.OnnxModel.Labels.Length + 4; // 4 = the boundingbox dimension (x, y, width, height)

            var inputWidth = _yoloCore.OnnxModel.Input.Width;
            var inputHeight = _yoloCore.OnnxModel.Input.Height;
            var scalingFactorW = (float)_yoloCore.OnnxModel.Outputs[1].Width / inputWidth;
            var scalingFactorH = (float)_yoloCore.OnnxModel.Outputs[1].Height / inputHeight;

            var pixelThreshold = (float)pixelConfidence; // ImageConfig.SEGMENTATION_PIXEL_THRESHOLD;

            foreach (var box in boundingBoxes)
            {
                var bboxWidth = box.BoundingBox.Width;
                var bboxHeight = box.BoundingBox.Height;

                var scaledBoundingBox = ScaleBoundingBox(box, scalingFactorW, scalingFactorH);
                var maskWeights = CollectMaskWeightsFromBoundingBoxArea(box, _yoloCore.OnnxModel.Outputs[0].Channels, _yoloCore.OnnxModel.Outputs[1].Channels, elements, ortSpan0);

                using var canvas = new SKCanvas(segmentedBitmap);
                canvas.Clear(SKColors.White);
                ApplyMaskToSegmentedPixels(segmentedBitmap,
                    _yoloCore.OnnxModel.Outputs[1].Width,
                    _yoloCore.OnnxModel.Outputs[1].Height,
                    _yoloCore.OnnxModel.Outputs[1].Channels,
                    scaledBoundingBox,
                    ortSpan1,
                    maskWeights);

                // Create an empty image used for cropping the bounding box
                croppedImage.Reset();
                croppedImage = new SKBitmap(new SKImageInfo(scaledBoundingBox.Width, scaledBoundingBox.Height, SKColorType.Gray8));

                CropSegmentedBoundingBox(croppedImage, segmentedBitmap, scaledBoundingBox, _yoloCore.OnnxModel.Outputs[1].Width);

                resizedBitmap.Reset();
                resizedBitmap = new SKBitmap(bboxWidth, bboxHeight, SKColorType.Gray8, SKAlphaType.Opaque);
                using var recanvas = new SKCanvas(resizedBitmap);

                recanvas.DrawBitmap(croppedImage.Resize(new SKSizeI(bboxWidth, bboxHeight), new SKSamplingOptions(4)), 0, 0);

                pixels.Clear();
                GetPixelsFromCroppedMask(resizedBitmap, pixelThreshold, pixels, box);
                box.SegmentedPixels = [.. pixels];
            }

            // Clean up the mess
            croppedImage.Dispose();
            resizedBitmap.Dispose();
            ortValues[0].Dispose();
            ortValues[1].Dispose();

            return [.. boundingBoxes.Select(x => (Segmentation)x)];
        }

        private static SKRectI ScaleBoundingBox(ObjectResult box, float scalingFactorW, float scalingFactorH)
        {
            var scaledLeft = (int)Math.Round(box.BoundingBoxUnscaled.Left * scalingFactorW);
            var scaledTop = (int)Math.Round(box.BoundingBoxUnscaled.Top * scalingFactorH);
            var scaledRight = (int)Math.Round(box.BoundingBoxUnscaled.Right * scalingFactorW);
            var scaledBottom = (int)Math.Round(box.BoundingBoxUnscaled.Bottom * scalingFactorH);

            return new SKRectI(scaledLeft, scaledTop, scaledRight, scaledBottom);
        }

        private static float[] CollectMaskWeightsFromBoundingBoxArea(ObjectResult box, int channelsFromOutput0, int channelsFromOutput1, int elements, ReadOnlySpan<float> ortSpan1)
        {
            var maskWeights = new float[channelsFromOutput1];
            var maskOffset = box.BoundingBoxIndex + (channelsFromOutput0 * elements);

            for (var m = 0; m < channelsFromOutput1; m++, maskOffset += channelsFromOutput0)
                maskWeights[m] = ortSpan1[maskOffset];

            return maskWeights;
        }

        unsafe private void ApplyMaskToSegmentedPixels(SKBitmap segmentedBitmap, int output1Width, int output1Height, int output1Channels, SKRectI scaledBoundingBox, ReadOnlySpan<float> ortSpan1, float[] maskWeights)
        {
            IntPtr pixelsPtr = segmentedBitmap.GetPixels();

            for (int y = 0; y < output1Height; y++)
            {
                for (int x = 0; x < output1Width; x++)
                {
                    if (x < scaledBoundingBox.Left || x > scaledBoundingBox.Right || y < scaledBoundingBox.Top || y > scaledBoundingBox.Bottom)
                        continue;

                    float pixelWeight = 0;
                    var offset = x + y * output1Width;
                    for (var p = 0; p < output1Channels; p++, offset += output1Width * output1Height)
                        pixelWeight += ortSpan1[offset] * maskWeights[p];

                    byte* pixelData = (byte*)pixelsPtr.ToPointer();
                    pixelData[y * output1Width + x] = YoloCore.CalculatePixelLuminance(YoloCore.Sigmoid(pixelWeight));
                }
            }
        }

        unsafe private void CropSegmentedBoundingBox(SKBitmap croppedImaged, SKBitmap segmentedBitmap, SKRectI scaledBoundingBox, int output1Width)
        {
            IntPtr croppedPixelsPtr = croppedImaged.GetPixels();
            IntPtr pixelsPtr = segmentedBitmap.GetPixels();

            Parallel.For(0, scaledBoundingBox.Height, _yoloCore.parallelOptions, y =>
            {
                int srcIndex = (scaledBoundingBox.Top + y) * output1Width + scaledBoundingBox.Left;
                int dstIndex = y * scaledBoundingBox.Width;

                byte* srcPixelData = (byte*)pixelsPtr.ToPointer();
                byte* dstPixelData = (byte*)croppedPixelsPtr.ToPointer();
                Buffer.MemoryCopy(srcPixelData + srcIndex, dstPixelData + dstIndex, scaledBoundingBox.Width, scaledBoundingBox.Width);
            });
        }

        unsafe private void GetPixelsFromCroppedMask(SKBitmap resizedBitmap, float confidenceThreshold, ConcurrentBag<Pixel> pixels, ObjectResult box)
        {
            var boundingBoxX = box.BoundingBox.Left;
            var boundingBoxY = box.BoundingBox.Top;

            IntPtr resizedPtr = resizedBitmap.GetPixels();

            Parallel.For(0, resizedBitmap.Height, _yoloCore.parallelOptions, y =>
            {
                byte* resizedPixelData = (byte*)resizedPtr.ToPointer();

                for (int x = 0; x < resizedBitmap.Width; x++)
                {
                    int index = y * resizedBitmap.Width + x;
                    byte pixelValue = resizedPixelData[index];
                    float confidence = YoloCore.CalculatePixelConfidence(pixelValue);

                    if (confidence > confidenceThreshold)
                        pixels.Add(new Pixel(boundingBoxX + x, boundingBoxY + y, confidence));
                }
            });
        }

        public void Dispose()
        {
            _objectDetectionModule?.Dispose();
            _yoloCore?.Dispose();

            GC.SuppressFinalize(this);
        }

        #endregion

    }
}
