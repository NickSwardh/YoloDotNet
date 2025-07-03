namespace YoloDotNet.Modules.V8
{
    internal class SegmentationModuleV8 : ISegmentationModule
    {
        private readonly YoloCore _yoloCore;
        private readonly ObjectDetectionModuleV8 _objectDetectionModule;
        private readonly float _scalingFactorW;
        private readonly float _scalingFactorH;
        private readonly SKBitmap _pixelMaskBitmap;
        private readonly SKCanvas _pixelMaskCanvas;
        private readonly SKPaint _paint;
        private readonly int _maskWidth;
        private readonly int _maskHeight;

        public OnnxModel OnnxModel => _yoloCore.OnnxModel;

        public SegmentationModuleV8(YoloCore yoloCore)
        {
            _yoloCore = yoloCore;
            _objectDetectionModule = new ObjectDetectionModuleV8(_yoloCore);

            // Get model input width and height
            var inputWidth = _yoloCore.OnnxModel.Input.Width;
            var inputHeight = _yoloCore.OnnxModel.Input.Height;

            // Get model pixel mask widh and height
            _maskWidth = _yoloCore.OnnxModel.Outputs[1].Width;
            _maskHeight = _yoloCore.OnnxModel.Outputs[1].Height;

            // Calculate scaling factor for downscaling boundingboxes to segmentation pixelmask proportions
            _scalingFactorW = (float)_maskWidth / inputWidth;
            _scalingFactorH = (float)_maskHeight / inputHeight;

            // Define bitmaps for processing the pixel mask
            _pixelMaskBitmap = new SKBitmap(_yoloCore.OnnxModel.Outputs[1].Width, _yoloCore.OnnxModel.Outputs[1].Height, SKColorType.Gray8, SKAlphaType.Opaque);
            _pixelMaskCanvas = new SKCanvas(_pixelMaskBitmap);

            _paint = new SKPaint
            {
                Style = SKPaintStyle.Fill,
                IsStroke = true
            };
        }

        public List<Segmentation> ProcessImage<T>(T image, double confidence, double pixelConfidence, double iou)
        {
            var (ortValues, imageSize) = _yoloCore.Run(image);

            return RunSegmentation(imageSize, ortValues, confidence, pixelConfidence, iou);
        }

        private List<Segmentation> RunSegmentation(SKSizeI imageSize, IDisposableReadOnlyCollection<OrtValue> ortValues, double confidence, double pixelConfidence, double iou)
        {
            var ortSpan0 = ortValues[0].GetTensorDataAsSpan<float>();
            var ortSpan1 = ortValues[1].GetTensorDataAsSpan<float>();

            var boundingBoxes = _objectDetectionModule.ObjectDetection(imageSize, ortSpan0, confidence, iou);

            foreach (var box in boundingBoxes)
            {
                var pixelMaskInfo = new SKImageInfo(box.BoundingBox.Width, box.BoundingBox.Height, SKColorType.Gray8, SKAlphaType.Opaque);
                var downScaledBoundingBox = DownscaleBoundingBoxToSegmentationOutput(box.BoundingBoxUnscaled);

                // 1) Get weights
                var maskWeights = GetMaskWeightsFromBoundingBoxArea(box, ortSpan0);

                // 2) Apply pixelmask based on mask-weights to canvas
                _pixelMaskCanvas.Clear(SKColors.Black);
                ApplySegmentationPixelMask(_pixelMaskCanvas, box.BoundingBoxUnscaled, ortSpan1, maskWeights);

                // 3) Crop downscaled boundingbox from the pixelmask canvas
                using var cropped = new SKBitmap();
                _pixelMaskBitmap.ExtractSubset(cropped, downScaledBoundingBox);

                // 4) Upscale cropped pixelmask to original boundingbox size. For smother edges, use an appropriate resampling method!
                using var resizedCrop = new SKBitmap(pixelMaskInfo);
                cropped.ScalePixels(resizedCrop, ImageConfig.SegmentationResamplingOptions);

                // 5) Pack the upscaled pixel mask into a compact bit array (1 bit per pixel)
                // for cleaner, memory-efficient storage of the mask in the detection box.
                box.BitPackedPixelMask = PackUpscaledMaskToBitArray(resizedCrop, pixelConfidence);
            }

            // Clean up
            ortValues[0].Dispose();
            ortValues[1].Dispose();

            return [.. boundingBoxes.Select(x => (Segmentation)x)];
        }

        private float[] GetMaskWeightsFromBoundingBoxArea(ObjectResult box, ReadOnlySpan<float> ortSpan0)
        {
            var elements = _yoloCore.OnnxModel.Labels.Length + 4; // 4 = the boundingbox dimension (x, y, width, height)
            var channelsFromOutput0 = _yoloCore.OnnxModel.Outputs[0].Channels;
            var channelsFromOutput1 = _yoloCore.OnnxModel.Outputs[1].Channels;

            var maskWeights = new float[channelsFromOutput1];
            var maskOffset = box.BoundingBoxIndex + (channelsFromOutput0 * elements);

            for (var m = 0; m < channelsFromOutput1; m++, maskOffset += channelsFromOutput0)
                maskWeights[m] = ortSpan0[maskOffset];

            return maskWeights;
        }

        private SKRectI DownscaleBoundingBoxToSegmentationOutput(SKRect box)
        {
            int left = (int)Math.Floor(box.Left * _scalingFactorW);
            int top = (int)Math.Floor(box.Top * _scalingFactorH);
            int right = (int)Math.Ceiling(box.Right * _scalingFactorW);
            int bottom = (int)Math.Ceiling(box.Bottom * _scalingFactorH);

            return new SKRectI(left, top, right, bottom);
        }

        private void ApplySegmentationPixelMask(SKCanvas canvas, SKRect bbox, ReadOnlySpan<float> outputOrtSpan, float[] maskWeights)
        {
            // Convert bounding box from original coordinates to segmentation mask scale
            var scaledBoundingBox = DownscaleBoundingBoxToSegmentationOutput(bbox);

            // Clamp bounding box to valid pixel indices within mask dimensions
            int startX = Math.Max(0, (int)scaledBoundingBox.Left);
            int endX = Math.Min(_maskWidth - 1, (int)scaledBoundingBox.Right);

            int startY = Math.Max(0, (int)scaledBoundingBox.Top);
            int endY = Math.Min(_maskHeight - 1, (int)scaledBoundingBox.Bottom);

            // Iterate only over pixels inside the bounding box
            for (int y = startY; y <= endY; y++)
            {
                for (int x = startX; x <= endX; x++)
                {
                    float pixelWeight = 0;
                    int offset = x + y * _maskWidth;

                    // Sum weighted values across 32 channels (or mask layers)
                    for (int p = 0; p < 32; p++, offset += _maskWidth * _maskHeight)
                        pixelWeight += outputOrtSpan[offset] * maskWeights[p];

                    // Apply sigmoid to map raw score to [0,1] confidence
                    pixelWeight = YoloCore.Sigmoid(pixelWeight);

                    // Convert confidence to alpha (opacity) byte value (0-255)
                    byte alpha = (byte)(pixelWeight * 255);

                    // Set paint color to white with calculated alpha transparency
                    _paint.Color = new SKColor(255, 255, 255, alpha);

                    // Draw the pixel point on the canvas
                    canvas.DrawPoint(x, y, _paint);
                }
            }
        }

        unsafe private byte[] PackUpscaledMaskToBitArray(SKBitmap resizedBitmap, double confidenceThreshold)
        {
            IntPtr resizedPtr = resizedBitmap.GetPixels();
            byte* resizedPixelData = (byte*)resizedPtr.ToPointer();

            var totalPixels = resizedBitmap.Width * resizedBitmap.Height;
            var bytes = new byte[CalculateBitMaskSize(totalPixels)];

            // Use bit-packing to efficiently store 8 pixels per byte (1 bit per pixel), 
            // significantly reducing memory usage compared to storing each pixel individually.
            for (int i = 0; i < totalPixels; i++)
            {
                var pixel = resizedPixelData[i];

                var confidence = YoloCore.CalculatePixelConfidence(pixel);

                if (confidence > confidenceThreshold)
                {
                    // Map this pixel's index to its bit in the byte array:
                    // - byteIndex: the byte containing this pixel's bit (1 byte = 8 pixels)
                    // - bitIndex: the bit position within that byte (0-7)
                    int byteIndex = i >> 3;     // Same as i / 8 (fast using bit shift)
                    int bitIndex = i & 0b0111;  // Same as i % 8 (fast using bit mask)

                    // Set the bit to 1 to indicate the pixel is present (confidence > threshold)
                    // Bits remain 0 by default to indicate absence (confidence <= threshold)
                    bytes[byteIndex] |= (byte)(1 << bitIndex);
                }
            }

            return bytes;
        }

        private static int CalculateBitMaskSize(int totalPixels) => (totalPixels + 7) / 8;

        public void Dispose()
        {
            _objectDetectionModule?.Dispose();
            _yoloCore?.Dispose();

            _pixelMaskBitmap?.Dispose();
            _pixelMaskCanvas?.Dispose();
            _paint?.Dispose();

            GC.SuppressFinalize(this);
        }
    }
}