// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023-2025 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

namespace YoloDotNet.Modules.V8
{
    internal class SegmentationModuleV8 : ISegmentationModule
    {
        private readonly YoloCore _yoloCore;
        private readonly ObjectDetectionModuleV8 _objectDetectionModule;
        private readonly float _scalingFactorW;
        private readonly float _scalingFactorH;
        private readonly int _maskWidth;
        private readonly int _maskHeight;
        private readonly int _elements;
        private readonly int _predictions;
        private readonly int _outputShapeMaskChannels;
        private List<Segmentation> _results = default!;

        public OnnxModel OnnxModel => _yoloCore.OnnxModel;

        // Represents a fixed-size float buffer of 32 elements for mask weights.
        // Uses the InlineArray attribute to avoid heap allocations entirely.
        // This structure is stack-allocated when used inside methods or structs,
        // making it ideal for high-performance scenarios where per-frame allocations must be avoided.
        [InlineArray(32)]
        internal struct MaskWeights32
        {
            private float _mask;
        }

        public SegmentationModuleV8(YoloCore yoloCore)
        {
            _yoloCore = yoloCore;
            _objectDetectionModule = new ObjectDetectionModuleV8(_yoloCore);

            // Get input shape from ONNX model. Format NCHW: [Batch (B), Channels (C), Height (H), Width (W)]
            var inputShape = _yoloCore.OnnxModel.InputShapes.ElementAt(0).Value;

            // Get output shape from ONNX model. Format: [Batch, Attributes, Predictions]
            var outputShape = _yoloCore.OnnxModel.OutputShapes.ElementAt(0).Value;

            // Get output shape from ONNX model. Format: [Batch (B), Channels (C), Height (H), Width (W)]
            var outputShapeMask = _yoloCore.OnnxModel.OutputShapes.ElementAt(1).Value;

            // Get model pixel mask widh and height
            _maskHeight = outputShapeMask[2];
            _maskWidth = outputShapeMask[3];

            _elements = _yoloCore.OnnxModel.Labels.Length + 4; // 4 = the boundingbox dimension (x, y, width, height)
            _predictions = outputShape[2];
            _outputShapeMaskChannels = outputShapeMask[1];

            // Get model input width and height
            var inputHeight = inputShape[2];
            var inputWidth = inputShape[3];

            // Calculate scaling factor for downscaling boundingboxes to segmentation pixelmask proportions
            _scalingFactorW = (float)_maskWidth / inputWidth;
            _scalingFactorH = (float)_maskHeight / inputHeight;

            _results = [];
        }

        public List<Segmentation> ProcessImage<T>(T image, double confidence, double pixelConfidence, double iou)
        {
            var inferenceResult = _yoloCore.Run(image);
            var detections = RunSegmentation(inferenceResult, confidence, pixelConfidence, iou);

            // Convert to List<PoseEstimation>
            _results.Clear();
            for (int i = 0; i < detections.Length; i++)
                _results.Add((Segmentation)detections[i]);

            return _results;
        }

        private Span<ObjectResult> RunSegmentation(InferenceResult inferenceResult, double confidence, double pixelConfidence, double iou)
        {
            var boundingBoxes = _objectDetectionModule.ObjectDetection(inferenceResult, confidence, iou);

            var ortSpan0 = inferenceResult.OrtSpan0;
            var ortSpan1 = inferenceResult.OrtSpan1;

            foreach (var box in boundingBoxes)
            {
                var pixelMaskInfo = new SKImageInfo(box.BoundingBox.Width, box.BoundingBox.Height, SKColorType.Gray8, SKAlphaType.Opaque);
                var downScaledBoundingBox = DownscaleBoundingBoxToSegmentationOutput(box.BoundingBoxUnscaled);

                // 1) Get weights
                var maskWeights = GetMaskWeightsFromBoundingBoxArea(box, ortSpan0);

                // 2) Apply pixelmask based on mask-weights to canvas
                using var pixelMaskBitmap = new SKBitmap(_maskWidth, _maskHeight, SKColorType.Gray8, SKAlphaType.Opaque);
                ApplySegmentationPixelMask(pixelMaskBitmap, box.BoundingBoxUnscaled, ortSpan1, maskWeights);

                // 3) Crop downscaled boundingbox from the pixelmask canvas
                using var cropped = new SKBitmap();
                pixelMaskBitmap.ExtractSubset(cropped, downScaledBoundingBox);

                // 4) Upscale cropped pixelmask to original boundingbox size. For smother edges, use an appropriate resampling method!
                using var resizedCrop = new SKBitmap(pixelMaskInfo);

                // Use AVX2-optimized upscaling if supported; otherwise, fall back to SkiaSharp's ScalePixels.
                if (Avx2.IsSupported)
                    Avx2LinearResizer.ScalePixels(cropped, resizedCrop);
                else
                    cropped.ScalePixels(resizedCrop, ImageConfig.SegmentationResamplingOptions);

                // 5) Pack the upscaled pixel mask into a compact bit array (1 bit per pixel)
                // for cleaner, memory-efficient storage of the mask in the detection box.
                box.BitPackedPixelMask = PackUpscaledMaskToBitArray(resizedCrop, pixelConfidence);
            }

            return boundingBoxes;
        }

        private MaskWeights32 GetMaskWeightsFromBoundingBoxArea(ObjectResult box, ReadOnlySpan<float> ortSpan0)
        {
            MaskWeights32 maskWeights = default;

            var maskOffset = box.BoundingBoxIndex + (_predictions * _elements);

            for (var m = 0; m < _outputShapeMaskChannels; m++, maskOffset += _predictions)
                maskWeights[m] = ortSpan0[maskOffset];

            return maskWeights;
        }

        private SKRectI DownscaleBoundingBoxToSegmentationOutput(SKRect box)
        {
            int left = (int)Math.Floor(box.Left * _scalingFactorW);
            int top = (int)Math.Floor(box.Top * _scalingFactorH);
            int right = (int)Math.Ceiling(box.Right * _scalingFactorW);
            int bottom = (int)Math.Ceiling(box.Bottom * _scalingFactorH);

            // Clamp to mask bounds (important!)
            left = Math.Clamp(left, 0, _maskWidth - 1);
            top = Math.Clamp(top, 0, _maskHeight - 1);
            right = Math.Clamp(right, 0, _maskWidth - 1);
            bottom = Math.Clamp(bottom, 0, _maskHeight - 1);

            return new SKRectI(left, top, right, bottom);
        }

        unsafe void ApplySegmentationPixelMask(SKBitmap bitmap, SKRect bbox, ReadOnlySpan<float> outputOrtSpan, MaskWeights32 maskWeights)
        {
            var scaledBoundingBox = DownscaleBoundingBoxToSegmentationOutput(bbox);

            int startX = Math.Max(0, (int)scaledBoundingBox.Left);
            int endX = Math.Min(_maskWidth - 1, (int)scaledBoundingBox.Right);
            int startY = Math.Max(0, (int)scaledBoundingBox.Top);
            int endY = Math.Min(_maskHeight - 1, (int)scaledBoundingBox.Bottom);

            int stride = bitmap.RowBytes;
            byte* ptr = (byte*)bitmap.GetPixels().ToPointer();

            for (int y = startY; y <= endY; y++)
            {
                byte* row = ptr + y * stride;

                for (int x = startX; x <= endX; x++)
                {
                    float pixelWeight = 0;
                    int offset = x + y * _maskWidth;

                    for (int p = 0; p < 32; p++, offset += _maskWidth * _maskHeight)
                        pixelWeight += outputOrtSpan[offset] * maskWeights[p];

                    pixelWeight = YoloCore.Sigmoid(pixelWeight);
                    row[x] = (byte)(pixelWeight * 255); // write directly to Gray8 bitmap
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

            GC.SuppressFinalize(this);
        }
    }
}