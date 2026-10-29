// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023-2026 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

namespace YoloDotNet.Core
{
    /// <summary>
    /// Initializes a new instance of the Yolo core class.
    /// </summary>
    internal class YoloCore(YoloOptions yoloOptions) : IDisposable
    {
        #region Private Fields
        private PinnedMemoryBufferPool _pinnedMemoryPool = default!;
        private readonly object _progressLock = new();
        private SKImageInfo _imageInfo;
        private long[] _inputShape = default!;
        private int _modelInputWidth;
        private int _modelInputHeight;
        private int _inputShapeSize;
        private float[] _floatInputBuffer = [];
        private ushort[] _halfInputBuffer = [];
        #endregion

        #region Properties
        public OnnxModel OnnxModel => yoloOptions.ExecutionProvider.OnnxData;
        public YoloOptions YoloOptions { get => yoloOptions; init => yoloOptions = value; }
        public ModelType ModelType => OnnxModel.ModelType;
        #endregion

        /// <summary>
        /// Initializes the YOLO model with the specified model type.
        /// </summary>
        public void InitializeYolo()
        {
            if (YoloOptions.ExecutionProvider is null)
                throw new YoloDotNetModelException("Execution Provider is missing. Please add an execution provider.", nameof(YoloOptions));

            _inputShape = OnnxModel.InputShapes.First().Value;
            _inputShapeSize = OnnxModel.InputShapeSize;
            _modelInputHeight = (int)_inputShape[2];
            _modelInputWidth = (int)_inputShape[3];

            VerifyExpectedModelType(OnnxModel.ModelType);

            var format = (_inputShape[2] == 1) ? SKColorType.Gray8 : SKColorType.Rgb888x;
            _imageInfo = new SKImageInfo(_modelInputWidth, _modelInputHeight, format, SKAlphaType.Opaque);

            _pinnedMemoryPool = new PinnedMemoryBufferPool(_imageInfo);

            if (OnnxModel.ModelDataType == ModelDataType.Float16)
                _halfInputBuffer = GC.AllocateUninitializedArray<ushort>(_inputShapeSize, pinned: true);
            else
                _floatInputBuffer = GC.AllocateUninitializedArray<float>(_inputShapeSize, pinned: true);

            FrameSaveService.Start();
        }

        /// <summary>
        /// Runs the YOLO model on the provided image and returns the inference results.
        /// </summary>
        /// <param name="image">The input image to process.</param>
        /// <param name="roi">Optional region of interest.</param>
        /// <param name="compress">为 false 时跳过缩放。调用方已将图片处理成模型输入尺寸时使用。</param>
        /// <returns>InferenceResult()</returns>
        public InferenceResult Run<T>(T image, SKRectI? roi, bool compress = true)
        {
            lock (_progressLock)
            {
                var sizeMatches = MatchesModelInputSize(image, roi, _modelInputWidth, _modelInputHeight);
                var skipResize = !compress || sizeMatches;

                // 尺寸已经对齐时直接写成 NCHW，不再经过 Canvas。
                if (sizeMatches && TryNormalizeMatched(image, roi, out var directSize))
                {
                    var directResult = ExecuteInference();
                    directResult.ImageOriginalSize = directSize;
                    return directResult;
                }

                var pinnedBuffer = _pinnedMemoryPool.Rent();

                try
                {
                    // 已是模型输入尺寸，或调用方明确关闭压缩时，按原像素写入，不再缩放。
                    var originalImageSize = skipResize
                        ? image.CopyImage(pinnedBuffer, roi)
                        : YoloOptions.ImageResize == ImageResize.Proportional
                            ? image.ResizeImageProportional(YoloOptions.SamplingOptions, pinnedBuffer, roi)
                            : image.ResizeImageStretched(YoloOptions.SamplingOptions, pinnedBuffer, roi);

                    if (OnnxModel.ModelDataType == ModelDataType.Float16)
                        pinnedBuffer.Pointer.NormalizePixelsToArray(_inputShape, _inputShapeSize, _halfInputBuffer, YoloOptions.ImageMean, YoloOptions.ImageStd);
                    else
                        pinnedBuffer.Pointer.NormalizePixelsToArray(_inputShape, _inputShapeSize, _floatInputBuffer, YoloOptions.ImageMean, YoloOptions.ImageStd);

                    var inferenceResult = ExecuteInference();
                    inferenceResult.ImageOriginalSize = originalImageSize;
                    return inferenceResult;
                }
                finally
                {
                    _pinnedMemoryPool.Return(pinnedBuffer, clear: !skipResize && YoloOptions.ImageResize == ImageResize.Proportional);
                }
            }
        }

        private bool TryNormalizeMatched<T>(T image, SKRectI? roi, out SKSizeI sourceSize)
        {
            var channels = (int)_inputShape[1];
            var mean = YoloOptions.ImageMean;
            var std = YoloOptions.ImageStd;

            if (OnnxModel.ModelDataType == ModelDataType.Float16)
            {
                return image.TryNormalizeMatched(
                    _modelInputWidth,
                    _modelInputHeight,
                    channels,
                    _halfInputBuffer,
                    out sourceSize,
                    roi,
                    mean,
                    std);
            }

            return image.TryNormalizeMatched(
                _modelInputWidth,
                _modelInputHeight,
                channels,
                _floatInputBuffer,
                out sourceSize,
                roi,
                mean,
                std);
        }

        private InferenceResult ExecuteInference()
        {
            if (OnnxModel.ModelDataType == ModelDataType.Float16)
                return YoloOptions.ExecutionProvider.Run<ushort>(_halfInputBuffer);

            return YoloOptions.ExecutionProvider.Run<float>(_floatInputBuffer);
        }

        /// <summary>
        /// 用最小堆从 <paramref name="scores"/> 选出最大的 <paramref name="keep"/> 个，并按分数从高到低写回。
        /// 分数保持原值，调用方再决定是否做 Sigmoid。
        /// </summary>
        internal static int SelectTopIndices(ReadOnlySpan<float> scores, int keep, int[] indices, float[] selected)
        {
            var count = scores.Length;
            if (count == 0 || keep <= 0 || indices.Length == 0 || selected.Length == 0)
                return 0;

            var k = Math.Min(keep, Math.Min(count, Math.Min(indices.Length, selected.Length)));

            for (var i = 0; i < k; i++)
            {
                indices[i] = i;
                selected[i] = scores[i];
            }

            for (var i = k / 2 - 1; i >= 0; i--)
                SiftMin(selected, indices, k, i);

            for (var i = k; i < count; i++)
            {
                var score = scores[i];
                if (score <= selected[0])
                    continue;

                selected[0] = score;
                indices[0] = i;
                SiftMin(selected, indices, k, 0);
            }

            for (var end = k - 1; end > 0; end--)
            {
                (selected[0], selected[end]) = (selected[end], selected[0]);
                (indices[0], indices[end]) = (indices[end], indices[0]);
                SiftMin(selected, indices, end, 0);
            }

            return k;
        }

        private static void SiftMin(float[] scores, int[] indices, int length, int parent)
        {
            while (true)
            {
                var left = parent * 2 + 1;
                if (left >= length)
                    return;

                var right = left + 1;
                var smallest = right < length && scores[right] < scores[left] ? right : left;
                if (scores[parent] <= scores[smallest])
                    return;

                (scores[parent], scores[smallest]) = (scores[smallest], scores[parent]);
                (indices[parent], indices[smallest]) = (indices[smallest], indices[parent]);
                parent = smallest;
            }
        }

        #region Helper methods

        /// <summary>
        /// 源图（或 ROI）宽高已经等于模型输入时，不需要再缩放。
        /// </summary>
        private static bool MatchesModelInputSize<T>(T image, SKRectI? roi, int modelWidth, int modelHeight)
        {
            int width;
            int height;

            if (roi.HasValue)
            {
                width = roi.Value.Width;
                height = roi.Value.Height;
            }
            else if (image is SKBitmap bitmap)
            {
                width = bitmap.Width;
                height = bitmap.Height;
            }
            else if (image is SKImage skImage)
            {
                width = skImage.Width;
                height = skImage.Height;
            }
            else
            {
                return false;
            }

            return width == modelWidth && height == modelHeight;
        }

        /// <summary>
        /// Removes overlapping bounding boxes in a list of object detection results.
        /// </summary>
        /// <param name="predictionSpan">A span with predition results</param>
        /// <param name="iouThreshold">Higher Iou-threshold result in fewer detections by excluding overlapping boxes.</param>
        /// <returns>A filtered Span<ObjectResult> with non-overlapping bounding boxes based on confidence scores.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public Span<ObjectResult> RemoveOverlappingBoxes(Span<ObjectResult> predictionSpan, double iouThreshold)
        {
            var totalPredictions = predictionSpan.Length;

            if (totalPredictions == 0)
                return Span<ObjectResult>.Empty; // ✓ Avoid array allocation

            // Sort by confidence
            MemoryExtensions.Sort(predictionSpan, ConfidenceComparer.Instance);

            var buffer = ArrayPool<ObjectResult>.Shared.Rent(totalPredictions);

            try
            {
                var counter = 0;
                for (int i = 0; i < totalPredictions; i++)
                {
                    var item = predictionSpan[i];

                    bool overlapFound = false;
                    for (int j = 0; j < counter; j++)
                    {
                        if (CalculateIoU(item.BoundingBox, buffer[j].BoundingBox) > iouThreshold)
                        {
                            overlapFound = true;
                            break;
                        }
                    }

                    if (!overlapFound)
                        buffer[counter++] = item;
                }
                return buffer.AsSpan(0, counter);
            }
            finally
            {
                ArrayPool<ObjectResult>.Shared.Return(buffer, false);
            }
        }

        /// <summary>
        /// Calculate buffer pool size as the next power of two to ensure array pool efficiency.
        /// </summary>
        public static int CalculateBufferPoolSize(int bufferSize) => 1 << (int)Math.Ceiling(Math.Log2(bufferSize));

        /// <summary>
        /// Squash value to a number between 0 and 1
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Sigmoid(float value) => 1 / (1 + MathF.Exp(-value));

        /// <summary>
        /// Calculate pixel luminance
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static byte CalculatePixelLuminance(float value) => (byte)(255 - value * 255);

        /// <summary>
        /// Calculate pixel by byte to confidence
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float CalculatePixelConfidence(byte value) => value / 255F;

        /// <summary>
        /// Calculate radian to degree
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float CalculateRadianToDegree(float value) => value * (180 / (float)Math.PI);

        /// <summary>
        /// Calculates the Intersection over Union (IoU) between two rectangles.
        /// </summary>
        /// <param name="a"></param>
        /// <param name="b"></param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float CalculateIoU(in SKRectI a, in SKRectI b)
        {
            // Use "in" keyword on parameters to pass by reference without allowing modification for better performance.

            int left = Math.Max(a.Left, b.Left);
            int top = Math.Max(a.Top, b.Top);
            int right = Math.Min(a.Right, b.Right);
            int bottom = Math.Min(a.Bottom, b.Bottom);

            int width = right - left;
            int height = bottom - top;

            if (width <= 0 || height <= 0)
                return 0f;

            int intersection = width * height;

            int areaA = (a.Right - a.Left) * (a.Bottom - a.Top);
            int areaB = (b.Right - b.Left) * (b.Bottom - b.Top);

            return (float)intersection / (areaA + areaB - intersection);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public (float, float, float, float) CalculateGain(SKSizeI size)
            => YoloOptions.ImageResize == ImageResize.Proportional ? CalculateProportionalGain(size) : CalculateStretchedGain(size);

        /// <summary>
        /// Calculates the padding and scaling factor needed to adjust the bounding box
        /// so that the detected object can be resized to match the original image size.
        /// </summary>
        /// <param name="size">The original image size.</param>
        public (float, float, float, float) CalculateProportionalGain(SKSizeI size)
        {
            int w = size.Width;
            int h = size.Height;

            float modelW = _modelInputWidth;
            float modelH = _modelInputHeight;

            float xPad, yPad;

            // If image is smaller than model input, it was centered without scaling.
            // In this case, gain is 1.0 and padding is half the difference between model input and image size.
            if (w < _modelInputWidth && h < _modelInputHeight)
            {
                xPad = (modelW - w) * 0.5f;
                yPad = (modelH - h) * 0.5f;

                return (xPad, yPad, 1.0f, 0f);  // gain = 1.0 (no scaling)
            }

            // compute scales as floats and use MathF to avoid double/float conversions
            float scaleW = w / modelW; // (float)w / model.Input.Width
            float scaleH = h / modelH; // (float)h / model.Input.Height

            float gain = MathF.Max(scaleW, scaleH);

            // ratio is how source maps into model dimensions
            float ratio = MathF.Min(modelW / w, modelH / h);

            xPad = (modelW - w * ratio) * 0.5f;
            yPad = (modelH - h * ratio) * 0.5f;

            return (xPad, yPad, gain, 0f);
        }

        /// <summary>
        /// Calculates the padding and scaling factor needed to adjust the bounding box
        /// so that the detected object can be resized to match the original image size.
        /// </summary>
        /// <param name="size">The original image size.</param>
        public (float, float, float, float) CalculateStretchedGain(SKSizeI size)
        {
            int w = size.Width;
            int h = size.Height;

            float modelW = _modelInputWidth;
            float modelH = _modelInputHeight;

            float xPad, yPad;

            // If image is smaller than model input, it was centered without scaling.
            // In this case, gains are 1.0 and padding is half the difference between model input and image size.
            if (w < _modelInputWidth && h < _modelInputHeight)
            {
                xPad = (modelW - w) * 0.5f;
                yPad = (modelH - h) * 0.5f;

                return (xPad, yPad, 1.0f, 1.0f);  // gains = 1.0 (no scaling)
            }

            float xGain = modelW / w;
            float yGain = modelH / h;

            xPad = (modelW - w * xGain) * 0.5f;
            yPad = (modelH - h * yGain) * 0.5f;

            return (xPad, yPad, xGain, yGain);
        }

        /// <summary>
        /// Crops the SKBitmap to the specified ROI.
        /// </summary>
        internal static SKImage CropToRoi(SKBitmap image, SKRectI roi)
        {
            using var cropped = new SKBitmap(roi.Width, roi.Height, image.ColorType, image.AlphaType);
            using var canvas = new SKCanvas(cropped);

            canvas.DrawBitmap(image, roi, new SKRect(0, 0, roi.Width, roi.Height));

            return SKImage.FromBitmap(cropped);
        }

        /// <summary>
        /// Crops the SKImage to the specified ROI.
        /// </summary>
        internal static SKImage CropToRoi(SKImage image, SKRectI roi)
            => image.Subset(roi) ?? throw new InvalidOperationException("Failed to create image subset.");

        /// <summary>
        /// Transforms a bounding box from ROI-relative coordinates to original image coordinates.
        /// </summary>
        internal static SKRectI TransformRoiBoundingBox(SKRectI box, SKRectI roi)
            => new(
                box.Left + roi.Left,
                box.Top + roi.Top,
                box.Right + roi.Left,
                box.Bottom + roi.Top
            );

        /// <summary>
        /// Transforms keypoints from ROI-relative coordinates to original image coordinates.
        /// </summary>
        internal static KeyPoint[] TransformKeyPoints(KeyPoint[] keyPoints, SKRectI roi)
            => keyPoints.Select(kp => kp with
            {
                X = kp.X + roi.Left,
                Y = kp.Y + roi.Top
            }).ToArray();

        /// <summary>
        /// Verify that loaded model is of the expected type
        /// </summary>
        public void VerifyExpectedModelType(ModelType expectedModelType)
        {
            if (expectedModelType.Equals(OnnxModel.ModelType) is false)
                throw new YoloDotNetModelMismatchException($"Loaded ONNX-model is of type {OnnxModel.ModelType} and can't be used for {expectedModelType}.");
        }

        /// <summary>
        /// Converts a collection of object detection results to a list of the specified type, optionally transforming
        /// bounding boxes and keypoints based on a region of interest.
        /// </summary>
        internal static List<T> InferenceResultsToType<T>(
            Span<ObjectResult> detections,
            SKRectI? roi,
            List<T> results,
            Func<ObjectResult, T> converter)
        {
            results.Clear();

            // If ROI is provided, transform bounding boxes and keypoints back to original image coordinates before converting to the desired type.
            if (roi.HasValue)
            {
                var roiValue = roi.Value;
                for (int i = 0; i < detections.Length; i++)
                {
                    detections[i].BoundingBox = TransformRoiBoundingBox(detections[i].BoundingBox, roiValue);

                    // Transform pose estimation keypoints back to original image coordinates.
                    if (typeof(T) == typeof(PoseEstimation))
                    {
                        detections[i].KeyPoints = TransformKeyPoints(detections[i].KeyPoints, roiValue);
                    }

                    results.Add(converter(detections[i]));
                }
            }
            else
            {
                for (int i = 0; i < detections.Length; i++)
                    results.Add(converter(detections[i]));
            }

            return results;
        }

        /// <summary>
        /// Releases resources and suppresses the finalizer for the current object.
        /// </summary>
        public void Dispose()
        {
            _pinnedMemoryPool?.Dispose();

            GC.SuppressFinalize(this);
        }

        #endregion
    }
}
