// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2026 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

namespace YoloDotNet.Modules.RFDETR
{
    internal class SegmentationModuleRFDETR : ISegmentationModule
    {
        private const string BackgroundClassPrefix = "background_class";

        private readonly YoloCore _yoloCore;
        private readonly int _predictions;
        private readonly int _totalLabels;
        private readonly int _modelInputWidth;
        private readonly int _modelInputHeight;
        private readonly int _maskWidth;
        private readonly int _maskHeight;
        private readonly int _maskPlaneSize;
        private readonly int _backgroundClassIndex;
        private readonly bool _hasBackgroundClass;
        private readonly LabelModel[] _labels;
        private int[] _topIndices = [];
        private float[] _topScores = [];
        private Candidate[] _candidates = [];
        private List<Segmentation> _results = default!;

        public event EventHandler VideoProgressEvent = delegate { };
        public event EventHandler VideoCompleteEvent = delegate { };
        public event EventHandler VideoStatusEvent = delegate { };

        public OnnxModel OnnxModel => _yoloCore.OnnxModel;

        public SegmentationModuleRFDETR(YoloCore yoloCore)
        {
            _yoloCore = yoloCore;

            var inputShape = _yoloCore.OnnxModel.InputShapes.ElementAt(0).Value;
            var detsShape = _yoloCore.OnnxModel.OutputShapes["dets"];
            var labelsShape = _yoloCore.OnnxModel.OutputShapes["labels"];
            var masksShape = _yoloCore.OnnxModel.OutputShapes["masks"];

            _modelInputHeight = (int)inputShape[2];
            _modelInputWidth = (int)inputShape[3];
            _predictions = detsShape[1];
            _totalLabels = labelsShape[2];
            _maskHeight = masksShape[2];
            _maskWidth = masksShape[3];
            _maskPlaneSize = _maskWidth * _maskHeight;
            _backgroundClassIndex = Array.FindIndex(
                _yoloCore.OnnxModel.Labels,
                label => label.Name.StartsWith(BackgroundClassPrefix, StringComparison.OrdinalIgnoreCase));
            _hasBackgroundClass = _backgroundClassIndex >= 0;
            _labels = _hasBackgroundClass
                ? [.. _yoloCore.OnnxModel.Labels
                    .Where(label => label.Index != _backgroundClassIndex)
                    .Select((label, index) => label with { Index = index })]
                : _yoloCore.OnnxModel.Labels;
            _results = [];
        }

        public List<Segmentation> ProcessImage<T>(T image, double confidence, double pixelConfidence, double iou, SKRectI? roi = null)
        {
            var inferenceResult = _yoloCore.Run(image, roi);
            var resultBuffer = ArrayPool<ObjectResult>.Shared.Rent(_predictions);

            try
            {
                var count = RunSegmentation(inferenceResult, confidence, resultBuffer);
                return YoloCore.InferenceResultsToType(resultBuffer.AsSpan(0, count), roi, _results, r => (Segmentation)r);
            }
            finally
            {
                ArrayPool<ObjectResult>.Shared.Return(resultBuffer, false);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private int RunSegmentation(InferenceResult inferenceResult, double confidenceThreshold, ObjectResult[] resultBuffer)
        {
            var imageSize = inferenceResult.ImageOriginalSize;
            var boxesSpan = inferenceResult.OrtSpan0;
            var logitsSpan = inferenceResult.OrtSpan1;
            var masksSpan = inferenceResult.OrtSpan2;

            if (boxesSpan.IsEmpty || logitsSpan.IsEmpty || masksSpan.IsEmpty)
                return 0;

            var (xPad, yPad, xGain, yGain) = _yoloCore.CalculateGain(imageSize);
            var (topIndices, topScores) = GetTopKBuffers(_predictions);
            var scoreCount = Math.Min(logitsSpan.Length, _predictions * _totalLabels);
            var topCount = YoloCore.SelectTopIndices(logitsSpan[..scoreCount], _predictions, topIndices, topScores);
            var crop = CreateMaskCrop(imageSize);
            EnsureCandidates(topCount);
            var candidateCount = 0;

            for (var candidateIndex = 0; candidateIndex < topCount; candidateIndex++)
            {
                var candidateConfidence = YoloCore.Sigmoid(topScores[candidateIndex]);

                if (candidateConfidence < confidenceThreshold)
                    break;

                var flatIndex = topIndices[candidateIndex];
                var prediction = flatIndex / _totalLabels;
                var labelIndex = flatIndex - prediction * _totalLabels;

                if (_hasBackgroundClass && labelIndex == _backgroundClassIndex)
                    continue;

                var mappedLabelIndex = MapLabelIndex(labelIndex);

                if (mappedLabelIndex < 0 || mappedLabelIndex >= _labels.Length)
                    continue;

                var boxOffset = prediction * 4;
                var cx = boxesSpan[boxOffset];
                var cy = boxesSpan[boxOffset + 1];
                var w = boxesSpan[boxOffset + 2];
                var h = boxesSpan[boxOffset + 3];

                var xMin = cx - w * 0.5f;
                var yMin = cy - h * 0.5f;
                var xMax = cx + w * 0.5f;
                var yMax = cy + h * 0.5f;

                SKRectI boundingBox;
                SKRect boundingBoxUnscaled;

                if (_yoloCore.YoloOptions.ImageResize == ImageResize.Proportional)
                {
                    var scaledXMin = xMin * _modelInputWidth - xPad;
                    var scaledYMin = yMin * _modelInputHeight - yPad;
                    var scaledXMax = xMax * _modelInputWidth - xPad;
                    var scaledYMax = yMax * _modelInputHeight - yPad;

                    boundingBox = ClampBox(
                        scaledXMin * xGain,
                        scaledYMin * xGain,
                        scaledXMax * xGain,
                        scaledYMax * xGain,
                        imageSize);
                    boundingBoxUnscaled = new SKRect(scaledXMin, scaledYMin, scaledXMax, scaledYMax);
                }
                else
                {
                    boundingBox = ClampBox(
                        xMin * imageSize.Width,
                        yMin * imageSize.Height,
                        xMax * imageSize.Width,
                        yMax * imageSize.Height,
                        imageSize);
                    boundingBoxUnscaled = new SKRect(
                        xMin * _modelInputWidth,
                        yMin * _modelInputHeight,
                        xMax * _modelInputWidth,
                        yMax * _modelInputHeight);
                }

                if (boundingBox.Width <= 0 || boundingBox.Height <= 0)
                    continue;

                _candidates[candidateCount++] = new Candidate(
                    prediction,
                    mappedLabelIndex,
                    candidateConfidence,
                    boundingBox,
                    boundingBoxUnscaled);
            }

            if (candidateCount == 0)
                return 0;

            // Official RF-DETR keeps a mask pixel when bilinear sampling (align_corners=False) is > 0.
            // The sample grid is capped so a detection that fills the frame does not add work.
            var masks = ArrayPool<byte[]>.Shared.Rent(candidateCount);
            var maskWidths = ArrayPool<int>.Shared.Rent(candidateCount);
            var maskHeights = ArrayPool<int>.Shared.Rent(candidateCount);

            try
            {
                unsafe
                {
                    fixed (float* maskBase = masksSpan)
                    {
                        var maskAddress = (IntPtr)maskBase;
                        var maskLength = masksSpan.Length;

                        if (candidateCount == 1)
                        {
                            PackCandidate(maskAddress, maskLength, crop, 0, masks, maskWidths, maskHeights);
                        }
                        else
                        {
                            Parallel.For(0, candidateCount, index =>
                                PackCandidate(maskAddress, maskLength, crop, index, masks, maskWidths, maskHeights));
                        }
                    }
                }

                for (var i = 0; i < candidateCount; i++)
                {
                    var candidate = _candidates[i];
                    resultBuffer[i] = new ObjectResult
                    {
                        Label = _labels[candidate.LabelIndex],
                        Confidence = candidate.Confidence,
                        BoundingBox = candidate.Box,
                        BoundingBoxUnscaled = candidate.Unscaled,
                        BoundingBoxIndex = candidate.Prediction,
                        BitPackedPixelMask = masks[i] ?? [],
                        MaskWidth = maskWidths[i],
                        MaskHeight = maskHeights[i]
                    };
                }
            }
            finally
            {
                ArrayPool<byte[]>.Shared.Return(masks, clearArray: true);
                ArrayPool<int>.Shared.Return(maskWidths, clearArray: false);
                ArrayPool<int>.Shared.Return(maskHeights, clearArray: false);
            }

            return candidateCount;
        }

        private MaskCrop CreateMaskCrop(SKSizeI imageSize)
        {
            var cropLeft = 0;
            var cropTop = 0;
            var cropRight = _maskWidth;
            var cropBottom = _maskHeight;

            if (_yoloCore.YoloOptions.ImageResize != ImageResize.Stretched
                && imageSize.Width > 0
                && imageSize.Height > 0
                && _maskWidth > 0
                && _maskHeight > 0)
            {
                var scale = MathF.Min((float)_modelInputWidth / imageSize.Width, (float)_modelInputHeight / imageSize.Height);
                var scaledWidth = (int)(imageSize.Width * scale);
                var scaledHeight = (int)(imageSize.Height * scale);
                var padX = (_modelInputWidth - scaledWidth) * 0.5f;
                var padY = (_modelInputHeight - scaledHeight) * 0.5f;

                cropLeft = Math.Clamp((int)MathF.Round(padX * _maskWidth / _modelInputWidth), 0, _maskWidth - 1);
                cropTop = Math.Clamp((int)MathF.Round(padY * _maskHeight / _modelInputHeight), 0, _maskHeight - 1);
                cropRight = Math.Clamp((int)MathF.Round((padX + scaledWidth) * _maskWidth / _modelInputWidth), cropLeft + 1, _maskWidth);
                cropBottom = Math.Clamp((int)MathF.Round((padY + scaledHeight) * _maskHeight / _modelInputHeight), cropTop + 1, _maskHeight);
            }

            var cropWidth = Math.Max(0, cropRight - cropLeft);
            var cropHeight = Math.Max(0, cropBottom - cropTop);
            return new MaskCrop(
                cropLeft,
                cropTop,
                cropWidth,
                cropHeight,
                imageSize.Width > 0 ? (float)cropWidth / imageSize.Width : 0,
                imageSize.Height > 0 ? (float)cropHeight / imageSize.Height : 0);
        }

        private void PackCandidate(
            IntPtr maskAddress,
            int maskLength,
            MaskCrop crop,
            int index,
            byte[][] masks,
            int[] maskWidths,
            int[] maskHeights)
        {
            var candidate = _candidates[index];
            var maskOffset = candidate.Prediction * _maskPlaneSize;
            if (candidate.Box.Width <= 0
                || candidate.Box.Height <= 0
                || crop.Width <= 0
                || crop.Height <= 0
                || crop.XScale <= 0
                || crop.YScale <= 0
                || maskOffset < 0
                || _maskPlaneSize <= 0
                || maskLength < _maskPlaneSize
                || maskOffset > maskLength - _maskPlaneSize)
            {
                masks[index] = [];
                maskWidths[index] = 0;
                maskHeights[index] = 0;
                return;
            }

            int packedWidth;
            int packedHeight;
            unsafe
            {
                var packed = PackBilinearMask((float*)maskAddress, maskOffset, candidate.Box, crop, out packedWidth, out packedHeight);
                if (!HasAnyBit(packed))
                {
                    masks[index] = [];
                    maskWidths[index] = 0;
                    maskHeights[index] = 0;
                    return;
                }

                masks[index] = packed;
            }

            maskWidths[index] = packedWidth;
            maskHeights[index] = packedHeight;
        }

        const int MaxPackedEdge = 384;

        private unsafe byte[] PackBilinearMask(
            float* masks,
            int maskOffset,
            SKRectI box,
            MaskCrop crop,
            out int packedWidth,
            out int packedHeight)
        {
            (packedWidth, packedHeight) = PackedSize(box);
            var stepX = box.Width / (float)packedWidth;
            var stepY = box.Height / (float)packedHeight;
            var packed = new byte[(packedWidth * packedHeight + 7) / 8];
            var pixel = 0;

            fixed (byte* destination = packed)
            {
                for (var y = 0; y < packedHeight; y++)
                {
                    var rawY = (box.Top + (y + 0.5f) * stepY) * crop.YScale - 0.5f;
                    if (rawY < 0)
                        rawY = 0;

                    var y0 = (int)rawY;
                    if (y0 >= crop.Height)
                        y0 = crop.Height - 1;
                    if (y0 < 0)
                        y0 = 0;
                    var y1 = y0 < crop.Height - 1 ? y0 + 1 : y0;
                    var yWeight = y1 == y0 ? 0f : rawY - y0;
                    var row0 = masks + maskOffset + (crop.Top + y0) * _maskWidth + crop.Left;
                    var row1 = masks + maskOffset + (crop.Top + y1) * _maskWidth + crop.Left;

                    for (var x = 0; x < packedWidth; x++)
                    {
                        var rawX = (box.Left + (x + 0.5f) * stepX) * crop.XScale - 0.5f;
                        if (rawX < 0)
                            rawX = 0;

                        var x0 = (int)rawX;
                        if (x0 >= crop.Width)
                            x0 = crop.Width - 1;
                        if (x0 < 0)
                            x0 = 0;
                        var x1 = x0 < crop.Width - 1 ? x0 + 1 : x0;
                        var xWeight = x1 == x0 ? 0f : rawX - x0;
                        var top = row0[x0] + (row0[x1] - row0[x0]) * xWeight;
                        var bottom = row1[x0] + (row1[x1] - row1[x0]) * xWeight;
                        if (top + (bottom - top) * yWeight <= 0)
                            continue;

                        var index = pixel + x;
                        destination[index >> 3] |= (byte)(1 << (index & 7));
                    }

                    pixel += packedWidth;
                }
            }

            return packed;
        }

        private static (int Width, int Height) PackedSize(SKRectI box)
        {
            var longest = Math.Max(box.Width, box.Height);
            if (longest <= MaxPackedEdge)
                return (box.Width, box.Height);

            var scale = MaxPackedEdge / (float)longest;
            return (
                Math.Max(1, (int)MathF.Round(box.Width * scale)),
                Math.Max(1, (int)MathF.Round(box.Height * scale)));
        }

        private static bool HasAnyBit(byte[] packed)
        {
            foreach (var value in packed)
            {
                if (value != 0)
                    return true;
            }

            return false;
        }

        private (int[] TopIndices, float[] TopScores) GetTopKBuffers(int size)
        {
            if (_topIndices.Length < size)
            {
                _topIndices = new int[size];
                _topScores = new float[size];
            }

            return (_topIndices, _topScores);
        }

        private void EnsureCandidates(int size)
        {
            if (_candidates.Length < size)
                _candidates = new Candidate[size];
        }

        private int MapLabelIndex(int labelIndex)
        {
            if (!_hasBackgroundClass)
                return labelIndex;

            return labelIndex > _backgroundClassIndex
                ? labelIndex - 1
                : labelIndex;
        }

        private static SKRectI ClampBox(float xMin, float yMin, float xMax, float yMax, SKSizeI imageSize)
        {
            var left = Math.Clamp((int)xMin, 0, imageSize.Width - 1);
            var top = Math.Clamp((int)yMin, 0, imageSize.Height - 1);
            var right = Math.Clamp((int)xMax, 0, imageSize.Width - 1);
            var bottom = Math.Clamp((int)yMax, 0, imageSize.Height - 1);

            return new SKRectI(left, top, right, bottom);
        }

        private readonly struct Candidate(
            int prediction,
            int labelIndex,
            float confidence,
            SKRectI box,
            SKRect unscaled)
        {
            public readonly int Prediction = prediction;
            public readonly int LabelIndex = labelIndex;
            public readonly float Confidence = confidence;
            public readonly SKRectI Box = box;
            public readonly SKRect Unscaled = unscaled;
        }

        private readonly struct MaskCrop(
            int left,
            int top,
            int width,
            int height,
            float xScale,
            float yScale)
        {
            public readonly int Left = left;
            public readonly int Top = top;
            public readonly int Width = width;
            public readonly int Height = height;
            public readonly float XScale = xScale;
            public readonly float YScale = yScale;
        }

        public void Dispose()
        {
            _yoloCore?.Dispose();
            GC.SuppressFinalize(this);
        }
    }
}
