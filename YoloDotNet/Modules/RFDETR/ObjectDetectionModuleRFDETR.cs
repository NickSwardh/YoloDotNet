// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2026 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

namespace YoloDotNet.Modules.RFDETR
{
    internal class ObjectDetectionModuleRFDETR : IObjectDetectionModule
    {
        private const string BackgroundClassPrefix = "background_class";

        private readonly YoloCore _yoloCore;
        private readonly int _predictions;
        private readonly int _totalLabels;
        private readonly int _modelInputWidth;
        private readonly int _modelInputHeight;
        private readonly int _backgroundClassIndex;
        private readonly bool _hasBackgroundClass;
        private readonly LabelModel[] _labels;
        private int[] _topIndices = [];
        private float[] _topScores = [];
        private List<ObjectDetection> _results = default!;

        public event EventHandler VideoProgressEvent = delegate { };
        public event EventHandler VideoCompleteEvent = delegate { };
        public event EventHandler VideoStatusEvent = delegate { };

        public OnnxModel OnnxModel => _yoloCore.OnnxModel;

        public ObjectDetectionModuleRFDETR(YoloCore yoloCore)
        {
            _yoloCore = yoloCore;

            var inputShape = _yoloCore.OnnxModel.InputShapes.ElementAt(0).Value;
            var detsShape = _yoloCore.OnnxModel.OutputShapes["dets"];
            var labelsShape = _yoloCore.OnnxModel.OutputShapes["labels"];

            _modelInputHeight = (int)inputShape[2];
            _modelInputWidth = (int)inputShape[3];
            _predictions = detsShape[1];
            _totalLabels = labelsShape[2];
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

        public List<ObjectDetection> ProcessImage<T>(T image, double confidence, double pixelConfidence, double iou, SKRectI? roi = null)
        {
            var inferenceResult = _yoloCore.Run(image, roi);
            var resultBuffer = ArrayPool<ObjectResult>.Shared.Rent(Math.Max(_predictions, 1));

            try
            {
                var count = DecodeDetections(inferenceResult, confidence, resultBuffer);
                return YoloCore.InferenceResultsToType(resultBuffer.AsSpan(0, count), roi, _results, r => (ObjectDetection)r);
            }
            finally
            {
                ArrayPool<ObjectResult>.Shared.Return(resultBuffer, false);
            }
        }

        private int DecodeDetections(InferenceResult inferenceResult, double confidenceThreshold, ObjectResult[] resultBuffer)
        {
            var imageSize = inferenceResult.ImageOriginalSize;
            var boxesSpan = inferenceResult.OrtSpan0;
            var logitsSpan = inferenceResult.OrtSpan1;

            if (boxesSpan.IsEmpty || logitsSpan.IsEmpty || resultBuffer.Length == 0)
                return 0;

            var (xPad, yPad, xGain, yGain) = _yoloCore.CalculateGain(imageSize);
            var validBoxCount = 0;
            var (topIndices, topScores) = GetTopKBuffers(_predictions);
            var scoreCount = Math.Min(logitsSpan.Length, _predictions * _totalLabels);
            var topCount = YoloCore.SelectTopIndices(logitsSpan[..scoreCount], _predictions, topIndices, topScores);

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

                resultBuffer[validBoxCount++] = new ObjectResult
                {
                    Label = _labels[mappedLabelIndex],
                    Confidence = candidateConfidence,
                    BoundingBox = boundingBox,
                    BoundingBoxUnscaled = boundingBoxUnscaled,
                    BoundingBoxIndex = prediction
                };
            }

            return validBoxCount;
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

        public void Dispose()
        {
            _yoloCore?.Dispose();
            GC.SuppressFinalize(this);
        }
    }
}
