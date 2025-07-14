// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2023-2025 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

namespace YoloDotNet.Modules.V8
{
    internal class ClassificationModuleV8 : IClassificationModule
    {
        private readonly YoloCore _yoloCore;

        public OnnxModel OnnxModel => _yoloCore.OnnxModel;

        public ClassificationModuleV8(YoloCore yoloCore)
        {
            _yoloCore = yoloCore;
        }

        public List<Classification> ProcessImage<T>(T image, double classes, double pixelConfidence, double iou)
        {
            var (ortValues, _) = _yoloCore.Run(image);

            using IDisposableReadOnlyCollection<OrtValue> _ = ortValues;
            using var ort = ortValues[0];
            return ClassifyTensor(ort, (int)classes);
        }

        #region Classicifation

        /// <summary>
        /// Classifies a tensor and returns a Classification list 
        /// </summary>
        /// <param name="numberOfClasses"></param>
        private List<Classification> ClassifyTensor(OrtValue ortTensor, int numberOfClasses)
        {
            var span = ortTensor.GetTensorDataAsSpan<float>();
            var labels = _yoloCore.OnnxModel.Labels.AsSpan();

            var queue = new PriorityQueue<Classification, float>();

            for (int i = 0; i < span.Length; i++)
            {
                var cls = new Classification
                {
                    Confidence = span[i],
                    Label = labels[i].Name
                };

                var addToQueue = false;

                if (queue.Count < numberOfClasses)
                {
                    addToQueue = true;
                }
                else if (cls.Confidence > queue.Peek().Confidence)
                {
                    addToQueue = true;
                    queue.Dequeue();
                }

                if (addToQueue)
                    queue.Enqueue(cls, (float)cls.Confidence);
            }

            // Dequeue into array
            var result = new Classification[queue.Count];
            for (int i = result.Length - 1; i >= 0; i--)
            {
                result[i] = queue.Dequeue();
            }

            return [.. result];
        }

        #endregion

        #region Helper methods

        public void Dispose()
        {
            _yoloCore.Dispose();

            GC.SuppressFinalize(this);
        }

        #endregion
    }
}
