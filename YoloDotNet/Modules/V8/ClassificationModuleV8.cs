// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023-2025 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

namespace YoloDotNet.Modules.V8
{
    internal class ClassificationModuleV8 : IClassificationModule
    {
        private readonly YoloCore _yoloCore;

        public OnnxModel OnnxModel => _yoloCore.OnnxModel;
        private ArrayPool<ClassificationEntry> _classificationPool = default!;

        public ClassificationModuleV8(YoloCore yoloCore)
        {
            _yoloCore = yoloCore;

            var poolSize = YoloCore.CalculateBufferPoolSize(_yoloCore.OnnxModel.Labels.Length);
            _classificationPool = ArrayPool<ClassificationEntry>.Create(poolSize, 10);
        }

        public List<Classification> ProcessImage<T>(T image, double classes, double pixelConfidence, double iou)
        {
            var inferenceResult = _yoloCore.Run(image);

            return ClassifyTensor(inferenceResult.OrtSpan0, (int)classes);
        }

        #region Classicifation

        /// <summary>
        /// Classifies a tensor and returns a Classification list 
        /// </summary>
        /// <param name="numberOfClasses"></param>
        private List<Classification> ClassifyTensor(ReadOnlySpan<float> span, int numberOfClasses)
        {
            var poolBuffer = _classificationPool.Rent(span.Length);

            try
            {
                // Fill poolbuffer with confidence and labelId
                for (int i = 0; i < span.Length; i++)
                {
                    poolBuffer[i] = new ClassificationEntry(span[i], i);
                }

                // Sort descending by confidence
                Array.Sort(poolBuffer, (a, b) => b.Confidence.CompareTo(a.Confidence));

                // Take the top-N classes based on numberOfClasses
                var results = new List<Classification>(numberOfClasses);
                for (int i = 0; i < numberOfClasses; i++)
                {
                    var entry = poolBuffer[i];
                    results.Add(new Classification
                    {
                        Confidence = entry.Confidence,
                        Label = _yoloCore.OnnxModel.Labels[entry.LabelId].Name
                    });
                }

                return results;
            }
            finally
            {
                _classificationPool.Return(array: poolBuffer, clearArray: true);
            }
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

    internal readonly struct ClassificationEntry(float confidence, int labelId)
    {
        public readonly float Confidence = confidence;  // confidence score
        public readonly int LabelId = labelId;          // index into labels array
    }
}
