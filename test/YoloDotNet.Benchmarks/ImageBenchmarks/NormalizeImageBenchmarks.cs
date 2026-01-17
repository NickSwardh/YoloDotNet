// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2025 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

using System.Linq;

namespace YoloDotNet.Benchmarks.ImageBenchmarks
{
    //[CPUUsageDiagnoser]
    [MemoryDiagnoser]
    public class NormalizeImageBenchmarks
    {
        private readonly string _testImage = SharedConfig.GetTestImage(ImageType.Classification);

        private Yolo _yolo;
        private PinnedMemoryBuffer _pinnedMemoryBuffer;
        private SKBitmap _image;
        private SKSamplingOptions _samplingOptions;

        private long[] _inputShape;
        private int _inputShapeSize;

        [GlobalSetup]
        public void GlobalSetup()
        {
            // Initialize a Yolo object detection 11 model
            _yolo = YoloCreator.Create(YoloType.V11_Obj_CPU);

            // Store input shape and size for normalization
            _inputShape = _yolo.OnnxModel.InputShapes.First().Value;
            _inputShapeSize = _yolo.OnnxModel.InputShapeSize;
            var width = (int)_inputShape[3];
            var height = (int)_inputShape[2];

            // Create a pinned memory buffer for the model input size
            var imageInfo = new SKImageInfo(width, height, SKColorType.Rgb888x, SKAlphaType.Opaque);
            _pinnedMemoryBuffer = new PinnedMemoryBuffer(imageInfo);

            // Load and resize the test image.
            // Resize it proportionally to fit the model input size and stored in the pinned memory buffer.
            _image = SKBitmap.Decode(_testImage);
            _image.ResizeImageProportional(_samplingOptions, _pinnedMemoryBuffer);
        }

        [GlobalCleanup]
        public void GlobalCleanup()
        {
            _yolo?.Dispose();
            _image?.Dispose();
            _pinnedMemoryBuffer?.Dispose();
        }

        [Benchmark]
        public void NormalizeImage()
        {
            // Rent a buffer from the array pool
            var normalizedPixelsFloatBuffer = ArrayPool<float>.Shared.Rent(_inputShapeSize);

            try
            {
                // Normalize the pixels in the pinned memory buffer to the rented float array
                _pinnedMemoryBuffer.Pointer.NormalizePixelsToArray(
                    _inputShape,
                    _inputShapeSize,
                    normalizedPixelsFloatBuffer);
            }
            finally
            {
                // Return the rented buffer to the array pool
                ArrayPool<float>.Shared.Return(normalizedPixelsFloatBuffer, false);
            }
        }
    }
}
