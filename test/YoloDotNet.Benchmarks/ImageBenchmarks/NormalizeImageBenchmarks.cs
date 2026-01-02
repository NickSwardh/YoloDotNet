namespace YoloDotNet.Benchmarks.ImageBenchmarks
{
    //[CPUUsageDiagnoser]
    [MemoryDiagnoser]
    public class NormalizeImageBenchmarks
    {
        private readonly string _testImage = SharedConfig.GetTestImage(ImageType.Hummingbird);

        private Yolo _yolo;
        private PinnedMemoryBuffer _pinnedMemoryBuffer;
        private SKBitmap _image;
        private readonly SKSamplingOptions _samplingOptions;

        private long[] _inputShape;
        private int _inputShapeSize;

        [GlobalSetup]
        public void GlobalSetup()
        {
            // Initialize a Yolo object detection 11 model
            _yolo = YoloCreator.Create(YoloType.V11_Obj_CPU);

            // Create a pinned memory buffer for the model input size
            var imageInfo = new SKImageInfo(_yolo.OnnxModel.Input.Width, _yolo.OnnxModel.Input.Height, SKColorType.Rgb888x, SKAlphaType.Opaque);
            _pinnedMemoryBuffer = new PinnedMemoryBuffer(imageInfo);

            // Load and resize the test image.
            // Resize it proportionally to fit the model input size and stored in the pinned memory buffer.
            _image = SKBitmap.Decode(_testImage);
            _image.ResizeImageProportional(_samplingOptions, _pinnedMemoryBuffer);

            // Store input shape and size for normalization
            _inputShape = _yolo.OnnxModel.InputShape;
            _inputShapeSize = _yolo.OnnxModel.InputShapeSize;
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
