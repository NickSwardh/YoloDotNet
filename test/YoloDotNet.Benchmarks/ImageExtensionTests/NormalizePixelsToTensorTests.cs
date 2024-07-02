namespace YoloDotNet.Benchmarks.ImageExtensionTests
{
    [MemoryDiagnoser]
    public class NormalizePixelsToTensorTests
    {
        #region Fields

        private static string _model = SharedConfig.GetTestModel(modelType: ModelType.ObjectDetection);
        private static string _testImage = SharedConfig.GetTestImage(imageType: ImageType.Street);
        private ArrayPool<float> _customSizeFloatPool;

        private int _tensorBufferSize;

        private Yolo _cpuYolo;

        private SKImage _image;
        private float[] _tensorArrayBuffer;

        private static SKBitmap _resizedBitmap;

        private SKImageInfo _skImageInfo;

        #endregion Fields

        #region Methods

        [GlobalSetup]
        public void GlobalSetup()
        {
            _cpuYolo = new Yolo(onnxModel: _model, cuda: false);
            _image = SKImage.FromEncodedData(_testImage);

            _skImageInfo = new SKImageInfo(_cpuYolo.OnnxModel.Input.Width, _cpuYolo.OnnxModel.Input.Height, SKColorType.Rgb888x, SKAlphaType.Opaque);

            _resizedBitmap = _image.ResizeImage(_skImageInfo);

            _tensorBufferSize = _cpuYolo.OnnxModel.Input.BatchSize * _cpuYolo.OnnxModel.Input.Channels * _cpuYolo.OnnxModel.Input.Width * _cpuYolo.OnnxModel.Input.Height;
            _customSizeFloatPool = ArrayPool<float>.Create(maxArrayLength: _tensorBufferSize + 1, maxArraysPerBucket: 10);
            _tensorArrayBuffer = _customSizeFloatPool.Rent(minimumLength: _tensorBufferSize);
        }

        [GlobalCleanup]
        public void GlobalCleanup()
        {
            _customSizeFloatPool.Return(array: _tensorArrayBuffer);
        }

        [Benchmark]
        public void NormalizePixelsToTensor()
        {
            _ = _resizedBitmap.NormalizePixelsToTensor(_cpuYolo.OnnxModel.InputShape, _tensorBufferSize, _tensorArrayBuffer);
        }

        #endregion Methods
    }
}
