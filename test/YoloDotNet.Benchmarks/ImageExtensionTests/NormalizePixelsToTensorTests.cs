namespace YoloDotNet.Benchmarks.ImageExtensionTests
{
    using System.Buffers;

    using SixLabors.ImageSharp;
    using BenchmarkDotNet.Attributes;
    using Microsoft.ML.OnnxRuntime.Tensors;
    using SixLabors.ImageSharp.PixelFormats;

    using YoloDotNet.Enums;
    using YoloDotNet.Extensions;
    using YoloDotNet.Test.Common;
    using YoloDotNet.Test.Common.Enums;

    [MemoryDiagnoser]
    public class NormalizePixelsToTensorTests
    {
        #region Fields

        private static string _model = SharedConfig.GetTestModel(modelType: ModelType.ObjectDetection);
        private static string _testImage = SharedConfig.GetTestImage(imageType: ImageType.Street);
        private ArrayPool<float> _customSizeFloatPool;

        private int _tensorBufferSize;

        private Yolo _cpuYolo;
        private Image<Rgb24> _image;
        private float[] _tensorArrayBuffer;

        #endregion Fields

        #region Methods

        [GlobalSetup]
        public void GlobalSetup()
        {
            _cpuYolo = new Yolo(onnxModel: _model, cuda: false);
            _image = Image.Load<Rgba32>(path: _testImage).ResizeImage(w: _cpuYolo.OnnxModel.Input.Width, h: _cpuYolo.OnnxModel.Input.Height).CloneAs<Rgb24>();

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
        public DenseTensor<float> NormalizePixelsToTensor()
        {
            return _image.NormalizePixelsToTensor(
                        inputBatchSize: _cpuYolo.OnnxModel.Input.BatchSize,
                        inputChannels: _cpuYolo.OnnxModel.Input.Channels,
                        tensorBufferSize: _tensorBufferSize,
                        tensorArrayBuffer: _tensorArrayBuffer);
        }

        #endregion Methods
    }
}
