namespace YoloDotNet.Benchmarks.ImageExtensionTests
{
    using System.Buffers;

    using SixLabors.ImageSharp;
    using BenchmarkDotNet.Attributes;
    using Microsoft.ML.OnnxRuntime.Tensors;
    using SixLabors.ImageSharp.PixelFormats;

    using YoloDotNet.Enums;
    using YoloDotNet.Extensions;
    using YoloDotNet.Test.Common.Enums;

    [MemoryDiagnoser]
    public class NormalizePixelsToTensorTests
    {
        #region Fields

        private static string model = SharedConfig.GetTestModel(modelType: ModelType.ObjectDetection);
        private static string testImage = SharedConfig.GetTestImage(imageType: ImageType.Street);
        private ArrayPool<float> customSizeFloatPool;

        private int tensorBufferSize;

        private Yolo cpuYolo;
        private Image<Rgb24> image;
        private float[] tensorArrayBuffer;

        #endregion Fields

        #region Methods

        [GlobalSetup]
        public void GlobalSetup()
        {
            cpuYolo = new Yolo(onnxModel: model, cuda: false);
            image = Image.Load<Rgba32>(path: testImage).ResizeImage(w: cpuYolo.OnnxModel.Input.Width, h: cpuYolo.OnnxModel.Input.Height).CloneAs<Rgb24>();

            tensorBufferSize = cpuYolo.OnnxModel.Input.BatchSize * cpuYolo.OnnxModel.Input.Channels * cpuYolo.OnnxModel.Input.Width * cpuYolo.OnnxModel.Input.Height;
            customSizeFloatPool = ArrayPool<float>.Create(maxArrayLength: tensorBufferSize + 1, maxArraysPerBucket: 10);
            tensorArrayBuffer = customSizeFloatPool.Rent(minimumLength: tensorBufferSize);
        }

        [GlobalCleanup]
        public void GlobalCleanup()
        {
            customSizeFloatPool.Return(array: tensorArrayBuffer);
        }

        [Benchmark]
        public DenseTensor<float> NormalizePixelsToTensor()
        {
            return image.NormalizePixelsToTensor(
                        inputBatchSize: cpuYolo.OnnxModel.Input.BatchSize,
                        inputChannels: cpuYolo.OnnxModel.Input.Channels,
                        tensorBufferSize: tensorBufferSize,
                        tensorArrayBuffer: tensorArrayBuffer);
        }

        #endregion Methods
    }
}
