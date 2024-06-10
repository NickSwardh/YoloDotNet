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
            this.cpuYolo = new Yolo(onnxModel: model, cuda: false);
            this.image = Image.Load<Rgba32>(path: testImage).ResizeImage(w: this.cpuYolo.OnnxModel.Input.Width, h: this.cpuYolo.OnnxModel.Input.Height).CloneAs<Rgb24>();

            this.tensorBufferSize = this.cpuYolo.OnnxModel.Input.BatchSize * this.cpuYolo.OnnxModel.Input.Channels * this.cpuYolo.OnnxModel.Input.Width * this.cpuYolo.OnnxModel.Input.Height;
            this.customSizeFloatPool = ArrayPool<float>.Create(maxArrayLength: this.tensorBufferSize + 1, maxArraysPerBucket: 10);
            this.tensorArrayBuffer = this.customSizeFloatPool.Rent(minimumLength: this.tensorBufferSize);
        }

        [GlobalCleanup]
        public void GlobalCleanup()
        {
            this.customSizeFloatPool.Return(array: this.tensorArrayBuffer);
        }

        [Benchmark]
        public DenseTensor<float> NormalizePixelsToTensor()
        {
            return this.image.NormalizePixelsToTensor(
                        inputBatchSize: this.cpuYolo.OnnxModel.Input.BatchSize,
                        inputChannels: this.cpuYolo.OnnxModel.Input.Channels,
                        tensorBufferSize: this.tensorBufferSize,
                        tensorArrayBuffer: tensorArrayBuffer);
        }

        #endregion Methods
    }
}
