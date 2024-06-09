namespace YoloDotNet.Benchmarks.ClassificationTests
{
    using System.Collections.Generic;

    using SixLabors.ImageSharp;
    using BenchmarkDotNet.Attributes;
    using SixLabors.ImageSharp.PixelFormats;

    using YoloDotNet.Enums;
    using YoloDotNet.Models;
    using YoloDotNet.Benchmarks;
    using YoloDotNet.Test.Common.Enums;

    [MemoryDiagnoser]
    public class SimpleClassificationTests
    {
        #region Fields

        private static string model = SharedConfig.GetTestModel(ModelType.Classification);
        private static string testImage = SharedConfig.GetTestImage(ImageType.Hummingbird);

        private Yolo cudaYolo;
        private Yolo cpuYolo;
        private Image image;

        #endregion Fields

        #region Methods

        [GlobalSetup]
        public void GlobalSetup()
        {
            this.cudaYolo = new Yolo(onnxModel: model, cuda: true);
            this.cpuYolo = new Yolo(onnxModel: model, cuda: false);
            this.image = Image.Load<Rgba32>(testImage);
        }

        [Benchmark]
        public List<Classification> RunSimpleClassificationGpu()
        {
            return this.cudaYolo.RunClassification(this.image, 1);
        }

        [Benchmark]
        public List<Classification> RunSimpleClassificationCpu()
        {
            return this.cpuYolo.RunClassification(this.image, 1);
        }

        #endregion Methods
    }
}
