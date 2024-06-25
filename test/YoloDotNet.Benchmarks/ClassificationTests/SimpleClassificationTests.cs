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

        private static string model = SharedConfig.GetTestModel(modelType: ModelType.Classification);
        private static string testImage = SharedConfig.GetTestImage(imageType: ImageType.Hummingbird);

        private Yolo cudaYolo;
        private Yolo cpuYolo;
        private Image image;

        #endregion Fields

        #region Methods

        [GlobalSetup]
        public void GlobalSetup()
        {
            cudaYolo = new Yolo(onnxModel: model, cuda: true);
            cpuYolo = new Yolo(onnxModel: model, cuda: false);
            image = Image.Load<Rgba32>(path: testImage);
        }

        [Benchmark]
        public List<Classification> RunSimpleClassificationGpu()
        {
            return cudaYolo.RunClassification(img: image, classes: 1);
        }

        [Benchmark]
        public List<Classification> RunSimpleClassificationCpu()
        {
            return cpuYolo.RunClassification(img: image, classes: 1);
        }

        #endregion Methods
    }
}
