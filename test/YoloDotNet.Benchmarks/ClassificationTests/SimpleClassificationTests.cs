namespace YoloDotNet.Benchmarks.ClassificationTests
{
    using System.Collections.Generic;

    using SixLabors.ImageSharp;
    using BenchmarkDotNet.Attributes;
    using SixLabors.ImageSharp.PixelFormats;

    using YoloDotNet.Enums;
    using YoloDotNet.Models;
    using YoloDotNet.Test.Common;
    using YoloDotNet.Test.Common.Enums;

    [MemoryDiagnoser]
    public class SimpleClassificationTests
    {
        #region Fields

        private static string _model = SharedConfig.GetTestModel(modelType: ModelType.Classification);
        private static string _testImage = SharedConfig.GetTestImage(imageType: ImageType.Hummingbird);

        private Yolo _cudaYolo;
        private Yolo _cpuYolo;
        private Image _image;

        #endregion Fields

        #region Methods

        [GlobalSetup]
        public void GlobalSetup()
        {
            _cudaYolo = new Yolo(onnxModel: _model, cuda: true);
            _cpuYolo = new Yolo(onnxModel: _model, cuda: false);
            _image = Image.Load<Rgba32>(path: _testImage);
        }

        [Benchmark]
        public List<Classification> RunSimpleClassificationGpu()
        {
            return _cudaYolo.RunClassification(img: _image, classes: 1);
        }

        [Benchmark]
        public List<Classification> RunSimpleClassificationCpu()
        {
            return _cpuYolo.RunClassification(img: _image, classes: 1);
        }

        #endregion Methods
    }
}
