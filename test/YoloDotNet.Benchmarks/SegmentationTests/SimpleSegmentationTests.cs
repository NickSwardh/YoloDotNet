namespace YoloDotNet.Benchmarks.SegmentationTests
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
    public class SimpleSegmentationTests
    {
        #region Fields

        private static string _model = SharedConfig.GetTestModel(modelType: ModelType.Segmentation);
        private static string _testImage = SharedConfig.GetTestImage(imageType: ImageType.People);

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
        public List<Segmentation> RunSimpleSegmentationGpu()
        {
            return _cudaYolo.RunSegmentation(img: _image, confidence: 0.25, iou: 0.45);
        }

        [Benchmark]
        public List<Segmentation> RunSimpleSegmentationCpu()
        {
            return _cpuYolo.RunSegmentation(img: _image, confidence: 0.25, iou: 0.45);
        }

        #endregion Methods
    }
}
