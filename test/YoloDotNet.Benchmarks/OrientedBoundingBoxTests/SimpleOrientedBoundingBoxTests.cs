namespace YoloDotNet.Benchmarks.OrientedBoundingBoxTests
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
    public class SimpleOrientedBoundingBoxTests
    {
        #region Fields

        private static string _model = SharedConfig.GetTestModel(modelType: ModelType.ObbDetection);
        private static string _testImage = SharedConfig.GetTestImage(imageType: ImageType.Island);

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
        public List<OBBDetection> RunSimpleObbDetectionGpu()
        {
            return _cudaYolo.RunObbDetection(img: _image, confidence: 0.25, iou: 0.45);
        }

        [Benchmark]
        public List<OBBDetection> RunSimpleObbDetectionCpu()
        {
            return _cpuYolo.RunObbDetection(img: _image, confidence: 0.25, iou: 0.45);
        }

        #endregion Methods
    }
}
