namespace YoloDotNet.Benchmarks.PoseEstimationTests
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
    public class SimplePostEstimationTests
    {
        #region Fields

        private static string _model = SharedConfig.GetTestModel(modelType: ModelType.PoseEstimation);
        private static string _testImage = SharedConfig.GetTestImage(imageType: ImageType.Crosswalk);

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
        public List<PoseEstimation> RunSimplePoseEstimationGpu()
        {
            return _cudaYolo.RunPoseEstimation(img: _image, confidence: 0.25, iou: 0.45);
        }

        [Benchmark]
        public List<PoseEstimation> RunSimplePoseEstimationCpu()
        {
            return _cpuYolo.RunPoseEstimation(img: _image, confidence: 0.25, iou: 0.45);
        }

        #endregion Methods
    }
}
