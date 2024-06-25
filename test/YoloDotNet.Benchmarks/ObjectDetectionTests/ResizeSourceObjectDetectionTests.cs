namespace YoloDotNet.Benchmarks.ObjectDetectionTests
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
    public class ResizeSourceObjectDetectionTests
    {
        #region Fields

        private static string _model = SharedConfig.GetTestModel(modelType: ModelType.ObjectDetection);
        private static string _originalSizeimagePath = SharedConfig.GetTestImage(imageType: ImageType.Street);
        private static string _modelSizeimagePath = SharedConfig.GetTestImage(imageName: "street640x640.jpg");

        private Yolo _cudaYolo;
        private Yolo _cpuYolo;
        private Image _originalSizeimage;
        private Image _modelSizeImage;

        #endregion Fields

        #region Methods

        [GlobalSetup]
        public void GlobalSetup()
        {
            _cudaYolo = new Yolo(onnxModel: _model, cuda: true);
            _cpuYolo = new Yolo(onnxModel: _model, cuda: false);
            _originalSizeimage = Image.Load<Rgba32>(path: _originalSizeimagePath);
            _modelSizeImage = Image.Load<Rgba32>(path: _modelSizeimagePath);
        }

        [Benchmark]
        public List<ObjectDetection> ObjectDetectionOriginalSizeGpu()
        {
            return _cudaYolo.RunObjectDetection(img: _originalSizeimage, confidence: 0.25, iou: 0.45);
        }

        [Benchmark]
        public List<ObjectDetection> ObjectDetectionOriginalSizeCpu()
        {
            return _cpuYolo.RunObjectDetection(img: _originalSizeimage, confidence: 0.25, iou: 0.45);
        }

        [Benchmark]
        public List<ObjectDetection> ObjectDetectionModelSizeGpu()
        {
            return _cudaYolo.RunObjectDetection(img: _modelSizeImage, confidence: 0.25, iou: 0.45);
        }

        [Benchmark]
        public List<ObjectDetection> ObjectDetectionModelSizeCpu()
        {
            return _cpuYolo.RunObjectDetection(img: _modelSizeImage, confidence: 0.25, iou: 0.45);
        }

        #endregion Methods
    }
}
