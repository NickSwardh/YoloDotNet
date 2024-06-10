namespace YoloDotNet.Benchmarks.ObjectDetectionTests
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
    public class ResizeSourceObjectDetectionTests
    {
        #region Fields

        private static string model = SharedConfig.GetTestModel(modelType: ModelType.ObjectDetection);
        private static string originalSizeimagePath = SharedConfig.GetTestImage(imageType: ImageType.Street);
        private static string modelSizeimagePath = SharedConfig.GetTestImage(imageName: "street640x640.jpg");

        private Yolo cudaYolo;
        private Yolo cpuYolo;
        private Image originalSizeimage;
        private Image modelSizeImage;

        #endregion Fields

        #region Methods

        [GlobalSetup]
        public void GlobalSetup()
        {
            this.cudaYolo = new Yolo(onnxModel: model, cuda: true);
            this.cpuYolo = new Yolo(onnxModel: model, cuda: false);
            this.originalSizeimage = Image.Load<Rgba32>(path: originalSizeimagePath);
            this.modelSizeImage = Image.Load<Rgba32>(path: modelSizeimagePath);
        }

        [Benchmark]
        public List<ObjectDetection> ObjectDetectionOriginalSizeGpu()
        {
            return this.cudaYolo.RunObjectDetection(img: this.originalSizeimage, confidence: 0.25, iou: 0.45);
        }

        [Benchmark]
        public List<ObjectDetection> ObjectDetectionOriginalSizeCpu()
        {
            return this.cpuYolo.RunObjectDetection(img: this.originalSizeimage, confidence: 0.25, iou: 0.45);
        }

        [Benchmark]
        public List<ObjectDetection> ObjectDetectionModelSizeGpu()
        {
            return this.cudaYolo.RunObjectDetection(img: this.modelSizeImage, confidence: 0.25, iou: 0.45);
        }

        [Benchmark]
        public List<ObjectDetection> ObjectDetectionModelSizeCpu()
        {
            return this.cpuYolo.RunObjectDetection(img: this.modelSizeImage, confidence: 0.25, iou: 0.45);
        }

        #endregion Methods
    }
}
