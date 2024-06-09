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

        private static string model = SharedConfig.GetTestModel(ModelType.ObjectDetection);
        private static string originalSizeimagePath = SharedConfig.GetTestImage(ImageType.Street);
        private static string modelSizeimagePath = SharedConfig.GetTestImage("street640x640.jpg");

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
            this.originalSizeimage = Image.Load<Rgba32>(originalSizeimagePath);
            this.modelSizeImage = Image.Load<Rgba32>(modelSizeimagePath);
        }

        [Benchmark]
        public List<ObjectDetection> ObjectDetectionOriginalSizeGpu()
        {
            return this.cudaYolo.RunObjectDetection(this.originalSizeimage, 0.25, 0.45);
        }

        [Benchmark]
        public List<ObjectDetection> ObjectDetectionOriginalSizeCpu()
        {
            return this.cpuYolo.RunObjectDetection(this.originalSizeimage, 0.25, 0.45);
        }

        [Benchmark]
        public List<ObjectDetection> ObjectDetectionModelSizeGpu()
        {
            return this.cudaYolo.RunObjectDetection(this.modelSizeImage, 0.25, 0.45);
        }

        [Benchmark]
        public List<ObjectDetection> ObjectDetectionModelSizeCpu()
        {
            return this.cpuYolo.RunObjectDetection(this.modelSizeImage, 0.25, 0.45);
        }

        #endregion Methods
    }
}
