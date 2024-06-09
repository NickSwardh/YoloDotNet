namespace YoloDotNet.Benchmarks.SegmentationTests
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
    public class SimpleSegmentationTests
    {
        #region Fields

        private static string model = SharedConfig.GetTestModel(ModelType.Segmentation);
        private static string testImage = SharedConfig.GetTestImage(ImageType.People);

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
        public List<Segmentation> RunSimpleSegmentationGpu()
        {
            return this.cudaYolo.RunSegmentation(this.image, 0.25, 0.45);
        }

        [Benchmark]
        public List<Segmentation> RunSimpleSegmentationCpu()
        {
            return this.cpuYolo.RunSegmentation(this.image, 0.25, 0.45);
        }

        #endregion Methods
    }
}
