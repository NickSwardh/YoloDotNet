namespace YoloDotNet.Benchmarks.OrientedBoundingBoxTests
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
    public class SimpleOrientedBoundingBoxTests
    {
        #region Fields

        private static string model = SharedConfig.GetTestModel(ModelType.ObbDetection);
        private static string testImage = SharedConfig.GetTestImage(ImageType.Island);

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
        public List<OBBDetection> RunSimpleObbDetectionGpu()
        {
            return this.cudaYolo.RunObbDetection(this.image, 0.25, 0.45);
        }

        [Benchmark]
        public List<OBBDetection> RunSimpleObbDetectionCpu()
        {
            return this.cpuYolo.RunObbDetection(this.image, 0.25, 0.45);
        }

        #endregion Methods
    }
}
