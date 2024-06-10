namespace YoloDotNet.Benchmarks.ImageExtensionTests
{
    using SixLabors.ImageSharp;
    using BenchmarkDotNet.Attributes;
    using SixLabors.ImageSharp.PixelFormats;

    using YoloDotNet.Enums;
    using YoloDotNet.Extensions;

    [MemoryDiagnoser]
    public class ResizeImageTests
    {
        #region Fields

        private static string model = SharedConfig.GetTestModel(modelType: ModelType.ObjectDetection);
        private static string testImage = SharedConfig.GetTestImage(imageName: "street640x640.jpg");

        private Yolo cpuYolo;
        private Image image;

        #endregion Fields

        #region Methods

        [GlobalSetup]
        public void GlobalSetup()
        {
            this.cpuYolo = new Yolo(onnxModel: model, cuda: false);
            this.image = Image.Load(path: testImage);
        }

        [Benchmark]
        public Image<Rgb24> ResizeImage()
        {
            return this.image.ResizeImage(
                        w: this.cpuYolo.OnnxModel.Input.BatchSize,
                        h: this.cpuYolo.OnnxModel.Input.Channels);
        }

        #endregion Methods
    }
}
