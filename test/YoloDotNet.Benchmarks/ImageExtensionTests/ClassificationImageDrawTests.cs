namespace YoloDotNet.Benchmarks.ImageExtensionTests
{
    using System.Collections.Generic;

    using SixLabors.ImageSharp;
    using BenchmarkDotNet.Attributes;

    using YoloDotNet.Enums;
    using YoloDotNet.Models;
    using YoloDotNet.Extensions;
    using YoloDotNet.Test.Common.Enums;

    [MemoryDiagnoser]
    public class ClassificationImageDrawTests
    {
        #region Fields

        private static string model = SharedConfig.GetTestModel(modelType: ModelType.Classification);
        private static string testImage = SharedConfig.GetTestImage(imageType: ImageType.Hummingbird);

        private Yolo cpuYolo;
        private Image image;
        private List<Classification> classifications;

        #endregion Fields

        #region Methods

        [GlobalSetup]
        public void GlobalSetup()
        {
            this.cpuYolo = new Yolo(onnxModel: model, cuda: false);
            this.image = Image.Load(path: testImage);
            this.classifications = this.cpuYolo.RunClassification(img: this.image, classes: 1);
        }

        [Params(true,false)]
        public bool DrawConfidence { get; set; }

        [Benchmark]
        public Image DrawClassification()
        {
            this.image.Draw(classifications: this.classifications, drawConfidence: this.DrawConfidence);

            return this.image;
        }

        #endregion Methods
    }
}
