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
    public class SegmentationImageDrawTests
    {
        #region Fields

        private static string model = SharedConfig.GetTestModel(modelType: ModelType.Segmentation);
        private static string testImage = SharedConfig.GetTestImage(imageType: ImageType.People);

        private Yolo cpuYolo;
        private Image image;
        private List<Segmentation> segmentations;

        #endregion Fields

        #region Methods

        [GlobalSetup]
        public void GlobalSetup()
        {
            this.cpuYolo = new Yolo(onnxModel: model, cuda: false);
            this.image = Image.Load(path: testImage);
            this.segmentations = this.cpuYolo.RunSegmentation(img: this.image, confidence: 0.25, iou: 0.45);
        }

        [Params(true,false)]
        public bool DrawConfidence { get; set; }

        [Benchmark]
        public Image DrawSegmentation()
        {
            this.image.Draw(segmentations: this.segmentations, drawConfidence: this.DrawConfidence);

            return this.image;
        }

        #endregion Methods
    }
}
