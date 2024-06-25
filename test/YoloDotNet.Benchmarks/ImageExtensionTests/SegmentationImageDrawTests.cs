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
            cpuYolo = new Yolo(onnxModel: model, cuda: false);
            image = Image.Load(path: testImage);
            segmentations = cpuYolo.RunSegmentation(img: image, confidence: 0.25, iou: 0.45);
        }

        [Params(true,false)]
        public bool DrawConfidence { get; set; }

        [Benchmark]
        public Image DrawSegmentation()
        {
            image.Draw(segmentations: segmentations, drawConfidence: DrawConfidence);

            return image;
        }

        #endregion Methods
    }
}
