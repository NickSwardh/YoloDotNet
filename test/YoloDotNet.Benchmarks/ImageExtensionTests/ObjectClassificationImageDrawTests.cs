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
    public class ObjectClassificationImageDrawTests
    {
        #region Fields

        private static string model = SharedConfig.GetTestModel(modelType: ModelType.ObjectDetection);
        private static string testImage = SharedConfig.GetTestImage(imageType: ImageType.Street);

        private Yolo cpuYolo;
        private Image image;
        private List<ObjectDetection> objectDetections;

        #endregion Fields

        #region Methods

        [GlobalSetup]
        public void GlobalSetup()
        {
            this.cpuYolo = new Yolo(onnxModel: model, cuda: false);
            this.image = Image.Load(path: testImage);
            this.objectDetections = cpuYolo.RunObjectDetection(img: this.image, confidence: 0.25, iou: 0.45);
        }

        [Params(true,false)]
        public bool DrawConfidence { get; set; }

        [Benchmark]
        public Image DrawObjectDetection()
        {
            this.image.Draw(detections: this.objectDetections, drawConfidence: this.DrawConfidence);

            return this.image;
        }

        #endregion Methods
    }
}
