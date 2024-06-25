namespace YoloDotNet.Benchmarks.ImageExtensionTests
{
    using System.Collections.Generic;

    using SixLabors.ImageSharp;
    using BenchmarkDotNet.Attributes;

    using YoloDotNet.Enums;
    using YoloDotNet.Models;
    using YoloDotNet.Extensions;
    using YoloDotNet.Test.Common;
    using YoloDotNet.Test.Common.Enums;

    [MemoryDiagnoser]
    public class PoseEstimationImageDrawTests
    {
        #region Fields

        private static string model = SharedConfig.GetTestModel(modelType: ModelType.PoseEstimation);
        private static string testImage = SharedConfig.GetTestImage(imageType: ImageType.Crosswalk);

        private Yolo cpuYolo;
        private Image image;
        private List<PoseEstimation> poseEstimations;

        #endregion Fields

        #region Methods

        [GlobalSetup]
        public void GlobalSetup()
        {
            cpuYolo = new Yolo(onnxModel: model, cuda: false);
            image = Image.Load(path: testImage);
            poseEstimations = cpuYolo.RunPoseEstimation(img: image, confidence: 0.25, iou: 0.45);
        }

        [Params(true,false)]
        public bool DrawConfidence { get; set; }

        [Benchmark]
        public Image DrawPoseEstimation()
        {
            image.Draw(segmentations: poseEstimations, CustomPoseMarkerColorMap.PoseMarkerOptions, drawConfidence: DrawConfidence);

            return image;
        }

        #endregion Methods
    }
}
