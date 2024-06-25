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

        private static string _model = SharedConfig.GetTestModel(modelType: ModelType.PoseEstimation);
        private static string _testImage = SharedConfig.GetTestImage(imageType: ImageType.Crosswalk);

        private Yolo _cpuYolo;
        private Image _image;
        private List<PoseEstimation> _poseEstimations;

        #endregion Fields

        #region Methods

        [GlobalSetup]
        public void GlobalSetup()
        {
            _cpuYolo = new Yolo(onnxModel: _model, cuda: false);
            _image = Image.Load(path: _testImage);
            _poseEstimations = _cpuYolo.RunPoseEstimation(img: _image, confidence: 0.25, iou: 0.45);
        }

        [Params(true,false)]
        public bool DrawConfidence { get; set; }

        [Benchmark]
        public Image DrawPoseEstimation()
        {
            _image.Draw(segmentations: _poseEstimations, CustomPoseMarkerColorMap.PoseMarkerOptions, drawConfidence: DrawConfidence);

            return _image;
        }

        #endregion Methods
    }
}
