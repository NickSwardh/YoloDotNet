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
    public class ClassificationImageDrawTests
    {
        #region Fields

        private static string _model = SharedConfig.GetTestModel(modelType: ModelType.Classification);
        private static string _testImage = SharedConfig.GetTestImage(imageType: ImageType.Hummingbird);

        private Yolo _cpuYolo;
        private Image _image;
        private List<Classification> _classifications;

        #endregion Fields

        #region Methods

        [GlobalSetup]
        public void GlobalSetup()
        {
            _cpuYolo = new Yolo(onnxModel: _model, cuda: false);
            _image = Image.Load(path: _testImage);
            _classifications = _cpuYolo.RunClassification(img: _image, classes: 1);
        }

        [Params(true,false)]
        public bool DrawConfidence { get; set; }

        [Benchmark]
        public Image DrawClassification()
        {
            _image.Draw(classifications: _classifications, drawConfidence: DrawConfidence);

            return _image;
        }

        #endregion Methods
    }
}
