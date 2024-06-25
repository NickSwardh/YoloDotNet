namespace YoloDotNet.Benchmarks.ImageExtensionTests
{
    [MemoryDiagnoser]
    public class SegmentationImageDrawTests
    {
        #region Fields

        private static string _model = SharedConfig.GetTestModel(modelType: ModelType.Segmentation);
        private static string _testImage = SharedConfig.GetTestImage(imageType: ImageType.People);

        private Yolo _cpuYolo;
        private Image _image;
        private List<Segmentation> _segmentations;

        #endregion Fields

        #region Methods

        [GlobalSetup]
        public void GlobalSetup()
        {
            _cpuYolo = new Yolo(onnxModel: _model, cuda: false);
            _image = Image.Load(path: _testImage);
            _segmentations = _cpuYolo.RunSegmentation(img: _image, confidence: 0.25, iou: 0.45);
        }

        [Params(true,false)]
        public bool DrawConfidence { get; set; }

        [Benchmark]
        public Image DrawSegmentation()
        {
            _image.Draw(segmentations: _segmentations, drawConfidence: DrawConfidence);

            return _image;
        }

        #endregion Methods
    }
}
